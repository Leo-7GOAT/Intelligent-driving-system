"""TRAIN-only native teacher/learner collection; never a policy evaluation."""
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


VERSION = 'native_executed_teacher_mixture_v1'


def validate_collection(protocol, config, cases):
    spec = protocol.get('native_recovery_collection', {})
    if (spec.get('version') != VERSION or protocol.get('methods') != ['B1']
            or protocol.get('evaluation_only') is not True
            or protocol.get('deterministic_training_route_evaluation') is not False
            or protocol.get('method_budgets') != {'B1': 0}
            or protocol.get('controlled_budget_per_method') != 0
            or config.method != 'B1' or config.flat_hold != 1
            or config.flat_longitudinal_mode != 'acceleration'
            or spec.get('teacher_probability') != .5 or spec.get('block_controls') != 10
            or spec.get('max_controls') != 3600 or config.deadline != 45
            or len(cases) != 4 or len(spec.get('geometry_ids', [])) != 4
            or spec.get('original_TRAIN_route_indices') != [4, 5, 6, 7]
            or [c['native_route']['geometry_id'] for c in cases] != spec['geometry_ids']
            or any(c.get('native_hazard') is not None for c in cases)):
        raise ValueError('Require the locked four-road TRAIN-only collection protocol')
    return spec


def tensor_digest(value):
    """Stable recursive fingerprint for policy, optimizer and frozen RMS state."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for part in item: visit(part)
        elif hasattr(item, 'detach'):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode()); digest.update(item.tobytes())
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def frozen_state(model, vec):
    rms = vec.__dict__['obs_rms']
    return tensor_digest({'policy': model.policy.state_dict(), 'optimizer': model.policy.optimizer.state_dict(),
        'timesteps': model.num_timesteps, 'updates': model._n_updates,
        'mean': rms.mean, 'var': rms.var, 'count': rms.count})


def collect(config, client, cases, model, vec, output, protocol, env_class, guard):
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv, rule_action
    from navstack.experiments import jsonable
    from navstack.imitation import command_action
    from navstack.mpc import MPC
    from navstack.research import atomic_json
    from navstack.vehicle import brake
    from scripts.query_native_persistent_teacher import HeldTeacher

    spec = validate_collection(protocol, config, cases)
    output = Path(output)
    teacher_path = output.parent / spec['teacher_config']
    if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != spec['teacher_config_sha256']:
        raise ValueError('Registered teacher configuration changed')
    teacher = EnvConfig.from_dict(json.loads(teacher_path.read_text(encoding='utf-8'))['env'])
    if (teacher.method != 'H' or teacher.vehicle != config.vehicle or teacher.reward != config.reward
            or teacher.control_dt != config.control_dt or teacher.decision_dt != config.decision_dt
            or teacher.road_geometry_version != config.road_geometry_version
            or teacher.planner.max_seconds is not None):
        raise ValueError('Teacher physical interface differs from frozen learner')
    vec.training = False
    before = frozen_state(model, vec)
    rng = np.random.default_rng(spec['mixture_seed'])
    environment = env_class(config, client, cases, output)
    observations, labels, groups, eligible = [], [], [], []
    learner_actions, sources = [], []
    episodes = []
    controls = teacher_controls = valid_labels = 0
    query_seconds = 0.
    started = time.monotonic(); initial_ticks = client.total_ticks
    trace_path = output / 'collection_trace.jsonl.gz'
    with gzip.open(trace_path, 'xt', encoding='utf-8') as stream:
        try:
            for case_id, case in enumerate(cases):
                guard()
                obs, _ = environment.reset(seed=spec['reset_seed'] + case_id, options={'case': case})
                expert = NavigationEnv(teacher)
                # The private planner reads the actual plant. It never resets,
                # integrates, advances the world, or sends a CARLA command.
                expert.world, expert.plant, expert.goal = environment.world, environment.plant, environment.goal
                expert.navigation_route = environment.navigation_route
                expert.last_behavior = np.zeros(4); expert.execution_outcome = np.zeros(6, dtype=np.float32)
                expert.reference = None; expert._reference_selection = None; expert._rule_plan_cache = None
                expert.episode = {'plan_times': [], 'plan_failures': 0}
                expert.controller = MPC(teacher.vehicle, teacher.control_dt, teacher.mpc_horizon)
                oracle = HeldTeacher(expert, expert.controller, rule_action, brake)
                index = 0; use_teacher = False; local_teacher = 0
                while True:
                    if controls >= spec['max_controls']:
                        raise RuntimeError('Locked maximum reached before all episodes terminated')
                    if index % 20 == 0:
                        guard()
                        atomic_json(output / 'collection_heartbeat.json', {'pid': os.getpid(), 'epoch': time.time(),
                            'phase': 'TRAIN_ONLY_MIXTURE_COLLECTION', 'case_id': case_id,
                            'collection_controls': controls, 'new_PPO_controls': 0, 'new_optimizer_steps': 0})
                    if index % spec['block_controls'] == 0:
                        use_teacher = bool(rng.random() < spec['teacher_probability'])
                    state = asdict(environment.state); frame = environment.plant.frame
                    previous = asdict(environment.plant.previous_command)
                    learner, _ = model.predict(vec.normalize_obs(obs[None]), deterministic=True)
                    learner = np.asarray(learner[0], dtype=np.float32)
                    queried = time.monotonic(); command, status = oracle.query(index)
                    elapsed = time.monotonic() - queried; query_seconds += elapsed
                    teacher_action = command_action(asdict(command), config.vehicle)
                    available = status == 'solved' or oracle.mode in ('wait', 'dynamic_brake', 'parking_brake')
                    action = teacher_action if use_teacher else learner
                    next_obs, reward, done, truncated, info = environment.step(action)
                    if len(info['trace']) != 1:
                        raise ValueError('Exactly one measured physical control per mixture action is required')
                    record = {'case_id': case_id, 'control_index': index,
                        'source': 'teacher' if use_teacher else 'learner',
                        'eligible_executed_teacher_label': bool(use_teacher and available),
                        'state_before': state, 'frame_before': frame, 'previous_command': previous,
                        'learner_action': learner, 'teacher_action': teacher_action, 'executed_action': action,
                        'teacher_requested_command': asdict(command), 'teacher_action_id': oracle.action,
                        'teacher_mode': oracle.mode, 'teacher_status': status,
                        'teacher_query_seconds': elapsed,
                        'reference_start_time': getattr(expert.reference, 'start_time', None),
                        'reward': reward, 'info': info}
                    stream.write(json.dumps(record, default=jsonable) + '\n')
                    if index % 20 == 0: stream.flush()
                    observations.append(obs.copy()); labels.append(teacher_action.copy()); groups.append(2048 + case_id)
                    eligible.append(bool(use_teacher and available)); learner_actions.append(learner.copy()); sources.append(use_teacher)
                    controls += 1; teacher_controls += int(use_teacher); local_teacher += int(use_teacher)
                    valid_labels += int(use_teacher and available); index += 1; obs = next_obs
                    if done or truncated:
                        episodes.append({'case_id': case_id, 'geometry_id': case['native_route']['geometry_id'],
                            'teacher_controls': local_teacher, 'learner_controls': index-local_teacher,
                            **info['episode_metrics']})
                        atomic_json(output / 'collection_progress.json', {'episodes': episodes, 'controls': controls})
                        print(json.dumps({'TRAIN_collection_case': case_id, 'controls': index,
                            'teacher_controls': local_teacher, 'termination': info['termination_reason']}), flush=True)
                        break
                # NavigationEnv.close is currently a no-op. Do not close the
                # shared plant via an expert cleanup path.
        finally:
            environment.close()
    after = frozen_state(model, vec)
    if before != after:
        raise ValueError('Mixture collection changed learner, PPO optimizer/counters or normalization')
    dataset = output / 'collection.npz'
    np.savez_compressed(dataset, observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(labels, dtype=np.float32), groups=np.asarray(groups, dtype=np.int64),
        fit_train_mask=np.asarray(eligible, dtype=bool), teacher_executed=np.asarray(sources, dtype=bool),
        learner_actions=np.asarray(learner_actions, dtype=np.float32))
    result = {'scope': __doc__, 'version': VERSION, 'partition': 'TRAIN', 'episodes': episodes,
        'collection_controls': controls, 'teacher_controls': teacher_controls,
        'learner_controls': controls-teacher_controls, 'eligible_executed_teacher_labels': valid_labels,
        'new_PPO_controls': 0, 'new_PPO_updates': 0, 'new_supervised_updates': 0,
        'collection_world_ticks': client.total_ticks-initial_ticks,
        'teacher_query_seconds': query_seconds, 'wall_seconds': time.monotonic()-started,
        'frozen_state_sha256_before': before, 'frozen_state_sha256_after': after,
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256((output/'model.zip').read_bytes()).hexdigest(),
        'normalization_sha256': hashlib.sha256((output/'normalization.pkl').read_bytes()).hexdigest(),
        'label_rule': 'Only fit_train_mask=True rows: valid teacher commands actually selected and physically executed. Other labels are counterfactual diagnostics only.',
        'performance_scope': 'Mixture outcomes are training data, never learned-policy evaluation or final confirmation.'}
    atomic_json(output/'collection.json', result)
    return {key: result[key] for key in ('collection_controls', 'collection_world_ticks', 'teacher_controls', 'learner_controls')}
