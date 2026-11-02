"""Join every TRAIN mixture control to CARLA, reconstruct inputs and source choices."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys


def materialize_dataset(path):
    """Decompress each NPZ member once, before the per-control audit loop."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def audit(batch):
    batch = Path(batch).resolve(); folder = batch/'B1'; runtime = batch/'source'
    if any(key.startswith('navstack.') for key in sys.modules):
        raise RuntimeError('Audit requires a fresh frozen-runtime interpreter')
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.measured_demo import measured_flat_env, measured_label, _actor_snapshot
    from navstack.world import MovingObstacle
    from navstack.parking_assist import goal_parking_command
    from navstack.native_cases import registered_native_cases, verify_registered_native_begin
    from navstack.vehicle import VehicleState, Command
    from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks
    from scripts.collect_native_recovery import validate_collection, frozen_state
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(batch)
    protocol = read(batch/'protocol.json'); result = read(folder/'collection.json'); summary = read(folder/'summary.json')
    config = EnvConfig.from_dict(read(folder/'config.json')['env'])
    cases = registered_native_cases(protocol); spec = validate_collection(protocol,config,cases)
    inventory = batch / spec['TRAIN_inventory']
    if sha(inventory) != spec['TRAIN_inventory_sha256']:
        raise ValueError('Original TRAIN inventory changed')
    if spec.get('route_profile') == 'broader_TRAIN36_v1':
        original = read(inventory)
        if original['routes'] != protocol['routes'] or original['native_case_manifest'] != protocol['native_case_manifest']:
            raise ValueError('Collection omitted or changed an original TRAIN condition')
    for name, digest in protocol['source_hashes'].items():
        if sha(runtime/name) != digest: raise ValueError('Frozen source changed: '+name)
    for name, key in [('collection.npz','dataset_sha256'),('collection_trace.jsonl.gz','trace_sha256'),
                      ('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/name) != result[key]: raise ValueError('Collection artifact changed: '+name)
    for name, key in [('model.zip','source_model_sha256'),('normalization.pkl','source_normalization_sha256')]:
        if sha(folder/name) != spec[key] or sha(folder/name) != sha(batch/'resume/B1'/name):
            raise ValueError('Collection changed learning checkpoint')
    if (any(summary[k] != 0 for k in ('physics_steps','actions','optimization_phases','optimizer_steps'))
            or summary['lineage_control_steps'] != spec.get('source_lineage_control_steps', 20480) or (folder/'optimization.jsonl').read_bytes()
            or any(result[k] != 0 for k in ('new_PPO_controls','new_PPO_updates','new_supervised_updates'))
            or (folder/'evaluation.json').exists()):
        raise ValueError('Collection contains optimization or is mislabeled as policy evaluation')
    actual, acknowledgements, owners, begins = {}, {}, {}, {}
    streams = sorted((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(streams) != 1: raise ValueError('Require exactly one complete collection bridge stream')
    responses, error = read_prefix(streams[0])
    if error: raise ValueError(error)
    for wrapper in responses:
        response = wrapper.get('response', {}); episode = response.get('episode_id')
        if episode is not None and response.get('request_id') is not None:
            key = (episode,response['request_id'])
            if key in acknowledgements: raise ValueError('Duplicate bridge acknowledgement')
            acknowledgements[key] = response
        if 'reset_ticks' in response:
            if episode in begins: raise ValueError('Duplicate reset episode')
            begins[episode] = response
        if 'vehicle_control' in response:
            key = (episode,response['frame'])
            if key in actual: raise ValueError('Duplicate physical control frame')
            actual[key] = response; owners[key] = streams[0].name
    rows, error = read_prefix(folder/'collection_trace.jsonl.gz')
    if error: raise ValueError(error)
    if not rows or len(rows) > spec['max_controls']: raise ValueError('Empty or over-budget collection')
    dataset = materialize_dataset(folder/'collection.npz')
    if set(dataset) != {'observations','actions','groups','fit_train_mask','teacher_executed','learner_actions'}:
        raise ValueError('Ambiguous collection dataset')
    if any(len(dataset[name]) != len(rows) for name in dataset): raise ValueError('Dataset drops recorded controls')
    torch.set_num_threads(1)
    model = load_ppo(folder/'model.zip',device='cpu')
    with (folder/'normalization.pkl').open('rb') as stream: norm = pickle.load(stream)
    norm.training = False
    before = frozen_state(model,norm)
    if before != result['frozen_state_sha256_before'] or before != result['frozen_state_sha256_after']:
        raise ValueError('Frozen policy/optimizer/RMS proof differs from saved learning bytes')
    rng = np.random.default_rng(spec['mixture_seed'])
    used, previous, episode_ids = set(), {}, set()
    completed = {}; indices = {}; teacher_counts = {}; case_episodes = {}
    view = None; current = -1; selected = False; teacher_total = eligible_total = 0; distance = 0.
    for position, row in enumerate(rows):
        case_id = row['case_id']; index = row['control_index']; info = row['info']
        ticks = info['trace']
        if len(ticks) != 1: raise ValueError('Mixture action lacks exactly one physical transition')
        tick = ticks[0]; episode = tick['backend_evidence']['episode_id']
        if current != case_id:
            if case_id != current+1 or index != 0 or current >= 0 and current not in completed:
                raise ValueError('Missing, reordered or unfinished collection episode')
            current = case_id
            distance = 0.
            begin = begins.get(episode)
            if begin is None: raise ValueError('Missing measured collection reset')
            verify_registered_native_begin(begin,protocol,case_id)
            view = measured_flat_env(config,begin['scene'],info['initial_scenario'],begin=begin)
        if (case_id in completed or index != indices.get(case_id,0)
                or case_episodes.setdefault(case_id,episode) != episode
                or row['frame_before'] != view._measured_frame or row['state_before'] != asdict(view.state)
                or row['previous_command'] != asdict(view.plant.previous_command)):
            raise ValueError('Collection state/history or episode ordering differs')
        if index % spec['block_controls'] == 0: selected = bool(rng.random() < spec['teacher_probability'])
        valid = row['teacher_status'] == 'solved' or row['teacher_mode'] in ('wait','dynamic_brake','parking_brake')
        eligible = selected and valid and not (spec.get('exclude_intervened_teacher_labels', False) and tick['intervention'])
        expected_source = 'teacher' if selected else 'learner'
        if row['source'] != expected_source or row['eligible_executed_teacher_label'] != eligible:
            raise ValueError('Mixture schedule or executed-label mask differs')
        np.testing.assert_array_equal(view._observation(),dataset['observations'][position])
        predicted = model.predict(norm.normalize_obs(dataset['observations'][position:position+1]),deterministic=True)[0][0]
        np.testing.assert_array_equal(predicted,row['learner_action'])
        np.testing.assert_array_equal(predicted,dataset['learner_actions'][position])
        np.testing.assert_array_equal(row['teacher_action'],dataset['actions'][position])
        if (dataset['fit_train_mask'][position] != eligible or dataset['teacher_executed'][position] != selected
                or dataset['groups'][position] != 2048+case_id): raise ValueError('Dataset source/label eligibility differs')
        action = np.asarray(row['teacher_action'] if selected else row['learner_action'],dtype=np.float32)
        np.testing.assert_array_equal(action,row['executed_action'])
        vehicle = config.vehicle
        requested = {'acceleration':float(action[0]*(vehicle.max_accel if action[0]>=0 else vehicle.max_decel)),
                     'steering':float(action[1]*vehicle.max_steer)}
        assisted, intervened, _ = goal_parking_command(view.state, view.goal, Command(**requested), config, distance > .1)
        if (asdict(assisted) != tick['requested_command']
                or bool(tick.get('goal_parking_intervention', False)) != intervened):
            raise ValueError('Selected action/shared parking helper differs from actual bridge request')
        # Independent production projection, measured frame and clock checks.
        measured_label(view,tick)
        _verify_ticks(ticks,actual,acknowledgements,owners,used,previous,episode_ids,config.actuator_observation)
        old_state = view.state; old_speed = old_state.v
        view.last_behavior = np.array([*action,0.,0.])
        view.plant.state = VehicleState(**tick['state']); view.plant.previous_command = Command(**tick['command'])
        view.plant.evidence = tick['backend_evidence']; view._measured_frame = tick['backend_frame']
        _, states = _actor_snapshot(view.plant.evidence.get('native_dynamic_actors', [] if not view._measured_actor_ids else None), view._measured_actor_ids)
        view.world.dynamic = [MovingObstacle(**state) for state in states]
        distance += float(np.hypot(view.state.x-old_state.x, view.state.y-old_state.y))
        if config.execution_feedback != 'off':
            view.execution_outcome = np.array([0.,0.,0.,float(tick['controller_status']=='failed'),
                float(tick['intervention']),(view.state.v-old_speed)/vehicle.max_speed],dtype=np.float32)
        indices[case_id] = index+1; teacher_counts[case_id] = teacher_counts.get(case_id,0)+int(selected)
        teacher_total += int(selected); eligible_total += int(eligible)
        if info.get('termination_reason'): completed[case_id] = info['episode_metrics']
    if set(completed) != set(range(len(cases))) or set(actual) != used or set(begins) != episode_ids:
        raise ValueError('Missing registered case or unaccounted physical controls/resets')
    if len(result['episodes']) != len(cases): raise ValueError('Incomplete collection result')
    for case_id, episode in enumerate(result['episodes']):
        wanted = {'case_id':case_id,'geometry_id':cases[case_id]['native_route']['geometry_id'],
            'teacher_controls':teacher_counts[case_id], 'learner_controls':indices[case_id]-teacher_counts[case_id], **completed[case_id]}
        if episode != wanted or indices[case_id] != completed[case_id]['physics_steps']:
            raise ValueError('Collection outcome differs from complete physical trace')
    expected = {'collection_controls':len(rows),'teacher_controls':teacher_total,'learner_controls':len(rows)-teacher_total,
        'eligible_executed_teacher_labels':eligible_total,
        'collection_world_ticks':len(rows)+sum(r['reset_ticks'] for r in begins.values())}
    if any(result[key] != value for key,value in expected.items()): raise ValueError('Actual collection cost differs')
    if any(summary[key] != expected[key] for key in ('collection_controls','teacher_controls','learner_controls','collection_world_ticks')):
        raise ValueError('Worker summary omits collection cost')
    if frozen_state(model,norm) != before: raise ValueError('Audit changed frozen learning state')
    return {'scope':__doc__,'batch':batch.name,'script_sha256':sha(Path(__file__)), 'protocol_sha256':sha(batch/'protocol.json'),
        'collection_result_sha256':sha(folder/'collection.json'),'trace_sha256':result['trace_sha256'],
        'dataset_sha256':result['dataset_sha256'],'bridge_response_stream':streams[0].name,
        'bridge_response_sha256':sha(streams[0]),'model_sha256':result['model_sha256'],
        'normalization_sha256':result['normalization_sha256'], **expected,
        'all_measured_observations_bit_exact':True,'all_learner_proposals_bit_exact':True,
        'all_source_choices_reconstructed':True,'no_unaccounted_applied_controls':True,
        'new_controls_during_audit':0,'new_optimizer_updates':0,
        'training_mixture_successes':sum(e['success'] for e in completed.values()),
        'training_mixture_unsafe_episodes':sum(e['termination_reason'] in ('collision','out_of_bounds') for e in completed.values()),
        'not_policy_evaluation':True,'frozen_state_sha256':before}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve completed collection audit')
    result=audit(args.batch)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
