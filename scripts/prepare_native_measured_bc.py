"""Bounded direct-control representation diagnostic using actual B3 frames.

This is disclosed extra supervision, never a PPO improvement or healthy-baseline
claim until the saved network actually completes a separate CARLA evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.env import NavigationEnv
from navstack.experiments import write_json, signature
from navstack.imitation import fit_demonstrations
from navstack.measured_demo import measured_flat_env, measured_label, advance_measured_flat
from navstack.process_lease import ProcessLease
from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy, RoadCandidateVecNormalize
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.audit_native_rule import audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config, learning_schedule, save_training_checkpoint
from scripts.prepare_native_hazard_learning import learning_protocol
from navstack.config import VehicleConfig
from navstack.research import resources_ok


def prepare(parent, seed=11, epochs=128):
    parent = Path(parent).resolve()
    if type(seed) is not int or not 0 <= seed < 2**31 or not 1 <= epochs <= 128:
        raise ValueError('Invalid bounded supervised diagnostic')
    require_finished_parent(parent)
    if audit(parent) != read(parent/'raw_audit.json'):
        raise ValueError('Teacher raw audit changed')
    protocol = learning_protocol(parent, 640, 1800, seed, ['B1'])
    protocol.update(flat_hold=1, flat_action_catalog='continuous', flat_longitudinal_mode='acceleration',
                    rollout_actions=320, ppo_batch_size=320, record_initial_parameters=False,
                    evaluation_only=True, controlled_budget_per_method=0, method_budgets={'B1': 0},
                    calibration_sha256=sha(parent/'calibration/calibration.json'),
                    scope='Frozen continuous direct B1 teacher-supervised representation diagnostic; zero PPO updates, no hierarchy or ordinary-PPO claim')
    vehicle = VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    config = training_config(protocol, 'B1', vehicle)
    registration = parent/f'measured_direct_bc_v2_seed{seed}_registration.json'
    with ProcessLease(parent/'measured_direct_bc_prepare.lock'):
        if registration.exists():
            raise FileExistsError('Inspect the registered diagnostic; do not repeat fitting silently')
        batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_measured_bc')
        batch.mkdir()
        write_json(registration, {'batch': batch.name, 'status': 'registered_before_supervised_fit'})
        write_json(batch/'fit_registration.json', {'seed': seed, 'epochs': epochs, 'fit_wall_seconds': 600,
            'version': 'measured_direct_bc_v2_clipped_qp',
            'label_bounds': 'Same normalized-action clipping as production; QP request residuals above1e-3 rejected; raw requests/acknowledgements unchanged',
            'teacher_batch': parent.name, 'teacher_raw_audit_sha256': sha(parent/'raw_audit.json'),
            'scope': 'All four development teacher cases including deadline; every measured control. Extra supervision, zero PPO updates.',
            'holdout': 'Every fifth control within each case; correlated fit diagnostic, not independent validation',
            'differences_from_prior_B1': 'Continuous acceleration/steering at20Hz plus disclosed teacher supervision; not a single-factor comparison',
            'fairness': 'Formal hierarchy comparisons require equivalent teacher access and separately charged teacher, supervised and on-policy costs.'})
        deadline = time.time()+600
        def health():
            healthy, reason = resources_ok(ROOT)
            if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3 or time.time() >= deadline:
                raise RuntimeError('Supervised diagnostic resource/deadline guard: '+str(reason))
        try:
            torch.set_num_threads(1)
            observations, actions, returns, masks, cases = [], [], [], [], []
            rows, error = read_prefix(parent/'evaluation_trace.jsonl.gz')
            if error:
                raise ValueError(error)
            for case in range(4):
                health()
                decisions = [r for r in rows if r['case_id'] == case]
                scene = read(parent/f'scene_{case:04d}.json')['scene']
                env = measured_flat_env(config, scene, decisions[0]['info']['initial_scenario'])
                ticks = [t for r in decisions for t in r['info']['trace']]
                rewards = [t['reward'] for t in ticks]
                discounted = np.zeros(len(ticks), dtype=np.float32)
                total = 0.
                for i in reversed(range(len(ticks))):
                    total = rewards[i]+config.gamma_low*total
                    discounted[i] = total
                for i, tick in enumerate(ticks):
                    if i % 128 == 0:
                        health()
                    observations.append(env._observation())
                    action = measured_label(env, tick)
                    actions.append(action)
                    advance_measured_flat(env, tick, action)
                    masks.append(i % 5 != 0)
                env.close()
                returns.extend(discounted)
                metrics = read(parent/f'case_{case:04d}.json')
                cases.append({'case_id': case, 'controls': len(ticks), 'termination_reason': metrics['termination_reason']})
                print(json.dumps(cases[-1]), flush=True)
            observations, actions = np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)
            np.savez_compressed(batch/'measured_demonstrations.npz', observations=observations, actions=actions,
                                returns=np.asarray(returns), train_mask=np.asarray(masks),
                                case_controls=np.asarray([r['controls'] for r in cases]))
            raw = DummyVecEnv([lambda: NavigationEnv(config)])
            raw.envs[0].case_index = 0
            norm = RoadCandidateVecNormalize(raw, norm_reward=False, gamma=config.gamma)
            model = PPO(FlatRoadCandidateActorCriticPolicy, norm,
                        **learning_schedule(protocol, config, 320), seed=seed, device='cpu')
            def progress(row):
                health()
                write_json(batch/'fit_heartbeat.json', {'epoch': time.time(), **row})
                print(json.dumps(row), flush=True)
            history = fit_demonstrations(model, norm, observations, actions, np.asarray(returns), np.asarray(masks),
                                         epochs=epochs, seed=seed, callback=progress, value_weight=0.)
            if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
                raise ValueError('Supervised fitting must not be reported as PPO interaction/updates')
            output = batch/'initialization';output.mkdir()
            write_json(output/'config.json', {'env': config.to_dict(), 'signature': signature(config), 'seed': seed,
                'backend': 'measured_CARLA_teacher_supervised_initialization', 'scope': protocol['scope']})
            metadata = {'physics_steps': 0, 'actions': 0, 'episodes': 0, 'optimization_phases': 0,
                'optimizer_steps': 0, 'world_ticks': 0, 'lineage_control_steps': 0,
                'charged_control_steps_lower_bound': len(actions), 'supervised_optimizer_steps': history[-1]['supervised_optimizer_steps']}
            save_training_checkpoint(output, 0, model, norm, raw, metadata)
            origin = output/read(output/'latest_checkpoint.json')['folder']
            shutil.copytree(origin, batch/'resume/B1')
            norm.close()
            (batch/'infrastructure').mkdir()
            shutil.copytree(parent/'calibration', batch/'calibration')
            for folder in ('navstack', 'scripts', 'tests'):
                for source in (ROOT/folder).rglob('*'):
                    if source.is_file() and '__pycache__' not in source.parts and (source.suffix in ('.py', '.ps1') or 'fixtures' in source.parts):
                        target = batch/'source'/source.relative_to(ROOT)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            now = time.time()
            protocol.update(created_epoch=now, deadline_epoch=now+1800, parent_batch=parent.name,
                scope='Frozen continuous direct B1 teacher-supervised representation diagnostic; zero PPO updates, no hierarchy or ordinary-PPO claim',
                extra_supervision={'teacher_batch': parent.name, 'cases': cases, 'teacher_controls': len(actions),
                    'dataset_sha256': sha(batch/'measured_demonstrations.npz'), 'epochs': epochs,
                    'supervised_optimizer_steps': history[-1]['supervised_optimizer_steps'], 'history': history,
                    'holdout_scope': 'Correlated teacher-control fit only, no independent condition claim'},
                recovery={'B1': {'checkpoint': 'resume/B1', 'parent_batch': parent.name,
                    'parent_checkpoint': 'teacher_supervised_initialization', 'lineage_controls': 0,
                    'consumed_controls_lower_bound': len(actions), 'retained_optimizer_steps': 0,
                    'unknown_unflushed_parent_tail': False}})
            protocol['source_hashes'] = {p.relative_to(batch/'source').as_posix(): sha(p)
                for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            write_json(batch/'protocol.json', protocol)
            write_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch/'protocol.json')})
            print(batch, flush=True)
            return batch
        except BaseException as exc:
            write_json(batch/'fit_failure.json', {'epoch': time.time(), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=128)
    args = parser.parse_args()
    prepare(args.parent, args.seed, args.epochs)
