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
from navstack.road_candidate_policy import FlatRoadCandidateActorCriticPolicy, OrderedFlatRoadCandidatePolicy, RoadCandidateVecNormalize
from scripts.audit_native_method_endpoint import read, read_prefix, sha
from scripts.audit_native_rule import audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config, learning_schedule, save_training_checkpoint
from scripts.prepare_native_hazard_learning import learning_protocol
from navstack.config import VehicleConfig
from navstack.research import resources_ok


def load_registered_views(views, reference_config, config, parent):
    """Verify the immutable measured dataset before any fitting or registration."""
    views, reference_config = Path(views).resolve(), Path(reference_config).resolve()
    manifest = read(views/'manifest.json')
    registration = read(views/'registration.json')
    metadata = read(reference_config)
    if (sha(reference_config) != manifest['reference_config_sha256']
            or manifest['reference_config_sha256'] != registration['reference_config_sha256']
            or metadata['env'] != config.to_dict() or metadata['signature'] != signature(config)
            or manifest['sources'] != registration['sources']
            or manifest['view_source_sha256'] != sha(ROOT/'navstack/measured_demo.py')
            or manifest['script_sha256'] != sha(ROOT/'scripts/build_native_measured_views.py')
            or any(manifest[key] != 0 for key in ('new_controls','supervised_optimizer_steps','ppo_updates'))):
        raise ValueError('Measured view configuration/source identity differs')
    sources = manifest['sources']
    if not sources or len({item['batch'] for item in sources}) != len(sources) or parent.name not in {item['batch'] for item in sources}:
        raise ValueError('Require unique registered teachers including the physical parent')
    for item in sources:
        if Path(item['batch']).name != item['batch']:
            raise ValueError('Teacher identity must be a batch name')
        donor = ROOT/'runs'/item['batch']
        require_finished_parent(donor)
        verified = audit(donor)
        if (verified != read(donor/'raw_audit.json') or sha(donor/'raw_audit.json') != item['raw_audit_sha256']
                or sha(donor/'protocol.json') != item['protocol_sha256']
                or verified['trace_sha256'] != item['trace_sha256']
                or verified['evaluation_controls_verified'] != item['controls']):
            raise ValueError('Registered teacher evidence changed')
    dataset = views/'measured_demonstrations.npz'
    if sha(dataset) != manifest['dataset_sha256']:
        raise ValueError('Measured dataset bytes changed')
    with np.load(dataset, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in ('observations','actions','returns','train_mask','groups')}
    count = manifest['teacher_unique_controls']
    if (count != sum(item['controls'] for item in sources)
            or arrays['observations'].shape != (count, 4705 if config.actuator_observation != 'off' else 4701)
            or arrays['actions'].shape != (count,2) or arrays['returns'].shape != (count,)
            or arrays['train_mask'].shape != (count,) or arrays['train_mask'].dtype != np.bool_
            or arrays['groups'].shape != (count,)
            or not np.issubdtype(arrays['groups'].dtype,np.integer)
            or not all(np.isfinite(arrays[key]).all() for key in ('observations','actions','returns'))
            or np.any(np.abs(arrays['actions']) > 1)):
        raise ValueError('Invalid measured dataset shape/values')
    offset = 0
    for group, case in enumerate(manifest['cases']):
        end = offset + case['controls']
        if (case['group'] != group or not np.all(arrays['groups'][offset:end] == group)
                or not np.array_equal(arrays['train_mask'][offset:end], np.arange(case['controls']) % 5 != 0)):
            raise ValueError('Measured case/holdout accounting differs')
        offset = end
    if offset != count:
        raise ValueError('Measured cases omit controls')
    return arrays, manifest


def shifted_holdout_mask(groups,original,phase):
    """Explicit correlated-fit split intervention; never change measured arrays."""
    if type(phase) is not int or phase not in (0,4):
        raise ValueError('Only the original phase0 or initial-covered phase4 is registered')
    groups=np.asarray(groups);original=np.asarray(original)
    if groups.ndim!=1 or original.shape!=groups.shape or original.dtype!=np.bool_:
        raise ValueError('Invalid registered groups/train mask')
    changed=original.copy();cases=[]
    for group in np.unique(groups):
        indices=np.flatnonzero(groups==group)
        if len(indices)<5 or not np.array_equal(original[indices],np.arange(len(indices))%5!=0):
            raise ValueError('Require complete original phase0 groups with a nonempty holdout')
        changed[indices]=np.arange(len(indices))%5!=phase
        cases.append({'group':int(group),'controls':len(indices),'train':int(changed[indices].sum()),
            'holdout':int((~changed[indices]).sum()),'initial_is_training':bool(changed[indices[0]])})
    return changed,cases


def prepare(parent, seed=11, epochs=128, views=None, reference_config=None,flat_head='pooled_v1',holdout_phase=0):
    parent = Path(parent).resolve()
    if type(seed) is not int or not 0 <= seed < 2**31 or not 1 <= epochs <= 128:
        raise ValueError('Invalid bounded supervised diagnostic')
    if flat_head not in ('pooled_v1','ordered_candidates_v1'):
        raise ValueError('Unknown versioned direct-policy actor head')
    if type(holdout_phase) is not int or holdout_phase not in (0,4) or (holdout_phase and views is None):
        raise ValueError('A shifted fit split requires explicit registered measured views')
    require_finished_parent(parent)
    if views is None and read(parent/'protocol.json').get('native_case_manifest') is not None:
        raise ValueError('Explicit mixed cases require audited registered views; refusing four-case subset supervision')
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
    if (views is None) != (reference_config is None):
        raise ValueError('Measured views require their explicit reference configuration')
    view_data, view_manifest = load_registered_views(views, reference_config, config, parent) if views else (None,None)
    view_suffix = '_views'+view_manifest['dataset_sha256'][:12] if views else ''
    if flat_head!='pooled_v1':view_suffix+='_head'+flat_head
    if holdout_phase:view_suffix+='_holdout'+str(holdout_phase)
    registration = parent/f'measured_direct_bc_v2_seed{seed}{view_suffix}_registration.json'
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
        fit_registration=read(batch/'fit_registration.json')
        fit_registration['flat_actor_head']=flat_head
        fit_registration['holdout_phase']=holdout_phase
        if holdout_phase:
            fit_registration['holdout']='Every fifth control with phase4; each measured initial state is TRAINING. Correlated fit diagnostic only, not independent validation.'
        write_json(batch/'fit_registration.json',fit_registration)
        if views:
            write_json(batch/'measured_view_provenance.json', view_manifest)
            fit_registration = read(batch/'fit_registration.json')
            fit_registration.update(version='measured_direct_bc_v3_registered_mixed_views',
                teacher_batches=[item['batch'] for item in view_manifest['sources']],
                scope='All registered static/dynamic teacher cases including failures; extra supervision, zero PPO updates.',
                label_bounds=view_manifest['brake_projection'],
                differences_from_prior_B1='Additional measured dynamic data and refitted train-only normalization; not a single-factor PPO comparison')
            write_json(batch/'fit_registration.json', fit_registration)
        deadline = time.time()+600
        def health():
            healthy, reason = resources_ok(ROOT)
            if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3 or time.time() >= deadline:
                raise RuntimeError('Supervised diagnostic resource/deadline guard: '+str(reason))
        try:
            torch.set_num_threads(1)
            observations, actions, returns, masks, cases = [], [], [], [], []
            rows, error = read_prefix(parent/'evaluation_trace.jsonl.gz') if not views else ([],None)
            if error:
                raise ValueError(error)
            for case in range(4) if not views else ():
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
            if views:
                observations, actions, returns, masks = (view_data[key] for key in ('observations','actions','returns','train_mask'))
                cases = view_manifest['cases']
                if holdout_phase:
                    masks,mask_cases=shifted_holdout_mask(view_data['groups'],masks,holdout_phase)
                    np.savez_compressed(batch/'measured_demonstrations.npz',**{**view_data,'train_mask':masks})
                    protocol['supervised_holdout_intervention']={'version':'initial_covered_phase4_v1',
                        'source_dataset_sha256':view_manifest['dataset_sha256'],'original_phase':0,'new_phase':holdout_phase,
                        'cases':mask_cases,'scope':'Only the correlated training/fit mask changes. Same measured observations/actions/returns/groups. Train-only normalization refits on the new mask; no independent holdout or driving-improvement claim.'}
                else:
                    shutil.copy2(Path(views)/'measured_demonstrations.npz', batch/'measured_demonstrations.npz')
            else:
                observations, actions = np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)
                np.savez_compressed(batch/'measured_demonstrations.npz', observations=observations, actions=actions,
                                    returns=np.asarray(returns), train_mask=np.asarray(masks),
                                    case_controls=np.asarray([r['controls'] for r in cases]))
            fit_dataset_sha=sha(batch/'measured_demonstrations.npz')
            raw = DummyVecEnv([lambda: NavigationEnv(config)])
            raw.envs[0].case_index = 0
            norm = RoadCandidateVecNormalize(raw, norm_reward=False, gamma=config.gamma)
            policy_type=OrderedFlatRoadCandidatePolicy if flat_head=='ordered_candidates_v1' else FlatRoadCandidateActorCriticPolicy
            model = PPO(policy_type, norm,
                        **learning_schedule(protocol, config, 320), seed=seed, device='cpu')
            def progress(row):
                health()
                write_json(batch/'fit_heartbeat.json', {'epoch': time.time(), **row})
                print(json.dumps(row), flush=True)
            history = fit_demonstrations(model, norm, observations, actions, np.asarray(returns), np.asarray(masks),
                                         epochs=epochs, seed=seed, callback=progress, value_weight=0.)
            if sha(batch/'measured_demonstrations.npz')!=fit_dataset_sha or (views and not holdout_phase and fit_dataset_sha!=view_manifest['dataset_sha256']):
                raise ValueError('Copied measured dataset changed during fitting')
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
            if views:
                protocol['extra_supervision'].update(teacher_batches=[s['batch'] for s in view_manifest['sources']],
                    measured_view_manifest_sha256=sha(batch/'measured_view_provenance.json'),
                    scope='Static and dynamic measured direct-control supervision; frozen evaluation uses every registered development condition. Not a single-factor PPO comparison or independent confirmation.')
            if flat_head!='pooled_v1':
                protocol['flat_actor_intervention']={'version':flat_head,
                    'scope':'Direct acceleration/steering from all ordered candidate embeddings; same inputs/data/limits, no planner or MPC execution by B1. Only actor fusion changes; larger parameter count and compute disclosed separately.',
                    'parameters':sum(p.numel() for p in model.policy.parameters())}
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
    parser.add_argument('--views')
    parser.add_argument('--reference-config')
    parser.add_argument('--flat-head',choices=['pooled_v1','ordered_candidates_v1'],default='pooled_v1')
    parser.add_argument('--holdout-phase',type=int,choices=[0,4],default=0)
    args = parser.parse_args()
    prepare(args.parent, args.seed, args.epochs, args.views, args.reference_config,args.flat_head,args.holdout_phase)
