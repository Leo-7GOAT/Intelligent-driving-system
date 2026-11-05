"""Independent native TRAIN48 H/H0 development cohort, with paired pristine actors."""
import argparse
import ast
import ctypes
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata

SEEDS = (11, 22, 33, 44, 55)
VERSION = 'independent_outcome_TRAIN48_v1'
BUDGET = 30720


def resource_admission(archive):
    gpu = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_ulonglong)
            for n in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('Cannot measure available physical memory')
    free=shutil.disk_usage(archive).free
    if gpu<4096 or m.avail_phys<16*1024**3 or free<16*1024**3:
        raise RuntimeError('Require4GiB freeGPU,16GiB freeRAM and16GiB archive disk before each native job')
    return dict(free_GPU_MiB=gpu,free_RAM_GiB=m.avail_phys/1024**3,free_disk_GiB=free/1024**3,
        logical_CPUs=os.cpu_count(),maximum_simultaneous_servers=2,unowned_2100_never_stopped=True)


def initialized_adam(model):
    """Materialize mathematically empty Adam states;never reset existing state."""
    import torch
    added = []
    for name, parameter in model.policy.named_parameters():
        state = model.policy.optimizer.state[parameter]
        if not state:
            state.update(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                         exp_avg_sq=torch.zeros_like(parameter))
            if model.policy.optimizer.param_groups[0].get('amsgrad'):
                state['max_exp_avg_sq'] = torch.zeros_like(parameter)
            added.append(name)
    return added


def migrate_pristine(model, norm, failure_off):
    from navstack.reference_context_policy import migrate_learner as append_reference
    from navstack.outcome_context_policy import migrate_learner as append_failure
    if model.num_timesteps or model.policy.optimizer.state:
        raise ValueError('This cohort starts at each independent zero-PPO actor,not a selected learner')
    migrated, extended_norm = append_reference(model, norm, gain_scale=100.)
    migrated = append_failure(migrated, outcome_ablated=failure_off)
    added = initialized_adam(migrated)
    return migrated, extended_norm, added


def prepare(cohort, archive):
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.experiments import actor_vector, signature
    from navstack.independent_clip_ppo import load_ppo
    from navstack.native_cases import registered_native_cases
    from navstack.native_case_schedule import rehearsal_schedule
    from scripts.audit_native_policy_sequence import observation_env
    from scripts.collect_native_recovery import tensor_digest
    cohort, archive = Path(cohort).resolve(), Path(archive).resolve()
    if cohort.exists():
        raise FileExistsError('Preserve all cohort preparations')
    if cohort.drive.lower() != 'd:' or shutil.disk_usage(archive).free < 32*1024**3:
        raise ValueError('Use D archive with32GiB headroom')
    pairs = read(ROOT/'protocols/native_feedback_development_pairs_20261031.json')['pairs']
    if tuple(p['seed'] for p in pairs) != SEEDS:
        raise ValueError('All five independent seeds are required')
    physical = archive/'20261104_shared_lowlevel_B3_r2_placements30'
    require_finished_parent(physical)
    pp = read(physical/'protocol.json')
    # All physical modules are the completed shared-rule version. Only the
    # explicit training admission gate gains a new fixed budget registration.
    for name, digest in pp['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name) != digest:
            if name != 'navstack/fixed_lowlevel_training.py':
                raise ValueError('Current physical module changed: '+name)
            def reduced(path):
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                tree.body = [n for n in tree.body if not
                    (isinstance(n, ast.FunctionDef) and n.name in ('registered','outcome_registered'))
                    and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                        and t.id == 'REGISTRATION_COHORT' for t in n.targets))]
                return ast.dump(tree, include_attributes=False)
            if reduced(ROOT/name) != reduced(physical/'source'/name):
                raise ValueError('Admission edit also changes another definition')
    template = read(ROOT/'runs/20261102T130625263733Z_native_canonical_parent_TRAIN48_PPO10240/protocol.json')
    schedule = rehearsal_schedule(registered_native_cases(template))
    from navstack.native_partitions import validate_native_partitions
    partition = validate_native_partitions(dict(template,native_validation=read(
        archive/'20261104_outcome_frozen_H_known24/protocol.json')['native_validation']))
    resources = resource_admission(archive)
    cfg = EnvConfig.from_dict(read(archive/'20261104_outcome_context_PPO10240_H/H/config.json')['env'])
    # Recorded static initial states are a read-only migration check,not an
    # environment integration or a new performance score.
    observation_samples = []
    for path in sorted((archive/'20261104_outcome_frozen_H_placements30/H').glob('scene_*.json')):
        evidence = read(path)
        env = observation_env(cfg, evidence['scene'], evidence['initial'])
        try:
            observation_samples.append(env._observation())
        finally:
            env.close()
    if len(observation_samples) != 30:
        raise ValueError('Require all30recorded static starts for migration checks')
    obs = np.stack(observation_samples)
    cohort.mkdir(); (cohort/'initializers').mkdir()
    shutil.copytree(physical/'calibration', cohort/'calibration')
    for folder in ('navstack', 'scripts'):
        shutil.copytree(ROOT/folder, cohort/'source'/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {p.relative_to(cohort/'source').as_posix():sha(p)
        for p in (cohort/'source').rglob('*') if p.is_file()}
    torch.set_num_threads(1); actors = set(); results = []; moments = None
    for pair in pairs:
        seed = pair['seed']; origin = ROOT/pair['h']/'resume/H'
        metadata = verified_checkpoint_metadata(origin)
        if metadata['physics_steps'] or metadata['lineage_control_steps'] or metadata['optimizer_steps']:
            raise ValueError('Pristine canonical checkpoint has prior PPO')
        model = load_ppo(origin/'model.zip', device='cpu')
        actor_sha = hashlib.sha256(actor_vector(model).detach().cpu().numpy().tobytes()).hexdigest()
        if actor_sha in actors:
            raise ValueError('Independent actors were duplicated')
        actors.add(actor_sha)
        with (origin/'normalization.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        norm.training = False
        if moments is None:
            moments = deepcopy((norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count))
        for a, b in zip(moments, (norm.obs_rms.mean, norm.obs_rms.var, norm.obs_rms.count)):
            np.testing.assert_array_equal(a, b)
        arms = {}
        for method in ('H', 'H0'):
            rng = torch.get_rng_state().clone()
            migrated, newnorm, added = migrate_pristine(model, norm, method == 'H0')
            if not torch.equal(rng, torch.get_rng_state()):
                raise ValueError('Zero-effect schema migration advanced Torch RNG')
            old = torch.as_tensor(norm.normalize_obs(obs))
            extended = np.c_[obs, np.zeros((len(obs), 31), dtype=np.float32)]
            new = torch.as_tensor(newnorm.normalize_obs(extended))
            np.testing.assert_array_equal(old.numpy(), new[:, :-31].numpy())
            with torch.no_grad():
                if (not torch.equal(model.policy.get_distribution(old).distribution.probs,
                                    migrated.policy.get_distribution(new).distribution.probs)
                        or not torch.equal(model.policy.predict_values(old), migrated.policy.predict_values(new))):
                    raise ValueError('Pristine probabilities/values changed at measured native starts')
            target = cohort/'initializers'/f'seed{seed}_{method}'
            target.mkdir(); migrated.save(target/'model.zip'); newnorm.save(target/'normalization.pkl')
            shutil.copy2(origin/'resume_state.pkl', target/'resume_state.pkl')
            config = read(origin/'config.json')
            atomic_json(target/'config_before_current_wrapper.json', config)
            atomic_json(target/'config.json', dict(config, env=cfg.to_dict(), signature=signature(cfg),
                observation_encoding={'version':'action_aligned_outcome_context_v1'},
                observation_shape=[4736], reference_gain_scale=100., failure_gain_scale=1000.,
                failure_feedback_ablated=method=='H0',
                explicit_wrapper_transition='Same current shared parking/planning/control version throughout new actual PPO;no old physical state resumed'))
            meta = dict(metadata, model_sha256=sha(target/'model.zip'),
                normalization_sha256=sha(target/'normalization.pkl'),
                resume_state_sha256=sha(target/'resume_state.pkl'), checkpoint_schema_intervention=True)
            atomic_json(target/'metadata.json', meta); verified_checkpoint_metadata(target)
            reload = load_ppo(target/'model.zip', device='cpu')
            if any(not torch.equal(v,reload.policy.state_dict()[n]) for n,v in migrated.policy.state_dict().items()):
                raise ValueError('Migrated weights changed on reload')
            arms[method] = dict(path=target.relative_to(cohort).as_posix(), **meta,
                materialized_zero_Adam_states=added, optimizer_sha256=tensor_digest(reload.policy.optimizer.state_dict()))
        a = load_ppo(cohort/arms['H']['path']/'model.zip', device='cpu')
        b = load_ppo(cohort/arms['H0']['path']/'model.zip', device='cpu')
        if (tensor_digest(a.policy.state_dict()) != tensor_digest(b.policy.state_dict())
                or arms['H']['optimizer_sha256'] != arms['H0']['optimizer_sha256']
                or arms['H']['normalization_sha256'] != arms['H0']['normalization_sha256']
                or arms['H']['resume_state_sha256'] != arms['H0']['resume_state_sha256']):
            raise ValueError('Paired H/H0 weights/Adam/RMS/RNG differ')
        results.append(dict(seed=seed, origin=pair['h'], source_model_sha256=sha(origin/'model.zip'),
            source_metadata_sha256=sha(origin/'metadata.json'), independent_actor_sha256=actor_sha,
            old30_parameter_tensors_retained=True, old_optimizer_empty=True,
            new_Adam_buffers_zero_equivalent_to_empty=True, all30measured_start_probabilities_values_exact=True,
            inherited_control_cost_lower_bound=metadata['charged_control_steps_lower_bound'], arms=arms))
    plan = dict(version=VERSION, seeds=list(SEEDS), source_hashes=source_hashes,
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        runtime_python=sys.executable,
        runtime_carla_python=str(ROOT/'.carla-python37/runtime/python.exe'),
        calibration_sha256=sha(cohort/'calibration/calibration.json'),
        physical_parent=physical.name, physical_parent_protocol_sha256=sha(physical/'protocol.json'),
        training_routes=template['routes'], training_manifest=template['native_case_manifest'],
        training_schedule=schedule, initializer_results=results, config=cfg.to_dict(),
        train_validation_partition=partition,initial_resource_admission=resources,
        requested_training_controls_per_arm=BUDGET, max_rollout_overshoot=319,
        per_native_job_wall_seconds=14400, automatic_retries=0, maximum_native_jobs=25,
        maximum_training_controls=10*(BUDGET+319), maximum_evaluation_controls=15*24*900,
        maximum_total_wall_seconds=25*15000, new_supervised_updates=0, new_final_conditions=0,
        selection='All five seeds,shared zero-PPO baseline once per pair,final fixed-budget H/H0 only;no score-based checkpoint/seed selection',
        interpretation='Independent warm-started actors with inherited TRAIN-only teacher supervision;not from scratch or a matched H/B1main comparison.All new learning executes native controls with fixed lowlevel.',
        development_metrics='Known24completion,unsafe outcomes,interventions,planning failures,physical controls and full-cycle compute;all conditions retained.No final hypothesis or thresholds changed.')
    atomic_json(cohort/'registration.json', plan)
    return plan


def checkpoint_ancestry(checkpoint,meta,cohort):
    checkpoint=Path(checkpoint)
    result=dict(parent_batch=Path(cohort).name,retained_optimizer_steps=meta['optimizer_steps'],
        unknown_unflushed_parent_tail=False)
    parent=checkpoint.parent.parent
    if checkpoint.parent.name=='H' and (parent/'protocol.json').is_file():
        prior=read(parent/'protocol.json').get('recovery',{}).get('H',{})
        result.update(parent_batch=parent.name,
            retained_optimizer_steps=meta['optimizer_steps']+prior.get('retained_optimizer_steps',0),
            unknown_unflushed_parent_tail=prior.get('unknown_unflushed_parent_tail',False))
    return result


def stage(cohort, archive, row, method, label, *, training, checkpoint, rpc_port=2110):
    if type(rpc_port) is not int or rpc_port not in (2110, 2120):
        raise ValueError('Use one of the two registered disjoint native slots')
    from navstack.fixed_lowlevel_training import REGISTRATION_COHORT, registered
    from navstack.mpc_reference_effort import settings
    cohort, archive, checkpoint = Path(cohort), Path(archive), Path(checkpoint)
    plan = read(cohort/'registration.json')
    name = f'20261104_independent_outcome_seed{row["seed"]}_{label}'
    output = archive/name
    if output.exists():
        raise FileExistsError('Preserve native attempt: '+name)
    base = archive/'20261104_outcome_frozen_H_known24'
    p = read(base/'protocol.json')
    for key in ('fixed_outcome_development_validation','outcome_initialization',
                'fixed_lowlevel_endpoint','lineage_targets','fixed_lowlevel_adaptation_registration'):
        p.pop(key, None)
    p.update(seed=row['seed'], methods=['H'], evaluation_only=not training,
        controlled_budget_per_method=BUDGET if training else 0,
        method_budgets={'H':BUDGET if training else 0},
        deterministic_training_route_evaluation=not training,
        record_initial_parameters=True, scope=plan['interpretation'],
        carla_rpc_port=rpc_port, carla_rpc_port_span=list(range(rpc_port,rpc_port+3)), allow_disjoint_carla=True,
        source_hashes=plan['source_hashes'], calibration_sha256=plan['calibration_sha256'])
    p['outcome_context_registration']['failure_feedback_ablated'] = method == 'H0'
    meta = verified_checkpoint_metadata(checkpoint)
    p['resource_admission'] = resource_admission(archive)
    p['recovery'] = {'H':dict(checkpoint='resume/H', parent_batch=cohort.name,
        lineage_controls=meta['lineage_control_steps'],
        consumed_controls_lower_bound=meta['charged_control_steps_lower_bound'],
        retained_optimizer_steps=meta['optimizer_steps'], unknown_unflushed_parent_tail=False,
        model_sha256=meta['model_sha256'], normalization_sha256=meta['normalization_sha256'])}
    p['recovery']['H'].update(checkpoint_ancestry(checkpoint,meta,cohort))
    if training:
        p.update(routes=plan['training_routes'], native_case_manifest=plan['training_manifest'],
            training_case_schedule=plan['training_schedule'], fixed_lowlevel_training=REGISTRATION_COHORT,
            freeze_observation_statistics=True, checkpoint_thresholds=list(range(320,BUDGET+1,320)))
        p.pop('native_validation', None)
        if not registered(p):
            raise ValueError('Complete fixed-cohort training registration required')
    else:
        for key in ('fixed_lowlevel_training', 'training_case_schedule','freeze_observation_statistics'):
            p.pop(key, None)
        p['checkpoint_thresholds'] = []
    settings(p)
    now = time.time(); p.update(created_epoch=now, deadline_epoch=now+14400,
        resource_limits={**p['resource_limits'], 'wall_seconds':14400, 'automatic_retries':0},
        independent_outcome_cohort=dict(version=VERSION, registration_sha256=sha(cohort/'registration.json'),
            seed=row['seed'], arm=method, phase=label, new_supervised_updates=0,new_final_conditions=0))
    output.mkdir(); (output/'infrastructure').mkdir()
    shutil.copytree(cohort/'source', output/'source')
    shutil.copytree(cohort/'calibration', output/'calibration')
    shutil.copytree(checkpoint, output/'resume/H')
    atomic_json(output/'protocol.json', p)
    atomic_json(output/'prepared.json', dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    return output


def learning_proof(batch, expected_budget=BUDGET):
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from scripts.collect_native_recovery import tensor_digest
    before = load_ppo(batch/'resume/H/model.zip',device='cpu')
    after = load_ppo(batch/'H/model.zip',device='cpu'); s = read(batch/'H/summary.json')
    fp = read(batch/'H/initial_parameter_fingerprints.json')
    if (not expected_budget <= s['physics_steps'] < expected_budget+320 or s['stopped_reason'] is not None
            or after.num_timesteps-before.num_timesteps != s['actions']
            or fp['parameters'] != {n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in before.policy.state_dict().items()}
            or fp['optimizer_sha256'] != tensor_digest(before.policy.optimizer.state_dict())):
        raise ValueError('Actual training initialization/budget differs')
    if tensor_digest(before.policy.state_dict()) == tensor_digest(after.policy.state_dict()):
        raise ValueError('No learned parameter changed')
    old,new = dict(before.policy.named_parameters()),dict(after.policy.named_parameters())
    if len(new) != 34 or set(new) != set(old):
        raise ValueError('Parameter inventory differs')
    for name, value in new.items():
        if float(after.policy.optimizer.state[value]['step'])-float(before.policy.optimizer.state[old[name]]['step']) != s['optimizer_steps']:
            raise ValueError('Saved Adam advancement differs: '+name)
    with (batch/'resume/H/resume_state.pkl').open('rb') as stream:
        rng = pickle.load(stream)
    if fp['torch_rng_sha256'] != hashlib.sha256(rng['torch'].numpy().tobytes()).hexdigest():
        raise ValueError('Actual initial RNG differs')
    with (batch/'resume/H/normalization.pkl').open('rb') as f:
        initial = pickle.load(f).obs_rms
    with (batch/'H/normalization.pkl').open('rb') as f:
        final = pickle.load(f).obs_rms
    for a,b in ((initial.mean,final.mean),(initial.var,final.var),(initial.count,final.count)):
        np.testing.assert_array_equal(a,b)
    if read(batch/'protocol.json')['outcome_context_registration']['failure_feedback_ablated']:
        if torch.count_nonzero(after.policy.action_net.reference_bias.failure_gain) or torch.count_nonzero(after.policy.value_net.failure_weight):
            raise ValueError('Ablated failure coordinates changed')
    return dict(parameter_count=34, all_parameter_Adam_states_advance=s['optimizer_steps'],
        actual_initialized_parameters_and_Adam_exact=True, initial_torch_RNG_exact=True,RMS_exact=True,
        new_supervision=0,model_sha256=sha(batch/'H/model.zip'),summary=s)


def resumed_selection(seeds, execution_name, resume_completed, rpc_port):
    selected=SEEDS if seeds is None else tuple(seeds)
    if (not selected or any(type(s) is not int or s not in SEEDS for s in selected)
            or len(set(selected))!=len(selected)):
        raise ValueError('Select unique registered seeds')
    if seeds is not None and not resume_completed:
        raise ValueError('Seed selection requires explicit completed-phase reuse')
    if execution_name is not None and not re.fullmatch(r'execution_[A-Za-z0-9_-]{1,90}',execution_name):
        raise ValueError('Execution namespace must be a direct execution_ child')
    if type(rpc_port) is not int or rpc_port not in (2110,2120):
        raise ValueError('Use a registered disjoint native port')
    return tuple(s for s in SEEDS if s in selected)


def completed_phase(cohort, archive, label, reuse):
    marker=Path(cohort)/'execution'/(label+'_complete.json')
    if not reuse or not marker.exists():return False
    receipt=read(marker)
    name=receipt['batch']
    if not isinstance(name,str) or name in ('.','..') or Path(name).name!=name or '/' in name or '\\' in name:
        raise ValueError('Completed batch must be a direct archive child')
    batch=Path(archive)/name
    require_finished_parent(batch)
    if sha(batch/'protocol.json')!=receipt['protocol_sha256']:
        raise ValueError('Completed job changed')
    return True


def run(cohort, archive, recovery_batch=None, *, execution_name=None,
        resume_completed=False, seeds=None, rpc_port=2110):
    selected=resumed_selection(seeds,execution_name,resume_completed,rpc_port)
    cohort, archive = Path(cohort).resolve(), Path(archive).resolve()
    plan = read(cohort/'registration.json'); original_execution = cohort/'execution'
    reuse=bool(recovery_batch or resume_completed)
    execution = cohort/(execution_name or ('execution_resume_20261105' if recovery_batch else 'execution')); execution.mkdir()
    def completed(label):
        return completed_phase(cohort,archive,label,reuse)
    from navstack.independent_clip_ppo import load_ppo
    import torch
    torch.set_num_threads(1)
    deadline = time.time()+plan['maximum_total_wall_seconds']
    def call(args, label, env=None, timeout=1800):
        remaining=deadline-time.time()
        if remaining<timeout:
            raise TimeoutError('Cohort wall budget cannot admit another complete bounded phase')
        atomic_json(execution/'status.json', dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=env,check=True,timeout=timeout)
    def execute(batch, label, training):
        for name, digest in plan['source_hashes'].items():
            if sha(cohort/'source'/name) != digest:
                raise ValueError('Frozen cohort source changed')
        call([shutil.which('pwsh'),'-NoProfile','-File',cohort/'source/scripts/run_registered_native.ps1',
            '-Batch',batch,'-Python',plan['runtime_python'],'-CarlaPython',plan['runtime_carla_python']],
            label+'_native',timeout=15000)
        require_finished_parent(batch)
        env = dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
        call([sys.executable,cohort/'source/scripts/audit_native_control_delivery.py','--batch',batch,
            '--output',batch/'strict_delivery.json'],label+'_engine',env)
        if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Engine delivery gate failed')
        if training:
            call([sys.executable,cohort/'source/scripts/audit_native_training_only.py','--batch',batch,
                '--output',batch/'training_only_raw_audit.json'],label+'_raw',env)
            call([sys.executable,cohort/'source/scripts/audit_native_training_schedule.py','--batch',batch,
                '--output',batch/'training_schedule_audit.json'],label+'_schedule',env)
            atomic_json(batch/'H/saved_learning_proof.json',learning_proof(batch,read(batch/'protocol.json')['method_budgets']['H']))
            expected_controls=read(batch/'H/summary.json')['physics_steps']
        else:
            call([sys.executable,cohort/'source/scripts/audit_native_method_endpoint.py','--batch',batch,
                '--method','H'],label+'_raw',env)
            # Reconstruct every static decision and every measured start;
            # moving-actor controls additionally remain covered by raw audit.
            for extra, name in (([], 'measured_start_inference.json'),
                                (['--all-static-actions'], 'static_action_inference.json')):
                call([sys.executable,cohort/'source/scripts/audit_native_reference_inference.py','--batch',batch,
                    '--method','H',*extra,'--output',batch/'H'/name],label+'_'+name,env)
            expected_controls=sum(e['physics_steps'] for e in read(batch/'H/evaluation.json')['episodes'])
        if read(batch/'strict_delivery.json')['counts']['controls'] != expected_controls:
            raise ValueError('Engine control count differs from the complete phase')
        atomic_json(execution/(label+'_complete.json'),dict(batch=batch.name,
            protocol_sha256=sha(batch/'protocol.json'),training=training))
        if reuse:
            marker=original_execution/(label+'_complete.json')
            if marker.exists():raise FileExistsError('Do not replace completed evidence')
            atomic_json(marker,read(execution/(label+'_complete.json')))
    try:
        for row in plan['initializer_results']:
            seed = row['seed']; checkpoint = cohort/row['arms']['H']['path']
            if seed not in selected:continue
            if not completed(f'seed{seed}_baseline'):
                baseline = stage(cohort,archive,row,'H','zero_PPO_known24',training=False,checkpoint=checkpoint,rpc_port=rpc_port)
                execute(baseline,f'seed{seed}_baseline',False)
            for method in ('H','H0'):
                initial = cohort/row['arms'][method]['path']
                training_label=f'seed{seed}_{method}_training'
                if completed(training_label):
                    training=archive/read(original_execution/(training_label+'_complete.json'))['batch']
                else:
                    if recovery_batch and seed==22 and method=='H':training=Path(recovery_batch)
                    else:training = stage(cohort,archive,row,method,method+'_TRAIN48_PPO30720',training=True,checkpoint=initial,rpc_port=rpc_port)
                    execute(training,training_label,True)
                checkpoint = training/'H'/read(training/'H/latest_checkpoint.json')['folder']
                if not completed(f'seed{seed}_{method}_endpoint'):
                    endpoint = stage(cohort,archive,row,method,method+'_known24',training=False,checkpoint=checkpoint,rpc_port=rpc_port)
                    execute(endpoint,f'seed{seed}_{method}_endpoint',False)
        atomic_json(execution/'complete.json',dict(seeds=list(selected),status='ALL_REGISTERED_JOBS_CLOSED' if selected==SEEDS else 'SELECTED_REGISTERED_JOBS_CLOSED',research_acceptance='UNMET'))
        if reuse and selected==SEEDS:atomic_json(original_execution/'complete.json',read(execution/'complete.json'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time())); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run')); parser.add_argument('--cohort',required=True)
    parser.add_argument('--archive',required=True);parser.add_argument('--recovery-batch')
    parser.add_argument('--execution-name');parser.add_argument('--resume-completed',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--rpc-port',type=int,default=2110);args=parser.parse_args()
    if args.phase=='prepare':prepare(args.cohort,args.archive)
    else:run(args.cohort,args.archive,args.recovery_batch,execution_name=args.execution_name,
        resume_completed=args.resume_completed,seeds=args.seeds,rpc_port=args.rpc_port)
