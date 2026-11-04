"""Gate every matured B1actor,then train all five only if every basic gate passes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_all_basic_gates(gates):
    if [r['seed'] for r in gates]!=[11,22,33,44,55] or any(type(r['passed']) is not bool for r in gates):
        raise ValueError('Require all five explicitly recorded basic-motion gates')
    if not all(r['passed'] for r in gates):
        raise RuntimeError('At least one basic-motion gate failed;no PPOseeds started,no seed omitted or gate waived')


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;frozen=execution/'source'
    if execution.parent.name!='runs' or not (root/'.git').exists():raise ValueError('Project execution required')
    plan=read(execution/'plan.json');manifest=read(execution/'source_manifest.json')
    if not 60<=plan['wait_seconds']<=86400:raise ValueError('Finite predecessor wait required')
    def verify():
        for name,digest in manifest.items():
            path=(frozen/name).resolve()
            if not path.is_relative_to(frozen) or sha(path)!=digest:raise ValueError('Frozen source changed:'+name)
        for name,digest in plan['workspace_bindings'].items():
            if sha(root/name)!=digest:raise ValueError('Bound preparation source changed:'+name)
    verify();sys.path.insert(0,str(frozen))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts import prepare_native_supervised_ppo as preparation
    from scripts.audit_native_ppo_configuration import verify as verify_learner
    from scripts.evaluate_native_rule import require_finished_parent
    preparation.ROOT=root
    created=[];gates=[];owner=None
    def status(phase,**extra):atomic_json(execution/'status.json',{'pid':os.getpid(),'epoch':time.time(),'phase':phase,'created':created,'gates':gates,**extra})
    def call(command,log,timeout,environment=None):
        with (execution/log).open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(execution/'last_child.json',{'pid':child.pid,'epoch':time.time(),'log':log})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:raise RuntimeError('Observer expired;native watchdog retains ownership PID'+str(child.pid))
        if code:raise RuntimeError('Registered child failed;no retry:'+log)
    def lifecycle(batch,prefix,timeout):
        verify();healthy,reason=resources_ok(root)
        if not healthy:raise RuntimeError(str(reason))
        call([shutil.which('pwsh'),'-NoProfile','-File',frozen/'scripts/run_registered_native.ps1',
            '-Batch',batch,'-Python',root/'.venv/Scripts/python.exe','-CarlaPython',root/'.carla-python37/runtime/python.exe',
            '-ActivateQueued','-UseCurrentGuard'],prefix+'_lifecycle.log',timeout)
        require_finished_parent(batch)
    def audit(batch,seed,prefix):
        environment=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))
        call([sys.executable,frozen/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','B1'],prefix+'_raw.log',300,environment)
        inference=batch/'B1/measured_start_inference.json'
        call([sys.executable,frozen/'scripts/audit_native_policy_inference.py','--batch',batch,'--method','B1','--output',inference],prefix+'_inference.log',300,environment)
        folder=batch/'B1';summary=read(folder/'summary.json');episodes=read(folder/'evaluation.json')['episodes'];raw=read(folder/'endpoint_raw_audit.json')
        if sorted(e['case_id'] for e in episodes)!=list(range(12)) or len(read(inference)['results'])!=12:
            raise ValueError('All12conditions and initial actions are required')
        result={'scope':'New all-five B1actor-maturation development cohort;all outcomes retained,not final confirmation',
            'seed':seed,'batch':batch.name,'cases':12,'successes':sum(e['success'] for e in episodes),
            'termination_counts':dict(Counter(e['termination_reason'] for e in episodes)),
            'new_training_controls':summary['physics_steps'],'new_optimizer_steps':summary['optimizer_steps'],
            'raw_controls_verified':raw['total_applied_controls_verified'],'evaluation_controls':raw['evaluation_control_frames_verified'],
            'model_sha256':sha(folder/'model.zip'),'normalization_sha256':sha(folder/'normalization.pkl'),
            'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'research_acceptance':'UNMET'}
        report=root/'reports'/('native_B1_matured_seed'+str(seed)+'_'+prefix.split('_')[-1]+'_20261031.json')
        with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
        return result,episodes
    try:
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x') as stream:json.dump({'pid':os.getpid(),'epoch':time.time(),'script_sha256':sha(Path(__file__)),'plan_sha256':sha(execution/'plan.json')},stream)
            dependency=root/'runs'/plan['dependency_execution'];until=time.monotonic()+plan['wait_seconds']
            expected='ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE'
            while read(dependency/'status.json')['phase']!=expected:
                if (dependency/'failure.json').exists() or time.monotonic()>until:raise RuntimeError('B2predecessor failed or finite wait expired')
                if owner is None:
                    identity=plan['dependency_process'];owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
                if owner.poll() is not None:
                    if read(dependency/'status.json')['phase']==expected:break
                    raise RuntimeError('B2predecessor exited before completion')
                status('WAITING_FOR_REGISTERED_B2_COHORT',dependency_pid=owner.pid,dependency_live_via_retained_handle=True);time.sleep(10)
            if (dependency/'failure.json').exists():raise RuntimeError('Retained predecessor failure')
            for row in read(dependency/'status.json')['completed']:require_finished_parent(root/'runs'/row['batch'])
            prepared=read(root/'runs'/plan['actor_preparation']/'status.json')
            if prepared['phase']!='ALL_FIVE_ACTORS_PREPARED_NOT_DRIVEN' or [r['seed'] for r in prepared['batches']]!=[11,22,33,44,55]:
                raise ValueError('All five fixed actor fits must finish before driving')
            for row in prepared['batches']:
                verify();batch=root/'runs'/row['batch'];seed=row['seed']
                if sha(batch/'protocol.json')!=row['protocol_sha256']:raise ValueError('Prepared actor gate changed')
                status('RUNNING_ALL_FIVE_INITIALIZATION_GATES',seed=seed,batch=batch.name)
                lifecycle(batch,'seed'+str(seed)+'_gate',2400)
                result,episodes=audit(batch,seed,'seed'+str(seed)+'_gate')
                summary=read(batch/'B1/summary.json')
                if summary['physics_steps'] or summary['optimizer_steps'] or summary['actor_l2_change']!=0:raise ValueError('Actor gate unexpectedly learned')
                try:
                    gate=preparation.basic_motion_training_gate(read(batch/'protocol.json'),{'episodes':episodes})
                    gates.append({'seed':seed,'batch':batch.name,'passed':True,'basic_gate':gate})
                except ValueError as error:
                    gates.append({'seed':seed,'batch':batch.name,'passed':False,'retained_gate_error':str(error)})
                status('INITIALIZATION_GATE_AUDITED',seed=seed)
            if not all(r['passed'] for r in gates):
                status('ALL_FIVE_GATES_AUDITED_REQUIRES_NEW_DEVELOPMENT')
            require_all_basic_gates(gates)
            for row in prepared['batches']:
                verify();seed=row['seed'];parent=root/'runs'/row['batch'];status('PREPARING_B1_PPO',seed=seed)
                units=read(parent/'protocol.json').get('PPO_reward_units')
                if units is not None and (units.get('version')!='fixed_physical_reward_units_v1' or units.get('scale')!=.05):
                    raise ValueError('Unknown inherited reward units')
                batch=preparation.prepare(parent,controls=20480,wall_seconds=7200,log_std=-4,learning_rate=3e-6,
                    rollout_actions=None,initialization_gate='basic_motion_training_v1',target_kl=.01,
                    freeze_observation_statistics=True,walker_spawn_clearance=True,ppo_reward_scale=None if units else .05)
                verify_learner(batch/'resume/B1',root/'runs'/plan['learner_reference']/'resume/B1')
                created.append({'seed':seed,'batch':batch.name})
                status('RUNNING_INDEPENDENT_B1_PPO',seed=seed,batch=batch.name)
                lifecycle(batch,'seed'+str(seed)+'_PPO',7800)
                result,_=audit(batch,seed,'seed'+str(seed)+'_PPO')
                if result['new_training_controls']<20480 or result['new_optimizer_steps']<=0:raise ValueError('Incomplete actual PPO budget')
                status('B1_SEED_COMPLETE',seed=seed)
            status('ALL_FIVE_MATURED_B1_PPO_SEEDS_COMPLETE')
    except BaseException as error:
        with (execution/'failure.json').open('x',encoding='utf-8') as stream:json.dump({'epoch':time.time(),'error':repr(error),'created':created,'gates':gates,'scope':'No experiment retry,seed omission,basic gate waiver or foreign kill'},stream,indent=2)
        raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)
