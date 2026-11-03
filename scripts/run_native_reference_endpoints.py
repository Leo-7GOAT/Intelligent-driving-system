"""Evaluate completed reference learners after an exact predecessor, without retraining."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_reference_evaluation import prepare


def run(parent,execution,after_identity,predecessor):
    parent,execution,predecessor=map(lambda p:Path(p).resolve(),(parent,execution,predecessor))
    execution.mkdir()
    identity=read(after_identity)
    owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
    atomic_json(execution/'registration.json',dict(parent=str(parent),after_identity=identity,
        parent_protocol_sha256=sha(parent/'protocol.json'),new_training_controls=0,new_PPO_updates=0,
        new_final_conditions=0,automatic_retries=0,
        scope='Restore declared4736 evaluation container only;use fixed completed budget endpoints. Earlier failed4705-input attempt remains preserved.'))
    def call(args,label,timeout=900,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                           timeout=timeout,env=env,check=True)
    try:
        atomic_json(execution/'status.json',dict(phase='WAITING_FOR_EXACT_STATIC_REMAINDER_COORDINATOR',pid=os.getpid(),epoch=time.time()))
        deadline=time.monotonic()+28800
        while owner.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite predecessor wait exhausted')
            time.sleep(10)
        if owner.poll()!=0 or not (predecessor/'complete.json').exists():
            raise ValueError('Predecessor did not finish;inspect retained failure without an automatic retry')
        require_finished_parent(parent)
        raw=read(parent/'training_only_raw_audit.json');delivery=read(parent/'strict_delivery.json')
        if (raw['protocol_sha256']!=sha(parent/'protocol.json') or not delivery['strict_delivery_gate_passed']
                or raw['training_control_frames_verified']!=sum(read(parent/m/'summary.json')['physics_steps'] for m in ('H','H0'))):
            raise ValueError('Completed learners lack the bound training-frame/control-delivery proof')
        results={}
        for phase,expected in (('static',12),('known_validation',24)):
            batch=parent.parent/(parent.name+'_'+phase+'_schema_fixed')
            prepare(parent,batch,phase)
            atomic_json(execution/(phase+'_prepared.json'),dict(batch=str(batch),protocol_sha256=sha(batch/'protocol.json')))
            call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1',
                  '-Batch',batch,'-UseCurrentGuard'],phase+'_native',read(batch/'protocol.json')['resource_limits']['wall_seconds']+600)
            require_finished_parent(batch)
            env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
            measured={}
            for method in ('H','H0'):
                call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method',method],phase+'_'+method+'_raw',env=env)
                call([sys.executable,ROOT/'scripts/audit_native_reference_inference.py','--batch',batch,'--method',method,
                      '--output',batch/method/'measured_start_inference.json'],phase+'_'+method+'_starts',env=env)
                call([sys.executable,ROOT/'scripts/audit_native_reference_inference.py','--batch',batch,'--method',method,
                      '--all-static-actions','--output',batch/method/'static_action_inference.json'],phase+'_'+method+'_static_actions',env=env)
                episodes=read(batch/method/'evaluation.json')['episodes']
                if len(episodes)!=expected:raise ValueError('Incomplete registered development inventory')
                measured[method]=dict(cases=len(episodes),successes=sum(e['success'] for e in episodes),
                    termination_counts=dict(Counter(e['termination_reason'] for e in episodes)),
                    model_sha256=sha(batch/method/'model.zip'),raw_audit_sha256=sha(batch/method/'endpoint_raw_audit.json'))
            call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,
                  '--output',batch/'strict_delivery.json'],phase+'_delivery')
            if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine acknowledgement failed')
            results[phase]=dict(batch=str(batch),results=measured)
            atomic_json(execution/(phase+'_complete.json'),results[phase])
        atomic_json(execution/'complete.json',dict(results=results,research_acceptance='UNMET',
            scope='One seed known development endpoints;no new learning,no final conditions. H0 removes old feedback and context together.'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True)
    p.add_argument('--execution',required=True);p.add_argument('--after-identity',required=True)
    p.add_argument('--predecessor',required=True);a=p.parse_args()
    run(a.parent,a.execution,a.after_identity,a.predecessor)
