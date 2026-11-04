"""Close both unchanged known inventories with one frozen steering-aware policy."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_refinement_pilot import prepare


def run(parent,execution):
    parent=Path(parent).resolve();execution=Path(execution).resolve();execution.mkdir()
    require_finished_parent(parent)
    atomic_json(execution/'registration.json',dict(parent=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
        phases={'static':12,'known_validation':24},all_outcomes_retained=True,new_PPO_updates=0,
        automatic_retries=0,new_final_conditions=0,scope='Full unchanged known development inventories;frozen-policy engineering transfer'))
    def call(args,label,timeout=1200,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=timeout,env=env)
    results={}
    try:
        batches={}
        for phase,count in [('static',12),('known_validation',24)]:
            batch=execution.parent/(execution.name+'_'+phase)
            prepare(parent,batch,None,True,phase)
            batches[phase]=batch
            atomic_json(execution/(phase+'_prepared.json'),dict(batch=str(batch),protocol_sha256=sha(batch/'protocol.json')))
        if read(batches['static']/'protocol.json')['source_hashes']!=read(batches['known_validation']/'protocol.json')['source_hashes']:
            raise ValueError('Both inventories must use exactly the same frozen source')
        for phase,count in [('static',12),('known_validation',24)]:
            batch=batches[phase]
            call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1',
                  '-Batch',batch,'-UseCurrentGuard','-ActivateQueued'],phase+'_native',read(batch/'protocol.json')['resource_limits']['wall_seconds']+600)
            require_finished_parent(batch)
            env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
            call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','H'],phase+'_raw',env=env)
            for flag,name in [([], 'measured_start_inference.json'),(['--all-static-actions'],'static_action_inference.json')]:
                call([sys.executable,ROOT/'scripts/audit_native_reference_inference.py','--batch',batch,'--method','H',
                      *flag,'--output',batch/'H'/name],phase+'_'+name,env=env)
            call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],phase+'_engine')
            if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine delivery failed')
            episodes=read(batch/'H/evaluation.json')['episodes']
            if len(episodes)!=count:raise ValueError('Incomplete registered inventory')
            result=dict(batch=str(batch),cases=count,successes=sum(e['success'] for e in episodes),
                reasons=dict(Counter(e['termination_reason'] for e in episodes)),raw_audit_sha256=sha(batch/'H/endpoint_raw_audit.json'))
            results[phase]=result;atomic_json(execution/(phase+'_complete.json'),result)
        atomic_json(execution/'complete.json',dict(results=results,research_acceptance='UNMET',new_PPO_updates=0))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--execution',required=True)
    a=p.parse_args();run(a.parent,a.execution)
