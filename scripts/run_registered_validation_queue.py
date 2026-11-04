"""Serial known-development endpoint verification;no retries or training."""
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


def run(batches,counts,execution):
    batches=[Path(x).resolve() for x in batches];execution=Path(execution).resolve()
    if len(batches)!=len(counts) or not batches or len(set(batches))!=len(batches):raise ValueError('Unique registered batch/count pairs required')
    protocols=[read(x/'protocol.json') for x in batches]
    for p in protocols:
        if p.get('methods')!=['H'] or p.get('evaluation_only') is not True or p.get('method_budgets')!={'H':0}:
            raise ValueError('This queue accepts zero-update Hknown-development endpoints only')
        if p['source_hashes']!=protocols[0]['source_hashes']:raise ValueError('Endpoint sources differ')
    for name in ['model.zip','normalization.pkl']:
        if len({sha(x/'resume/H'/name) for x in batches})!=1:raise ValueError('Endpoint model/normalizer differ')
    execution.mkdir();atomic_json(execution/'registration.json',dict(batches=[dict(name=x.name,cases=n,protocol_sha256=sha(x/'protocol.json')) for x,n in zip(batches,counts)],
        new_training=0,new_final_conditions=0,automatic_retries=0))
    def call(args,label,timeout=1800,env=None):
        atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
    results=[]
    try:
        for index,(batch,count) in enumerate(zip(batches,counts)):
            label=str(index)
            call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',batch,
                '-UseCurrentGuard','-ActivateQueued'],label+'_native',read(batch/'protocol.json')['resource_limits']['wall_seconds']+600)
            require_finished_parent(batch)
            env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'),PYTHONIOENCODING='utf-8')
            call([sys.executable,ROOT/'scripts/audit_native_method_endpoint.py','--batch',batch,'--method','H'],label+'_raw',env=env)
            for flags,name in [([], 'measured_start_inference.json'),(['--all-static-actions'],'static_action_inference.json')]:
                call([sys.executable,ROOT/'scripts/audit_native_reference_inference.py','--batch',batch,'--method','H',*flags,
                    '--output',batch/'H'/name],label+'_'+name,env=env)
            call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],label+'_engine')
            if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Engine acknowledgement failed')
            episodes=read(batch/'H/evaluation.json')['episodes']
            if len(episodes)!=count:raise ValueError('Registered inventory incomplete')
            result=dict(batch=batch.name,cases=count,successes=sum(e['success'] for e in episodes),
                reasons=dict(Counter(e['termination_reason'] for e in episodes)),raw_sha256=sha(batch/'H/endpoint_raw_audit.json'))
            results.append(result);atomic_json(execution/(label+'_complete.json'),result)
        atomic_json(execution/'complete.json',dict(results=results,research_acceptance='UNMET',new_training=0,new_final_conditions=0))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',nargs='+',required=True)
    p.add_argument('--counts',nargs='+',type=int,required=True);p.add_argument('--execution',required=True)
    a=p.parse_args();run(a.batches,a.counts,a.execution)
