"""Run two current fixed-reference native controller jobs after deployment probes."""
import argparse
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


def validate(output):
    from scripts.native_curve_control import make_controller
    from navstack.config import VehicleConfig
    from navstack.reference_manifest import load_fixed_reference
    output=Path(output).resolve();plan=read(output/'registration.json')
    if len(plan['jobs'])!=2 or {r['controller'] for r in plan['jobs']}!={'mpc20','pursuit_pi'}:
        raise ValueError('Both fixed-reference controllers required')
    for name,digest in plan['source_hashes'].items():
        if sha(output/'source'/name)!=digest:raise ValueError('Frozen fixed-control driver changed')
    for row in plan['jobs']:
        if Path(row['batch']).name!=row['batch']:raise ValueError('Fixed job must be a sibling batch')
        batch=output.parent/row['batch'];p=read(batch/'protocol.json')
        if (sha(batch/'protocol.json')!=row['queued_protocol_sha256']
                or read(batch/'prepared.json')['protocol_sha256']!=row['queued_protocol_sha256']
                or p['comparison_controller']!=row['controller']):
            raise ValueError('Fixed controller protocol changed')
        if (sha(batch/'fixed_references/manifest.json')!=plan['reference_manifest_sha256']
                or sha(batch/'calibration/calibration.json')!=plan['calibration_sha256']):
            raise ValueError('Fixed references or calibration changed')
        for name,digest in plan['source_hashes'].items():
            if sha(batch/'source'/name)!=digest:raise ValueError('Frozen fixed-control worker changed')
        vehicle=VehicleConfig(**read(batch/'calibration/calibration.json')['vehicle']);make_controller(vehicle,p)
        for i,route in enumerate(p['routes']):load_fixed_reference(batch/'fixed_references',i,route['geometry_id'])
    return plan


def deployment_closed(dependency):
    from scripts.run_current_native_deployment_controls import job_inventory
    dependency=Path(dependency);path=dependency/'execution/complete.json'
    if not path.exists():return False
    plan=read(dependency/'registration.json');rows=read(path)['results']
    if len(rows)!=8 or {r['label'] for r in rows}!={r['label'] for r in job_inventory(plan)}:
        raise ValueError('All eight deployment results required')
    for row in rows:
        batch=dependency.parent/row['batch'];require_finished_parent(batch)
        if sha(batch/'protocol.json')!=row['protocol_sha256'] or not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Closed deployment evidence changed')
    return True


def run(output):
    from navstack.owned_process import OwnedProcess
    from scripts.run_independent_outcome_cohort import resource_admission
    output=Path(output).resolve();plan=validate(output);execution=output/'execution';execution.mkdir()
    dependency=Path(plan['dependency_cohort'])
    if sha(dependency/'registration.json')!=plan['dependency_sha256']:raise ValueError('Deployment dependency changed')
    def status(phase,**fields):atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**fields))
    def call(args,label,env=None,timeout=1800):
        with (execution/(label+'.log')).open('xb') as log:
            subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    atomic_json(execution/'registration.json',dict(plan_sha256=sha(output/'registration.json'),driver_sha256=sha(Path(__file__)),maximum_native_jobs=2,automatic_retries=0))
    try:
        if not deployment_closed(dependency):
            owner=OwnedProcess(**plan['dependency_identity'],allow_terminate=False);until=time.monotonic()+plan['maximum_wait_seconds']
            while not deployment_closed(dependency):
                if owner.poll() is not None or (dependency/'execution/failure.json').exists() or time.monotonic()>=until:
                    raise RuntimeError('Deployment dependency incomplete/stopped or bounded wait expired;no fixed-control launch')
                status('WAITING_FOR_DEPLOYMENT_CONTROLS',dependency_pid=plan['dependency_identity']['pid'],new_native_jobs=0);time.sleep(30)
        results=[]
        for row in plan['jobs']:
            batch=output.parent/row['batch'];name=row['controller'];status(name+'_NATIVE',completed_jobs=len(results))
            atomic_json(execution/(name+'_resources.json'),resource_admission(output.parent))
            call([shutil.which('pwsh'),'-NoProfile','-File',ROOT/'scripts/run_registered_native.ps1','-Batch',batch,
                '-Kind','control','-Python',sys.executable,'-CarlaPython',plan['runtime_carla_python'],'-UseCurrentGuard','-ActivateQueued'],name+'_native',timeout=15000)
            require_finished_parent(batch)
            env=dict(os.environ,PYTHONIOENCODING='utf-8',NAVSTACK_CARLA_PYTHON=plan['runtime_carla_python'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            call([sys.executable,batch/'source/scripts/analyze_native_control.py','--batch',batch],name+'_raw',env)
            call([sys.executable,ROOT/'scripts/audit_native_control_delivery.py','--batch',batch,'--output',batch/'strict_delivery.json'],name+'_engine',env)
            if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:raise ValueError('Fixed-control engine delivery mismatch')
            summary=read(batch/'analysis/verified_summary.json')
            if len(summary['episodes'])!=4:raise ValueError('Fixed-control condition omitted')
            results.append(dict(controller=name,batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),summary=summary))
            atomic_json(execution/(name+'_complete.json'),results[-1])
        first,second=[output.parent/r['batch'] for r in plan['jobs']]
        call([sys.executable,ROOT/'scripts/compare_native_control.py','--baseline',first,'--candidate',second],'paired_comparison')
        comparison=read(second/'analysis/paired_comparison.json')
        atomic_json(execution/'complete.json',dict(results=results,comparison=comparison,new_PPO_updates=0,new_final_conditions=0,
            research_acceptance='UNMET',scope='Four known fixed-reference controller conditions;no PPO or five-seed superiority claim'))
        status('COMPLETE',completed_jobs=2)
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:validate(args.output);print('PASS: two current fixed-reference controllers validated;no native launch')
    else:run(args.output)
