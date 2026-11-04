"""Collect and audit one fixed diverse TRAIN inventory with the unchanged strong B3."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json,resources_ok
from scripts.audit_native_method_endpoint import read,sha,read_prefix
from scripts.audit_native_control_delivery import control_errors
from scripts.evaluate_native_rule import require_finished_parent

PARENT=ROOT/'runs/20261102T053259603507Z_native_B3_broader_TRAIN_teacher'
REFERENCE=ROOT/'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/B1/config.json'


def prepare(inventory,output):
    inventory,output=Path(inventory).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve the registered teacher collection')
    require_finished_parent(PARENT)
    p=deepcopy(read(PARENT/'protocol.json'));layout=read(inventory)
    registration=layout['registration']
    if (registration['version']!='diverse_native_TRAIN36_geometry_v2'
            or registration['base_paths']!=12 or registration['route_prefix_augmentation'] is not False
            or layout.get('native_validation') is not None or layout['data_role']!='TRAIN_EXTRA_SUPERVISION'
            or len(registered_native_cases(layout))!=36):
        raise ValueError('Require the complete geometry-only twelve-base-path TRAIN inventory')
    for name,digest in p['source_hashes'].items():
        if sha(PARENT/'source'/name)!=digest:raise ValueError('Parent frozen physical source changed')
    okay,reason=resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free<18*1024**3:raise RuntimeError(str(reason))
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_diverse_TRAIN36_teacher')
    batch.mkdir();(batch/'infrastructure').mkdir()
    for name in ('source','calibration'):
        shutil.copytree(PARENT/name,batch/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PARENT/'source_config.json',batch/'source_config.json')
    shutil.copy2(inventory,batch/'TRAIN_inventory.json')
    atomic_json(batch/'case_manifest.json',layout['native_case_manifest'])
    now=time.time()
    p.update(parent_batch=PARENT.name,parent_root_relative=PARENT.relative_to(ROOT).as_posix(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),routes=layout['routes'],native_case_manifest=layout['native_case_manifest'],
        case_manifest_sha256=sha(batch/'case_manifest.json'),created_epoch=now,deadline_epoch=now+7200,
        resource_limits={**p['resource_limits'],'wall_seconds':7200,'automatic_retries':0},
        scope='Unchanged strongB3 on all36newdiverseTRAINconditions;extra supervision only,noPPOorvalidation/finaltest.',
        teacher_collection=dict(version='diverse_TRAIN36_teacher_v2',inventory_sha256=sha(inventory),
            base_paths=12,route_prefix_augmentation=False,all_cases_and_failures_retained=True,
            maximum_controls=32400,automatic_retries=0,labels='Actual executed controls;successful episodes eligible for nominal teaching,all failures retained.'))
    atomic_json(batch/'protocol.json',p)
    atomic_json(batch/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    output.mkdir()
    atomic_json(output/'registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),
        inventory_sha256=sha(inventory),base_paths=12,cases=36,maximum_controls=32400,
        data_role='TRAIN_EXTRA_SUPERVISION',validation_conditions=0,final_conditions=0,new_PPO_updates=0))
    print(batch,flush=True)


def run(output):
    output=Path(output).resolve()
    if (output/'started.json').exists():raise FileExistsError('Never relaunch started collection')
    r=read(output/'registration.json');batch=ROOT/'runs'/r['batch']
    if sha(batch/'protocol.json')!=r['protocol_sha256']:raise ValueError('Registered protocol changed')
    source=output/'operational_source'
    for name in ('scripts','navstack'):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    atomic_json(output/'source_manifest.json',{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
    shutil.copy2(__file__,output/'executed_observer.py')
    atomic_json(output/'started.json',dict(pid=os.getpid(),epoch=time.time(),automatic_retries=0))
    deadline=time.monotonic()+12000
    def status(phase,**extra):atomic_json(output/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**extra))
    def call(args,label,timeout):
        with (output/(label+'.log')).open('xb') as stream:
            child=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json(output/'last_child.json',dict(pid=child.pid,label=label,epoch=time.time()))
            code=child.wait(timeout=min(timeout,max(1,deadline-time.monotonic())))
        if code:raise RuntimeError('Owned child failed;stop without retry:'+label)
    try:
        status('COLLECTING_ALL36_DIVERSE_TRAIN_CONDITIONS')
        call([shutil.which('pwsh'),'-NoProfile','-File',source/'scripts/run_registered_native.ps1',
            '-Batch',batch,'-Kind','rule','-Python',ROOT/'.venv/Scripts/python.exe',
            '-CarlaPython',ROOT/'.carla-python37/runtime/python.exe','-UseCurrentGuard'],'native',7800)
        require_finished_parent(batch)
        status('AUDITING_ALL_DIVERSE_TRAIN_EXECUTED_CONTROLS')
        call([sys.executable,batch/'source/scripts/audit_native_rule.py','--batch',batch],'raw_audit',1200)
        raw=read(batch/'raw_audit.json');streams=list((batch/'infrastructure').glob('*_responses.jsonl.gz'))
        if len(streams)!=1 or sha(streams[0])!=raw['bridge_response_sha256']:raise ValueError('Bridge stream differs')
        rows,error=read_prefix(streams[0])
        if error:raise ValueError(error)
        count=0
        for row in rows:
            x=row['response']
            if 'vehicle_control' not in x:continue
            applied=x['applied_vehicle_control']
            if (control_errors(x['vehicle_control'],applied) or x.get('control_delivery')!='batch_sync_v1'
                    or x.get('applied_control_matches_request') is not True
                    or applied['manual_gear_shift'] is not True or applied['gear']!=1
                    or applied['hand_brake'] is not False or applied['reverse'] is not False):
                raise ValueError('Engine acknowledgement differs')
            count+=1
        if count!=raw['evaluation_controls_verified']:raise ValueError('Incomplete engine inventory')
        atomic_json(output/'strict_delivery.json',dict(verified_controls=count,all_engine_controls_acknowledged=True,
            tolerance=1e-5,new_controls=0))
        status('BUILDING_MEASURED_DIVERSE_TRAIN_DIRECT_VIEWS')
        call([sys.executable,batch/'source/scripts/build_native_measured_views.py','--reference-config',REFERENCE,
            '--teachers',batch,'--output',output/'direct_views','--wall-seconds','1800'],'direct_views',1860)
        status('ALL36_DIVERSE_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE',physical_controls=count)
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),epoch=time.time()))
        status('STOPPED_WITH_FAILURE_NO_RETRY')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--inventory');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run(args.output)
    elif args.inventory:prepare(args.inventory,args.output)
    else:parser.error('Preparation requires --inventory')
