"""Audited per-case parallel reconstruction; measured states only, no plant stepping."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time


def convert_case(task):
    import numpy as np
    sys.path.insert(0,task['source'])
    from navstack.config import EnvConfig
    from navstack.measured_demo import measured_flat_env,measured_label,advance_measured_flat
    from navstack.research import resources_ok
    cfg=EnvConfig.from_dict(task['config']);ticks=task['ticks'];initial=task['initial']
    env=measured_flat_env(cfg,task['begin']['scene'],initial,begin=task['begin'])
    observations=[];labels=[];changed=projections=0;discounted=np.empty(len(ticks),dtype=np.float32);total=0.
    for i in reversed(range(len(ticks))):
        total=ticks[i]['reward']+cfg.gamma_low*total;discounted[i]=total
    try:
        for index,tick in enumerate(ticks):
            if index%128==0:
                okay,reason=resources_ok(task['workspace'])
                if not okay or time.monotonic()>=task['deadline']:raise RuntimeError('Conversion guard:'+str(reason))
            observations.append(env._observation());label=measured_label(env,tick);labels.append(label)
            request=tick['requested_command']
            projections+=(abs(request['steering'])>cfg.vehicle.max_steer+1e-3
                or not -cfg.vehicle.max_decel-1e-3<=request['acceleration']<=cfg.vehicle.max_accel+1e-3)
            before=[asdict(actor) for actor in env.world.dynamic]
            advance_measured_flat(env,tick,label)
            changed+=before!=[asdict(actor) for actor in env.world.dynamic]
    finally:env.close()
    destination=Path(task['output']);destination.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(destination/'arrays.npz',observations=np.asarray(observations,dtype=np.float32),
        actions=np.asarray(labels,dtype=np.float32),returns=discounted,train_mask=np.arange(len(ticks))%5!=0,
        groups=np.full(len(ticks),task['case_id'],dtype=np.int64))
    case=dict(group=task['case_id'],source_id=0,case_id=task['case_id'],controls=len(ticks),
        termination_reason=task['termination_reason'],dynamic_actors=len(initial['dynamic']),
        measured_actor_changed_frames=changed,acknowledged_brake_projection_controls=projections)
    (destination/'case.json').write_text(json.dumps(case,indent=2),encoding='utf-8')
    return case


def merge_arrays(folders,cases,expected_controls):
    import numpy as np
    if ([c['case_id'] for c in cases]!=list(range(len(cases)))
            or any(c['group']!=c['case_id'] for c in cases) or sum(c['controls'] for c in cases)!=expected_controls):
        raise ValueError('Missing/reordered case or measured control')
    members={k:[] for k in ('observations','actions','returns','train_mask','groups')}
    for folder,case in zip(folders,cases):
        with np.load(Path(folder)/'arrays.npz',allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        if set(arrays)!=set(members) or any(len(v)!=case['controls'] for v in arrays.values()):
            raise ValueError('Incomplete case arrays')
        if not np.all(arrays['groups']==case['group']) or not np.array_equal(arrays['train_mask'],np.arange(case['controls'])%5!=0):
            raise ValueError('Case/split identity differs')
        for key in members:members[key].append(arrays[key])
    return {key:np.concatenate(value) for key,value in members.items()}


def run(batch,reference_config,execution,workers=4,wall_seconds=1800,wait_worker=False):
    batch,reference_config,execution=map(lambda p:Path(p).resolve(),(batch,reference_config,execution))
    if workers not in (1,2,4) or wall_seconds!=1800:raise ValueError('Require the locked bounded conversion profile')
    if (execution/'started.json').exists():raise FileExistsError('Preserve previous conversion attempts')
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Use a fresh frozen-runtime interpreter')
    runtime=batch/'source';sys.path.insert(0,str(runtime))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.experiments import signature
    from navstack.native_cases import registered_native_cases
    from navstack.research import atomic_json
    from scripts.audit_native_method_endpoint import read,read_prefix,sha
    from scripts.audit_native_rule import audit
    from scripts.audit_native_control_delivery import control_errors
    from scripts.evaluate_native_rule import require_finished_parent
    execution.mkdir(parents=True,exist_ok=True)
    atomic_json(execution/'started.json',dict(pid=os.getpid(),epoch=time.time(),workers=workers,new_controls=0,wait_worker=wait_worker))
    atomic_json(execution/'registration.json',dict(batch=batch.name,protocol_sha256=sha(batch/'protocol.json'),data_role='TRAIN_EXTRA_SUPERVISION'))
    shutil.copy2(__file__,execution/'executed_parallel_converter.py')
    if wait_worker:
        from navstack.owned_process import OwnedProcess
        identity=read(batch/'worker_process.json')
        owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
        wait_deadline=time.monotonic()+7500
        try:
            if owner.poll() is not None:raise ValueError('Physical worker already ended;use explicit completed-data mode')
            while owner.poll() is None:
                if time.monotonic()>wait_deadline:raise TimeoutError('Finite physical dependency wait exhausted')
                atomic_json(execution/'status.json',dict(phase='WAITING_FOR_EXACT_PHYSICAL_DIVERSE_TEACHER',pid=os.getpid(),epoch=time.time()))
                time.sleep(10)
            if owner.poll()!=0:raise RuntimeError('Physical teacher failed;no conversion of incomplete data')
        finally:owner.kernel.CloseHandle(owner.handle)
        cleanup_deadline=time.monotonic()+300
        while True:
            if time.monotonic()>cleanup_deadline:raise TimeoutError('Owned cleanup/raw audit did not finish')
            try:
                require_finished_parent(batch)
                if (batch/'raw_audit.json').exists():
                    read(batch/'raw_audit.json');break
            except (FileNotFoundError,ValueError,json.JSONDecodeError):pass
            time.sleep(2)
    require_finished_parent(batch);protocol=read(batch/'protocol.json')
    if protocol.get('data_role')!='TRAIN_EXTRA_SUPERVISION' or protocol.get('native_validation') is not None:
        raise ValueError('Only the completed TRAIN teacher may supply labels')
    if len(registered_native_cases(protocol))!=36:raise ValueError('Require all36 registered diverse TRAIN cases')
    for name,digest in protocol['source_hashes'].items():
        if sha(runtime/name)!=digest:raise ValueError('Frozen teacher source changed')
    verified=audit(batch)
    if verified!=read(batch/'raw_audit.json'):raise ValueError('Complete original raw audit differs')
    cfg=EnvConfig.from_dict(read(reference_config)['env']);donor=EnvConfig.from_dict(protocol['config'])
    if (signature(cfg)!=read(reference_config)['signature'] or cfg.vehicle!=donor.vehicle
            or cfg.reward!=donor.reward or cfg.control_dt!=donor.control_dt or cfg.actuator_observation!=donor.actuator_observation):
        raise ValueError('Physical interface differs')
    started=time.monotonic();deadline=started+wall_seconds
    def status(phase,**extra):atomic_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),epoch=time.time(),**extra))
    try:
        streams=list((batch/'infrastructure').glob('*_responses.jsonl.gz'))
        if len(streams)!=1 or sha(streams[0])!=verified['bridge_response_sha256']:raise ValueError('Bridge stream differs')
        responses,error=read_prefix(streams[0])
        if error:raise ValueError(error)
        begins={};count=0
        for wrapper in responses:
            r=wrapper['response']
            if 'scene' in r:begins[r['episode_id']]=r
            if 'vehicle_control' not in r:continue
            a=r['applied_vehicle_control']
            if (control_errors(r['vehicle_control'],a) or r.get('control_delivery')!='batch_sync_v1'
                    or r.get('applied_control_matches_request') is not True or a['manual_gear_shift'] is not True
                    or a['gear']!=1 or a['hand_brake'] is not False or a['reverse'] is not False):raise ValueError('Engine acknowledgement differs')
            count+=1
        if count!=verified['evaluation_controls_verified']:raise ValueError('Engine inventory differs')
        atomic_json(execution/'strict_delivery.json',dict(verified_controls=count,all_engine_controls_acknowledged=True,tolerance=1e-5,new_controls=0))
        rows,error=read_prefix(batch/'evaluation_trace.jsonl.gz')
        if error:raise ValueError(error)
        tasks=[];workspace=batch.parent.parent
        for case_id in range(len(registered_native_cases(protocol))):
            decisions=[r for r in rows if r['case_id']==case_id];ticks=[t for r in decisions for t in r['info']['trace']]
            tasks.append(dict(case_id=case_id,source=str(runtime),workspace=str(workspace),config=cfg.to_dict(),ticks=ticks,
                initial=decisions[0]['info']['initial_scenario'],begin=begins[ticks[0]['backend_evidence']['episode_id']],
                termination_reason=read(batch/('case_%04d.json'%case_id))['termination_reason'],
                output=str(execution/'parallel_cases'/('case_%04d'%case_id)),deadline=deadline))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
        status('PARALLEL_MEASURED_DIVERSE_TRAIN_CONVERSION',completed_cases=0)
        completed={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(convert_case,t):t['case_id'] for t in tasks}
            for future in as_completed(futures):
                case=future.result();completed[case['case_id']]=case
                status('PARALLEL_MEASURED_DIVERSE_TRAIN_CONVERSION',completed_cases=len(completed))
        cases=[completed[i] for i in range(len(tasks))]
        check={**tasks[0],'output':str(execution/'serial_check_case0')};serial=convert_case(check)
        if serial!=cases[0]:raise ValueError('Serial/parallel case metadata differs')
        with np.load(Path(tasks[0]['output'])/'arrays.npz',allow_pickle=False) as a,np.load(Path(check['output'])/'arrays.npz',allow_pickle=False) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        arrays=merge_arrays([t['output'] for t in tasks],cases,count)
        output=execution/'direct_views';output.mkdir();dataset=output/'measured_demonstrations.npz';np.savez_compressed(dataset,**arrays)
        provenance=[dict(batch=batch.name,raw_audit_sha256=sha(batch/'raw_audit.json'),protocol_sha256=sha(batch/'protocol.json'),
            trace_sha256=verified['trace_sha256'],controls=count,bridge_projection_source_sha256=protocol['source_hashes']['scripts/carla_bridge37.py'],
            target_config_differences={k:[donor.to_dict().get(k),cfg.to_dict().get(k)] for k in set(donor.to_dict())|set(cfg.to_dict()) if donor.to_dict().get(k)!=cfg.to_dict().get(k)})]
        manifest=dict(scope='Same frozen measured-history functions,independent cases in parallel;no plant commands or new policy fit.',
            sources=provenance,cases=cases,teacher_unique_controls=count,dataset_sha256=sha(dataset),reference_config_sha256=sha(reference_config),
            view_source_sha256=sha(runtime/'navstack/measured_demo.py'),script_sha256=sha(Path(__file__)),
            original_serial_converter_sha256=sha(runtime/'scripts/build_native_measured_views.py'),
            conversion_wall_seconds=time.monotonic()-started,conversion_wall_budget_seconds=wall_seconds,
            serial_parallel_first_case_arrays_exact=True,workers=workers,holdout='Every fifth control per case;correlated diagnostic only',
            returns='Recorded teacher returns;never direct-policy value targets',new_controls=0,supervised_optimizer_steps=0,ppo_updates=0)
        atomic_json(output/'manifest.json',manifest)
        status('ALL36_DIVERSE_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE',physical_controls=count)
        print(json.dumps(dict(teacher_unique_controls=count,conversion_wall_seconds=manifest['conversion_wall_seconds'],workers=workers)),flush=True)
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));status('STOPPED_WITH_FAILURE_NO_RETRY');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','reference-config','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--wait-worker',action='store_true',help='Wait on exact live physical worker,then require owned cleanup and completed original raw audit')
    args=parser.parse_args();run(args.batch,args.reference_config,args.output,args.workers,wait_worker=args.wait_worker)
