"""Prepare current fine-grid,synchronous-control MPC versus pursuit/PI inputs."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def current_profile():
    return dict(version='current_fixed_reference_control_v1',reference_centered_effort=True,
                new_training_controls=0,new_optimizer_updates=0)


def prepare(output,training,catalog,dependency,owner_pid):
    from navstack.owned_process import OwnedProcess
    from navstack.pursuit_controller import PARAMETERS
    from navstack.reference_manifest import load_fixed_reference
    from navstack.config import VehicleConfig
    from scripts.prepare_contextual_outcome_pilot import live_owner,available_memory
    output,training,catalog,dependency=map(lambda p:Path(p).resolve(),(output,training,catalog,dependency))
    if output.exists() or any((output.parent/(output.name+'_'+name)).exists() for name in ('mpc20','pursuit_pi')):
        raise FileExistsError('Preserve all fixed-reference preparations')
    if output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<24*1024**3 or available_memory()<4:
        raise ValueError('Require D24GiB disk and4GiB RAM for CPU preparation')
    require_finished_parent(training);parent=read(training/'protocol.json');manifest=read(catalog/'manifest.json')
    if (parent['methods']!=['H'] or parent.get('evaluation_only')
            or not read(training/'strict_delivery.json')['strict_delivery_gate_passed']
            or manifest['parent_protocol_sha256']!=sha(training/'protocol.json')
            or manifest['new_physical_controls'] or manifest['PPO_updates']
            or len(manifest['references'])!=4):raise ValueError('Complete current four-reference provenance required')
    identity,command=live_owner(owner_pid)
    if (command.count('--output')!=1 or Path(command[command.index('--output')+1]).resolve()!=dependency
            or not any(Path(arg).name=='run_current_native_deployment_controls.py' for arg in command)):
        raise ValueError('Dependency must identify the live eight-probe coordinator')
    owner=OwnedProcess(**identity,allow_terminate=False)
    if owner.poll() is not None:raise ValueError('Deployment dependency is not live')
    previous=read(dependency/'registration.json');routes=parent['routes'][:4]
    durations=[]
    for i,route in enumerate(routes):
        ref,payload,_=load_fixed_reference(catalog,i,route['geometry_id'])
        durations.append(float(ref.t[-1]))
        if payload.get('policy_called') is not False or payload.get('speed_cap_m_s')!=1.:
            raise ValueError('Explicit fixed production query/current retiming required')
    if max(durations)>35:raise ValueError('Fixed references require nominal<=35s before the common45s test window')
    calibration=read(training/'calibration/calibration.json')
    if calibration['recommended_adapter'].get('control_delivery')!='batch_sync_v1':
        raise ValueError('Current synchronous engine-control interface required')
    vehicle=VehicleConfig(**calibration['vehicle'])
    output.mkdir();shutil.copytree(catalog,output/'fixed_references')
    shutil.copytree(training/'source/navstack',output/'source/navstack',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'scripts',output/'source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(training/'source/scripts/carla_bridge37.py',output/'source/scripts/carla_bridge37.py')
    source_hashes={q.relative_to(output/'source').as_posix():sha(q) for q in (output/'source').rglob('*') if q.is_file()}
    for name,digest in parent['source_hashes'].items():
        if name.startswith('navstack/') and source_hashes[name]!=digest:raise ValueError('Fixed-control physical source changed:'+name)
    jobs=[]
    for name in ('mpc20','pursuit_pi'):
        batch=output.parent/(output.name+'_'+name);batch.mkdir();(batch/'infrastructure').mkdir()
        shutil.copytree(output/'source',batch/'source');shutil.copytree(catalog,batch/'fixed_references')
        shutil.copytree(training/'calibration',batch/'calibration')
        now=time.time()
        p=dict(scope='Current fixed-reference MPC20versus unchanged pursuit/PI;four known TRAINcurves,not PPO/final confirmation',
            current_fixed_reference_control=current_profile(),comparison_controller=name,
            controller_parameters=deepcopy(PARAMETERS) if name=='pursuit_pi' else {},
            fixed_references_manifest_sha256=sha(batch/'fixed_references/manifest.json'),
            carla_rpc_host='127.0.0.1',carla_rpc_port=2120,carla_rpc_port_span=[2120,2121,2122],
            allow_disjoint_carla=True,cleanup_owned_carla=True,created_epoch=now,deadline_epoch=now+14400,
            source_hashes=source_hashes,calibration_sha256=sha(batch/'calibration/calibration.json'),
            routes=routes,road_geometry_version=parent['road_geometry_version'],
            longitudinal_adapter=calibration['recommended_adapter']['longitudinal_mode'],control_delivery='batch_sync_v1',
            registered_hybrid_plans=None,target_speed_m_s=1.,episode_seconds=45.,dt=.05,mpc_horizon=20,
            goal_tolerance_m=.8,goal_speed_m_s=.25,goal_yaw_tolerance_rad=.35,
            resource_limits={**parent['resource_limits'],'wall_seconds':14400,'automatic_retries':0},
            fixed_reference_current_provenance=dict(parent=training.name,parent_protocol_sha256=sha(training/'protocol.json'),
                reference_manifest_sha256=sha(catalog/'manifest.json'),new_training_controls=0,new_PPO_updates=0,
                nominal_reference_seconds=durations,geometry_scope='Four first registered basic TRAINroads,not new final conditions'))
        atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',dict(status='QUEUED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
        jobs.append(dict(controller=name,batch=batch.name,queued_protocol_sha256=sha(batch/'protocol.json')))
    plan=dict(version='current_fixed_reference_control_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        jobs=jobs,source_hashes=source_hashes,reference_manifest_sha256=sha(catalog/'manifest.json'),
        calibration_sha256=sha(training/'calibration/calibration.json'),nominal_reference_seconds=durations,
        dependency_cohort=str(dependency),dependency_sha256=sha(dependency/'registration.json'),dependency_identity=identity,
        runtime_carla_python=previous['runtime_carla_python'],maximum_wait_seconds=86400,maximum_native_jobs=2,
        maximum_evaluation_controls=2*4*900,maximum_simultaneous_servers=2,automatic_retries=0,
        new_training_controls=0,new_PPO_updates=0,new_final_conditions=0,
        scope='Fixed-controller component study,not hierarchical/PPO performance or five independent learners')
    atomic_json(output/'registration.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','training','catalog','dependency'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--owner-pid',type=int,required=True);args=parser.parse_args()
    result=prepare(args.output,args.training,args.catalog,args.dependency,args.owner_pid)
    print('PREPARED_NOT_STARTED: two current fixed-reference controllers')
