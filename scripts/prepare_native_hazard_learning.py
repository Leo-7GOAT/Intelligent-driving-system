"""Fresh real-PPO hazard learning under a completed shared B3 execution version.

The rule supplies configuration/scene provenance only, never labels or actions.
All selected methods start fresh; the known roads remain development conditions.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig,VehicleConfig
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from scripts.audit_native_rule import audit
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import registered_native_cases,training_config


def apply_guarded_progress(protocol,config):
    """Shared, bounded task reward; independent of the selected architecture."""
    from navstack.native_route import local_layout
    import math
    distances=[]
    for route in protocol['routes']:
        layout=local_layout(route,route['map'],route['opendrive_sha256'])
        distances.append(math.dist(layout['path'][0][:2],layout['goal'][:2]))
    reward=config['reward'];horizon=config['deadline']
    if (len(distances)!=4 or max(distances)>40 or horizon>45 or reward['gamma_per_second']!=.99
            or reward['time_weight']!=.1 or reward['risk_weight']!=.4 or reward['comfort_weight']!=.04
            or reward['beta']!=1. or reward['heading_potential_weight']!=0.
            or reward['distance_scale']!=40. or reward['deadline']!=-10.):
        raise ValueError('Guarded progress profile requires its declared distance/time/reward bounds')
    protocol.update(task_progress_weight=.5,unsafe_event_penalty=-90.,
        reward_intervention={'version':'guarded_progress_v1','progress_weight_per_meter':.5,
            'collision':-90.,'out_of_bounds':-90.,'max_initial_distance_m':40.,'max_episode_seconds':45.,
            'scope':'Explicit non-PBRS global-distance progress reward; same physical-step parts for every selected method. Other rewards and task termination gates unchanged. No planner/local-goal reward or safety guarantee.'})


def learning_protocol(parent,controls,wall_seconds,seed,methods,endpoint_filter=False,guarded_progress=False,failure_memo=False):
    old=read(parent/'protocol.json');config=EnvConfig.from_dict(old['config']).to_dict()
    if type(failure_memo) is not bool:raise ValueError('Failure memo must be explicitly enabled or disabled')
    if (not old.get('native_hazard') or config['subgoal_mode']!='route_frame_terminal'
            or len(old['routes'])!=4):
        raise ValueError('Require four full native hazard development roads')
    if (type(controls) is not int or not 640<=controls<=100000
            or type(wall_seconds) is not int or not 900<=wall_seconds<=7200
            or type(seed) is not int or not 0<=seed<2**31):
        raise ValueError('Invalid bounded fresh learning budget or seed')
    if not methods or len(set(methods))!=len(methods) or set(methods)-{'H0','H','B1'}:
        raise ValueError('Select unique H0/H/B1 methods')
    protocol={'scope':'Fresh real CARLA native-hazard PPO development; known roads, no rule teacher, replay, inherited model or formal superiority claim',
        'methods':list(methods),'seed':seed,'routes':old['routes'],'native_hazard':old['native_hazard'],
        'record_initial_parameters':True,
        'controlled_budget_per_method':controls,'episode_seconds':config['deadline'],
        'deterministic_training_route_evaluation':True,'road_candidate_policy':True,'feedback_ablation':True,
        'continuation_feedback':True,'subgoal_mode':'route_frame_terminal',
        'route_goal_candidate':config.get('route_goal_candidate',False),
        'route_candidate_geometry':config.get('route_candidate_geometry','route_offsets_v1'),
        'reference_reuse':config.get('reference_reuse',False),'candidate_head':'unshared_endpoint_v1' if endpoint_filter else 'unshared_v1',
        'flat_action_catalog':'speed_steer_3x7_v1','flat_longitudinal_mode':'speed_target_v1','flat_hold':10,
        'actuator_observation':config.get('actuator_observation','off'),'road_geometry_version':config['road_geometry_version'],
        'road_collision_backend':config.get('road_collision_backend','scalar'),'planner_nodes':config['planner']['max_nodes'],
        'planner_heuristic':config['planner'].get('heuristic_mode','euclidean'),'task_progress_weight':config['reward'].get('progress_weight',0.),
        'rollout_actions':32,'ppo_batch_size':32,'ppo_epochs':10,
        'checkpoint_thresholds':sorted({min(controls,n) for n in (1280,2560,controls)}),
        'reset_ticks_counted_separately':True,'cleanup_owned_carla':True,
        'carla_rpc_host':old['carla_rpc_host'],'carla_rpc_port':old['carla_rpc_port'],
        'carla_rpc_port_span':old['carla_rpc_port_span'],
        'resource_limits':{'wall_seconds':wall_seconds,'disk_min_GiB':12,'ram_min_GiB':4,'gpu_stop_temperature_C':85,'automatic_retries':0},
        'comparison':'H0/H share unshared actor architecture, task and low level; H0 ablates execution/continuation feedback. Equally informed B1 directly selects categorical speed/steering. All selected methods have the same new physical budget; unfinished phases remain incomplete.',
        'stopping_rule':'One fixed development endpoint per selected method. No automatic retry, early success selection or final-test claim.'}
    vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    resolved=training_config(protocol,'H',vehicle).to_dict()
    if endpoint_filter:
        resolved.pop('endpoint_candidate_filter',None)
        protocol['policy_intervention']={'version':'endpoint_clear_v1','scope':'Necessary endpoint freedom only; no local/global connection mask; wait retained. Ablated/unknown feedback remains unmasked. Same condition is already required by B3/planner; no rule actions or teacher.'}
    if guarded_progress:
        apply_guarded_progress(protocol,config)
        # This is a declared common task-reward intervention, not a change to
        # parent vehicle, road, planner, trajectory or MPC configuration.
        protocol_check=training_config(protocol,'H',vehicle).to_dict()
        if endpoint_filter:protocol_check.pop('endpoint_candidate_filter',None)
        for key in ('progress_weight','collision','out_of_bounds'):protocol_check['reward'][key]=config['reward'][key]
        resolved=protocol_check
    if resolved!=config:
        raise ValueError('Fresh H would change the parent shared task/controller/vehicle configuration')
    registered_native_cases(protocol)
    if failure_memo:
        from navstack.planner_memo import VERSION
        protocol['planner_failure_memo']=VERSION
        protocol['planner_compute_intervention']={'scope':'Exact static deterministic failed searches only; unchanged actions, rewards and task gates. Cache hits charge zero new nodes and report original search nodes separately. No driving-performance or realtime claim.'}
    return protocol


def prepare(parent,controls=4096,wall_seconds=7200,seed=11,methods=('H','H0','B1'),endpoint_filter=False,guarded_progress=False,failure_memo=False):
    parent=Path(parent).resolve();require_finished_parent(parent)
    verified=audit(parent)
    if verified!=read(parent/'raw_audit.json'):raise ValueError('Completed B3 raw audit changed')
    protocol=learning_protocol(parent,controls,wall_seconds,seed,methods,endpoint_filter,guarded_progress,failure_memo)
    healthy,reason=resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    registration=parent/f'hazard_learning_seed{seed}{"_endpoint" if endpoint_filter else ""}{"_guarded_progress" if guarded_progress else ""}{"_memo" if failure_memo else ""}_registration.json'
    with ProcessLease(parent/'hazard_learning_prepare.lock'):
        if registration.exists():raise FileExistsError('This parent/seed already has a fresh learning batch; preserve and continue it')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_hazard_learning')
        batch.mkdir();(batch/'infrastructure').mkdir();shutil.copytree(parent/'calibration',batch/'calibration')
        for folder in ('navstack','scripts','tests'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and (source.suffix in ('.py','.ps1') or source.is_relative_to(ROOT/'tests/fixtures')):
                    if '__pycache__' in source.parts:continue
                    destination=batch/'source'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,destination)
        now=time.time()
        protocol.update(created_epoch=now,deadline_epoch=now+wall_seconds,
            parent_batch=parent.name,parent_rule_protocol_sha256=sha(parent/'protocol.json'),
            parent_rule_raw_audit_sha256=sha(parent/'raw_audit.json'),
            calibration_sha256=sha(batch/'calibration/calibration.json'),disk_free_bytes=shutil.disk_usage(ROOT).free,
            source_hashes={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file()})
        write_json(batch/'protocol.json',protocol)
        write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
        with registration.open('x',encoding='utf-8') as stream:
            import json
            json.dump({'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),'methods':list(methods)},stream,indent=2)
    print(batch,flush=True);return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--controls',type=int,default=4096)
    parser.add_argument('--wall-seconds',type=int,default=7200)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--methods',nargs='+',choices=['H','H0','B1'],default=['H','H0','B1'])
    parser.add_argument('--endpoint-filter',action='store_true')
    parser.add_argument('--guarded-progress',action='store_true')
    parser.add_argument('--failure-memo',action='store_true')
    args=parser.parse_args();prepare(args.parent,args.controls,args.wall_seconds,args.seed,args.methods,args.endpoint_filter,args.guarded_progress,args.failure_memo)
