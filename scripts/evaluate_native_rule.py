"""Existing B3 catalog policy on the exact native training roads, with no PPO claim."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.carla_bridge import BridgeClient, CarlaNavigationEnv
from navstack.config import EnvConfig
from navstack.env import rule_action
from navstack.experiments import write_json, jsonable, signature
from navstack.metrics import timing_summary
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.carla_steering_audit import guard


def require_finished_parent(parent):
    """No handoff into another live or failed training lifecycle."""
    for name in ('summary.json', 'guard_final.json', 'infrastructure/server_stop.json'):
        if not (parent/name).is_file():
            raise ValueError('Parent training and owned-server cleanup must finish first')
    final = read(parent/'guard_final.json')
    if ((parent/'failure.json').exists() or read(parent/'summary.json')['status'] != 'complete'
            or type(final.get('exit_code')) is not int or final['exit_code'] != 0
            or final.get('reason') is not None
            or read(parent/'infrastructure/server_stop.json')['status'] != 'owned_empty_server_stopped'):
        raise ValueError('Parent did not complete and clean up normally')


def feedforward_calibration(path, original):
    """Reuse the measured fit without changing vehicle geometry or physical limits."""
    from navstack.carla_actuation import validate_actuator_interface, AccelerationPI
    path = Path(path).resolve()
    calibration = read(path)
    excluded = {'recommended_adapter', 'throttle_fit_sha256', 'status'}
    if {k:v for k,v in calibration.items() if k not in excluded} != {k:v for k,v in original.items() if k not in excluded}:
        raise ValueError('Feedforward calibration changes physical calibration')
    fit_path = path.with_name('throttle_fit.json')
    if sha(fit_path) != calibration.get('throttle_fit_sha256'):
        raise ValueError('Measured throttle fit hash mismatch')
    fit = read(fit_path)
    gains = calibration['recommended_adapter']
    if gains.get('throttle_model') != fit['coefficients']:
        raise ValueError('Adapter coefficients differ from measured throttle fit')
    validate_actuator_interface(gains, 'ff_state_v1', True, 'native_curve')
    AccelerationPI(gains)
    return path, fit_path


def prepare(parent, source_method, wall_seconds=1800, catalog=None, continuation_feedback=False, speed_policy='native_score', calibration=None, planner_nodes=None, planner_heuristic=None, road_collision_backend=None,hazard_kind=None,rpc_port=None,final_goal_candidate=False,hazard_lateral=None,reference_reuse=False,static_geometry=None,candidate_geometry=None):
    parent = Path(parent).resolve()
    require_finished_parent(parent)
    if not 300 <= wall_seconds <= 3600:
        raise ValueError('Rule evaluation needs a bounded 300..3600 second window')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    old = read(parent/'protocol.json')
    selected_port=old['carla_rpc_port'] if rpc_port is None else rpc_port
    if type(selected_port) is not int or not 1024<=selected_port<=65533:raise ValueError('Invalid native rule RPC port')
    hazard=None
    if static_geometry is not None and hazard_kind!='static_vehicle':raise ValueError('Static body geometry requires a declared parked vehicle')
    if hazard_kind is not None:
        if catalog!='route_frame_terminal' or hazard_kind not in ('static_vehicle','crossing_walker'):
            raise ValueError('Native hazard evaluation requires the explicit new route catalog source')
        from navstack.native_hazards import resolve_native_hazard
        from navstack.native_route import local_layout
        hazard={'version':'native_hazard_v1','kind':hazard_kind,'distance_m':10.}
        if static_geometry is not None:hazard['static_geometry']=static_geometry
        if hazard_lateral is not None:
            if hazard_kind!='crossing_walker':raise ValueError('Explicit hazard lateral requires a walker')
            hazard['lateral_m']=hazard_lateral
        for route in old['routes']:resolve_native_hazard(local_layout(route,route['map'],route['opendrive_sha256']),hazard)
    elif hazard_lateral is not None:
        raise ValueError('Hazard lateral requires a declared native hazard')
    config_path = parent/source_method/'config.json'
    env = EnvConfig.from_dict(read(config_path)['env'])
    if env.method != 'H' or not env.road_observation or env.planner.max_seconds is not None:
        raise ValueError('Require the measured-road hierarchical node-budget interface')
    if catalog not in (None, 'route_frame_terminal'):
        raise ValueError('Unsupported explicit catalog intervention')
    if continuation_feedback and catalog != 'route_frame_terminal':
        raise ValueError('Continuation diagnostic requires the explicit route catalog')
    if speed_policy not in ('native_score', 'paired_low_speed'):
        raise ValueError('Unknown rule speed selection policy')
    if planner_nodes is not None and (type(planner_nodes) is not int or planner_nodes not in (1800,7200) or catalog is None):
        raise ValueError('Explicit route planner contrast supports1800 or7200nodes')
    if planner_heuristic is not None and (planner_heuristic not in ('euclidean','dubins_v1') or catalog is None):
        raise ValueError('Explicit route planner heuristic must be euclidean or dubins_v1')
    parent_planner_nodes=env.planner.max_nodes
    parent_heuristic=env.planner.heuristic_mode
    parent_collision_backend=env.road_collision_backend
    if road_collision_backend is not None:
        if catalog is None:
            raise ValueError('Road collision backend contrast requires an explicit versioned catalog')
        env=EnvConfig.from_dict({**env.to_dict(),'road_collision_backend':road_collision_backend})
    if planner_nodes is not None:
        env.planner.max_nodes=planner_nodes
    if planner_heuristic is not None:
        env.planner.heuristic_mode=planner_heuristic
    adapter_paths = None
    if calibration is not None:
        if catalog != 'route_frame_terminal':
            raise ValueError('Feedforward intervention requires the explicit versioned catalog')
        adapter_paths = feedforward_calibration(calibration, read(parent/'calibration/calibration.json'))
        env = EnvConfig.from_dict({**env.to_dict(), 'actuator_observation':'ff_state_v1'})
    if catalog:
        env = EnvConfig.from_dict({**env.to_dict(), 'subgoal_mode': catalog,
            'route_goal_candidate':final_goal_candidate,
            'route_candidate_geometry':candidate_geometry or env.route_candidate_geometry,
            'reference_reuse':reference_reuse,
            'execution_feedback': 'continuation' if continuation_feedback else env.execution_feedback})
    elif final_goal_candidate or reference_reuse or candidate_geometry:
        raise ValueError('Final goal candidate requires an explicit new catalog experiment')
    for name, digest in old['source_hashes'].items():
        if sha(parent/'source'/name) != digest:
            raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json') != old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_rule')
    batch.mkdir()
    (batch/'infrastructure').mkdir()
    shutil.copytree(parent/'calibration', batch/'calibration')
    if adapter_paths:
        shutil.copy2(adapter_paths[0], batch/'calibration/calibration.json')
        shutil.copy2(adapter_paths[1], batch/'calibration/throttle_fit.json')
    shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('evaluate_native_rule.py', 'audit_native_rule.py', 'watch_existing_pair.py'):
        shutil.copy2(ROOT/'scripts'/name, batch/'source/scripts'/name)
    if catalog:
        # New development version, never mutate or resume the parent's model.
        for folder in ('navstack', 'scripts', 'tests'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and (source.suffix in ('.py', '.ps1')
                        or source.suffix=='.json' and source.is_relative_to(ROOT/'tests/fixtures')):
                    target = batch/'source'/source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        write_json(batch/'source_config.json', {'env': env.to_dict(), 'signature': signature(env),
            'scope': 'New route-catalog rule diagnostic; no inherited learned model'})
    else:
        shutil.copy2(config_path, batch/'source_config.json')
    now = time.time()
    protocol = {'scope': 'Existing rule-only B3 catalog diagnostic on all four native training roads; no learned-policy, independent-seed or final-confirmation claim',
                'parent_batch': parent.name, 'parent_root_relative': parent.relative_to(ROOT).as_posix(),
                'parent_protocol_sha256': sha(parent/'protocol.json'),
                'source_method': source_method, 'source_config_sha256': sha(batch/'source_config.json'),
                'config': env.to_dict(), 'routes': old['routes'],
                'rule': 'existing_navstack.env.rule_action',
                'rule_speed_policy': speed_policy,
                'rule_limitation': 'Existing ego-arc heuristic uses endpoint and primitive checks; do not assume it is the strongest possible B3',
                'created_epoch': now, 'deadline_epoch': now+wall_seconds,
                'calibration_sha256': sha(batch/'calibration/calibration.json'), 'cleanup_owned_carla': True,
                'carla_rpc_host': old['carla_rpc_host'], 'carla_rpc_port': selected_port,
                'carla_rpc_port_span': list(range(selected_port,selected_port+3)),
                'resource_limits': {'wall_seconds': wall_seconds, 'disk_min_GiB': 12, 'ram_min_GiB': 4,
                                    'gpu_stop_temperature_C': 85, 'automatic_retries': 0},
                'disk_free_bytes': shutil.disk_usage(ROOT).free,
                'training_physics_steps': 0, 'optimizer_updates': 0,
                'source_hashes': {p.relative_to(batch/'source').as_posix(): sha(p)
                                  for p in (batch/'source').rglob('*') if p.is_file() and
                                  (p.suffix in ('.py', '.ps1') or p.is_relative_to(batch/'source/tests/fixtures'))}}
    if catalog:
        protocol.update(scope='Rule-only route-frame catalog development coverage on all four training roads; not learned-policy evidence',
            catalog_intervention={'from': read(config_path)['env']['subgoal_mode'], 'to': catalog,
                'continuation_feedback': continuation_feedback, 'rule_missing_continuation_penalty_m': 5. if continuation_feedback else 0.,
                'parent_config_sha256': sha(config_path), 'new_model_required': True,
                'shared_information': 'All policies receive known-route candidate geometry; no future actor truth'},
            rule_limitation='Same connector scoring and up to three bounded candidate searches as goal-frame rule; computation recorded')
    if adapter_paths:
        protocol['actuator_intervention'] = {'mode':'acceleration_ff_v2','observation':'ff_state_v1',
            'parent_calibration_sha256':old['calibration_sha256'],
            'fit_sha256':sha(adapter_paths[1]),'new_model_required':True,
            'scope':'Previously measured feedforward fit; shared four-state observation and matched fresh adaptation required for H/H0/B1'}
    if planner_nodes is not None or planner_heuristic is not None:
        protocol['planner_intervention']={'from_nodes':parent_planner_nodes,'to_nodes':env.planner.max_nodes,
            'from_heuristic':parent_heuristic,'to_heuristic':env.planner.heuristic_mode,
            'new_model_required':True,'scope':'Versioned Hybrid A* node-budget/search-order contrast, shared by H/H0/B3; B1 remains direct control. Physical task and vehicle limits unchanged; compute costs reported separately.'}
    if road_collision_backend is not None:
        protocol['road_collision_intervention']={'from':parent_collision_backend,'to':env.road_collision_backend,
            'new_model_required':True,'scope':'Exact full-body road SAT batching; unchanged scalar oracle for contact-sensitive poses and single-pose checks. Share the version with H/H0/B1/B3 and report compute separately.'}
    if hazard is not None:
        protocol.update(native_hazard=hazard,scope='Real native-road hazard development under shared rule-only B3 planning/MPC; actual parked vehicle or crossing walker, all4known routes retained; not PPO training or final confirmation')
    write_json(batch/'protocol.json', protocol)
    write_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch/'protocol.json')})
    print(batch, flush=True)
    return batch


def selected_speed_action(env, proposal, speed_policy):
    """Explicit B3 speed ablation; no learned policy calls or environment changes."""
    if speed_policy == 'native_score' or proposal == 0:
        return proposal
    if speed_policy != 'paired_low_speed':
        raise ValueError('Unknown rule speed selection policy')
    action = 2*((int(proposal)-1)//2)+1
    target, speed = env.decode_action(action)
    original_target, original_speed = env.decode_action(proposal)
    if target != original_target or speed != .75 or original_speed not in (.75, 1.5):
        raise ValueError('Rule speed ablation requires paired identical-pose catalog entries')
    return action


def evaluate_cases(env, cases, output, health, speed_policy='native_score'):
    """One unchanged rule call and one production environment step per decision."""
    results, controls = [], 0
    with gzip.open(output/'evaluation_trace.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for case_id, case in enumerate(cases):
            health()
            env.reset(seed=7301+case_id, options={'case': case})
            write_json(output/f'scene_{case_id:04d}.json', {'scene': env.scene_evidence, 'initial': env.initial_scenario})
            selection_times, complete_cycle_times = [], []
            while not env.done:
                started = time.perf_counter()
                proposal = rule_action(env)
                action = selected_speed_action(env, proposal, speed_policy)
                selection_seconds = time.perf_counter()-started
                _, _, _, _, info = env.step(action)
                cycle_seconds = time.perf_counter()-started
                controls += info['executed_substeps']
                selection_times.append(selection_seconds)
                complete_cycle_times.append(cycle_seconds)
                stream.write(json.dumps({'case_id': case_id, 'policy_kind': 'rule_only_B3',
                    'policy_action': action, 'rule_proposal': proposal, 'rule_speed_policy': speed_policy,
                    'selection_seconds': selection_seconds,
                    'full_decision_seconds': cycle_seconds, 'info': info}, default=jsonable)+'\n')
                atomic_json(output/'heartbeat.json', {'epoch': time.time(), 'case_id': case_id,
                    'actions': len(selection_times), 'physics_steps': controls})
                if len(selection_times) % 20 == 0:
                    health()
            results.append({'case_id': case_id, **info['episode_metrics'],
                'rule_selection_times': timing_summary(selection_times, env.config.decision_dt),
                'full_decision_times': timing_summary(complete_cycle_times, env.config.decision_dt)})
            write_json(output/f'case_{case_id:04d}.json', results[-1])
            print(json.dumps({'case_id': case_id, 'termination': info['termination_reason'],
                              'physics_steps': info['episode_metrics']['physics_steps']}), flush=True)
    return results, controls


def run(batch):
    batch = Path(batch).resolve()
    protocol = read(batch/'protocol.json')
    # Recorded parent path is a workspace-relative source reference, not a command.
    workspace = batch.parent.parent
    parent = (workspace/protocol['parent_root_relative']).resolve()
    if not parent.is_relative_to(workspace/'runs'):
        raise ValueError('Parent outside workspace run archive')
    require_finished_parent(parent)
    if sha(parent/'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    for name, digest in protocol['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Execute only the frozen source')
    if sha(batch/'source_config.json') != protocol['source_config_sha256']:
        raise ValueError('Source configuration changed')
    if EnvConfig.from_dict(read(batch/'source_config.json')['env']).to_dict() != protocol['config']:
        raise ValueError('Rule configuration differs from the hierarchical interface')
    if sha(batch/'calibration/calibration.json') != protocol['calibration_sha256']:
        raise ValueError('Calibration changed')
    guard(batch, protocol)
    with (batch/'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'epoch': time.time(), 'policy_kind': 'rule_only_B3'}, stream)
    client, env = None, None
    started = time.time()
    try:
        client = BridgeClient(batch)
        config = EnvConfig.from_dict(protocol['config'])
        # Retain H environment semantics; policy_kind identifies rule selection.
        env = CarlaNavigationEnv(config, client)
        cases = [{'scenario': 'native_curve', 'condition': i, 'native_route': r}
                 for i, r in enumerate(protocol['routes'])]
        if protocol.get('native_hazard'):
            cases=[{**case,'native_hazard':protocol['native_hazard']} for case in cases]
        episodes, controls = evaluate_cases(env, cases, batch, lambda: guard(batch, protocol),
                                           protocol.get('rule_speed_policy', 'native_score'))
        write_json(batch/'summary.json', {'status': 'complete', 'policy_kind': 'rule_only_B3',
            'scope': protocol['scope'], 'episodes': episodes, 'evaluation_controls': controls,
            'world_ticks': client.total_ticks, 'reset_ticks': client.total_ticks-controls,
            'training_physics_steps': 0, 'optimizer_updates': 0, 'wall_seconds': time.time()-started})
    except BaseException as exc:
        write_json(batch/'failure.json', {'epoch': time.time(), 'error': repr(exc)})
        raise
    finally:
        if env is not None:
            env.close()
        if client is not None:
            client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare')
    group.add_argument('--batch')
    parser.add_argument('--source-method', default='H_rollout32')
    parser.add_argument('--wall-seconds', type=int, default=1800)
    parser.add_argument('--catalog', choices=['route_frame_terminal'])
    parser.add_argument('--continuation-feedback', action='store_true')
    parser.add_argument('--speed-policy', choices=['native_score', 'paired_low_speed'], default='native_score')
    parser.add_argument('--calibration', type=Path, help='Explicit measured feedforward calibration with adjacent throttle_fit.json')
    parser.add_argument('--planner-nodes',type=int,choices=[1800,7200])
    parser.add_argument('--planner-heuristic',choices=['euclidean','dubins_v1'])
    parser.add_argument('--road-collision-backend',choices=['scalar','batch_sat_v1'])
    parser.add_argument('--hazard-kind',choices=['static_vehicle','crossing_walker'])
    parser.add_argument('--rpc-port',type=int)
    parser.add_argument('--final-goal-candidate',action='store_true')
    parser.add_argument('--hazard-lateral',type=float)
    parser.add_argument('--reference-reuse',action='store_true')
    parser.add_argument('--static-geometry',choices=['axis_aligned_box_v1','projected_box_strips12_v1'])
    parser.add_argument('--candidate-geometry',choices=['route_offsets_v1','kinematic_arcs_goal_v1'])
    args = parser.parse_args()
    prepare(args.prepare, args.source_method, args.wall_seconds, args.catalog, args.continuation_feedback, args.speed_policy, args.calibration, args.planner_nodes, args.planner_heuristic, args.road_collision_backend,args.hazard_kind,args.rpc_port,args.final_goal_candidate,args.hazard_lateral,args.reference_reuse,args.static_geometry,args.candidate_geometry) if args.prepare else run(args.batch)
