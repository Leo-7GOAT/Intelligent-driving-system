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
from navstack.experiments import write_json, jsonable
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


def prepare(parent, source_method, wall_seconds=1800):
    parent = Path(parent).resolve()
    require_finished_parent(parent)
    if not 300 <= wall_seconds <= 3600:
        raise ValueError('Rule evaluation needs a bounded 300..3600 second window')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    old = read(parent/'protocol.json')
    config_path = parent/source_method/'config.json'
    env = EnvConfig.from_dict(read(config_path)['env'])
    if env.method != 'H' or not env.road_observation or env.planner.max_seconds is not None:
        raise ValueError('Require the measured-road hierarchical node-budget interface')
    for name, digest in old['source_hashes'].items():
        if sha(parent/'source'/name) != digest:
            raise ValueError('Parent source changed')
    if sha(parent/'calibration/calibration.json') != old['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_rule')
    batch.mkdir()
    (batch/'infrastructure').mkdir()
    shutil.copytree(parent/'calibration', batch/'calibration')
    shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('evaluate_native_rule.py', 'audit_native_rule.py', 'watch_existing_pair.py'):
        shutil.copy2(ROOT/'scripts'/name, batch/'source/scripts'/name)
    shutil.copy2(config_path, batch/'source_config.json')
    now = time.time()
    protocol = {'scope': 'Existing rule-only B3 catalog diagnostic on all four native training roads; no learned-policy, independent-seed or final-confirmation claim',
                'parent_batch': parent.name, 'parent_root_relative': parent.relative_to(ROOT).as_posix(),
                'parent_protocol_sha256': sha(parent/'protocol.json'),
                'source_method': source_method, 'source_config_sha256': sha(config_path),
                'config': env.to_dict(), 'routes': old['routes'],
                'rule': 'existing_navstack.env.rule_action',
                'rule_limitation': 'Existing ego-arc heuristic uses endpoint and primitive checks; do not assume it is the strongest possible B3',
                'created_epoch': now, 'deadline_epoch': now+wall_seconds,
                'calibration_sha256': old['calibration_sha256'], 'cleanup_owned_carla': True,
                'carla_rpc_host': old['carla_rpc_host'], 'carla_rpc_port': old['carla_rpc_port'],
                'carla_rpc_port_span': old['carla_rpc_port_span'],
                'resource_limits': {'wall_seconds': wall_seconds, 'disk_min_GiB': 12, 'ram_min_GiB': 4,
                                    'gpu_stop_temperature_C': 85, 'automatic_retries': 0},
                'disk_free_bytes': shutil.disk_usage(ROOT).free,
                'training_physics_steps': 0, 'optimizer_updates': 0,
                'source_hashes': {p.relative_to(batch/'source').as_posix(): sha(p)
                                  for p in (batch/'source').rglob('*') if p.is_file() and p.suffix in ('.py', '.ps1')}}
    write_json(batch/'protocol.json', protocol)
    write_json(batch/'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch/'protocol.json')})
    print(batch, flush=True)
    return batch


def evaluate_cases(env, cases, output, health):
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
                action = rule_action(env)
                selection_seconds = time.perf_counter()-started
                _, _, _, _, info = env.step(action)
                cycle_seconds = time.perf_counter()-started
                controls += info['executed_substeps']
                selection_times.append(selection_seconds)
                complete_cycle_times.append(cycle_seconds)
                stream.write(json.dumps({'case_id': case_id, 'policy_kind': 'rule_only_B3',
                    'policy_action': action, 'selection_seconds': selection_seconds,
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
        episodes, controls = evaluate_cases(env, cases, batch, lambda: guard(batch, protocol))
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
    args = parser.parse_args()
    prepare(args.prepare, args.source_method, args.wall_seconds) if args.prepare else run(args.batch)
