"""Reproduce repeated static-search failures from an audited CARLA endpoint.

No environment step, plant integration, policy update, or new CARLA frame is
performed. Each failing case contributes its most frequent exact search input.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.planner_memo import ExactStaticFailureMemo
from navstack.planning import HybridAStar
from scripts.audit_native_method_endpoint import audit, read, read_prefix, sha
from scripts.audit_native_policy_sequence import observation_env, advance_observation


def canonical(result):
    return result.status, result.failure_reason, result.cost, result.path.shape, result.path.tobytes()


def benchmark(batch, method, output, repeats=100):
    batch, output = Path(batch).resolve(), Path(output)
    if output.exists() or type(repeats) is not int or not 2 <= repeats <= 1000:
        raise ValueError('Require a new output and bounded lookup count')
    folder = batch / method
    if audit(batch, method) != read(folder / 'endpoint_raw_audit.json'):
        raise ValueError('Parent raw audit changed')
    config = EnvConfig.from_dict(read(folder / 'config.json')['env'])
    if config.planner.max_seconds is not None:
        raise ValueError('Only deterministic node-budget endpoints are eligible')
    for name in ('navstack/planning.py', 'navstack/world.py', 'navstack/road_grid.py', 'navstack/subgoals.py'):
        if sha(ROOT / name) != sha(batch / 'source' / name):
            raise ValueError('Geometric planner implementation differs from the recorded source')
    rows, error = read_prefix(folder / 'evaluation_trace.jsonl.gz')
    if error:
        raise ValueError(error)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case_id']].append(row)
    bridge = batch / 'infrastructure' / read(folder / 'endpoint_raw_audit.json')['bridge_response_stream']
    bridge_rows, bridge_error = read_prefix(bridge)
    if bridge_error: raise ValueError(bridge_error)
    begins = {wrapper['response']['episode_id']: wrapper['response']
              for wrapper in bridge_rows if 'scene' in wrapper.get('response', {})}
    results = []
    for case, decisions in grouped.items():
        # A training+evaluation endpoint need not store scene_*.json for its
        # evaluation. The bridge begin response contains that exact scene.
        episode = decisions[0]['info']['trace'][0]['backend_evidence']['episode_id']
        begin = begins[episode]
        initial = decisions[0]['info']['initial_scenario']
        env = observation_env(config, begin['scene'], initial)
        env.world.road_collision_backend = config.road_collision_backend
        inputs = {}
        for row in decisions:
            plan = row['info'].get('plan') or {}
            if row['policy_action'] and row['info']['mode'] == 'plan_failed' and plan.get('status') in ('timeout', 'no_path'):
                start = (env.state.x, env.state.y, env.state.yaw)
                target, _ = env.decode_action(row['policy_action'])
                key = np.asarray([start, target], dtype=np.float64).tobytes()
                entry = inputs.setdefault(key, {'start': start, 'goal': target, 'count': 0, 'recorded': plan})
                entry['count'] += 1
            advance_observation(env, row)
        if not inputs: continue
        selected = max(inputs.values(), key=lambda entry: entry['count'])
        start, goal = selected['start'], selected['goal']
        base = HybridAStar(config.vehicle, config.planner)
        begin_time = time.perf_counter()
        expected = base.plan(start, goal, env.world)
        uncached_seconds = time.perf_counter() - begin_time
        recorded = selected['recorded']
        if (expected.status, expected.failure_reason, expected.expanded_nodes) != (
                recorded['status'], recorded['failure_reason'], recorded['expanded_nodes']):
            raise ValueError('Recorded deterministic search failure does not reproduce')
        memo = ExactStaticFailureMemo(HybridAStar(config.vehicle, config.planner))
        fill_started = time.perf_counter()
        fill = memo.plan(start, goal, env.world)
        fill_seconds = time.perf_counter() - fill_started
        assert canonical(fill) == canonical(expected) and fill.expanded_nodes == expected.expanded_nodes
        durations = []
        for _ in range(repeats):
            begin_time = time.perf_counter()
            result = memo.plan(start, goal, env.world)
            durations.append(time.perf_counter() - begin_time)
            assert canonical(result) == canonical(expected) and result.cache_hit
            assert result.expanded_nodes == 0 and result.cached_expanded_nodes == expected.expanded_nodes
        results.append({'case_id': case, 'start': start, 'goal': goal,
                        'observed_exact_queries': selected['count'], 'status': expected.status,
                        'failure_reason': expected.failure_reason, 'original_expanded_nodes': expected.expanded_nodes,
                        'uncached_single_wall_seconds': uncached_seconds,
                        'memo_fill_wall_seconds': fill_seconds, 'cache_lookup_count': repeats,
                        'cache_lookup_wall_seconds_mean': float(np.mean(durations)),
                        'cache_lookup_wall_seconds_max': max(durations),
                        'canonical_search_outcomes_identical': True, 'new_nodes_per_cache_lookup': 0})
    result = {'scope': 'Offline geometric search benchmark on completed audited CARLA states. '
                       'Per failing case, most frequent exact query; lookup timings exclude first fill. '
                       'Not end-to-end speedup, driving improvement or physical infeasibility proof.',
              'batch': batch.name, 'method': method, 'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json'),
              'script_sha256': sha(Path(__file__)), 'memo_source_sha256': sha(ROOT / 'navstack/planner_memo.py'),
              'new_controls': 0, 'optimizer_updates': 0, 'cases': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--method', default='H')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    benchmark(args.batch, args.method, args.output)
