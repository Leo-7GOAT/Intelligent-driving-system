"""Compare an archived road kernel with exact batched corners, offline only."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.planning import HybridAStar
from navstack.road_grid_batch import ExactRoadBatch
from scripts.benchmark_road_batch import ExperimentalWorld, plan_record, timing


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark(baseline, output, repeats=3, fixture=None):
    baseline, output = Path(baseline), Path(output)
    if output.exists() or not 2 <= repeats <= 10:
        raise ValueError('Use a new output and2..10interleaved repetitions')
    protocol = json.loads((baseline/'protocol.json').read_text(encoding='utf-8'))
    name = 'navstack/road_grid_batch.py'
    original = baseline/'source'/name
    if sha(original) != protocol['source_hashes'][name]:
        raise ValueError('Archived baseline source changed')
    dependencies = {}
    for name in ('config', 'planning', 'world', 'road_grid', 'vehicle', 'dubins_guide'):
        relative = 'navstack/'+name+'.py'
        archived, current = baseline/'source'/relative, ROOT/relative
        if (sha(archived) != protocol['source_hashes'][relative]
                or archived.read_text(encoding='utf-8') != current.read_text(encoding='utf-8')):
            raise ValueError('Comparison changes another physical/search dependency: '+relative)
        dependencies[relative] = {'baseline_sha256': sha(archived), 'current_sha256': sha(current), 'text_equal': True}
    spec = importlib.util.spec_from_file_location('navstack._archived_exact_road_batch', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = Path(fixture) if fixture is not None else ROOT/'tests/fixtures/native_wait_state_v1.json'
    case = json.loads(fixture.read_text(encoding='utf-8'))
    config = EnvConfig.from_dict(case['config'])
    config.planner.heuristic_mode = 'dubins_v1'
    if config.planner.max_seconds is not None:
        raise ValueError('Fixed node-budget comparison required')
    pose = tuple(case['state'][key] for key in ('x', 'y', 'yaw'))

    def world(kernel, capture=False):
        instance = ExperimentalWorld(case['world'], 'batch', capture=capture)
        instance._road_batch = kernel(instance.road)
        return instance

    captured = world(module.ExactRoadBatch, True)
    def signature(plan):
        record = plan_record(plan)
        if record['cost'] is not None and not np.isfinite(record['cost']):
            record['cost'] = None
        return record

    expected_plan = signature(HybridAStar(config.vehicle, config.planner).plan(pose, case['goal'], captured))
    if not captured.calls:
        raise ValueError('No actual planner collision batches captured')
    kernels = {'baseline': module.ExactRoadBatch(captured.road), 'batched_corners': ExactRoadBatch(captured.road)}
    expected = [kernels['baseline'].intersects_many(poses, config.vehicle, margin) for poses, margin in captured.calls]
    for (poses, margin), flags in zip(captured.calls, expected):
        if not np.array_equal(flags, kernels['batched_corners'].intersects_many(poses, config.vehicle, margin)):
            raise ValueError('Actual planner collision flags changed')
    kernel_times = {key: [] for key in kernels}
    search_times = {key: [] for key in kernels}
    for repetition in range(repeats):
        order = list(kernels) if repetition % 2 == 0 else list(reversed(kernels))
        for key in order:
            start = time.perf_counter()
            for poses, margin in captured.calls:
                kernels[key].intersects_many(poses, config.vehicle, margin)
            kernel_times[key].append(time.perf_counter()-start)
            instance = world(type(kernels[key]))
            start = time.perf_counter()
            plan = HybridAStar(config.vehicle, deepcopy(config.planner)).plan(pose, case['goal'], instance)
            search_times[key].append(time.perf_counter()-start)
            if signature(plan) != expected_plan:
                raise ValueError('Full search status/cost/nodes/path changed')
    result = {'scope': 'Offline saved-query compute comparison;no CARLA ticks,PPO updates or active source changes.',
              'baseline_batch': baseline.name, 'fixture_sha256': sha(fixture),
              'baseline_kernel_sha256': sha(original), 'candidate_kernel_sha256': sha(ROOT/'navstack/road_grid_batch.py'),
              'dependencies': dependencies, 'repeats': repeats, 'actual_planner_batches': len(captured.calls),
              'actual_planner_poses': sum(len(poses) for poses, _ in captured.calls), 'collision_flags_exact': True,
              'full_search_exact': expected_plan, 'kernel_timings': {k: timing(v) for k, v in kernel_times.items()},
              'search_timings': {k: timing(v) for k, v in search_times.items()},
              'concurrent_load': 'An independently frozen CARLA development phase may be running;interleaved samples are not a hardware/realtime certification.',
              'new_physical_controls': 0, 'optimizer_updates': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fixture', help='Optional recorded planning-query fixture;no simulator execution')
    args = parser.parse_args()
    benchmark(args.baseline, args.output, args.repeats, args.fixture)
