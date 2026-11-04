"""Offline exact-SAT experiment; never connects to or advances CARLA.

Compare actual planner call batches and scattered poses, then interleave full
searches with unchanged node budgets. Scalar remains the production default.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json
from navstack.planning import HybridAStar
from navstack.road_grid import RoadGrid
from navstack.road_grid_batch import ExactRoadBatch
from navstack.world import World, body_corners


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExperimentalWorld(World):
    """Capture production calls and select its explicit backend for this probe."""

    def __init__(self, data, backend, capture=False):
        super().__init__(data['width'], data['height'], data['static'],
                         road=RoadGrid(data['road_grid']),
                         road_collision_backend='batch_sat_v1' if backend == 'batch' else 'scalar')
        self.capture = capture
        self.calls = []

    def free_many(self, poses, vehicle, margin=0.):
        poses = np.asarray(poses, dtype=float)
        if self.capture:
            self.calls.append((poses.copy(), margin))
        return super().free_many(poses, vehicle, margin)


def plan_record(plan):
    return dict(status=plan.status, failure_reason=plan.failure_reason,
                expanded_nodes=plan.expanded_nodes, cost=plan.cost,
                path_sha256=hashlib.sha256(plan.path.tobytes()).hexdigest(),
                path_samples=len(plan.path))


def timing(values):
    return dict(median=float(np.median(values)), min=float(min(values)),
                max=float(max(values)), samples_seconds=values)


def analyze(output, repeats):
    if repeats < 2 or repeats > 10:
        raise ValueError('Use 2..10 interleaved repetitions')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior benchmark evidence')
    fixture = ROOT / 'tests/fixtures/native_wait_state_v1.json'
    case = json.loads(fixture.read_text())
    config = EnvConfig.from_dict(case['config'])
    if config.planner.max_seconds is not None:
        raise ValueError('Reproducible node-budget experiment required')
    data = case['world']
    pose = tuple(case['state'][k] for k in ('x', 'y', 'yaw'))
    archive = output.with_name(output.stem + '_source')
    archive.mkdir(parents=True)
    sources = list((ROOT / 'navstack').glob('*.py')) + [Path(__file__), fixture]
    for source in sources:
        target = archive / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    capture_config = deepcopy(config)
    capture_config.planner.heuristic_mode = 'dubins_v1'
    captured = ExperimentalWorld(data, 'scalar', capture=True)
    captured_plan = HybridAStar(config.vehicle, capture_config.planner).plan(pose, case['goal'], captured)
    if captured_plan.status != 'success' or not captured.calls:
        raise ValueError('Expected production guided-search collision workload')

    # Kernel batches retain the actual planner's batch boundaries and margins.
    # Scattered random poses expose any cost of union-of-chunk candidate cells.
    road = RoadGrid(data['road_grid'])
    batch = ExactRoadBatch(road)
    rng = np.random.default_rng(438)
    scattered = np.c_[rng.uniform(-4, data['width'] + 4, 2048),
                      rng.uniform(-4, data['height'] + 4, 2048),
                      rng.uniform(-4 * np.pi, 4 * np.pi, 2048)]
    workloads = {'actual_guided_calls': captured.calls,
                 'scattered_2048': [(scattered, .1)]}
    kernel = {}
    for label, calls in workloads.items():
        def execute(backend):
            if backend == 'scalar':
                return [np.array([road.intersects(body_corners(p, config.vehicle, m))
                                  for p in poses]) for poses, m in calls]
            return [batch.intersects_many(poses, config.vehicle, m) for poses, m in calls]
        expected = execute('scalar')
        candidate = execute('batch')
        if any(not np.array_equal(a, b) for a, b in zip(expected, candidate)):
            raise ValueError('Collision decisions differ: ' + label)
        durations = {'scalar': [], 'batch': []}
        for repetition in range(repeats):
            for backend in (('scalar', 'batch') if repetition % 2 == 0 else ('batch', 'scalar')):
                start = time.perf_counter()
                execute(backend)
                durations[backend].append(time.perf_counter() - start)
        kernel[label] = dict(batches=len(calls), poses=sum(len(p) for p, _ in calls),
                             exact_flags=True, timings={k: timing(v) for k, v in durations.items()})
        print(json.dumps({'kernel': label, **kernel[label]}), flush=True)

    searches = []
    for mode in ('euclidean', 'dubins_v1'):
        durations = {'scalar': [], 'batch': []}
        expected = None
        for repetition in range(repeats):
            for backend in (('scalar', 'batch') if repetition % 2 == 0 else ('batch', 'scalar')):
                candidate_config = deepcopy(config)
                candidate_config.planner.heuristic_mode = mode
                world = ExperimentalWorld(data, backend)
                start = time.perf_counter()
                plan = HybridAStar(config.vehicle, candidate_config.planner).plan(pose, case['goal'], world)
                durations[backend].append(time.perf_counter() - start)
                record = plan_record(plan)
                if expected is None:
                    expected = record
                if record != expected:
                    raise ValueError('Full search changed status/cost/nodes/path: ' + mode)
        searches.append(dict(heuristic=mode, **expected, exact_search=True,
                             timings={k: timing(v) for k, v in durations.items()}))
        print(json.dumps(searches[-1]), flush=True)
    result = dict(scope='Offline experimental full-body road SAT batching; no live runtime change, CARLA controls, PPO updates or realtime claim',
                  fixture_sha256=sha(fixture), seed=438, repeats=repeats,
                  source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                  new_controls=0, optimizer_updates=0, kernel=kernel, searches=searches)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    analyze(args.output, args.repeats)
