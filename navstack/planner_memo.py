"""Opt-in memoization of exact, deterministic static Hybrid A* failures.

No pose bins, action filtering, successful-path reuse, or physical-infeasibility
claim. Node-budget failure is only the result of this particular bounded search.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import time

import numpy as np

from .planning import HybridAStar, PlanResult
from .road_grid import RoadGrid
from .world import World

VERSION = 'exact_static_failure_v1'


@dataclass
class MemoizedFailure(PlanResult):
    cache_hit: bool = True
    cached_expanded_nodes: int = 0


class ExactStaticFailureMemo:
    def __init__(self, planner, max_entries=128):
        if type(planner) is not HybridAStar or type(max_entries) is not int or not 1 <= max_entries <= 512:
            raise ValueError('Require the unchanged Hybrid A* backend and a bounded cache')
        self.planner = planner
        self.max_entries = max_entries
        self._scope = None
        self._failures = OrderedDict()
        self.hits = self.misses = self.bypasses = 0

    @property
    def vehicle(self):
        return self.planner.vehicle

    @property
    def config(self):
        return self.planner.config

    def _geometry(self, world):
        # Import lazily to avoid coupling geometric planner imports to CARLA.
        from .carla_bridge import CarlaWorld
        if (self.config.max_seconds is not None or type(world) not in (World, CarlaWorld)
                or world.dynamic or type(world.road) not in (type(None), RoadGrid)):
            return None
        road = world.road
        # Content is read on every call: in-place geometry edits invalidate the
        # cache too. Do not trust just a payload's original digest or object id.
        road_key = None if road is None else (
            road.width, road.height, road.dx, road.dy,
            road.drivable.shape, road.drivable.dtype.str, road.drivable.tobytes())
        return (id(world), world.width, world.height,
                tuple(tuple(rect) for rect in world.static),
                world.road_collision_backend, bool(getattr(world, 'measured_offroad', False)),
                road_key, json.dumps(asdict(self.vehicle), sort_keys=True, allow_nan=False),
                json.dumps(asdict(self.config), sort_keys=True, allow_nan=False))

    def plan(self, start, goal, world):
        started = time.perf_counter()
        poses = np.asarray([start, goal], dtype=np.float64)
        scope = self._geometry(world)
        if scope is None or poses.shape != (2, 3) or not np.isfinite(poses).all():
            self._scope = None
            self._failures.clear()
            self.bypasses += 1
            return self.planner.plan(start, goal, world)
        if scope != self._scope:
            self._failures.clear()
            self._scope = scope
        key = poses.tobytes()  # Exact float64 inputs, including signed zero.
        previous = self._failures.get(key)
        if previous is not None:
            self.hits += 1
            self._failures.move_to_end(key)
            return MemoizedFailure(previous.status, previous.path.copy(), previous.cost,
                                   0, time.perf_counter() - started, previous.failure_reason,
                                   cached_expanded_nodes=previous.expanded_nodes)
        self.misses += 1
        result = self.planner.plan(start, goal, world)
        # Wall-clock timeouts are never eligible. Do not remember invalid input,
        # trajectory failures, successful paths, or arbitrary backend errors.
        eligible = (result.status == 'no_path' and result.failure_reason == 'open_set_exhausted'
                    or result.status == 'timeout' and result.failure_reason == 'search_budget'
                    and result.expanded_nodes == self.config.max_nodes)
        if eligible and result.path.shape == (0, 3) and result.cost is None:
            self._failures[key] = replace(result, path=result.path.copy())
            if len(self._failures) > self.max_entries:
                self._failures.popitem(last=False)
        return result


def configure_planner_memo(env, protocol):
    version = protocol.get('planner_failure_memo')
    if version is None:
        return env
    if version != VERSION:
        raise ValueError('Unknown registered planner failure memo version')
    env.planner = ExactStaticFailureMemo(env.planner)
    return env
