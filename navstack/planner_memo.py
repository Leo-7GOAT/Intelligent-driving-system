"""Opt-in exact static search memoization, with failure-only defaults.

Keys preserve exact inputs. Optional spatial-result caching still creates and
validates each timed reference anew. Search failure is no infeasibility proof.
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


@dataclass
class MemoizedStaticPlan(MemoizedFailure):
    cache_scope: str = 'exact_static_steering_plan_v3'


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
            road.collision_drivable.shape, road.collision_drivable.dtype.str, road.collision_drivable.tobytes())
        key = (id(world), world.width, world.height,
                tuple(tuple(rect) for rect in world.static),
                world.road_collision_backend, bool(getattr(world, 'measured_offroad', False)),
                road_key, json.dumps(asdict(self.vehicle), sort_keys=True, allow_nan=False),
                json.dumps(asdict(self.config), sort_keys=True, allow_nan=False))
        if getattr(world,'_reference_point_constraints',None) is not None:
            from .reference_point_constraints import signature
            key=key+(signature(world),)
        return key

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


class ExactSteeringFailureMemo(ExactStaticFailureMemo):
    """Exact static failures also keyed by measured steering and speed ceiling.

    Private planning-view identities may change; the entire supported geometry
    remains in the key. Optional success caching stores immutable spatial search
    results; every reference is still parameterized and natively validated anew.
    """
    def __init__(self,planner,max_entries=128,cache_success=False):
        super().__init__(planner,max_entries)
        if type(cache_success) is not bool:raise ValueError('Explicit success-cache boolean required')
        self.cache_success=cache_success

    def plan(self,start,goal,world,*,initial_steering,reference_speed):
        started=time.perf_counter()
        values=np.asarray([*start,*goal,initial_steering,reference_speed],dtype=np.float64)
        scope=self._geometry(world)
        if scope is None or values.shape!=(8,) or not np.isfinite(values).all():
            self.bypasses+=1
            return self.planner.plan(start,goal,world,initial_steering=initial_steering,reference_speed=reference_speed)
        key=(scope[1:],values.tobytes())
        previous=self._failures.get(key)
        if previous is not None:
            self.hits+=1;self._failures.move_to_end(key)
            kind=MemoizedStaticPlan if self.cache_success else MemoizedFailure
            return kind(previous.status,previous.path.copy(),previous.cost,0,
                time.perf_counter()-started,previous.failure_reason,cached_expanded_nodes=previous.expanded_nodes)
        self.misses+=1
        result=self.planner.plan(start,goal,world,initial_steering=initial_steering,reference_speed=reference_speed)
        eligible=(result.status=='no_path' and result.failure_reason=='open_set_exhausted'
            or result.status=='timeout' and result.failure_reason=='search_budget' and result.expanded_nodes==self.config.max_nodes)
        eligible_failure=eligible and result.path.shape==(0,3) and result.cost is None
        eligible_success=(self.cache_success and result.status=='success' and len(result.path)>=2
            and result.cost is not None and np.isfinite(result.cost) and np.isfinite(result.path).all())
        if eligible_failure or eligible_success:
            self._failures[key]=replace(result,path=result.path.copy())
            if len(self._failures)>self.max_entries:self._failures.popitem(last=False)
        return result


def configure_planner_memo(env, protocol):
    version = protocol.get('planner_failure_memo')
    if version is None:
        return env
    if version != VERSION:
        raise ValueError('Unknown registered planner failure memo version')
    env.planner = ExactStaticFailureMemo(env.planner)
    return env
