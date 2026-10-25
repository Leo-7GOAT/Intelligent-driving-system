"""Forward-only Hybrid A* and a separate, corrected 8-connected grid A*."""

from dataclasses import dataclass
import heapq
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from .vehicle import wrap
from .dubins_guide import dubins_distance


@dataclass
class PlanResult:
    status: str
    path: np.ndarray  # columns x,y,yaw, in m,m,rad
    cost: float | None
    expanded_nodes: int
    planning_time: float
    failure_reason: str | None = None


def grid_search(start, goal, obstacles, shape, dijkstra=False):
    """Euclidean edges and admissible octile heuristic, no diagonal corner cutting."""
    width, height = (shape, shape) if isinstance(shape, int) else shape

    def free(p):
        return 0 <= p[0] < width and 0 <= p[1] < height and p not in obstacles

    def heuristic(p):
        dx, dy = abs(p[0] - goal[0]), abs(p[1] - goal[1])
        return 0 if dijkstra else max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    if not free(start) or not free(goal):
        return [], math.inf, 0
    best, parent, queue, expanded = (
        {start: 0.0},
        {},
        [(heuristic(start), 0.0, start)],
        0,
    )
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != best[current]:
            continue
        expanded += 1
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1], cost, expanded
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            p = (current[0] + dx, current[1] + dy)
            if not free(p) or (
                dx
                and dy
                and (
                    not free((current[0] + dx, current[1]))
                    or not free((current[0], current[1] + dy))
                )
            ):
                continue
            g = cost + math.hypot(dx, dy)
            if g < best.get(p, math.inf):
                best[p], parent[p] = g, current
                heapq.heappush(queue, (g + heuristic(p), g, p))
    return [], math.inf, expanded


def primitive(pose, steer, length, wheelbase, step):
    """Exact constant-steer bicycle arc, sampled for full-body collision checking."""
    distances = np.linspace(0, length, max(2, math.ceil(length / step) + 1))[1:]
    x, y, yaw = pose
    k = math.tan(steer) / wheelbase
    angles = yaw + k * distances
    if abs(k) < 1e-10:
        return np.column_stack(
            (x + distances * math.cos(yaw), y + distances * math.sin(yaw), angles)
        )
    return np.column_stack(
        (
            x + (np.sin(angles) - math.sin(yaw)) / k,
            y - (np.cos(angles) - math.cos(yaw)) / k,
            angles,
        )
    )


def analytic_connection(start, goal, world, vehicle, spacing=0.06):
    """Bounded endpoint-tangent connector, checked against the full vehicle body.

    This is a sufficient feasibility witness, not proof of unreachability when
    rejected. The Hybrid A* search remains the fallback. Three fixed tangent
    scales and deterministic dense sampling avoid a wall-time acceptance rule.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    distance = float(np.linalg.norm(goal[:2] - start[:2]))
    if distance < 0.1:
        return None
    direction = (goal[:2] - start[:2]) / distance
    if np.dot(direction, [math.cos(start[2]), math.sin(start[2])]) <= 0:
        return None
    if not world.free(start, vehicle) or not world.free(goal, vehicle):
        return None
    for scale in (1.0, 1.3, .75):
        derivatives = distance * scale * np.array([[math.cos(p[2]), math.sin(p[2])] for p in (start, goal)])
        curve = CubicHermiteSpline([0, 1], [start[:2], goal[:2]], derivatives)
        # Oversample the parameter; reject any resulting chord above spacing.
        u = np.linspace(0, 1, max(16, math.ceil(distance * 4 / spacing) + 1))
        xy, first, second = curve(u), curve(u, 1), curve(u, 2)
        speed = np.linalg.norm(first, axis=1)
        if np.any(speed < .05) or np.any(first @ direction <= 0):
            continue
        curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / speed**3
        if np.max(np.abs(curvature)) > 0.98 / vehicle.min_turn_radius:
            continue
        if np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)) > spacing:
            continue
        yaw = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
        path = np.column_stack((xy, yaw))
        margin = spacing / 2 * (1 + math.hypot(vehicle.front, vehicle.width/2) / vehicle.min_turn_radius)
        if world.free_many(path, vehicle, margin).all():
            return path
    return None


class HybridAStar:
    def __init__(self, vehicle, config):
        self.vehicle, self.config = vehicle, config

    def plan(self, start, goal, world):
        started = time.perf_counter()
        c, v = self.config, self.vehicle

        def result(status, path=None, cost=None, count=0, reason=None):
            return PlanResult(
                status,
                np.array(path).reshape(-1, 3) if path is not None else np.empty((0, 3)),
                cost,
                count,
                time.perf_counter() - started,
                reason,
            )

        if not world.free(start, v) or not world.free(goal, v):
            return result("invalid", reason="start_or_goal_body_collision_or_bounds")
        if c.analytic_expansion:
            direct = analytic_connection(start, goal, world, v)
            if direct is not None:
                return result("success", direct, float(np.linalg.norm(np.diff(direct[:, :2], axis=0), axis=1).sum()), 1)
        steers = np.linspace(-v.max_steer, v.max_steer, 5)
        local_segments = None
        if c.vectorized_primitives:
            local_segments = np.asarray([primitive((0,0,0),delta,c.primitive_length,v.wheelbase,c.collision_step) for delta in steers])

        def key(pose, steer_id):
            # Previous steering is part of search state because transition cost uses it.
            return (
                round(pose[0] / c.xy_resolution),
                round(pose[1] / c.xy_resolution),
                round((wrap(pose[2]) + math.pi) / c.yaw_resolution),
                steer_id,
            )

        def geometric_heuristic(p):
            distance = max(
                0, math.hypot(p[0] - goal[0], p[1] - goal[1]) - c.goal_tolerance
            )
            turn = (
                max(0, abs(wrap(p[2] - goal[2])) - c.yaw_tolerance) * v.min_turn_radius
            )
            return max(distance, turn)

        def heuristic(p):
            base=geometric_heuristic(p)
            if c.heuristic_mode=='dubins_v1' and base>0:
                # Exact-pose obstacle-free forward geometry guides queue order.
                # With terminal tolerances/weighted search this is an engineering
                # guide, not an admissibility or optimality claim. It never
                # changes goal acceptance or substitutes an unvalidated path.
                return max(base,dubins_distance(p,goal,v.min_turn_radius))
            return base

        nodes = [(np.array(start, dtype=float), -1, np.empty((0, 3)), 2, 0.0)]
        queue, best = [(c.heuristic_weight * heuristic(start), 0)], {key(start, 2): 0.0}
        expanded = 0
        # Inflated envelopes cover interpolation between samples, including body rotation.
        margin = (
            c.collision_step
            / 2
            * (1 + math.hypot(v.front, v.width / 2) / v.min_turn_radius)
        )
        while queue:
            if expanded >= c.max_nodes or (
                c.max_seconds is not None
                and time.perf_counter() - started >= c.max_seconds
            ):
                return result("timeout", count=expanded, reason="search_budget")
            _, idx = heapq.heappop(queue)
            pose, parent, segment, si, g = nodes[idx]
            if g > best.get(key(pose, si), math.inf) + 1e-9:
                continue
            expanded += 1
            if c.analytic_expansion and expanded % 32 == 0:
                connection = analytic_connection(pose, goal, world, v)
                if connection is not None:
                    pieces = [connection[1:]]
                    current_idx = idx
                    while nodes[current_idx][1] >= 0:
                        pieces.append(nodes[current_idx][2])
                        current_idx = nodes[current_idx][1]
                    path = np.vstack([np.asarray(start).reshape(1, 3)] + pieces[::-1])
                    return result("success", path, g + float(np.linalg.norm(np.diff(connection[:, :2], axis=0), axis=1).sum()), expanded)
            if (
                geometric_heuristic(pose) <= 1e-9
                and abs(wrap(pose[2] - goal[2])) <= c.yaw_tolerance
            ):
                pieces = []
                while nodes[idx][1] >= 0:
                    pieces.append(nodes[idx][2])
                    idx = nodes[idx][1]
                path = np.vstack([np.array(start).reshape(1, 3)] + pieces[::-1])
                return result("success", path, g, expanded)
            transformed = None
            if local_segments is not None:
                co,si_yaw = math.cos(pose[2]),math.sin(pose[2])
                transformed = np.empty_like(local_segments)
                transformed[:,:,0] = pose[0]+co*local_segments[:,:,0]-si_yaw*local_segments[:,:,1]
                transformed[:,:,1] = pose[1]+si_yaw*local_segments[:,:,0]+co*local_segments[:,:,1]
                transformed[:,:,2] = pose[2]+local_segments[:,:,2]
                valid_segments = world.free_many(transformed.reshape(-1,3),v,margin).reshape(5,-1).all(axis=1)
            for next_si, delta in enumerate(steers):
                seg = transformed[next_si] if transformed is not None else primitive(pose,delta,c.primitive_length,v.wheelbase,c.collision_step)
                if not (valid_segments[next_si] if transformed is not None else all(world.free(p,v,margin) for p in seg)):
                    continue
                end = seg[-1]
                ng = (
                    g
                    + c.primitive_length
                    + c.steer_cost * abs(delta)
                    + c.steer_change_cost * abs(delta - steers[si])
                )
                nk = key(end, next_si)
                if ng + 1e-9 < best.get(nk, math.inf):
                    best[nk] = ng
                    nodes.append((end, idx, seg, next_si, ng))
                    heapq.heappush(
                        queue,
                        (ng + c.heuristic_weight * heuristic(end), len(nodes) - 1),
                    )
        return result("no_path", count=expanded, reason="open_set_exhausted")


class GridAStarPlanner:
    """B2: grid A* followed by endpoint-tangent cubic curves, revalidated in full."""

    def __init__(self, vehicle, config):
        self.vehicle, self.config = vehicle, config

    def plan(self, start, goal, world):
        begin = time.perf_counter()
        v, c = self.vehicle, self.config

        def result(status, path=None, cost=None, n=0, reason=None):
            return PlanResult(
                status,
                np.array(path).reshape(-1, 3) if path is not None else np.empty((0, 3)),
                cost,
                n,
                time.perf_counter() - begin,
                reason,
            )

        if not world.free(start, v) or not world.free(goal, v):
            return result("invalid", reason="start_or_goal_body_collision_or_bounds")
        r = c.grid_resolution
        shape = (math.floor(world.width / r) + 1, math.floor(world.height / r) + 1)
        # Inflate by half width; final oriented-envelope test handles front/rear extent.
        blocked = set()
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                x, y = ix * r, iy * r
                if (
                    x < v.width / 2
                    or y < v.width / 2
                    or x > world.width - v.width / 2
                    or y > world.height - v.width / 2
                    or (world.road is not None and world.road.disc_intersects(x, y, v.width/2))
                    or any(
                        a - v.width / 2 <= x <= d + v.width / 2
                        and b - v.width / 2 <= y <= e + v.width / 2
                        for a, b, d, e in world.static
                    )
                ):
                    blocked.add((ix, iy))
        raw, cost, n = grid_search(
            (round(start[0] / r), round(start[1] / r)),
            (round(goal[0] / r), round(goal[1] / r)),
            blocked,
            shape,
        )
        if len(raw) < 2:
            return result("no_path", n=n, reason="grid_search_empty")
        xy = np.array(raw, dtype=float) * r
        xy[0], xy[-1] = start[:2], goal[:2]
        if c.grid_postprocess == "body_spline_v1":
            from .grid_smoothing import body_spline
            path = body_spline(xy, start, goal, world, v)
            if path is None:
                return result("infeasible", n=n, reason="body_spline_uncertified")
            return result("success", path, float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()), n)
        # Shortcut greedily through visible center-line segments on the inflated grid.
        simple = [xy[0]]
        i = 0
        while i < len(xy) - 1:
            next_i = i + 1
            for j in range(len(xy) - 1, i, -1):
                pts = np.linspace(
                    xy[i],
                    xy[j],
                    max(2, math.ceil(np.linalg.norm(xy[j] - xy[i]) / (r * 0.3))),
                )
                if all((round(p[0] / r), round(p[1] / r)) not in blocked for p in pts):
                    next_i = j
                    break
            simple.append(xy[next_i])
            i = next_i
        # Try multiple waypoint densities; accept only collision-free, curvature-feasible fits.
        for points in (
            np.array(simple),
            xy[::2] if len(xy) % 2 else np.vstack([xy[::2], xy[-1]]),
            xy,
        ):
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
            points = points[keep]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            if len(s) < 2:
                continue
            curve = CubicSpline(
                s,
                points,
                bc_type=(
                    (1, np.array([math.cos(start[2]), math.sin(start[2])])),
                    (1, np.array([math.cos(goal[2]), math.sin(goal[2])])),
                ),
            )
            u = np.linspace(0, s[-1], math.ceil(s[-1] / 0.06) + 1)
            pos, d1, d2 = curve(u), curve(u, 1), curve(u, 2)
            norm = np.linalg.norm(d1, axis=1)
            k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(norm**3, 1e-10)
            path = np.column_stack([pos, np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))])
            if (
                min(norm) > 0.1
                and max(abs(k)) <= 1 / v.min_turn_radius + 1e-4
                and all(world.free(p, v, 0.10) for p in path)
            ):
                return result("success", path, float(s[-1]), n)
        return result("infeasible", n=n, reason="postprocess_collision_or_curvature")
