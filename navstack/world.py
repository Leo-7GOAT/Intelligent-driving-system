from dataclasses import dataclass, field
import math
import numpy as np
from .vehicle import VehicleState


@dataclass
class MovingObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.65

    def advance(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt


def body_corners(pose, vehicle, margin=0.0):
    x, y, yaw = pose[:3]
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array(
        [
            [-vehicle.rear - margin, -vehicle.width / 2 - margin],
            [vehicle.front + margin, -vehicle.width / 2 - margin],
            [vehicle.front + margin, vehicle.width / 2 + margin],
            [-vehicle.rear - margin, vehicle.width / 2 + margin],
        ]
    )
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def rectangles_overlap(a, b):
    # Separating-axis test for convex oriented rectangles (touching is collision).
    for p in (a, b):
        for edge in (p[1] - p[0], p[3] - p[0]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


@dataclass
class World:
    width: float
    height: float
    static: list = field(default_factory=list)  # xmin,ymin,xmax,ymax (m)
    dynamic: list = field(default_factory=list)

    def out_of_bounds(self, pose, vehicle, margin=0.0):
        p = body_corners(pose, vehicle, margin)
        return bool(
            np.any(p < 0)
            or np.any(p[:, 0] > self.width)
            or np.any(p[:, 1] > self.height)
        )

    def static_collision(self, pose, vehicle, margin=0.0):
        body = body_corners(pose, vehicle, margin)
        lo, hi = body.min(axis=0), body.max(axis=0)
        for x0, y0, x1, y1 in self.static:
            if hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1:
                continue
            if rectangles_overlap(
                body, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            ):
                return True
        return False

    def clearance(self, pose, vehicle, obstacle, future=0.0):
        # Exact circle-to-oriented-rectangle signed clearance (negative means overlap).
        x, y, yaw = pose[:3]
        dx, dy = (
            obstacle.x + obstacle.vx * future - x,
            obstacle.y + obstacle.vy * future - y,
        )
        lx, ly = (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        )
        qx = max(-vehicle.rear - lx, 0, lx - vehicle.front)
        qy = max(abs(ly) - vehicle.width / 2, 0)
        return math.hypot(qx, qy) - obstacle.radius

    def collision(self, pose, vehicle, future=0.0):
        return self.static_collision(pose, vehicle) or any(
            self.clearance(pose, vehicle, o, future) <= 0 for o in self.dynamic
        )

    def free(self, pose, vehicle, margin=0.0):
        return not self.out_of_bounds(
            pose, vehicle, margin
        ) and not self.static_collision(pose, vehicle, margin)

    def free_many(self, poses, vehicle, margin=0.0):
        """Vectorized SAT against static axis-aligned rectangles and map bounds."""
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("poses must be N x 3")
        c, s = np.cos(poses[:,2]), np.sin(poses[:,2])
        half_length = (vehicle.front + vehicle.rear) / 2 + margin
        half_width = vehicle.width / 2 + margin
        shift = (vehicle.front - vehicle.rear) / 2
        cx, cy = poses[:,0]+shift*c, poses[:,1]+shift*s
        extent_x = half_length*np.abs(c)+half_width*np.abs(s)
        extent_y = half_length*np.abs(s)+half_width*np.abs(c)
        valid = (np.isfinite(poses).all(axis=1) & (cx-extent_x >= 0) & (cy-extent_y >= 0)
                 & (cx+extent_x <= self.width) & (cy+extent_y <= self.height))
        for x0,y0,x1,y1 in self.static:
            rx,ry = (x1-x0)/2,(y1-y0)/2
            dx,dy = (x0+x1)/2-cx,(y0+y1)/2-cy
            tolerance = 1e-12  # conservatively classify floating-point contact
            overlap = ((np.abs(dx) <= rx+extent_x+tolerance) & (np.abs(dy) <= ry+extent_y+tolerance)
                       & (np.abs(dx*c+dy*s) <= half_length+rx*np.abs(c)+ry*np.abs(s)+tolerance)
                       & (np.abs(-dx*s+dy*c) <= half_width+rx*np.abs(s)+ry*np.abs(c)+tolerance))
            valid &= ~overlap
        return valid

    def occupied(self, x, y):
        return (
            x < 0
            or y < 0
            or x > self.width
            or y > self.height
            or any(a <= x <= c and b <= y <= d for a, b, c, d in self.static)
        )

    def advance(self, dt):
        for o in self.dynamic:
            o.advance(dt)

    def risk(self, state, vehicle):
        pose = (state.x, state.y, state.yaw)
        # Current and constant-velocity one-second clearance, no future ground truth.
        risks = []
        for o in self.dynamic:
            clear = self.clearance(pose, vehicle, o)
            future_pose = (
                state.x + state.v * math.cos(state.yaw),
                state.y + state.v * math.sin(state.yaw),
                state.yaw,
            )
            predicted = self.clearance(future_pose, vehicle, o, 1.0)
            risks.append(math.exp(-max(0.0, min(clear, predicted)) / 1.5))
        corners = body_corners(pose, vehicle)
        for rect in self.static:
            other = np.array(
                [
                    [rect[0], rect[1]],
                    [rect[2], rect[1]],
                    [rect[2], rect[3]],
                    [rect[0], rect[3]],
                ]
            )
            if rectangles_overlap(corners, other):
                risks.append(1.0)
            else:
                # Minimum vertex-to-edge distance over both polygons.
                distances = []
                for p, q in ((corners, other), (other, corners)):
                    for a, b in zip(q, np.roll(q, -1, axis=0)):
                        ab = b - a
                        t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
                        distances.extend(
                            np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
                        )
                risks.append(math.exp(-min(distances) / 1.5))
        return float(max(risks, default=0.0))


def make_scenario(config, rng, route_length=None):
    w, h = config.width, config.height
    if w < 24 or h < 18:
        raise ValueError("map must fit the full-sized vehicle: >=24 x 18 m")
    name = config.scenario
    if name == "mixed":
        name = str(rng.choice(["empty", "static", "narrow", "dynamic"]))
    offset = {"train": 0.0, "validation": 1.8, "test": -2.3}[config.split]
    y = h / 2 + float(rng.uniform(-0.5, 0.5))
    start, goal = VehicleState(4.0, y), (w - 7.0, y, 0.0)
    if route_length is not None:
        goal = (min(w - 7.0, 4.0 + route_length), y, 0.0)
    world = World(w, h)
    if name == "static":
        world.static.append(
            (w * 0.46 + offset, y - 1.3, w * 0.46 + 1.4 + offset, y + 1.3)
        )
    elif name == "narrow":
        world.static.extend(
            [
                (10 + offset, 1, 20 + offset, y - 2),
                (10 + offset, y + 2, 20 + offset, h - 1),
            ]
        )
    elif name == "dynamic":
        world.dynamic.append(MovingObstacle(w * 0.46 + offset, y - 4, 0, 0.65))
    elif name == "unseen":
        world.static.extend(
            [
                (13 + offset, y - 4, 15 + offset, y + 0.5),
                (21 + offset, y + 2, 23 + offset, y + 6),
            ]
        )
    elif name != "empty":
        raise ValueError(f"unknown scenario {name}")
    # Additional sampled rectangles actually affect geometry; rejection protects start/goal only.
    if name in ("static", "unseen"):
        added = 0
        for _ in range(500):
            if added >= config.obstacle_num:
                break
            x0, y0 = rng.uniform(8, w - 9), rng.uniform(2, h - 3)
            rect = (float(x0), float(y0), float(x0 + 1), float(y0 + 1))
            world.static.append(rect)
            if world.free(
                (start.x, start.y, start.yaw), config.vehicle, 0.5
            ) and world.free(goal, config.vehicle, 0.5):
                added += 1
            else:
                world.static.pop()
        if added < config.obstacle_num:
            raise ValueError("could not place requested obstacle count")
    return world, start, goal, name
