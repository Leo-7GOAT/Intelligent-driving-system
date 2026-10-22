"""Shared measured-road geometry; occupied raster cells are closed rectangles."""
import hashlib
import json
import math
import numpy as np


ROAD_SIZE = 64


def road_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class RoadGrid:
    def __init__(self, payload):
        data = dict(payload)
        digest = data.pop("sha256")
        if road_digest(data) != digest:
            raise ValueError("Road grid hash mismatch")
        self.width, self.height = float(data["width"]), float(data["height"])
        cells = np.asarray(data["drivable"])
        if (data["version"] not in ("carla_driving_64_v1","carla_driving_refined_union_64_v2","carla_driving_indexed_union_64_v3") or cells.shape != (ROAD_SIZE, ROAD_SIZE)
                or not np.isin(cells, [0, 1]).all() or not np.isfinite([self.width, self.height]).all()
                or not 0 < min(self.width, self.height) <= max(self.width, self.height) <= 64):
            raise ValueError("Invalid measured road grid")
        self.drivable = cells.astype(bool)
        self.dx, self.dy = self.width / ROAD_SIZE, self.height / ROAD_SIZE
        self.sha256 = digest
        self.payload = payload

    def occupied(self, x, y):
        if not math.isfinite(x+y) or not 0 <= x <= self.width or not 0 <= y <= self.height:
            return True
        return not self.drivable[min(ROAD_SIZE-1, int(y/self.dy)), min(ROAD_SIZE-1, int(x/self.dx))]

    def rectangles(self, corners, reach=0.):
        lo, hi = corners.min(axis=0)-reach, corners.max(axis=0)+reach
        first = np.maximum(0, np.floor(lo/[self.dx, self.dy]).astype(int)-1)
        last = np.minimum(ROAD_SIZE, np.floor(hi/[self.dx, self.dy]).astype(int)+2)
        if np.any(last <= first):
            return np.empty((0, 4))
        yy, xx = np.nonzero(~self.drivable[first[1]:last[1], first[0]:last[0]])
        xx, yy = xx+first[0], yy+first[1]
        return np.c_[xx*self.dx, yy*self.dy, (xx+1)*self.dx, (yy+1)*self.dy]

    def intersects(self, corners):
        rects = self.rectangles(corners)
        if not len(rects):
            return False
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        overlap = ((rects[:, 0] <= hi[0]+1e-12) & (rects[:, 2] >= lo[0]-1e-12)
                   & (rects[:, 1] <= hi[1]+1e-12) & (rects[:, 3] >= lo[1]-1e-12))
        centers = (rects[:, :2]+rects[:, 2:])/2
        radii = (rects[:, 2:]-rects[:, :2])/2
        for edge in (corners[1]-corners[0], corners[3]-corners[0]):
            axis = np.array([-edge[1], edge[0]])
            bounds = corners @ axis
            middle, radius = centers @ axis, radii @ np.abs(axis)
            overlap &= (middle+radius >= bounds.min()-1e-12) & (middle-radius <= bounds.max()+1e-12)
        return bool(overlap.any())

    def disc_intersects(self, x, y, radius):
        """Centre-space inflation for the B2 grid; final full-body SAT remains."""
        rects = self.rectangles(np.array([[x, y]]), reach=radius)
        if not len(rects):
            return False
        delta = np.maximum(np.maximum(rects[:, :2]-[x,y], [x,y]-rects[:, 2:]), 0.)
        return bool(np.any(np.sum(delta*delta, axis=1) <= radius*radius+1e-12))

    def clearance(self, corners):
        # Exact polygon distance to nearby occupied cells; farther cells have
        # negligible exp(-distance/1.5) risk and are conservatively capped at 6m.
        if self.intersects(corners):
            return 0.
        rects = self.rectangles(corners, reach=6.)
        if not len(rects):
            return 6.
        points = np.stack((rects[:, [0, 1]], rects[:, [2, 1]], rects[:, [2, 3]], rects[:, [0, 3]]), axis=1).reshape(-1, 2)
        distance = 6.
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            edge = b-a
            t = np.clip((points-a) @ edge / (edge @ edge), 0, 1)
            distance = min(distance, float(np.linalg.norm(points-a-t[:, None]*edge, axis=1).min()))
        for point in corners:
            delta = np.maximum(np.maximum(rects[:, :2]-point, point-rects[:, 2:]), 0.)
            distance = min(distance, float(np.linalg.norm(delta, axis=1).min()))
        return distance
