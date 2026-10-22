"""Native CARLA route geometry, compatible with the isolated Python3.7 bridge."""
import hashlib
import json
import math


def select_routes(survey):
    """Four development curves selected solely by signed geometry before driving."""
    result = []
    for target in (-30., -60., 30., 60.):
        eligible = [r for r in survey["candidates"] if r["heading_change_degrees"]*target > 0]
        if not eligible:
            raise ValueError("Native survey lacks the requested turn direction")
        route = min(eligible, key=lambda r: (abs(r["heading_change_degrees"]-target), r["geometry_id"]))
        if any(r["geometry_id"] == route["geometry_id"] for r in result):
            raise ValueError("Geometry selection duplicates a route")
        result.append({**route, "map": survey["map"], "opendrive_sha256": survey["opendrive_sha256"], "selection_target_degrees": target})
    return result


def local_layout(route, map_name, map_digest):
    if route["map"] != map_name or route["opendrive_sha256"] != map_digest:
        raise ValueError("Native geometry is from a different map")
    points = route["points"]
    if not 2 <= len(points) <= 1000 or not all(math.isfinite(p[k]) for p in points for k in ("x", "y", "z", "yaw_degrees_carla")):
        raise ValueError("Invalid native route points")
    digest = hashlib.sha256(json.dumps([[round(p[k], 3) for k in ("x", "y", "yaw_degrees_carla")] for p in points]).encode()).hexdigest()
    if digest != route["geometry_id"]:
        raise ValueError("Native route geometry hash mismatch")
    first = points[0]
    angle = first["yaw_degrees_carla"]
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    raw = [[(p["x"]-first["x"])*co+(p["y"]-first["y"])*si,
            (p["x"]-first["x"])*si-(p["y"]-first["y"])*co,
            (math.radians(angle-p["yaw_degrees_carla"])+math.pi) % (2*math.pi)-math.pi] for p in points]
    x0, y0 = 6-min(p[0] for p in raw), 6-min(p[1] for p in raw)
    path = [[p[0]+x0, p[1]+y0, p[2]] for p in raw]
    return {"angle": angle, "origin": [first["x"]-x0*co-y0*si, first["y"]-x0*si+y0*co],
            "spawn_z": first["z"]+.6, "path": path, "width": max(p[0] for p in path)+6,
            "height": max(p[1] for p in path)+6, "goal": path[-1], "geometry_id": digest}
