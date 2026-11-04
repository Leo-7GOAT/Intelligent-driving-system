"""Native CARLA route geometry, compatible with the isolated Python3.7 bridge."""
import hashlib
import json
import math

REFINED_GRID_VERSION = "carla_driving_indexed_union_256_v4"
ROAD_VERSIONS = ("carla_driving_64_v1", "carla_driving_refined_union_64_v2",
                 "carla_driving_indexed_union_64_v3", REFINED_GRID_VERSION)


def sample_road_grid(layout, is_driving, version="carla_driving_64_v1"):
    """Sample actual map at cell corners and centre, without advancing physics.

    Entire cells are blocked if any sampled point is outside Driving lanes.
    This is conservative relative to samples, not a continuous-surface proof.
    Shared grid dimensions stay fixed across policies; metre resolution is saved.
    """
    width, height = layout["width"], layout["height"]
    if not 0 < min(width, height) <= max(width, height) <= 64:
        raise ValueError("Native road extent exceeds the registered grid")
    size = 64
    dx, dy = width/size, height/size
    vertices = [[bool(is_driving(x*dx, y*dy)) for x in range(size+1)] for y in range(size+1)]
    cells = [[int(vertices[y][x] and vertices[y][x+1] and vertices[y+1][x]
                  and vertices[y+1][x+1] and is_driving((x+.5)*dx, (y+.5)*dy))
              for x in range(size)] for y in range(size)]
    if version not in ROAD_VERSIONS:
        raise ValueError("Unknown measured-road geometry version")
    data = {"version": version, "width": width, "height": height,
            "drivable": cells, "geometry_id": layout["geometry_id"],
            "sampling": "four cell corners plus centre; Driving lane and height check; not continuous-surface proof"}
    if version == "carla_driving_refined_union_64_v2":
        data["sampling"] = "four cell corners plus centre; refined local Driving-lane union; no lane-width dilation; not continuous mesh proof"
    if version == "carla_driving_indexed_union_64_v3":
        data["sampling"] = "four cell corners plus centre; refined indexed Driving-lane union across roads; no lane-width dilation; not continuous mesh proof"
    if version == REFINED_GRID_VERSION:
        # Keep the common policy observation unchanged. Only the explicitly
        # versioned shared collision/risk/planning raster has finer cells.
        fine_size = 256
        fx, fy = width/fine_size, height/fine_size
        fine_vertices = [[bool(is_driving(x*fx, y*fy)) for x in range(fine_size+1)]
                         for y in range(fine_size+1)]
        fine = [[int(fine_vertices[y][x] and fine_vertices[y][x+1]
                     and fine_vertices[y+1][x] and fine_vertices[y+1][x+1]
                     and is_driving((x+.5)*fx, (y+.5)*fy))
                 for x in range(fine_size)] for y in range(fine_size)]
        data.update(collision_drivable=fine,
                    sampling="64x64 common policy observation;256x256 shared collision/risk/planning;four corners plus centre;indexed Driving-lane union;no width dilation;not continuous mesh proof")
    data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return data


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
