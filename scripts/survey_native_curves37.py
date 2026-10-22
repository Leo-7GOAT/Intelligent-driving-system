"""Read-only installed-map survey; candidate geometry is not driving evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def wrap(degrees):
    return (degrees+180) % 360-180


def survey(output, calibration):
    import carla
    output = Path(output)
    if output.exists():
        raise RuntimeError("Preserve existing map survey")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(15.)
    if client.get_client_version() != "0.9.15" or client.get_server_version() != "0.9.15":
        raise RuntimeError("Require matching0.9.15")
    world = client.get_world()
    road_map = world.get_map()
    cal = json.loads(Path(calibration).read_text(encoding="utf-8"))
    vehicle = cal["vehicle"]
    max_curvature = math.tan(.5)/vehicle["wheelbase"]
    candidates, rejected, seen = [], {}, set()
    started = time.time()
    initial_frame = world.get_snapshot().frame
    waypoints = sorted(road_map.generate_waypoints(8.), key=lambda p: (p.road_id, p.lane_id, p.s))
    for start in waypoints:
        if time.time()-started > 120:
            raise RuntimeError("Bounded map survey timed out; no partial pool promoted")
        path = [start]
        for _ in range(24):
            nexts = path[-1].next(1.)
            if not nexts:
                break
            # Fixed deterministic continuation; not selected by policy success.
            path.append(min(nexts, key=lambda p: (abs(wrap(p.transform.rotation.yaw-path[-1].transform.rotation.yaw)), p.road_id, p.lane_id, p.s)))
        if len(path) != 25:
            rejected["short"] = rejected.get("short", 0)+1
            continue
        positions = [(p.transform.location.x, p.transform.location.y, p.transform.location.z) for p in path]
        lengths = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(positions, positions[1:])]
        heading_change = sum(wrap(b.transform.rotation.yaw-a.transform.rotation.yaw) for a, b in zip(path, path[1:]))
        curvature = [math.radians(wrap(b.transform.rotation.yaw-a.transform.rotation.yaw))/max(d, 1e-9)
                     for a, b, d in zip(path, path[1:], lengths)]
        if not 15 <= abs(heading_change) <= 90 or max(abs(k) for k in curvature) > max_curvature:
            rejected["heading_or_curvature"] = rejected.get("heading_or_curvature", 0)+1
            continue
        if max(p[2] for p in positions)-min(p[2] for p in positions) > .5:
            rejected["slope"] = rejected.get("slope", 0)+1
            continue
        misses = []
        for index, point in enumerate(path):
            yaw = math.radians(point.transform.rotation.yaw)
            for forward in (-vehicle["rear"], 0., vehicle["front"]*.5, vehicle["front"]):
                for lateral in (-vehicle["width"]/2, 0., vehicle["width"]/2):
                    origin = point.transform.location
                    probe = carla.Location(x=origin.x+forward*math.cos(yaw)-lateral*math.sin(yaw),
                                           y=origin.y+forward*math.sin(yaw)+lateral*math.cos(yaw), z=origin.z)
                    if road_map.get_waypoint(probe, project_to_road=False, lane_type=carla.LaneType.Driving) is None:
                        misses.append([index, forward, lateral])
        if misses:
            rejected["sampled_body_off_driving_surface"] = rejected.get("sampled_body_off_driving_surface", 0)+1
            continue
        points = [{"x": p.transform.location.x, "y": p.transform.location.y, "z": p.transform.location.z,
                   "yaw_degrees_carla": p.transform.rotation.yaw, "road_id": p.road_id,
                   "lane_id": p.lane_id, "s": p.s, "lane_width": p.lane_width, "is_junction": p.is_junction} for p in path]
        identity = hashlib.sha256(json.dumps([[round(p[k], 3) for k in ("x", "y", "yaw_degrees_carla")] for p in points]).encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append({"geometry_id": identity, "length_m": sum(lengths), "heading_change_degrees": heading_change,
                           "max_abs_curvature_m_inv": max(abs(k) for k in curvature), "points": points})
    result = {"scope": "Native-road geometry candidates only; no vehicle execution or PPO evaluation. Routes can overlap and are not independent final conditions.",
              "map": road_map.name, "opendrive_sha256": hashlib.sha256(road_map.to_opendrive().encode()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "calibration_sha256": hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
              "vehicle": vehicle, "max_curvature_m_inv": max_curvature,
              "body_probe_scope": "12 footprint samples at each1m route point; not continuous swept-body or collision proof",
              "selection": "All8m-spaced map starts,24successor1msteps,deterministic smallest heading branch; absolute heading15..90deg,vertical range<=0.5m; preserve rejection counts",
              "start_frame": initial_frame, "end_frame": world.get_snapshot().frame,
              "world_ticks_requested": 0, "starts_examined": len(waypoints), "rejections": rejected,
              "elapsed_seconds": time.time()-started, "candidates": candidates}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("map", "starts_examined", "rejections", "elapsed_seconds", "world_ticks_requested")}))
    print("candidate_count="+str(len(candidates)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()
    survey(args.output, args.calibration)
