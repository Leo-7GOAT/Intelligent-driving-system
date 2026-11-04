"""Read-only CARLA map-query audit of rejected raster samples; no actor or tick."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import carla


def audit(batch, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Do not replace prior map-query evidence")
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(15.)
    if client.get_client_version() != "0.9.15" or client.get_server_version() != "0.9.15":
        raise RuntimeError("Require matching CARLA0.9.15")
    world = client.get_world(); road = world.get_map()
    rows = []
    started = time.time()
    for file in sorted((Path(batch)/"B1").glob("scene_*.json")):
        scene = json.loads(file.read_text(encoding="utf-8"))["scene"]
        assert scene["map"] == road.name
        grid, layout = scene["road_grid"], scene["native_layout"]
        co, si = math.cos(math.radians(scene["angle"])), math.sin(math.radians(scene["angle"]))
        dx, dy = scene["width"]/64, scene["height"]/64
        failed, near_inside, height_rejections = [], 0, 0
        for iy, row in enumerate(grid["drivable"]):
            for ix, drivable in enumerate(row):
                if drivable:
                    continue
                for ox, oy in ((0,0),(1,0),(0,1),(1,1),(.5,.5)):
                    if time.time()-started > 120:
                        raise RuntimeError("Map audit wall budget exceeded")
                    x,y = (ix+ox)*dx,(iy+oy)*dy
                    location = carla.Location(x=scene["origin"][0]+x*co+y*si,
                        y=scene["origin"][1]+x*si-y*co,z=layout["spawn_z"])
                    direct = road.get_waypoint(location,project_to_road=False,lane_type=carla.LaneType.Driving)
                    if direct is not None:
                        height_rejections += int(abs(direct.transform.location.z-(layout["spawn_z"]-.6))>=1.5)
                        continue
                    nearest = road.get_waypoint(location,project_to_road=True,lane_type=carla.LaneType.Driving)
                    tr = nearest.transform
                    vx,vy = location.x-tr.location.x,location.y-tr.location.y
                    angle = math.radians(tr.rotation.yaw)
                    lateral = abs(-vx*math.sin(angle)+vy*math.cos(angle))
                    longitudinal = abs(vx*math.cos(angle)+vy*math.sin(angle))
                    distance = math.hypot(vx,vy)
                    inside = distance < nearest.lane_width/2-.05
                    near_inside += int(inside)
                    failed.append({"cell":[ix,iy],"local":[x,y],"world":[location.x,location.y,location.z],
                        "nearest_road_id":nearest.road_id,"nearest_lane_id":nearest.lane_id,"nearest_s":nearest.s,
                        "nearest_xyz":[tr.location.x,tr.location.y,tr.location.z],"width":nearest.lane_width,
                        "distance":distance,"lateral":lateral,"longitudinal":longitudinal,
                        "strictly_inside_nearest_width":inside})
        rows.append({"geometry_id":layout["geometry_id"],"road_sha256":grid["sha256"],
            "none_samples":len(failed),"strictly_inside_nearest_width":near_inside,
            "height_rejections":height_rejections,"failed_samples":failed})
    result = {"scope":"Read-only map-query discrepancy diagnosis; zero requested ticks/actors, not driving or corrected road truth",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "map":road.name,"opendrive_sha256":hashlib.sha256(road.to_opendrive().encode()).hexdigest(),
        "seconds":time.time()-started,"routes":rows}
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{k:v for k,v in row.items() if k!="failed_samples"} for row in rows]))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();audit(args.batch,args.output)
