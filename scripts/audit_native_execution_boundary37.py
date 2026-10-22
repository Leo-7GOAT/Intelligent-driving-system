"""Read-only nearest-lane geometry at every saved failed footprint sample."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import carla


def audit(batch, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise ValueError("Preserve existing audit evidence")
    client=carla.Client("127.0.0.1",2000);client.set_timeout(15.)
    if client.get_client_version() != "0.9.15" or client.get_server_version() != "0.9.15":
        raise ValueError("Require matching0.9.15")
    road=client.get_world().get_map();rows=[]
    protocol=json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(road.to_opendrive().encode()).hexdigest() != protocol["routes"][0]["opendrive_sha256"]:
        raise ValueError("Audit map differs from execution")
    for case in sorted(batch.glob("case_*")):
        terminal=json.loads((case/"terminal_measurement.json").read_text(encoding="utf-8"))
        for xyz in terminal["road_surface_probe"]["offroad_points_world"]:
            loc=carla.Location(x=xyz[0],y=xyz[1],z=xyz[2])
            wp=road.get_waypoint(loc,project_to_road=True,lane_type=carla.LaneType.Driving)
            tr=wp.transform;dx=loc.x-tr.location.x;dy=loc.y-tr.location.y;a=math.radians(tr.rotation.yaw)
            distance=math.hypot(dx,dy)
            rows.append({"case":case.name,"point":xyz,"nearest":[tr.location.x,tr.location.y,tr.location.z],
                "road_id":wp.road_id,"lane_id":wp.lane_id,"s":wp.s,"width":wp.lane_width,
                "planar_distance":distance,"lateral":abs(-dx*math.sin(a)+dy*math.cos(a)),
                "longitudinal":abs(dx*math.cos(a)+dy*math.sin(a)),"outside_half_width":distance-wp.lane_width/2})
    result={"scope":"read-only nearest-lane diagnosis; zero actor/tick requests","map":road.name,
        "opendrive_sha256":hashlib.sha256(road.to_opendrive().encode()).hexdigest(),"points":rows}
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(rows))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.batch,a.output)
