"""Audit saved rejected footprint points against local refined Driving lanes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import carla
from navstack.lane_projection import local_driving_union


def audit(batches,output):
    output=Path(output)
    if output.exists():raise ValueError("Preserve existing audit")
    started=time.time()
    client=carla.Client("127.0.0.1",2000);client.set_timeout(15.)
    if client.get_client_version()!="0.9.15" or client.get_server_version()!="0.9.15":
        raise ValueError("Require matching0.9.15")
    world=client.get_world();road=world.get_map();digest=hashlib.sha256(road.to_opendrive().encode()).hexdigest()
    actors_before=[a.id for a in world.get_actors() if a.type_id.startswith(("vehicle.","walker.","sensor."))]
    rows=[]
    for batch in map(Path,batches):
        protocol=json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
        if protocol["routes"][0]["opendrive_sha256"]!=digest:raise ValueError("Map differs")
        for file in sorted(batch.glob("case_*/terminal_measurement.json")):
            record=json.loads(file.read_text(encoding="utf-8"))
            for point in record["road_surface_probe"]["offroad_points_world"]:
                if time.time()-started>120:raise RuntimeError("120s read-only budget")
                loc=carla.Location(x=point[0],y=point[1],z=point[2])
                rows.append({"batch":batch.name,"case":file.parent.name,"point":point,
                    "measurement_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                    "raw_inside_now":road.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving) is not None,
                    "refined_union":local_driving_union(road,loc,carla.LaneType.Driving)})
    actors_after=[a.id for a in world.get_actors() if a.type_id.startswith(("vehicle.","walker.","sensor."))]
    if actors_before!=actors_after:raise RuntimeError("Actor inventory changed")
    result={"scope":"Read-only local lane-union diagnosis; no physics ticks/actor creation/training gate changes",
            "opendrive_sha256":digest,"rows":rows,"seconds":time.time()-started,
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path(__file__).resolve().parents[1]/"navstack/lane_projection.py"]}}
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batches",nargs="+",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.batches,a.output)
