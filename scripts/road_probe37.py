"""Read the installed map geometry; no ticking or actor changes."""
import json,math,argparse
from pathlib import Path
import carla

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);args=p.parse_args()
    c=carla.Client("127.0.0.1",2000);c.set_timeout(10);w=c.get_world();m=w.get_map()
    candidates=[]
    for index in (1,10,11,12,15,16,17,18):
        tr=m.get_spawn_points()[index];angle=math.radians(tr.rotation.yaw)
        columns=[]
        for x in (0,5,10,15,20,25,30,35,40):
            valid=[]
            for yi in range(-40,41):
                y=yi*.25
                loc=carla.Location(x=tr.location.x+x*math.cos(angle)+y*math.sin(angle),
                                   y=tr.location.y+x*math.sin(angle)-y*math.cos(angle),z=tr.location.z)
                wp=m.get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Driving)
                if wp is not None: valid.append(y)
            columns.append({"x":x,"road_y":valid})
        candidates.append({"spawn":index,"location":[tr.location.x,tr.location.y,tr.location.z],
                           "yaw":tr.rotation.yaw,"columns":columns})
    Path(args.output).write_text(json.dumps({"map":m.name,"candidates":candidates},indent=2))
    for r in candidates:
        print(r["spawn"],[(v["x"],min(v["road_y"],default=None),max(v["road_y"],default=None)) for v in r["columns"]])
