"""Infer units/coordinate space from measured movement, then fit limited low-speed gains."""
import argparse,json,math
from pathlib import Path
import numpy as np


def analyze(folder):
    raw=json.loads((folder/"raw_calibration.json").read_text(encoding="utf-8"))
    samples=[g[key] for g in raw["geometry_samples"] for key in ("geometry","end_geometry")]
    hypotheses=[]
    for scale in (1,.01):
        for space in ("world","actor_local"):
            local=[]
            for g in samples:
                p=np.array([w["position"] for w in g["wheels"]])*scale
                yaw=math.radians(g["yaw_degrees"])
                if space=="world":
                    p=p-np.array(g["location"])
                    p[:,:2]=p[:,:2]@np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
                local.append(p)
            local=np.array(local); wheelbase=abs(float(local[:,:,0][:,:2].mean()-local[:,:,0][:,2:].mean()))
            residual=float(np.std(local[:,:,:2],axis=0).mean())
            hypotheses.append({"scale_to_m":scale,"space":space,"wheelbase":wheelbase,"motion_invariance_residual_m":residual,
                               "plausible":bool(1<wheelbase<5 and residual<.05),"mean_local_wheels":local.mean(axis=0).tolist()})
    valid=[h for h in hypotheses if h["plausible"]]
    if len(valid)!=1: raise RuntimeError("Geometry units/space remain ambiguous")
    selected=valid[0]; wheels=np.array(selected["mean_local_wheels"])
    rear=float(wheels[2:,0].mean()); box=samples[0]
    dimensions={"wheelbase":selected["wheelbase"],"front":box["bbox_location"][0]+box["bbox_extent"][0]-rear,
                "rear":rear-(box["bbox_location"][0]-box["bbox_extent"][0]),"width":2*box["bbox_extent"][1]}
    rows=[json.loads(s) for s in (folder/"responses.jsonl").read_text(encoding="utf-8").splitlines()]
    drive=[r for r in rows if r["phase"].startswith("throttle") and .3<r["v"]<4 and 25<r["step"]<79]
    X=np.array([[r["throttle"],1,r["v"]] for r in drive]);y=np.array([r["a"] for r in drive])
    coeff=np.linalg.lstsq(X,y,rcond=None)[0]
    brakes=[-r["a"]/r["brake"] for r in rows if 85<r["step"]<130 and .3<r["v"]<4 and r["a"]<0]
    steering=[-math.atan(np.mean(np.tan(np.radians(r["wheel_steer_degrees"]))))/r["steer"]
              for r in rows if r["steer"] and 40<r["step"]<80]
    phases={}
    for phase in sorted({r["phase"] for r in rows}):
        part=[r for r in rows if r["phase"]==phase]
        phases[phase]={"first_motion_seconds":next((r["t"] for r in part if r["v"]>.1),None),
                       "peak_speed":max(r["v"] for r in part),"final_speed":part[-1]["v"],
                       "yaw_change_degrees":part[-1]["yaw_degrees"]-part[0]["yaw_degrees"]}
    result={"status":"measured_initial_calibration","geometry_hypotheses":hypotheses,"selected":selected,
            "vehicle":dimensions,"rear_axle_actor_x":rear,"steer_radians_per_normalized":float(np.median(steering)),
            "acceleration_regression_throttle_intercept_speed":coeff.tolist(),"brake_decel_per_normalized":float(np.median(brakes)),
            "phases":phases,"limitations":["low-speed piecewise response and startup delay; one linear throttle model is inadequate",
                                           "only normalized steering +/-0.25 sampled; no claim of full range calibration"],
            "recommended_adapter":{"throttle_feedforward":.2,"acceleration_kp":.08,"speed_kp":.35,
                                   "brake_decel":float(np.median(brakes)),"steer_gain":abs(float(np.median(steering)))}}
    (folder/"calibration.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder");analyze(Path(p.parse_args().folder))
