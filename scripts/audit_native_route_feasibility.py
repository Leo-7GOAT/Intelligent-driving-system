"""Search all saved native geometries with the actual shared road constraints."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navstack.config import EnvConfig, PlannerConfig
from navstack.planning import HybridAStar, GridAStarPlanner
from navstack.road_grid import RoadGrid
from navstack.world import World
from navstack.experiments import write_json
from navstack.trajectory import parameterize


def audit(source, output, grid_postprocess="legacy"):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cfg = EnvConfig.from_dict(json.loads((source/"B1/config.json").read_text(encoding="utf-8"))["env"])
    planner = PlannerConfig(max_seconds=None, max_nodes=10000, analytic_expansion=True, vectorized_primitives=True, grid_postprocess=grid_postprocess)
    write_json(output/"protocol.json", {"source_batch":source.name, "planner":asdict(planner),
        "scope":"Offline development feasibility, no physical driving or PPO performance. All four geometries retained; search failure does not prove no feasible path.",
        "created_epoch":time.time(), "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/"navstack/planning.py",ROOT/"navstack/grid_smoothing.py",ROOT/"navstack/world.py",ROOT/"navstack/road_grid.py",ROOT/"navstack/trajectory.py"]}})
    fig, axes = plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    results = []
    for index, file in enumerate(sorted((source/"B1").glob("scene_*.json"))):
        data = json.loads(file.read_text(encoding="utf-8")); scene = data["scene"]
        road = RoadGrid(scene["road_grid"]); world = World(scene["width"],scene["height"],road=road)
        initial = data["initial"]["initial"]
        start = (initial["x"],initial["y"],initial["yaw"])
        for method_index, (method, constructor) in enumerate((("H",HybridAStar),("B2",GridAStarPlanner))):
            plan = constructor(cfg.vehicle,planner).plan(start,scene["goal"],world)
            record = {"case_id":index,"method":method,"geometry_id":scene["native_layout"]["geometry_id"],
                "scene_sha256":hashlib.sha256(file.read_bytes()).hexdigest(), "road_sha256":road.sha256, **asdict(plan)}
            if plan.status == "success":
                assert world.free_many(plan.path,cfg.vehicle).all()
                reference = parameterize(plan.path,1.,0.,cfg.vehicle,0.,stop=True)
                reference_path = np.c_[reference.x,reference.y,reference.yaw]
                record["reference_free"] = bool(world.free_many(reference_path,cfg.vehicle).all())
                record["reference_max_curvature"] = float(np.abs(reference.kappa).max())
                record["reference_length_m"] = float(reference.s[-1])
                write_json(output/f"case_{index:02d}_{method}_reference.json",asdict(reference))
            write_json(output/f"case_{index:02d}_{method}.json",record)
            results.append({k:v for k,v in record.items() if k!="path"})
            axis = axes[method_index,index]
            axis.imshow(~getattr(road,'collision_drivable',road.drivable),origin="lower",extent=(0,road.width,0,road.height),cmap="Greys",vmin=0,vmax=1,alpha=.5)
            original = np.asarray(scene["native_layout"]["path"])
            axis.plot(original[:,0],original[:,1],"--",label="original survey")
            if len(plan.path):
                axis.plot(plan.path[:,0],plan.path[:,1],label="searched path")
            axis.set(title=f"{method} route {index}: {plan.status}\n{plan.expanded_nodes} nodes, {plan.planning_time:.2f}s",xlabel="x (m)",ylabel="y (m)",aspect="equal")
            print(json.dumps(results[-1]),flush=True)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Native map: full-body search and trajectory checks, all failures retained")
    fig.savefig(output/"native_planner_feasibility.png",dpi=160); plt.close(fig)
    write_json(output/"results.json",results)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--grid-postprocess",choices=["legacy","body_spline_v1"],default="legacy")
    a=p.parse_args();audit(a.source,a.output,a.grid_postprocess)
