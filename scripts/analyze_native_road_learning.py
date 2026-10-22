"""Rebuild shared-road training evidence from recorded CARLA frames and cells."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from navstack.config import EnvConfig
from navstack.road_grid import RoadGrid
from navstack.world import World
from navstack.experiments import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(batch):
    batch = Path(batch)
    protocol = read(batch/"protocol.json")
    for relative, digest in protocol["source_hashes"].items():
        assert hashlib.sha256((batch/"source"/relative).read_bytes()).hexdigest() == digest
    actual = {}
    for file in (batch/"infrastructure").glob("*_responses.jsonl.gz"):
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)["response"]
                if "vehicle_control" in row:
                    actual[(row["episode_id"], row["frame"])] = row
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    methods = []
    used = set()
    for method_index, method in enumerate(protocol["methods"]):
        directory = batch/method
        config = EnvConfig.from_dict(read(directory/"config.json")["env"])
        scenes = [read(p) for p in sorted(directory.glob("scene_*.json"))]
        traces = {i: [] for i in range(len(scenes))}
        reasons, controllers, plans = Counter(), Counter(), Counter()
        frames = 0
        previous = {}
        grid_terminal_events = 0
        for path in directory.glob("transitions*.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    action = json.loads(line)
                    episode = action["episode_id"]
                    if action["plan"]:
                        plans[action["plan"]["status"]] += 1
                    for tick in action["trace"]:
                        evidence = tick["backend_evidence"]
                        key = (evidence["episode_id"], evidence["frame"])
                        measured = actual[key]
                        assert key not in used
                        used.add(key)
                        assert tick["state"] == measured["state"]
                        assert evidence["imu_frame"] == evidence["frame"] == measured["imu_frame"]
                        if episode in previous:
                            old_frame, old_time = previous[episode]
                            assert evidence["frame"] == old_frame+1
                            assert abs(tick["state"]["time"]-old_time-.05) < 1e-6
                        previous[episode] = evidence["frame"], tick["state"]["time"]
                        controllers[tick["controller_status"]] += 1
                        traces[episode].append(tick["state"])
                        frames += 1
                    if action["termination_reason"]:
                        reasons[action["termination_reason"]] += 1
                        scene = scenes[episode]["scene"]
                        world = World(scene["width"], scene["height"], road=RoadGrid(scene["road_grid"]))
                        state = traces[episode][-1]
                        if world.out_of_bounds((state["x"], state["y"], state["yaw"]), config.vehicle):
                            grid_terminal_events += 1
                            assert action["termination_reason"] in ("out_of_bounds", "collision")
        summary = read(directory/"summary.json")
        assert frames == summary["physics_steps"]
        assert summary["actor_l2_change"] > 0 and summary["reload_exact"]
        assert summary["optimization_phases"] >= 2 and summary["optimizer_steps"] >= 4
        for index, axis in enumerate(axes[method_index]):
            if index >= len(scenes):
                axis.axis("off")
                continue
            scene = scenes[index]["scene"]
            road = RoadGrid(scene["road_grid"])
            axis.imshow(~road.drivable, origin="lower", extent=(0, road.width, 0, road.height), cmap="Greys", vmin=0, vmax=1, alpha=.55)
            reference = np.asarray(scene["native_layout"]["path"])
            axis.plot(reference[:, 0], reference[:, 1], "--", color="tab:blue", label="survey centreline (not policy reference)")
            trajectory = traces[index]
            if trajectory:
                axis.plot([s["x"] for s in trajectory], [s["y"] for s in trajectory], color="tab:orange", label="actual PPO execution")
                axis.scatter(trajectory[-1]["x"], trajectory[-1]["y"], c="tab:red", s=25)
            axis.set(title=f"{method}: episode {index}, {len(trajectory)} control frames", xlabel="local x (m)", ylabel="local y (m)", aspect="equal")
        methods.append({"method": method, "control_frames_verified": frames, "world_ticks": summary["world_ticks"],
            "termination_reasons": dict(reasons), "grid_terminal_events": grid_terminal_events,
            "controller_statuses": dict(controllers), "plan_statuses": dict(plans),
            "unfinished_episodes": sum(bool(traces[i]) for i in traces)-sum(reasons.values()),
            "distinct_route_grids": len({s["scene"]["road_grid"]["sha256"] for s in scenes}),
            "actor_l2_change": summary["actor_l2_change"], "optimizer_steps": summary["optimizer_steps"]})
    assert len(used) == len(actual), "Unaccounted applied CARLA controls"
    fig.suptitle("Development interface training only: short rollouts, failures and unfinished episodes retained")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    output = batch/"analysis"
    output.mkdir(exist_ok=True)
    fig.savefig(output/"native_road_training.png", dpi=160)
    plt.close(fig)
    result = {"scope": protocol["scope"], "raw_control_frames": len(actual), "methods": methods,
        "limitations": "Sampled raster, four development geometries; rollout overshoot differs; no completion or hierarchy advantage established"}
    write_json(output/"verified_summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--batch", required=True)
    analyze(parser.parse_args().batch)
