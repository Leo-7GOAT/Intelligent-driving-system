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


def verify_trace(trace, actual, used, previous):
    """Match every applied control to its measured state, frame and clock."""
    for tick in trace:
        evidence = tick["backend_evidence"]
        key = (evidence["episode_id"], evidence["frame"])
        measured = actual[key]
        assert key not in used, "Control counted more than once"
        used.add(key)
        assert tick["state"] == measured["state"]
        assert evidence["imu_frame"] == evidence["frame"] == measured["imu_frame"]
        episode = evidence["episode_id"]
        if episode in previous:
            old_frame, old_time = previous[episode]
            assert evidence["frame"] == old_frame+1
            assert abs(tick["state"]["time"]-old_time-.05) < 1e-6
        previous[episode] = evidence["frame"], tick["state"]["time"]


def verify_evaluation(directory, actual, used):
    """Evaluation controls share the bridge log but never the training count."""
    if not (directory/"evaluation.json").exists():
        assert not (directory/"evaluation_trace.jsonl.gz").exists(), "Incomplete evaluation"
        return None
    evaluation = read(directory/"evaluation.json")
    for name, key in (("model.zip", "model_sha256"), ("normalization.pkl", "normalization_sha256")):
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == evaluation[key]
    previous, cases, failures = {}, {}, Counter()
    with gzip.open(directory/"evaluation_trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            info = row["info"]
            case = cases.setdefault(row["case_id"], {"frames": 0, "actions": 0, "terminal": None})
            assert case["terminal"] is None, "Control after episode termination"
            verify_trace(info["trace"], actual, used, previous)
            assert len(info["trace"]) == info["executed_substeps"]
            case["frames"] += len(info["trace"])
            case["actions"] += 1
            if info.get("plan") and info["plan"].get("failure_reason"):
                failures[info["plan"]["failure_reason"]] += 1
            if info["termination_reason"]:
                case["terminal"] = info["episode_metrics"]
    assert set(cases) == {e["case_id"] for e in evaluation["episodes"]}
    for episode in evaluation["episodes"]:
        case = cases[episode["case_id"]]
        assert case["terminal"] == {k:v for k,v in episode.items() if k != "case_id"}
        assert (case["frames"], case["actions"]) == (episode["physics_steps"], episode["actions"])
    return {"scope": evaluation["scope"], "control_frames_verified": sum(c["frames"] for c in cases.values()),
            "world_ticks": evaluation["world_ticks"], "successes": sum(e["success"] for e in evaluation["episodes"]),
            "episodes": len(cases), "termination_reasons": dict(Counter(e["termination_reason"] for e in evaluation["episodes"])),
            "plan_failure_reasons": dict(failures)}


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
    fig, axes = plt.subplots(len(protocol["methods"]), 4, figsize=(16, 4*len(protocol["methods"])), constrained_layout=True, squeeze=False)
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
                    verify_trace(action["trace"], actual, used, previous)
                    for tick in action["trace"]:
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
            axis.imshow(~getattr(road,'collision_drivable',road.drivable), origin="lower", extent=(0, road.width, 0, road.height), cmap="Greys", vmin=0, vmax=1, alpha=.55)
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
            "evaluation": verify_evaluation(directory, actual, used),
            "actor_l2_change": summary["actor_l2_change"], "optimizer_steps": summary["optimizer_steps"]})
        if protocol.get('recovery'):
            recovery=protocol['recovery'].get(method,{})
            retained=recovery.get('lineage_controls',0)
            consumed=recovery.get('consumed_controls_lower_bound',0)
            assert summary['lineage_control_steps']==retained+frames
            assert summary['charged_control_steps_lower_bound']==consumed+frames
            assert summary['inherited_controls_not_in_checkpoint']==consumed-retained
            methods[-1]['recovery_accounting']={
                'new_verified_controls':frames,'retained_model_lineage_controls':retained+frames,
                'charged_controls_lower_bound':consumed+frames,'inherited_controls_not_in_checkpoint':consumed-retained,
                'unknown_unflushed_parent_tail':recovery.get('unknown_unflushed_parent_tail',False),
                'scope':'Parent lost controls remain charged; this recovery is not a budget-matched comparison'}
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
