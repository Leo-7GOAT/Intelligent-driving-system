"""Summarize actual decisions and terminal states, including failed and stationary episodes."""

import argparse, gzip, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.experiments import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def audit(folder):
    folder = Path(folder)
    episodes = lines(folder / "episodes.jsonl")
    final = (folder / "scenes.jsonl").exists()
    key = "case_id" if final else "episode"
    scenes = {
        r[key]: r.get("initial_scenario", r)
        for r in lines(folder / ("scenes.jsonl" if final else "scenarios.jsonl"))
    }
    data = {}
    files = (
        sorted(folder.glob("transitions*.jsonl.gz"))
        if final
        else [folder / "transitions.jsonl"]
    )
    for file in files:
        current = defaultdict(
            lambda: {
                "plans": Counter(),
                "actions": Counter(),
                "continuous": [],
                "ticks": 0,
                "interventions": 0,
            }
        )
        with (
            gzip.open(file, "rt", encoding="utf-8")
            if file.suffix == ".gz"
            else file.open(encoding="utf-8")
        ) as stream:
            for line in stream:
                r = json.loads(line)
                d = current[r[key]]
                action = r.get("action", r.get("policy_action"))
                d["plans"][(r.get("plan") or {}).get("status", "not_applicable")] += 1
                if isinstance(action, list):
                    d["continuous"].append(action)
                else:
                    d["actions"][str(action)] += 1
                for tick in r["trace"]:
                    d["ticks"] += 1
                    d["interventions"] += int(tick["intervention"])
                    d["final_state"] = tick["state"]
        data.update(current)
    result = []
    for episode in episodes:
        i = episode[key]
        d = data[i]
        state = d["final_state"]
        goal = scenes[i]["goal"]
        action_values = np.asarray(d.pop("continuous"), float)
        continuous = (
            {
                "mean": action_values.mean(axis=0).tolist(),
                "std": action_values.std(axis=0).tolist(),
                "nonpositive_acceleration_action_fraction": float(
                    np.mean(action_values[:, 0] <= 0)
                ),
            }
            if action_values.size
            else None
        )
        result.append(
            {
                "episode": i,
                "scenario": episode.get("scenario"),
                "reason": episode["termination_reason"],
                "success": episode["success"],
                "progress": episode["progress"],
                "final_goal_distance": math.hypot(
                    state["x"] - goal[0], state["y"] - goal[1]
                ),
                "continuous_action_summary": continuous,
                **d,
            }
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch)
    records = []
    training = []
    for version in ("formal_v5", "supplementary_v5"):
        for run in sorted((batch / version).glob("*")):
            if not (run / "latest.json").exists():
                continue
            latest = read(run / "latest.json")
            episodes = (
                lines(run / "episodes.jsonl")
                if (run / "episodes.jsonl").exists()
                else []
            )
            episodes = [
                r for r in episodes if r["at_physics_steps"] <= latest["physics_steps"]
            ][-50:]
            training.append(
                {
                    "version": version,
                    "run": run.name,
                    "physics_steps": latest["physics_steps"],
                    "modes": latest["modes"],
                    "mode_fractions": {
                        k: v / max(1, latest["actions"])
                        for k, v in latest["modes"].items()
                    },
                    "last_50_completed_episode_reasons": dict(
                        Counter(r["termination_reason"] for r in episodes)
                    ),
                    "last_50_mean_progress": (
                        float(np.mean([r["progress"] for r in episodes]))
                        if episodes
                        else None
                    ),
                    "course_note": "Early samples may use short goals; do not substitute them for full-route validation",
                }
            )
            for marker in sorted(run.glob("validation_*.json")):
                r = read(marker)
                records.append(
                    {
                        "version": version,
                        "run": run.name,
                        "split": "development",
                        "budget": int(marker.stem.split("_")[-1]),
                        "episodes": audit(r["evaluation"]),
                    }
                )
            for marker in sorted(run.glob("final_*.json")):
                r = read(marker)
                if r.get("complete"):
                    records.append(
                        {
                            "version": version,
                            "run": run.name,
                            "split": marker.stem,
                            "budget": 1000000,
                            "episodes": audit(r["run"]),
                        }
                    )
    write_json(batch / "analysis/decision_failure_audit.json", records)
    write_json(batch / "analysis/training_failure_audit.json", training)
    report = [
        "# 决策与失败实测审计",
        "",
        "保留所有已完成回合。以下每条均来自实际动作、规划状态、执行位置与干预日志。",
        "",
    ]
    for r in records:
        report.extend(
            [
                f"## {r['version']}/{r['run']} — {r['split']} {r['budget']}",
                "",
                "|case|场景|终止|进度m|终点距离m|最终速度|干预/物理步|规划状态次数|",
                "|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for ep in r["episodes"]:
            report.append(
                f"|{ep['episode']}|{ep['scenario']}|{ep['reason']}|{ep['progress']:.3f}|{ep['final_goal_distance']:.3f}|{ep['final_state']['v']:.3f}|{ep['interventions']/max(1,ep['ticks']):.3f}|{dict(ep['plans'])}|"
            )
        report.append("")
    (batch / "analysis/decision_failure_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {"evaluations_audited": len(records), "training_runs": len(training)}
        )
    )
