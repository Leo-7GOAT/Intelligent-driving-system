"""Describe completed CARLA episodes from actual PPO actions and frame-aligned controls."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.experiments import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def audit_evaluation(folder):
    folder = Path(folder)
    summary = read(folder / "evaluation_summary.json")
    episodes = lines(folder / "episodes.jsonl")
    attempts = {
        r["case_id"]: r["attempt"] for r in episodes if r["status"] == "completed"
    }
    latest = {}
    for file in sorted(folder.glob("transitions*.jsonl.gz")):
        session = defaultdict(
            lambda: {"actions": [], "modes": Counter(), "plans": Counter(), "ticks": []}
        )
        with gzip.open(file, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["attempt"] != attempts.get(row["case_id"]):
                    continue
                case = session[row["case_id"]]
                case["actions"].append(row["policy_action"])
                case["modes"][row["mode"]] += 1
                case["plans"][
                    (row.get("plan") or {}).get("status", "not_applicable")
                ] += 1
                case["ticks"].extend(row["trace"])
        # A budget-censored episode restarts in a new session. Never join those paths.
        latest.update(session)
    records = []
    for ep in episodes:
        if ep["status"] != "completed":
            records.append(
                {"case_id": ep["case_id"], "case": ep["case"], "status": ep["status"]}
            )
            continue
        raw = latest[ep["case_id"]]
        ticks = raw["ticks"]
        if len(ticks) != ep["physics_steps"]:
            raise ValueError(
                f"Incomplete completed-case trace: {folder}, case {ep['case_id']}"
            )
        frames = [t["backend_frame"] for t in ticks]
        if any(b != a + 1 for a, b in zip(frames, frames[1:])):
            raise ValueError("Nonconsecutive CARLA control frames")
        actions = np.asarray(raw["actions"], dtype=float)
        controls = [
            t["backend_evidence"]["vehicle_control"]
            for t in ticks
            if "vehicle_control" in t["backend_evidence"]
        ]
        continuous = actions.ndim == 2
        record = {
            "case_id": ep["case_id"],
            "case": ep["case"],
            "status": "completed",
            "success": ep["success"],
            "termination_reason": ep["termination_reason"],
            "progress_m": ep["progress"],
            "physics_steps": len(ticks),
            "actions": len(actions),
            "modes": dict(raw["modes"]),
            "plans": dict(raw["plans"]),
            "stationary_tick_fraction_abs_speed_below_0_05": float(
                np.mean([abs(t["state"]["v"]) < 0.05 for t in ticks])
            ),
            "intervention_tick_fraction": float(
                np.mean([t["intervention"] for t in ticks])
            ),
            "actuator_evidence_tick_coverage": len(controls) / len(ticks),
            "throttle_positive_tick_fraction": (
                float(np.mean([u["throttle"] > 1e-6 for u in controls]))
                if controls
                else None
            ),
            "brake_positive_tick_fraction": (
                float(np.mean([u["brake"] > 1e-6 for u in controls]))
                if controls
                else None
            ),
            "requested_acceleration_nonpositive_tick_fraction": float(
                np.mean([t["requested_command"]["acceleration"] <= 0 for t in ticks])
            ),
            "actual_acceleration_min_mps2": min(
                t["state"]["acceleration"] for t in ticks
            ),
            "actual_acceleration_max_mps2": max(
                t["state"]["acceleration"] for t in ticks
            ),
            "actual_jerk_abs_max_mps3": max(abs(t["physical"]["jerk"]) for t in ticks),
            "plan_over_period_fraction": ep["plan_times"].get("over_period_fraction"),
            "control_cycle_over_period_fraction": ep["control_cycle_times"].get(
                "over_period_fraction"
            ),
            "reward_components": ep["reward_components"],
            "final_state": ticks[-1]["state"],
            "continuous_policy_action_mean": (
                actions.mean(axis=0).tolist() if continuous else None
            ),
            "policy_action_nonpositive_acceleration_fraction": (
                float(np.mean(actions[:, 0] <= 0)) if continuous else None
            ),
            "policy_action_steering_at_bound_fraction": (
                float(np.mean(abs(actions[:, 1]) >= 0.999)) if continuous else None
            ),
            "discrete_policy_action_histogram": (
                None if continuous else dict(Counter(int(a) for a in actions))
            ),
        }
        records.append(record)
    return {
        "run": str(folder),
        "method": summary["method"],
        "seed": summary["seed"],
        "transfer": summary["transfer"],
        "split": summary["split"],
        "cases": records,
    }


def audit_batch(batch):
    batch = Path(batch)
    records = []
    for directory in (
        "carla_v5_100k_development",
        "carla_v5_100k_source_finetune_warmup1",
        "carla_formal_evaluation_v5",
    ):
        for marker in sorted((batch / directory).glob("*/evaluation_summary.json")):
            if read(marker).get("complete"):
                records.append(audit_evaluation(marker.parent))
    write_json(batch / "analysis/carla_decision_failure_audit.json", records)
    report = [
        "# CARLA 决策与执行失败审计",
        "",
        "仅汇总完成的实测评估；基础设施失败和未完成样本另列，不能当作零碰撞成功。静止阈值为实际 |v|<0.05 m/s；比例按实际控制 tick 计算。历史日志若缺少实车油门/制动字段，标缺失，并在JSON报告覆盖率，不补零。诊断描述行为，不单凭相关性归因。",
        "",
    ]
    for run in records:
        report += [
            f"## {run['method']} seed {run['seed']} / {run['transfer']} / {run['split']}",
            "",
            "|case|场景/条件|终止|进度m|静止比例|干预比例|油门>0比例|制动>0比例|规划状态次数|",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in run["cases"]:
            case = row["case"]
            if row["status"] != "completed":
                report.append(f"|{row['case_id']}|{case}|{row['status']}|—|—|—|—|—|—|")
                continue
            throttle = row["throttle_positive_tick_fraction"]
            brake = row["brake_positive_tick_fraction"]
            throttle_text = "缺失" if throttle is None else f"{throttle:.3f}"
            brake_text = "缺失" if brake is None else f"{brake:.3f}"
            report.append(
                f"|{row['case_id']}|{case['scenario']}/{case['condition']}|{row['termination_reason']}|{row['progress_m']:.3f}|{row['stationary_tick_fraction_abs_speed_below_0_05']:.3f}|{row['intervention_tick_fraction']:.3f}|{throttle_text}|{brake_text}|{row['plans']}|"
            )
        report.append("")
    (batch / "analysis/carla_decision_failure_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    records = audit_batch(parser.parse_args().batch)
    print(
        json.dumps(
            {
                "carla_evaluations_audited": len(records),
                "cases": sum(len(r["cases"]) for r in records),
            }
        )
    )
