"""Describe actual training episodes by their initialized goal distance; no performance selection."""

import argparse, json, math, time
from collections import Counter, defaultdict
from pathlib import Path


def audit(batch):
    batch = Path(batch)
    records = []
    before_first_update = []
    optimization_diagnostics = []
    for version in ("formal_v5", "supplementary_v5"):
        for path in sorted((batch / version).glob("*/episodes.jsonl")):
            groups = defaultdict(list)
            complete_rows = []
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(keepends=True)
            for line in lines:
                if not line.endswith("\n"):
                    continue  # A running writer can have an incomplete last record.
                row = json.loads(line)
                complete_rows.append(row)
                scene = row["initial_scenario"]
                initial = scene["initial"]
                distance = round(
                    math.hypot(
                        scene["goal"][0] - initial["x"], scene["goal"][1] - initial["y"]
                    ),
                    3,
                )
                groups[distance].append(row)
            optimization = path.parent / "optimization.jsonl"
            if optimization.exists():
                optimization_rows = [
                    json.loads(line)
                    for line in optimization.read_text(encoding="utf-8").splitlines(
                        keepends=True
                    )
                    if line.endswith("\n")
                ]
                if not optimization_rows:
                    continue
                first_update = optimization_rows[0]
                last_update = optimization_rows[-1]
                recent = optimization_rows[-50:]
                diagnostics = {
                    "version": version,
                    "run": path.parent.name,
                    "physics_steps": last_update["physics_steps"],
                    "optimization_phases": last_update["phase"],
                    "optimizer_steps": last_update["optimizer_steps"],
                    "reported_epoch_counter": last_update["ppo_epoch_updates"],
                    "recent_phases": len(recent),
                }
                for metric in (
                    "train/approx_kl",
                    "train/clip_fraction",
                    "train/explained_variance",
                    "actor_update_l2",
                ):
                    values = [r[metric] for r in recent if metric in r]
                    diagnostics[metric] = (
                        {
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }
                        if values
                        else None
                    )
                optimization_diagnostics.append(diagnostics)
                initial_episodes = [
                    r
                    for r in complete_rows
                    if r["at_physics_steps"] <= first_update["physics_steps"]
                ]
                before_first_update.append(
                    {
                        "version": version,
                        "run": path.parent.name,
                        "first_update_after_physics_steps": first_update[
                            "physics_steps"
                        ],
                        "completed_episodes_before_first_optimizer_update": len(
                            initial_episodes
                        ),
                        "successes_before_first_optimizer_update": sum(
                            r["success"] for r in initial_episodes
                        ),
                        "initial_goal_distances_m": sorted(
                            {
                                round(
                                    math.hypot(
                                        r["initial_scenario"]["goal"][0]
                                        - r["initial_scenario"]["initial"]["x"],
                                        r["initial_scenario"]["goal"][1]
                                        - r["initial_scenario"]["initial"]["y"],
                                    ),
                                    3,
                                )
                                for r in initial_episodes
                            }
                        ),
                        "interpretation": "These episodes precede the first optimizer update; their success is not evidence of learned improvement",
                    }
                )
            for distance, rows in sorted(groups.items()):
                n = len(rows)
                ticks = sum(r["physics_steps"] for r in rows)
                components = sorted({k for r in rows for k in r["reward_components"]})
                records.append(
                    {
                        "version": version,
                        "run": path.parent.name,
                        "actual_initialized_goal_distance_m": distance,
                        "completed_training_episodes": n,
                        "last_episode_end_physics_step": max(
                            r["at_physics_steps"] for r in rows
                        ),
                        "physics_steps_in_completed_episodes": ticks,
                        "success_count": sum(r["success"] for r in rows),
                        "success_fraction_training_only": sum(
                            r["success"] for r in rows
                        )
                        / n,
                        "failures": dict(
                            Counter(r["termination_reason"] for r in rows)
                        ),
                        "mean_progress_m": sum(r["progress"] for r in rows) / n,
                        "mean_episode_reward": sum(r["reward"] for r in rows) / n,
                        "mean_episode_reward_components": {
                            k: sum(r["reward_components"].get(k, 0) for r in rows) / n
                            for k in components
                        },
                        "physical_step_intervention_rate": sum(
                            r["interventions"] for r in rows
                        )
                        / max(1, ticks),
                    }
                )
    result = {
        "observed_epoch": time.time(),
        "scope": "On-policy training episodes, not held-out evaluation or independent seed evidence; only completed episodes included",
        "records": records,
        "before_first_optimizer_update": before_first_update,
        "optimization_diagnostics": optimization_diagnostics,
        "epoch_counter_note": "SB3 epoch counter includes an entered epoch that can stop early on KL. optimizer_steps records actual parameter-update calls, not a nominal epoch multiplier.",
    }
    out = batch / "analysis"
    (out / "training_curriculum_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# 实际训练课程审计",
        "",
        "按回合初始化时的真实目标距离分组，不用名义步数猜课程。仅完整训练回合；不是验证/测试成绩，也不能把回合数当独立训练样本。",
        "",
        "|版本/运行|目标米|完整训练回合|成功数|训练成功比例|平均进度米|平均奖励|干预/物理步|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"|{r['version']}/{r['run']}|{r['actual_initialized_goal_distance_m']}|{r['completed_training_episodes']}|{r['success_count']}|{r['success_fraction_training_only']:.3f}|{r['mean_progress_m']:.2f}|{r['mean_episode_reward']:.2f}|{r['physical_step_intervention_rate']:.3f}|"
        )
    lines.extend(
        [
            "",
            "逐项奖励分量、失败原因和末回合累计物理步见同名JSON。未完成回合留在原始轨迹中，不补写结果。",
        ]
    )
    lines.extend(
        [
            "",
            "## 首次优化之前",
            "",
            "以下回合在第一次optimizer.step之前已经完成，不是训练取得的学习收益。H的模型/规划/停车先验与B1初始动作分布必须计入解释。",
            "",
            "|运行|首次优化前物理步|完整回合|成功数|",
            "|---|---:|---:|---:|",
        ]
    )
    for r in before_first_update:
        lines.append(
            f"|{r['version']}/{r['run']}|{r['first_update_after_physics_steps']}|{r['completed_episodes_before_first_optimizer_update']}|{r['successes_before_first_optimizer_update']}|"
        )
    lines.extend(
        [
            "",
            "## 真实参数更新",
            "",
            "SB3的epoch计数包含因KL提前退出的已进入epoch，不能解读为完整遍历数据的次数。以下optimizer.step计数来自真实优化器hook；近50阶段KL、剪切比例与value解释方差详见JSON，不把参数有变化等同于任务收敛。",
            "",
            "|运行|实际物理步|优化阶段|实际optimizer.step|SB3 epoch计数|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in optimization_diagnostics:
        lines.append(
            f"|{r['run']}|{r['physics_steps']}|{r['optimization_phases']}|{r['optimizer_steps']}|{r['reported_epoch_counter']}|"
        )
    (out / "training_curriculum_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"curriculum_groups": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    audit(parser.parse_args().batch)
