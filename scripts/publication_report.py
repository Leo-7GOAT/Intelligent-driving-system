"""Paper-facing measured tables; missing experiments stay missing and seeds stay independent."""

import argparse, hashlib, json, sys, time, subprocess, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.statistics import paired_bootstrap, learning_curve_auc
from navstack.experiments import write_json
from research_report import report as base_report
from training_curriculum_audit import audit as curriculum_audit
from carla_failure_audit import audit_batch as carla_failure_audit
from verify_batch_integrity import verify as verify_batch_integrity
from plot_final_metrics import plot as plot_final_metrics
from audit_carla_conditions import audit as audit_carla_conditions


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return (
        [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]
        if Path(path).exists()
        else []
    )


def mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def preserve_report_sources(batch):
    root = Path(__file__).resolve().parents[1]
    files = sorted(list((root / "scripts").glob("*.py")) + list((root / "scripts").glob("*.ps1")))
    payloads = {str(path.relative_to(root)): path.read_bytes() for path in files}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    destination = batch / "report_source_versions" / bundle
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = destination / name
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("Existing report source bundle was modified")
        else:
            path.write_bytes(payload)
    manifest = {"bundle": bundle, "directory": str(destination.resolve()), "hashes": hashes,
                "scope": "Report and orchestration source snapshot; training and CARLA runtime snapshots remain separately registered."}
    if not (destination / "manifest.json").exists():
        write_json(destination / "manifest.json", manifest)
    write_json(batch / "analysis/report_source_manifest.json", manifest)
    return manifest


def metrics(episodes):
    completed = [r for r in episodes if r.get("status", "completed") == "completed"]
    n = len(episodes)
    ticks = sum(r["physics_steps"] for r in completed)
    result = {
        "scheduled_resolved_cases": n,
        "completed_cases": len(completed),
        "infrastructure_unresolved": n - len(completed),
        "safe_completion": mean([float(r.get("success", False)) for r in episodes]),
        "collision_rate": mean(
            [float(r.get("termination_reason") == "collision") for r in episodes]
        ),
        "out_of_bounds_rate": mean(
            [float(r.get("termination_reason") == "out_of_bounds") for r in episodes]
        ),
        "deadline_rate": mean(
            [float(r.get("termination_reason") == "deadline") for r in episodes]
        ),
        "progress_m": mean([r["progress"] for r in completed]),
        "successful_travel_time_s": mean(
            [r.get("travel_time_success_only") for r in completed]
        ),
        "physical_step_intervention_rate": sum(r["interventions"] for r in completed)
        / max(1, ticks),
        "mean_episode_plan_failure_rate": mean(
            [r.get("plan_failure_rate") for r in completed]
        ),
        "mean_episode_mpc_failure_rate": mean(
            [r.get("mpc_failure_rate") for r in completed]
        ),
        "failures": dict(
            Counter(r.get("termination_reason", r.get("status")) for r in episodes)
        ),
        "physics_steps": ticks,
    }
    for key in ("jerk", "lateral_acceleration", "risk"):
        available = [r for r in completed if r.get(key + "_rms") is not None]
        result[key + "_pooled_rms"] = (
            float(
                np.sqrt(
                    sum(r[key + "_rms"] ** 2 * r["physics_steps"] for r in available)
                    / sum(r["physics_steps"] for r in available)
                )
            )
            if available
            else None
        )
    for key in ("plan_times", "control_times", "qp_times", "control_cycle_times"):
        result["mean_episode_" + key + "_p95_s"] = mean(
            [r.get(key, {}).get("p95") for r in completed]
        )
    result["scheduled_completion_lower_bound"] = sum(bool(r.get("success", False)) for r in episodes) / n if n else None
    if len(completed) != n:
        # A failed RPC is not an observed vehicle collision-free failure or a policy score.
        for key in ("safe_completion", "collision_rate", "out_of_bounds_rate", "deadline_rate"):
            result[key] = None
    if not ticks:
        result["physical_step_intervention_rate"] = None
    return result


def comparison(records, left, right):
    records = [r for r in records if r["metrics"]["infrastructure_unresolved"] == 0]
    names = sorted(
        set(
            k
            for r in records
            for k, v in r["metrics"].items()
            if isinstance(v, (float, int))
            and k
            not in (
                "physics_steps",
                "completed_cases",
                "scheduled_resolved_cases",
                "infrastructure_unresolved",
                "scheduled_completion_lower_bound",
            )
        )
    )
    out = {}
    for metric in names:
        a = {
            r["seed"]: r["metrics"][metric]
            for r in records
            if r["name"] == left and r["metrics"].get(metric) is not None
        }
        b = {
            r["seed"]: r["metrics"][metric]
            for r in records
            if r["name"] == right and r["metrics"].get(metric) is not None
        }
        out[metric] = paired_bootstrap(a, b)
    return out


def main(batch):
    batch = Path(batch)
    status=subprocess.run([sys.executable,str(Path(__file__).with_name("batch_status.py")),"--batch",str(batch)],
                          capture_output=True,text=True,encoding="utf-8",check=True,env={**os.environ,"PYTHONIOENCODING":"utf-8"})
    (batch/"run_status.md").write_text("# 本批真实运行状态\n\n"+status.stdout.replace("|Version","\n|Version"),encoding="utf-8")
    curriculum_audit(batch)
    carla_failure_audit(batch)
    audit_carla_conditions(batch)
    integrity = verify_batch_integrity(batch)
    if not integrity["checks_passed"]:
        raise ValueError("Batch integrity check failed; inspect analysis/batch_integrity.json")
    report_source = preserve_report_sources(batch)
    base_report(batch)
    out = batch / "analysis"
    evaluations = []
    dev = []
    for version in ("formal_v5", "supplementary_v5"):
        for run in sorted((batch / version).glob("*")):
            if not run.is_dir():
                continue
            name, seed = run.name.rsplit("_seed", 1)
            for f in sorted(run.glob("validation_*.json")):
                val = read(f)
                episodes = rows(Path(val["evaluation"]) / "episodes.jsonl")
                dev.append(
                    {
                        "version": version,
                        "name": name,
                        "seed": int(seed),
                        "nominal_budget": int(f.stem.split("_")[-1]),
                        "actual_physics_steps": val["actual_physics_steps"],
                        "metrics": metrics(episodes),
                    }
                )
            for label in ("primary", "generalization"):
                f = run / ("final_" + label + ".json")
                if not f.exists() or not read(f).get("complete"):
                    continue
                summary = read(f)
                episodes = rows(Path(summary["run"]) / "episodes.jsonl")
                scenarios = {}
                for scenario in sorted({r["case"]["scenario"] for r in episodes}):
                    scenarios[scenario] = metrics(
                        [r for r in episodes if r["case"]["scenario"] == scenario]
                    )
                if label == "generalization":
                    scenarios["density_6_obstacles"] = metrics(
                        [
                            r
                            for r in episodes
                            if r["case"].get("overrides", {}).get("obstacle_count") == 6
                            or r["case"].get("overrides", {}).get("obstacle_num") == 6
                        ]
                    )
                    scenarios["vehicle_parameter_shift"] = metrics(
                        [
                            r
                            for r in episodes
                            if "vehicle" in r["case"].get("overrides", {})
                        ]
                    )
                evaluations.append(
                    {
                        "backend": "lightweight",
                        "version": version,
                        "label": label,
                        "name": name,
                        "seed": int(seed),
                        "actual_training_steps": summary["training_physics_steps"],
                        "manifest_sha256": summary["manifest_sha256"],
                        "metrics": metrics(episodes),
                        "scenarios": scenarios,
                        "run": summary["run"],
                    }
                )
    for f in sorted(
        (batch / "carla_formal_evaluation_v5").glob("*/evaluation_summary.json")
    ):
        summary = read(f)
        if not summary.get("complete"):
            continue
        episodes = rows(f.parent / "episodes.jsonl")
        evaluated_checkpoint = Path(summary["source_checkpoint"])
        training = read(evaluated_checkpoint / "training_summary.json")
        checkpoint_config = read(evaluated_checkpoint / "config.json")
        light_origin = Path(checkpoint_config.get("source_checkpoint", evaluated_checkpoint))
        source_training = read(light_origin / "training_summary.json")
        scenarios = {
            s: metrics([r for r in episodes if r["case"]["scenario"] == s])
            for s in ("static", "curve", "dynamic")
        }
        evaluations.append(
            {
                "backend": "CARLA_0.9.15",
                "version": "formal_v5",
                "label": summary["split"] + "_" + summary["transfer"],
                "name": summary["method"],
                "seed": summary["seed"],
                "lightweight_source_training_steps": source_training["physics_steps"],
                "carla_finetuning_training_steps": (
                    training["physics_steps"] if training.get("backend") == "CARLA_0.9.15" else 0
                ),
                "carla_controlled_training_steps": training.get("controlled_physics_steps", 0),
                "carla_reset_training_steps": training.get("reset_physics_steps", 0),
                "manifest_sha256": summary["manifest_sha256"],
                "metrics": metrics(episodes),
                "scenarios": scenarios,
                "run": str(f.parent),
            }
        )
    write_json(out / "measured_final_metrics.json", evaluations)
    plot_final_metrics(evaluations, out / "final_seed_metrics.png")
    write_json(out / "formal_development_metrics.json", dev)
    comparisons = []
    for backend, label in sorted({(r["backend"], r["label"]) for r in evaluations}):
        group = [
            r for r in evaluations if (r["backend"], r["label"]) == (backend, label)
        ]
        if len({r["manifest_sha256"] for r in group}) != 1:
            raise ValueError("Cannot pair unequal case manifests")
        for left, right in (
            ("H", "B1"),
            ("H", "B2"),
            ("H", "B1_hold10"),
            ("H_shared", "B1_shared"),
            ("H", "H_shared"),
            ("B1", "B1_shared"),
        ):
            if {left, right} <= set(r["name"] for r in group):
                values = comparison(group, left, right)
                planned = (
                    [11, 22, 33, 44, 55]
                    if left == "H" and right == "B1"
                    else [11, 22, 33]
                )
                paired = values.get("safe_completion", {})
                comparisons.append(
                    {
                        "backend": backend,
                        "label": label,
                        "left": left,
                        "right": right,
                        "metrics": values,
                        "planned_seeds": planned,
                        "unobserved_infrastructure_records": [{"method":r["name"],"seed":r["seed"],"unresolved_conditions":r["metrics"]["infrastructure_unresolved"]} for r in group if r["metrics"]["infrastructure_unresolved"]],
                        "all_planned_pairs_complete": paired.get(
                            "independent_training_seeds"
                        )
                        == planned,
                    }
                )
    write_json(out / "paired_final_metric_intervals.json", comparisons)
    primary = next(
        (
            r
            for r in comparisons
            if r["backend"] == "lightweight"
            and r["label"] == "primary"
            and r["left"] == "H"
            and r["right"] == "B1"
        ),
        None,
    )
    decision = {
        "metric": "1M independent-test safe-completion H minus B1",
        "status": "incomplete",
        "comparison": primary,
    }
    if primary and primary["all_planned_pairs_complete"]:
        ci = primary["metrics"]["safe_completion"]["ci95"]
        decision["status"] = (
            "supports_positive_complete_system_difference"
            if ci and ci[0] > 0
            else "does_not_establish_positive_primary_difference"
        )
    write_json(out / "primary_decision.json", decision)
    auc = []
    for name, seed in sorted({(r["name"], r["seed"]) for r in dev}):
        points = [
            (r["nominal_budget"], r["metrics"]["safe_completion"])
            for r in dev
            if (r["name"], r["seed"]) == (name, seed)
        ]
        auc.append(
            {
                "name": name,
                "seed": seed,
                **learning_curve_auc(points),
                "first_evaluated_budget_at_80_percent": next(
                    (x for x, y in sorted(points) if y >= 0.8), None
                ),
            }
        )
    auc_interval = paired_bootstrap(
        {
            r["seed"]: r["value"]
            for r in auc
            if r["name"] == "H" and r["value"] is not None
        },
        {
            r["seed"]: r["value"]
            for r in auc
            if r["name"] == "B1" and r["value"] is not None
        },
    )
    write_json(
        out / "formal_auc_and_threshold.json",
        {
            "per_seed": auc,
            "H_minus_B1_auc": auc_interval,
            "threshold_note": "Only checkpoint resolution; null means not observed, not infinite or interpolated success",
        },
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, color in (("H", "#2563eb"), ("B1", "#dc2626")):
        group = [r for r in dev if r["name"] == name]
        budgets = sorted({r["nominal_budget"] for r in group})
        means = []
        std = []
        for b in budgets:
            values = [
                r["metrics"]["safe_completion"]
                for r in group
                if r["nominal_budget"] == b
            ]
            means.append(np.mean(values))
            std.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            ax.annotate(f"n={len(values)}", (b, means[-1]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
        for seed in sorted({r["seed"] for r in group}):
            points = sorted(
                (r["nominal_budget"], r["metrics"]["safe_completion"])
                for r in group
                if r["seed"] == seed
            )
            ax.plot(
                [x for x, y in points],
                [y for x, y in points],
                color=color,
                alpha=0.2,
                lw=1,
            )
        ax.plot(budgets, means, "o-", color=color, label=name + " seed mean")
        ax.fill_between(
            budgets,
            np.maximum(0, np.array(means) - std),
            np.minimum(1, np.array(means) + std),
            color=color,
            alpha=0.12,
        )
    ax.set(
        xlabel="Cumulative nominal physics budget (actual overshoot in tables)",
        ylabel="Development safe completion",
        ylim=(-0.03, 1.03),
        title="Frozen formal scheme; thin lines = seeds, band = 1 sample SD",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "formal_learning_curve.png", dpi=170)
    plt.close(fig)
    lines = [
        "# 正式方案实测统计",
        "",
        f"生成时间 UTC epoch {time.time():.0f}。所有数值来自保存的检查点和完整回合；不补齐未完成实验。",
        f"报告及调度源码快照：{report_source['bundle']}，文件hash见report_source_manifest.json；算法执行快照另行登记。",
        "当前确认版本为formal_v5；formal_v4及更早累计训练存在检查点验证重置RNG的问题，只保留为历史开发记录，不计入本表或主指标。",
        "",
        "## 固定主指标",
        "",
        "共同名义1M物理步、40项独立轻量测试的安全完成率H−B1。独立单位为训练种子，20,000次配对bootstrap，统计seed 20261016。",
        "下表中CARLA的development条目仅为冻结后的开发诊断；test条目才对应30项独立CARLA测试。各划分分别统计，不混合样本。",
        "",
        "|backend/测试|方法|seed|基础设施缺测条件|安全完成率|碰撞率|越界率|进度m|接管/物理步|成功条件通行秒|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "不适用/未取得" if x is None else f"{x:.4f}"

    for r in evaluations:
        m = r["metrics"]
        lines.append(
            f"|{r['backend']}/{r['label']}|{r['name']}|{r['seed']}|{m['infrastructure_unresolved']}|{fmt(m['safe_completion'])}|{fmt(m['collision_rate'])}|{fmt(m['out_of_bounds_rate'])}|{fmt(m['progress_m'])}|{fmt(m['physical_step_intervention_rate'])}|{fmt(m['successful_travel_time_s'])}|"
        )
    if not evaluations:
        lines.append("\n尚无已完成的正式最终评估。等待对应累计检查点。")
    for c in comparisons:
        s = c["metrics"].get("safe_completion")
        if not s:
            continue
        lines.extend(
            [
                "",
                f"### {c['backend']} / {c['label']} / {c['left']}−{c['right']}",
                "",
                f"配对seed={s['independent_training_seeds']}；计划seed是否全部完成={c['all_planned_pairs_complete']}；两方均值={fmt(s['left_mean'])}/{fmt(s['right_mean'])}；样本标准差={fmt(s['left_std'])}/{fmt(s['right_std'])}；差值={fmt(s['difference_mean'])}，95%区间={s['ci95']}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 次要指标与限制",
            "",
            f"100k–1M开发曲线归一化AUC配对区间：{json.dumps(auc_interval,ensure_ascii=False)}",
            "",
            "控制与规划P95列明确是各回合P95的均值，并非合并样本的P95。离线并行机器负载下的耗时不能直接证明部署硬实时。",
            "control_cycle计时覆盖环境内单物理tick的控制、执行与反馈处理，不含外层PPO.predict、该宏动作之前的规划/时间参数化及末端观测编码。plan_times主要覆盖规划搜索。因此这些分项不能冒充完整端到端决策延迟；端到端部署时延尚未单独测量。",
            "",
            "成功条件通行时间没有成功时为空值；碰撞、越界和超时全部保留。无参考的B1不构造MPC误差。基础设施故障单列且不记成功。",
            "若基础设施重试后仍缺少物理回合，策略完成/碰撞/越界率保留未取得，不把无观测写成零碰撞或零策略完成。JSON另保留全部计划条件的保守完成率下界scheduled_completion_lower_bound；此类缺测记录不进入策略种子区间，unobserved_infrastructure_records明确列出。",
            "",
            "主比较为完整系统：H含车辆模型、几何规划、终点参考/制动、安全机制和较低决策频率。不能把差值全部归给PPO层级这一单一因素。",
            "",
            "区间条件于固定测试清单，表示独立训练种子的随机性；不是全地图总体区间或安全认证。全部五个主配对完成且95%差值区间下界大于0，才记为支持正向主指标；次要结果不能替代主指标。",
            "若计划种子未全部完成，已完成子集的区间仅描述该子集。搜索耗时可能与策略行为相关，墙钟截尾并非随机缺失，不能将较快完成的种子外推为完整五种子结果。",
            "",
            "CARLA迁移已重新标定低层车辆；微调预算另列且包含重置tick。弯曲行驶任务位于直道路面上的障碍路线，原生弯道路形尚未验证。",
            "CARLA实际初始状态、障碍/目标几何与动态控制条件的配对核对见carla_condition_audit.json；逐类同时报告固定条件的实际差异性。该审计不假设未来物理过程位级确定。",
            "",
            "详细分场景与指标区间：measured_final_metrics.json、paired_final_metric_intervals.json。原始状态/动作/奖励/碰撞帧见各run的transitions*.jsonl.gz。",
            "",
            "![正式开发曲线](formal_learning_curve.png)",
            "",
            "![独立测试逐训练seed结果](final_seed_metrics.png)",
        ]
    )
    (out / "formal_results.md").write_text("\n".join(lines), encoding="utf-8")
    baselines=[]
    for file in (batch/"baselines").glob("*/evaluation_summary.json"):
        row=read(file);baselines.append({"run":str(file.parent),"policy":"rule_only_B3","training_physics_steps":0,
                                       "split":row["split"],"metrics":metrics(rows(file.parent/"episodes.jsonl"))})
    rule=batch/"baselines/formal_B3_final_v5"
    if (rule/"summary.json").exists() and read(rule/"summary.json").get("complete"):
        baselines.append({"run":str(rule),"policy":"rule_only_B3","training_physics_steps":0,"split":"test","metrics":metrics(rows(rule/"episodes.jsonl"))})
    write_json(out/"rule_baseline_metrics.json",baselines)
    with (out/"formal_results.md").open("a",encoding="utf-8") as stream:
        stream.write("\n\n## B3规则对照\n\nB3不含PPO训练，不生成虚假的多训练seed区间。\n")
        for row in baselines:stream.write(f"\n- {row['split']}：完成率{row['metrics']['safe_completion']:.4f}，评估物理步{row['metrics']['physics_steps']}，PPO训练步0。\n")
    rule_pairs = []
    if (rule / "summary.json").exists() and read(rule / "summary.json").get("complete"):
        rule_cases = {r["case_id"]: r for r in rows(rule / "episodes.jsonl")}
        for evaluation in evaluations:
            if (evaluation["backend"], evaluation["label"], evaluation["name"]) != ("lightweight", "primary", "H"):
                continue
            learned_cases = {r["case_id"]: r for r in rows(Path(evaluation["run"]) / "episodes.jsonl")}
            if set(learned_cases) != set(rule_cases) or any(
                learned_cases[i]["case"] != rule_cases[i]["case"] for i in rule_cases
            ):
                raise ValueError("H/B3 condition identities differ; refusing an unpaired comparison")
            outcomes = Counter()
            by_scenario = defaultdict(Counter)
            for i, r in learned_cases.items():
                key = ("both_success" if r["success"] and rule_cases[i]["success"] else
                       "H_only_success" if r["success"] else
                       "B3_only_success" if rule_cases[i]["success"] else "both_failed")
                outcomes[key] += 1
                by_scenario[r["scenario"]][key] += 1
            rule_pairs.append({"seed": evaluation["seed"], "cases": len(rule_cases),
                               "outcomes": dict(outcomes), "scenarios": dict(by_scenario),
                               "interpretation": "Descriptive condition-matched diagnostic; one deterministic rule policy is not an independent training-seed sample."})
    write_json(out / "paired_rule_outcomes.json", rule_pairs)
    with (out / "formal_results.md").open("a", encoding="utf-8") as stream:
        for row in rule_pairs:
            stream.write(f"\nH seed{row['seed']}与B3在同一{row['cases']}条件上逐项核对：{json.dumps(row['outcomes'], ensure_ascii=False)}。")
        if rule_pairs:
            stream.write("\n\n这是描述性反例检查，不把同一规则策略复制成多个独立训练seed，也不据此声称PPO相对规则有统计优势。分场景见paired_rule_outcomes.json。\n")
    with (batch / "research_results.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n\n正式方案多指标与独立最终测试详见 [formal_results.md](analysis/formal_results.md)。\n"
        )
    print(
        json.dumps(
            {
                "completed_formal_evaluation_records": len(evaluations),
                "comparisons": len(comparisons),
                "development_records": len(dev),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    main(p.parse_args().batch)
