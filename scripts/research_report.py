"""Regenerate tables and figures from actual immutable checkpoints and episode logs."""

import argparse, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from navstack.statistics import paired_bootstrap, learning_curve_auc
from navstack.experiments import write_json
from training_time_accounting import session_training_seconds


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report(batch):
    batch = Path(batch)
    out = batch / "analysis"
    out.mkdir(exist_ok=True)
    historical_note = (
        "检查点审计：formal_v4及更早累计训练受保存重载时重置全局RNG的实现缺陷影响。"
        "这些观测保留为历史开发记录，不计入确认结论。formal_v5从头训练；"
        "主指标仅使用该版本的完整独立最终测试，未完成时不作优势结论。"
    )
    statuses = []
    evals = []
    if (batch / "debug_pair.json").exists():
        for entry in read(batch / "debug_pair.json"):
            r = dict(entry["training"])
            r.update(version="debug", name=f"{entry['method']}_seed{r['seed']}", backend="lightweight", reason="debug_budget_reached")
            statuses.append(r)
            episodes=[json.loads(line) for line in (Path(entry["validation"])/"episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            evals.append({"version":"debug","name":r["name"],"seed":r["seed"],"method":r["method"],"budget":20000,
                          "actual_physics_steps":r["physics_steps"],"metrics":entry["evaluation"],"episodes":episodes})
    for latest in sorted(batch.glob("*/*/latest.json")):
        if latest.parent.parent.name == "carla_resume_verification":
            continue
        r = read(latest)
        r["version"] = latest.parent.parent.name
        r["name"] = latest.parent.name
        statuses.append(r)
        optimization = latest.parent / "optimization.jsonl"
        if optimization.exists():
            optimization_rows = [json.loads(line) for line in optimization.read_text(encoding="utf-8").splitlines(keepends=True) if line.endswith("\n")]
            accounting = session_training_seconds([row for row in optimization_rows if row["physics_steps"] <= r["physics_steps"]])
            r["measured_training_wall_seconds_from_sessions"] = accounting["seconds"]
            r["training_session_time_accounting"] = accounting
        for validation in sorted(latest.parent.glob("validation_*.json")):
            e = read(validation)
            budget = int(validation.stem.split("_")[-1])
            episodes = [
                json.loads(x)
                for x in (Path(e["evaluation"]) / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            evals.append(
                {
                    "version": r["version"],
                    "name": r["name"],
                    "seed": r["seed"],
                    "method": r["method"],
                    "budget": budget,
                    "actual_physics_steps": e["actual_physics_steps"],
                    "metrics": e["metrics"],
                    "episodes": episodes,
                }
            )
    statistics = []
    for version, budget in sorted({(e["version"], e["budget"]) for e in evals}):
        group = [e for e in evals if e["version"] == version and e["budget"] == budget]
        paired = paired_bootstrap(
            {
                e["seed"]: e["metrics"]["success_rate"]
                for e in group
                if e["name"].startswith("H_seed")
            },
            {
                e["seed"]: e["metrics"]["success_rate"]
                for e in group
                if e["name"].startswith("B1_seed")
            },
        )
        statistics.append(
            {
                "version": version,
                "nominal_budget": budget,
                "split": "development",
                "metric": "safe_completion",
                **paired,
            }
        )
    write_json(out / "status.json", statuses)
    write_json(out / "confirmation_scope.json", {
        "active_version": "formal_v5", "supplementary_version": "supplementary_v5",
        "versions_not_used_for_primary_confirmation": sorted({r["version"] for r in statuses
            if r["version"] not in ("formal_v5", "supplementary_v5", "carla_formal_training_v5")}),
        "debug_note": "Debug trainer reloads only after training; it is not the cumulative checkpoint RNG failure path.",
        "note": historical_note,
    })
    write_json(out / "development_statistics.json", statistics)
    versions = sorted({e["version"] for e in evals})
    nrows = max(1, (len(versions) + 1) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.4 * nrows), squeeze=False)
    for ax, version in zip(axes.flat, versions):
        group = [e for e in evals if e["version"] == version]
        for name in sorted({e["name"].rsplit("_seed", 1)[0] for e in group}):
            series = [e for e in group if e["name"].rsplit("_seed", 1)[0] == name]
            budgets = sorted({e["budget"] for e in series})
            means, errors = [], []
            for budget in budgets:
                values = [e["metrics"]["success_rate"] for e in series if e["budget"] == budget]
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            ax.errorbar(budgets, means, yerr=errors, marker="o", capsize=3, label=name)
        ax.set(xlabel="Nominal physics budget (actual counts in table)",
               ylabel="Development completion", ylim=(-.05, 1.05),
               title=version + (" (current)" if version == "formal_v5" else " (historical)"))
        if len({e["budget"] for e in group}) == 1:
            ax.set_xticks([group[0]["budget"]])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(versions):]:
        ax.set_visible(False)
    fig.suptitle("Measured development only; means and sample SD. Current confirmation: formal_v5", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "learning_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    failures = []
    for e in evals:
        byscene = {}
        for scene in sorted({x["scenario"] for x in e["episodes"]}):
            rows = [x for x in e["episodes"] if x["scenario"] == scene]
            byscene[scene] = {
                "episodes": len(rows),
                "success_rate": float(np.mean([x["success"] for x in rows])),
                "failures": dict(Counter(x["termination_reason"] for x in rows)),
                "mean_progress": float(np.mean([x["progress"] for x in rows])),
                "mean_intervention_rate": float(
                    np.mean([x["intervention_rate"] for x in rows])
                ),
            }
        failures.append(
            {
                "version": e["version"],
                "name": e["name"],
                "budget": e["budget"],
                "scenarios": byscene,
            }
        )
    write_json(out / "scenario_results.json", failures)
    final_statistics = []
    for version in sorted({r["version"] for r in statuses}):
        for label in ("primary", "generalization"):
            rows = []
            for marker in (batch / version).glob("*/final_" + label + ".json"):
                row = read(marker)
                if row.get("complete"):
                    row["run_name"] = marker.parent.name
                    rows.append(row)
            if rows:
                hashes = {r["manifest_sha256"] for r in rows}
                if len(hashes) != 1:
                    raise ValueError("Final paired results use different manifests")
                left = {
                    r["seed"]: r["success_rate"]
                    for r in rows
                    if r["run_name"].startswith("H_seed")
                }
                right = {
                    r["seed"]: r["success_rate"]
                    for r in rows
                    if r["run_name"].startswith("B1_seed")
                }
                if not left and not right:
                    continue
                final_statistics.append(
                    {
                        "version": version,
                        "label": label,
                        "manifest_sha256": list(hashes)[0],
                        "planned_seeds": [11, 22, 33, 44, 55],
                        **paired_bootstrap(left, right),
                    }
                )
    write_json(out / "final_statistics.json", final_statistics)
    auc = []
    for version, name in sorted({(e["version"], e["name"]) for e in evals}):
        points = [
            (e["budget"], e["metrics"]["success_rate"])
            for e in evals
            if (e["version"], e["name"]) == (version, name)
        ]
        auc.append(
            {
                "version": version,
                "name": name,
                **learning_curve_auc(points),
                "first_milestone_at_80_percent": next(
                    (x for x, y in sorted(points) if y >= 0.8), None
                ),
            }
        )
    write_json(out / "learning_auc.json", auc)
    lines = [
        "# 实测研究进度",
        "",
        "以下数字来自本批实际日志；开发集结果不能代替独立最终测试。",
        "",
        "|版本/方法/seed|backend|检查点物理步|策略动作|优化阶段|optimizer.step|Actor L2变化|精确重载|状态|",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in statuses:
        lines.append(
            f"|{r['version']}/{r['name']}|{r['backend']}|{r['physics_steps']}|{r['actions']}|{r['optimization_phases']}|{r.get('optimizer_steps','旧日志未记录')}|{r['actor_l2_change']:.4f}|{r['reload_exact']}|{r['reason']}|"
        )
    lines += ["", "## 当前版本交互量与训练时长", "",
              "物理秒按固定dt=0.05折算；CARLA世界步包含重置，控制/重置分列。训练秒截至该检查点之前最后一次优化日志，含会话内保存耗时，不含排队、会话开始前加载、最后保存和评估；多个进程并行，其秒数不能相加当作本批墙钟。进行中的额外交互见run_status.md。", "",
              "|版本/运行|世界物理秒|控制/重置物理步|已记录训练秒|", "|---|---:|---:|---:|"]
    for r in statuses:
        if r["version"] not in ("formal_v5", "supplementary_v5", "carla_formal_training_v5", "carla_v5_100k_source_finetune_warmup1"):
            continue
        control = r.get("controlled_physics_steps", r["physics_steps"])
        reset = r.get("reset_physics_steps", 0)
        seconds = r.get("measured_training_wall_seconds_from_sessions")
        seconds_text = "未记录" if seconds is None else f"{seconds:.2f}"
        lines.append(f"|{r['version']}/{r['name']}|{r['physics_steps']*.05:.2f}|{control}/{reset}|{seconds_text}|")
    lines += [
        "",
        "## 配对开发结果",
        "",
        "统计单位为独立训练种子；不足3个配对种子不计算区间。20,000次配对bootstrap，统计种子20261016。",
    ]
    for s in statistics:
        lines.append(
            f"- {s['version']} / {s['nominal_budget']}：配对seed={s['independent_training_seeds']}，H均值={s['left_mean']}，B1均值={s['right_mean']}，差值={s['difference_mean']}，95%区间={s['ci95']}。"
        )
    lines += [
        "",
        "## 独立最终测试",
        "",
        (
            json.dumps(final_statistics, ensure_ascii=False, indent=2)
            if final_statistics
            else "尚无已完成的1M独立最终测试。"
        ),
    ]
    carla = []
    for path in sorted(
        [
            *(batch / "carla").glob("*/evaluation_summary.json"),
            *(batch / "carla_v5_100k_development").glob("*/evaluation_summary.json"),
            *(batch / "carla_v5_100k_source_finetune").glob("*/evaluation_summary.json"),
            *(batch / "carla_v5_100k_source_finetune_warmup1").glob("*/evaluation_summary.json"),
            *(batch / "carla_formal_evaluation_v5").glob("*/evaluation_summary.json"),
        ]
    ):
        row = read(path)
        carla.append(row)
    write_json(out / "carla_evaluations.json", carla)
    lines += [
        "",
        "## CARLA实测",
        "",
        "|方法|迁移阶段|划分|完成/计划case|成功率|策略控制步|含重置物理步|基础设施未解决|",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in carla:
        lines.append(
            f"|{r['method']}|{r['transfer']}|{r['split']}|{r['completed']}/{r['cases']}|{r['success_rate_all_scheduled']:.3f}|{r['physics_steps']}|{r.get('world_physics_ticks_including_reset','未单列')}|{r['infrastructure_unresolved']}|"
        )
    lines += [
        "",
        "早期CARLA微调仅seed11；正式多seed结果另见analysis/formal_results.md。回合数不等于独立训练seed数。最终关闭桥接的一帧清理开销不参与PPO，另视为基础设施开销。",
    ]
    lines += [
        "",
        "主指标仍为共同1M预算、独立最终测试的安全完成率差值；未运行阶段不外推，也不以开发集区间替代该指标。",
        "",
        "![学习曲线](analysis/learning_curve.png)",
    ]
    lines.insert(2, historical_note + "\n")
    (batch / "research_results.md").write_text("\n".join(lines), encoding="utf-8")
    failure_lines = [
        "# 失败分析",
        "",
        "全部任务失败保留；基础设施故障单列，不能作为零碰撞成功。",
        "本批末尾CARLA seed44在H冻结开发之后出现独占锁残留。旧PID37360已经退出，后续B1开发和双方最终评估在tick前被拒绝，H微调也在首个reset/optimizer之前失败。确认无桥接进程且匹配0.9.15世界为空后，原锁归档于infrastructure/stale_tick_owner_37360_*.lock；退出清理为什么留下锁仍需加固调查，未擅自增加最终评估重试。原始失败日志和缺测条件全部保留，不能称该模型0/30策略失败。",
        "",
        "已确认的接口瓶颈：31动作目录在窄通道起点有较多不可行转向目标，接近终点时长距离目标越界；独立几何见证没有按H成功筛选场景。",
        "",
        "CARLA原生Tesla轴距3.005m、宽2.163m，与轻量训练车不同；迁移使用实测几何和共同执行器，标记为低层重新标定。",
        "",
        "规则高层静态/弯道诊断超时，但完整可行参考经Hybrid A*/MPC实车停车成功。两种诊断均非PPO成绩。",
        "",
        "正式候选动作复核：停车位置的可规划非等待候选由42/90提高到90/90；窄通道起点仍为30/90。该审计是几何接口诊断，不是PPO完成率。见analysis/action_feasibility_summary.json及对应图。",
        "CARLA逐条件动作与执行审计见analysis/carla_decision_failure_audit.md/json：同时列实际静止比例、规划状态、干预比例、油门/制动证据覆盖率。没有记录的旧执行器字段保留缺失，不补零；静止无碰撞仍按超时失败。",
        "",
        "formal_v4的300k seed11开发反例：H静态场景3/3超时，制动干预约78%–100%；其75%总体完成率来自空场景、窄通道与动态场景。B1同档主要停驶或进度很小。后续里程碑必须独立报告，不能用这条开发观察概括最终结果。",
        "",
        "formal_v5的100k seed11开发反例：H空场景第0项距终点3.10m后选择等待63/90次；静态第1项重复不可规划动作83/90次，对应830/900物理步干预；窄通道第2项71/90次不可规划。B1第0项全程非正加速度、进度0。该固定种子诊断不能替代其余种子或后续预算。新版100k真实CARLA冻结迁移H/B1均0/6，全部超时，保留全部失败。",
        "",
        "逐回合实际最终位置、目标距离、动作分布、规划失败状态和干预比例见[决策失败审计](analysis/decision_failure_audit.md)。原生H/B1是完整系统比较；低碰撞或低jerk若伴随停驶，不能单独称优势。",
        "",
        "训练期间6m/16m/23m真实初始化目标下的完整回合、成功数、进度和奖励分量见[训练课程审计](analysis/training_curriculum_audit.md)。这些是采样策略的训练记录，不能当成确定性验证或最终成绩。",
        "",
        "PPO连续动作基线尚有探索/停车与退化停驶问题；target_kl导致实际epochs不同，真实optimizer.step次数另存，不能把名义10 epochs误称每轮都执行了10次。B3规则有竞争力时也保留该反例，不以规则轨迹冒充PPO。",
        "正式CARLA的1M源模型在共同50k微调后，seed33的B1静态场景平均有效进度15.53m，高于H的3.39m，但双方均0/10完成停车；不能把B1的零完成率描述成完全没学会运动。H同seed的curve平均进度20.61m仍0/10完成，表明到达附近不等于满足位置/朝向/速度终止条件。该H模型最终测试干预物理步比例65.57%，也不能仅以动态场景10/10成功声称高层独立学会安全。所有值均是固定已完成seed33配对测试的描述，不外推为全部种子。",
        "最终测试暴露后的探索性几何审计：固定主清单全部10个unseen布局的有限解析轨迹搜索均未找到见证，原始零执行结果保留于analysis/posthoc_feasibility_witness/results.json。随后独立的有界三次样条搜索对同一全部10项寻找整车无碰撞/曲率合法路径，找到的参考由原MPC和真实轻量车辆模型执行；结果见analysis/posthoc_spline_feasibility/results.json及all_conditions.png。未找到见证不证明地图无解；找到的全局特权参考也不是PPO成绩。任何布局、策略成绩及确认规则均未改变。",
    ]
    for diagnostic in ("posthoc_spline_feasibility", "posthoc_shared_witness_library"):
        diagnostic_path = out / diagnostic / "results.json"
        if diagnostic_path.exists():
            result = read(diagnostic_path)
            successful_ids = [r["case_id"] for r in result["results"] if r.get("outcome") == "success"]
            failure_lines.extend(["", f"独立诊断 {diagnostic}：检查{len(result['results'])}/{result['scheduled_cases']}条件，实际成功见证条件{successful_ids}；实际车辆执行{result['physical_steps_executed']}步。这些是诊断交互，不计PPO训练或完成率；重复条件不能当成新增独立场景。"])
    for row in failures:
        failure_lines += [
            "",
            f"## {row['version']}/{row['name']} @ {row['budget']}",
            "",
            json.dumps(row["scenarios"], ensure_ascii=False, indent=2),
        ]
    failure_lines.insert(2, historical_note + "\n")
    (batch / "failure_analysis.md").write_text(
        "\n".join(failure_lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {"statuses": len(statuses), "evaluations": len(evals), "output": str(out)}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    report(p.parse_args().batch)
