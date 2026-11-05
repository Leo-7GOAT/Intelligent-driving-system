"""Rebuild a descriptive comparison from every closed native development endpoint.

Repeated seeds are not unique final conditions. This script makes no inferential
or PPO-only attribution claim and never drives the simulator.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basic_qualification(episodes):
    if len(episodes) != 4 or sorted(e['case_id'] for e in episodes) != list(range(4)):
        raise ValueError('Require all four registered basic probes')
    unsafe=sum(e['termination_reason'] in ('collision','out_of_bounds') for e in episodes)
    successes=sum(e['success'] for e in episodes)
    return dict(successes=successes,episodes=4,unsafe=unsafe,passed=successes>=3 and unsafe==0)


def endpoint(batch, seed, label):
    p = read(batch / "protocol.json")
    method, = p["methods"]
    folder = batch / method
    summary = read(batch / "summary.json")
    guard = read(batch / "guard_final.json")
    if summary.get("status") != "complete" or guard.get("exit_code") != 0 or guard.get("reason") is not None:
        raise ValueError("Endpoint is not normally closed: " + batch.name)
    if read(batch / "infrastructure/server_stop.json")["status"] != "owned_empty_server_stopped":
        raise ValueError("Owned server cleanup is missing")
    ev = read(folder / "evaluation.json")
    audit = read(folder / "endpoint_raw_audit.json")
    for name, key in (("evaluation.json", "evaluation_sha256"), ("summary.json", "summary_sha256"), ("config.json", "config_sha256")):
        if digest(folder / name) != audit[key]:
            raise ValueError("Audited endpoint changed: " + name)
    if digest(batch / "protocol.json") != audit["protocol_sha256"]:
        raise ValueError("Audited protocol changed")
    for name, key in (("model.zip", "model_sha256"), ("normalization.pkl", "normalization_sha256")):
        if digest(folder / name) != ev[key]:
            raise ValueError("Evaluated model changed")
    episodes = ev["episodes"]
    if len(episodes) != 24 or sorted(e["case_id"] for e in episodes) != list(range(24)):
        raise ValueError("Require the complete known24 inventory")
    controls = sum(e["physics_steps"] for e in episodes)
    engine = read(batch / "strict_delivery.json")
    if not engine["strict_delivery_gate_passed"] or engine["counts"]["controls"] != controls or audit["evaluation_control_frames_verified"] != controls:
        raise ValueError("Control accounting differs")
    results = [{k: e[k] for k in ("case_id", "scenario", "success", "termination_reason", "physics_steps", "actions", "interventions", "plan_failures")} for e in episodes]
    inventory = p['native_validation']['manifest']['cases']
    if len(inventory) != 24:
        raise ValueError('Validation condition inventory differs')
    for result in results:
        hazard = inventory[result['case_id']].get('hazard')
        result['core_class'] = hazard['kind'] if hazard else 'curve'
    ancestry = dict(p["recovery"][method])
    correction = batch / "recovery_ancestry_correction.json"
    basic = None
    basic_batch=batch.with_name(batch.name.replace('_known24','_basic'))
    if label=='B1' and (basic_batch/'B1/evaluation.json').exists():
        basic_path=basic_batch/'B1/evaluation.json'
        if digest(basic_path)!=read(basic_batch/'B1/endpoint_raw_audit.json')['evaluation_sha256']:
            raise ValueError('Basic qualification input changed')
        basic=basic_qualification(read(basic_path)['episodes'])
    return dict(seed=seed, method=label, native_method=method, batch=batch.name,
        successes=sum(e["success"] for e in episodes), episodes=len(episodes),
        unsafe=sum(e["termination_reason"] in ("collision", "out_of_bounds") for e in episodes),
        reasons=dict(Counter(e["termination_reason"] for e in episodes)),
        by_scenario={s: dict(successes=sum(e["success"] for e in episodes if e["scenario"] == s), episodes=sum(e["scenario"] == s for e in episodes)) for s in sorted({e["scenario"] for e in episodes})},
        by_core_class={s: dict(successes=sum(e['success'] for e in results if e['core_class']==s), episodes=sum(e['core_class']==s for e in results)) for s in sorted({e['core_class'] for e in results})},
        controls=controls, model_sha256=ev["model_sha256"], normalization_sha256=ev["normalization_sha256"],
        protocol_sha256=digest(batch / "protocol.json"), raw_audit_sha256=digest(folder / "endpoint_raw_audit.json"),
        ancestry=ancestry, ancestry_correction=read(correction) if correction.exists() else None,
        basic_probe=basic,cases=results)


def build(archive, output, figure):
    archive, output, figure = map(Path, (archive, output, figure))
    rows = []
    for seed in (11, 22, 33, 44, 55):
        for label in ("H", "H0", "B1"):
            name = (f"20261104_matched_flat_seed{seed}_known24" if label == "B1" else f"20261104_independent_outcome_seed{seed}_{label}_known24")
            batch = archive / name
            if (batch / "summary.json").exists() and read(batch / "summary.json").get("status") == "complete":
                rows.append(endpoint(batch, seed, label))
    paired = sorted({r["seed"] for r in rows if r["method"] == "H"} & {r["seed"] for r in rows if r["method"] == "B1"})
    aggregates = {}
    for method in ("H", "B1"):
        chosen = [r for r in rows if r["seed"] in paired and r["method"] == method]
        if not chosen:
            raise ValueError("No complete paired endpoints")
        aggregates[method] = {k: sum(r[k] for r in chosen) for k in ("successes", "episodes", "unsafe", "controls")}
        aggregates[method]["success_rate"] = aggregates[method]["successes"] / aggregates[method]["episodes"]
    # Match task geometry, vehicle and case identity, not just list lengths.
    protocols = [read(archive / r["batch"] / "protocol.json") for r in rows if r["seed"] in paired and r["method"] in ("H", "B1")]
    for p in protocols[1:]:
        if (p["native_validation"] != protocols[0]["native_validation"]
                or p["calibration_sha256"] != protocols[0]["calibration_sha256"]):
            raise ValueError("Endpoints do not share registered conditions and calibration")
    data = dict(schema_version=1, date="2026-11-05", scope="Descriptive native CARLA development comparison; all closed endpoints retained, no new driving",
        research_acceptance="UNMET", paired_seeds=paired, unique_known_conditions=24,
        aggregates=aggregates, H_minus_B1_percentage_points=100*(aggregates["H"]["success_rate"]-aggregates["B1"]["success_rate"]),
        endpoints=rows, new_controls=0, new_optimizer_updates=0,
        limitations=["Repeated known24 conditions across learners are not new unique final conditions.",
            "Hierarchy includes planning/MPC and inherited teacher initialization; the gap is not attributable to PPO alone.",
            "Seed22 H has a preserved interruption, discarded readable tail and unknown unflushed ancestral tail; its correction sidecar governs ancestry.",
            "B1 basic qualification is per seed: seeds11/22 pass, seed33 fails2/4with2unsafe; complex-task convergence and equal historical development cost are not established.",
            "H0 and the strong historical B3 leave PPO-specific and added-feedback benefit unestablished.",
            "Five complete paired learners, final120, and the formal paired-confidence gate remain unmet."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"H": "#187b73", "B1": "#bd6238"}
    x = np.arange(len(paired))
    for method, offset in (("H", -.18), ("B1", .18)):
        selected = [next(r for r in rows if r["seed"] == seed and r["method"] == method) for seed in paired]
        for ax, key in zip(axes, ("successes", "unsafe")):
            vals = [100*r[key]/r["episodes"] for r in selected]
            bars = ax.bar(x+offset, vals, .36, color=colors[method], label="Hierarchy" if method == "H" else "Flat PPO")
            ax.bar_label(bars, labels=[f'{r[key]}/24' for r in selected], padding=3, fontsize=9)
            ax.set_xticks(x, [str(seed) for seed in paired]); ax.set_xlabel("Independent learner seed")
            ax.set_ylim(0, 113); ax.set_ylabel("Episodes (%)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Safe task completion"); axes[1].set_title("Collision or bounds violation")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("CARLA 0.9.15: 24 known development conditions per learner\nRepeated conditions; descriptive evidence, not final confirmation", fontsize=11)
    figure.parent.mkdir(parents=True, exist_ok=True); fig.savefig(figure, dpi=170); plt.close(fig)
    print(json.dumps({"paired_seeds": paired, "aggregates": aggregates, "gap_pp": data["H_minus_B1_percentage_points"]}))
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True); p.add_argument("--output", required=True); p.add_argument("--figure", required=True)
    a = p.parse_args(); build(a.archive, a.output, a.figure)
