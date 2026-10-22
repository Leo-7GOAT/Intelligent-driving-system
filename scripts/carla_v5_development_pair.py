"""Fixed seed11 100k development transfer for the RNG-repaired version; no tuning or test use."""

import argparse, hashlib, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json
from navstack.experiments import append_json
from carla_evaluate import evaluate_carla


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    folder = batch / "carla_v5_100k_development"
    folder.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    cases = root / "configs/carla_development_cases.json"
    deadline = read(batch / "batch.json")["deadline_epoch"]
    protocol = {
        "epoch": time.time(), "source_version": "formal_v5", "seed": 11,
        "lightweight_budget": 100000, "methods": ["H", "B1"],
        "case_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        "purpose": "Development-only real execution of corrected-version paired checkpoints; no hyperparameter changes or model selection",
        "transfer": "low_level_recalibrated_frozen_policy_100k_v5",
        "final_test_used": False, "carla_training_steps": 0,
    }
    if not (folder / "protocol.json").exists():
        atomic_json(folder / "protocol.json", protocol)
    while not all((batch / "formal_v5" / f"{m}_seed11/validation_100000.json").exists() for m in ("H", "B1")):
        if time.time() > deadline - 180 or (batch / "STOP").exists():
            return
        time.sleep(3)
    pipeline = read(batch / "carla_formal_pipeline_v5/status.json")
    if pipeline["status"] != "waiting_for_paired_seed_1M":
        raise RuntimeError("The formal CARLA owner is active; refuse concurrent use")
    checkpoints = {}
    for method in ("H", "B1"):
        source_run = batch / "formal_v5" / f"{method}_seed11"
        target_steps = read(source_run / "validation_100000.json")["actual_physics_steps"]
        candidates = [f.parent for f in source_run.glob("checkpoint_*/training_summary.json")
                      if read(f)["physics_steps"] == target_steps]
        if not candidates:
            raise RuntimeError("Exact 100k milestone checkpoint missing")
        checkpoints[method] = str(sorted(candidates)[0])
    atomic_json(folder / "checkpoints.json", checkpoints)
    for method in ("H", "B1"):
        if time.time() > deadline - 180 or (batch / "STOP").exists():
            return
        run = folder / method
        if (run / "evaluation_summary.json").exists() and read(run / "evaluation_summary.json").get("complete"):
            continue
        atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "evaluating", "method": method, "epoch": time.time()})
        _, result = evaluate_carla(batch, checkpoints[method], cases, "development", protocol["transfer"], run=run)
        append_json(folder / "results.jsonl", result)
    atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "complete", "epoch": time.time()})


if __name__ == "__main__":
    main()
