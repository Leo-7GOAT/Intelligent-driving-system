"""Matched 50k real CARLA development fine-tuning from fixed seed11 100k sources."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json
from navstack.experiments import append_json
from carla_finetune_pair import finetune
from carla_evaluate import evaluate_carla


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--retry-h", action="store_true", help="One fresh attempt after the preserved pre-update H infrastructure failure")
    p.add_argument("--warmup-version", action="store_true", help="Fresh matched pair after validated sensor-startup repair")
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    root = Path(__file__).resolve().parents[1]
    folder = batch / ("carla_v5_100k_source_finetune" + ("_warmup1" if args.warmup_version else ""))
    folder.mkdir(exist_ok=True)
    checkpoints = read(batch / "carla_v5_100k_development/checkpoints.json")
    deadline = read(batch / "batch.json")["deadline_epoch"]
    protocol = {
        "epoch": time.time(), "source_checkpoints": checkpoints, "seed": 11,
        "cumulative_carla_physics_budget_each": 50000, "includes_reset_ticks": True,
        "split": "development", "final_tests_used": False,
        "hyperparameters": "Inherit the frozen source PPO settings; no new tuning",
        "stage_order": ["H_finetune", "B1_finetune", "H_development", "B1_development"],
        "formal_1M_pipeline_temporarily_stopped_to_ensure_exclusive_CARLA_ownership": True,
        "resume_formal_pipeline_afterward": True,
        "sensor_startup_repair": bool(args.warmup_version),
        "prior_failed_attempts_preserved": "carla_v5_100k_source_finetune/H_seed11 and H_seed11_retry1",
    }
    if not (folder / "protocol.json").exists():
        atomic_json(folder / "protocol.json", protocol)
    if args.retry_h and (folder / "H_seed11_retry1/failures.jsonl").exists():
        raise RuntimeError("The one permitted H infrastructure retry already failed")
    try:
        trained = {}
        for method in ("H", "B1"):
            if time.time() > deadline-180 or (batch / "STOP").exists():
                return
            atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "training", "method": method, "epoch": time.time()})
            run = folder / (f"{method}_seed11" + ("_retry1" if method == "H" and args.retry_h else ""))
            result = finetune(batch, checkpoints[method], 50000, 11, run)
            append_json(folder / "results.jsonl", {"stage": "training", "method": method, "result": result})
            if result["physics_steps"] < 50000:
                return
            trained[method] = result["checkpoint"]
        for method in ("H", "B1"):
            if time.time() > deadline-180 or (batch / "STOP").exists():
                return
            atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "evaluating", "method": method, "epoch": time.time()})
            marker = folder / (method + "_development/evaluation_summary.json")
            if marker.exists() and read(marker).get("complete"):
                continue
            _, result = evaluate_carla(batch, trained[method], root / "configs/carla_development_cases.json", "development",
                                      "carla_finetuned_50k_from_v5_100k", run=folder / (method + "_development"))
            append_json(folder / "results.jsonl", {"stage": "development", "method": method, "result": result})
        atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "complete", "epoch": time.time()})
    except Exception as exc:
        atomic_json(folder / "status.json", {"pid": os.getpid(), "status": "infrastructure_or_training_failure", "error": repr(exc), "epoch": time.time()})
        raise
    finally:
        if time.time() < deadline-180 and not (batch / "STOP").exists():
            log = (batch / ("carla_formal_resume_after_development_" + str(time.time_ns()) + ".log")).open("w", encoding="utf-8")
            process = subprocess.Popen([sys.executable, str(root / "scripts/carla_formal_pipeline.py"), "--batch", str(batch)],
                                       cwd=root, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            log.close()
            atomic_json(folder / "formal_pipeline_resumed.json", {"launcher_pid": process.pid, "epoch": time.time()})


if __name__ == "__main__":
    main()
