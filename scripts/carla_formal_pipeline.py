"""Serial CARLA owner: preregistered paired 1M transfer and matched 100k fine-tuning."""

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json, resources_ok
from navstack.experiments import append_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_for_fine_budget(run, budget):
    candidates = []
    for path in Path(run).glob("checkpoint_*/training_summary.json"):
        row = read(path)
        if (
            row.get("requested_physics_budget") == budget
            and row["physics_steps"] >= budget
        ):
            candidates.append((row["physics_steps"], str(path.parent.resolve())))
    if not candidates:
        raise RuntimeError(
            f"No completed CARLA checkpoint registered for {budget} steps"
        )
    return min(candidates)[1]


def worker(args):
    from carla_evaluate import evaluate_carla
    from carla_finetune_pair import finetune

    job = read(args.job)
    batch = Path(args.batch).resolve()
    run = Path(job["run"])
    if job["kind"] == "train":
        result = finetune(batch, job["checkpoint"], job["budget"], job["seed"], run=run)
    else:
        _, result = evaluate_carla(
            batch, job["checkpoint"], job["cases"], job["split"], job["label"], run=run
        )
    print(json.dumps(result), flush=True)


def main(args):
    batch = Path(args.batch).resolve()
    root = Path(__file__).resolve().parents[1]
    deadline = read(batch / "batch.json")["deadline_epoch"]
    folder = batch / "carla_formal_pipeline_v5"
    folder.mkdir(exist_ok=True)
    specs = [(s, m) for s in (11, 22, 33, 44, 55) for m in ("H", "B1")]
    code = [
        *sorted((root / "navstack").glob("*.py")),
        root / "scripts/carla_bridge37.py",
        root / "scripts/carla_evaluate.py",
        root / "scripts/carla_finetune_pair.py",
        root / "scripts/carla_diagnostic.py",
        Path(__file__),
    ]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code
    }
    protocol = {
        "epoch": time.time(),
        "source_version": "formal_v5",
        "source_physics_budget": 1000000,
        "seeds": [11, 22, 33, 44, 55],
        "fine_tuning_physics_budget_each": 100000,
        "fine_tuning_cumulative_milestones": [50000, 100000],
        "includes_reset_ticks": True,
        "inference": "deterministic",
        "fine_tuning_hyperparameters": "Inherit each source PPO optimizer and its frozen settings; no further tuning",
        "fine_tuning_cases": "30 training conditions: static/curve/dynamic x 0..9; disjoint from development 11/22 and final 1000..1009",
        "order": [
            "frozen_development",
            "frozen_final",
            "matched_finetune",
            "finetuned_development",
            "finetuned_final",
        ],
        "final_manifest_sha256": hashlib.sha256(
            (root / "configs/carla_final_cases.json").read_bytes()
        ).hexdigest(),
        "code_sha256": hashes,
        "source_code_is_recorded_and_checked_before_every_job": True,
        "scheduling": "Choose the first ready paired 1M seed in fixed priority 11/22/33/44/55; run all matched stages for it, then choose among remaining seeds. Selection uses checkpoint readiness only, never performance. All five pairs remain required for a complete result.",
        "scope": "Secondary CARLA external validation, separate from the primary lightweight 1M endpoint",
    }
    if (folder / "protocol.json").exists():
        prior = read(folder / "protocol.json")
        if prior["code_sha256"] != hashes:
            raise RuntimeError("CARLA pipeline code changed after registration")
    else:
        atomic_json(folder / "protocol.json", protocol)

    def stopped():
        return time.time() > deadline - 90 or (batch / "STOP").exists()

    completed = []
    failed = []
    def ready_jobs():
        for budget in protocol["fine_tuning_cumulative_milestones"]:
            pending_seeds = list(protocol["seeds"])
            while pending_seeds:
                ready = next((s for s in pending_seeds if all(
                    (batch / "formal_v5" / f"{m}_seed{s}" / "validation_1000000.json").exists()
                    for m in ("H", "B1"))), None)
                if ready is None:
                    atomic_json(folder / "status.json", {"pid": os.getpid(), "epoch": time.time(),
                        "status": "waiting_for_ready_paired_1M", "pending_seeds": pending_seeds,
                        "fine_tuning_milestone": budget, "completed": completed})
                    if stopped():
                        return
                    time.sleep(5)
                    continue
                pending_seeds.remove(ready)
                for stage in protocol["order"]:
                    yield budget, ready, stage

    for budget, seed, stage in ready_jobs():
        while not all(
            (
                batch / "formal_v5" / f"{m}_seed{seed}" / "validation_1000000.json"
            ).exists()
            for m in ("H", "B1")
        ):
            atomic_json(
                folder / "status.json",
                {
                    "pid": os.getpid(),
                    "epoch": time.time(),
                    "status": "waiting_for_paired_seed_1M",
                    "seed": seed,
                    "completed": completed,
                },
            )
            if stopped():
                return
            time.sleep(5)
        checkpoints = {
            (seed, m): read(batch / "formal_v5" / f"{m}_seed{seed}" / "latest.json")[
                "checkpoint"
            ]
            for m in ("H", "B1")
        }
        for method in ("H", "B1"):
            if stopped():
                break
            ok, reason = resources_ok(batch)
            if not ok:
                atomic_json(
                    folder / "resource_stop.json",
                    {"reason": reason, "epoch": time.time()},
                )
                return
            for rel, digest in hashes.items():
                if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
                    raise RuntimeError("Registered CARLA code changed: " + rel)
            training_run = batch / "carla_formal_training_v5" / f"{method}_seed{seed}"
            source = checkpoints[(seed, method)]
            if stage.startswith("finetuned"):
                state = read(training_run / "latest.json")
                if state["physics_steps"] < budget:
                    raise RuntimeError("Paired CARLA fine-tuning stage incomplete")
                source = checkpoint_for_fine_budget(training_run, budget)
            if stage == "matched_finetune":
                run = training_run
                marker = run / "latest.json"
                done = marker.exists() and read(marker)["physics_steps"] >= budget
                job = {"kind": "train", "budget": budget}
            else:
                run = batch / "carla_formal_evaluation_v5" / f"{stage}_{method}_seed{seed}"
                if stage.startswith("finetuned"):
                    run = run.with_name(run.name + f"_budget{budget}")
                marker = run / "evaluation_summary.json"
                done = marker.exists() and read(marker).get("complete", False)
                split = "development" if stage.endswith("development") else "test"
                job = {
                    "kind": "evaluate",
                    "split": split,
                    "cases": str(
                        root
                        / "configs"
                        / (
                            "carla_development_cases.json"
                            if split == "development"
                            else "carla_final_cases.json"
                        )
                    ),
                    "label": (
                        f"carla_finetuned_{budget}"
                        if stage.startswith("finetuned")
                        else "low_level_recalibrated_frozen_policy_1M"
                    ),
                }
            job.update(
                {
                    "seed": seed,
                    "method": method,
                    "checkpoint": source,
                    "run": str(run),
                    "stage": stage,
                    "fine_tuning_milestone": budget,
                }
            )
            if done:
                completed.append({**job, "status": "already_complete"})
                continue
            file = folder / f"{stage}_{method}_seed{seed}_budget{budget}.json"
            atomic_json(file, job)
            stamp = str(time.time_ns())
            log_path = folder / f"{file.stem}_{stamp}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--batch",
                        str(batch),
                        "--job",
                        str(file),
                    ],
                    cwd=root,
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                last_progress = time.time()
                while process.poll() is None:
                    stamp = max(
                        log_path.stat().st_mtime,
                        (
                            (run / "heartbeat.json").stat().st_mtime
                            if (run / "heartbeat.json").exists()
                            else 0
                        ),
                    )
                    last_progress = max(last_progress, stamp)
                    atomic_json(
                        folder / "status.json",
                        {
                            "pid": os.getpid(),
                            "child_pid": process.pid,
                            "epoch": time.time(),
                            "status": stage,
                            "job": job,
                            "completed": completed,
                        },
                    )
                    if time.time() > deadline or time.time() - last_progress > 300:
                        append_json(
                            folder / "failures.jsonl",
                            {
                                "job": job,
                                "reason": "deadline_or_no_progress_300s",
                                "epoch": time.time(),
                                "owned_child_pid": process.pid,
                            },
                        )
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        break
                    time.sleep(5)
            done = marker.exists() and (
                read(marker)["physics_steps"] >= budget
                if job["kind"] == "train"
                else read(marker).get("complete", False)
            )
            result = {
                **job,
                "exit_code": process.poll(),
                "status": "complete" if done else "incomplete",
            }
            (completed if done else failed).append(result)
            append_json(folder / "results.jsonl", result)
            if not done:
                break
        if stopped() or failed:
            break
    atomic_json(
        folder / "summary.json",
        {
            "epoch": time.time(),
            "completed": completed,
            "failed": failed,
            "budget_or_stop": stopped(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--job")
    args = p.parse_args()
    worker(args) if args.job else main(args)
