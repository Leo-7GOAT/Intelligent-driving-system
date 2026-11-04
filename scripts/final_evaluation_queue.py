"""Evaluate each ready 1M checkpoint once; the primary decision still requires all five pairs."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json, resources_ok


def main(args):
    batch = Path(args.batch).resolve()
    root = Path(__file__).resolve().parents[1]
    deadline = json.loads((batch / "batch.json").read_text(encoding="utf-8"))[
        "deadline_epoch"
    ]
    folder = batch / (
        "final_evaluation_queue_" + Path(args.queue).stem
        if args.queue
        else "final_evaluation_queue_v5"
    )
    folder.mkdir(exist_ok=True)
    runs = [
        batch / args.version / f"{m}_seed{s}"
        for s in (11, 22, 33, 44, 55)
        for m in ("H", "B1")
    ]
    if args.queue:
        jobs = json.loads(Path(args.queue).read_text(encoding="utf-8"))
        runs = [
            batch / j["version"] / j["name"] for j in jobs if j["budget"] == 1000000
        ]
    pending = [
        (r, label, cases)
        for label, cases in (
            ("primary", "lightweight_final_cases.json"),
            ("generalization", "generalization_cases.json"),
        )
        for r in runs
    ]
    active = []
    completed = []
    failed = []
    while pending or active:
        ok, reason = resources_ok(batch)
        if not ok or time.time() > deadline - 90 or (batch / "STOP").exists():
            pending = []
        while pending and len(active) < args.workers:
            ready_index = next(
                (
                    i
                    for i, (run, _, _) in enumerate(pending)
                    if (run / "validation_1000000.json").exists()
                ),
                None,
            )
            if ready_index is None:
                break
            run, label, cases = pending.pop(ready_index)
            marker = run / ("final_" + label + ".json")
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get(
                "complete"
            ):
                completed.append(
                    {"run": str(run), "label": label, "status": "already_complete"}
                )
                continue
            checkpoint = json.loads((run / "latest.json").read_text(encoding="utf-8"))[
                "checkpoint"
            ]
            log = (
                folder / (run.name + "_" + label + "_" + str(time.time_ns()) + ".log")
            ).open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(root / "scripts/evaluate_manifest.py"),
                "--batch",
                str(batch),
                "--checkpoint",
                checkpoint,
                "--source-root",
                str(Path(args.source_root).resolve()),
                "--cases",
                str(root / "configs" / cases),
                "--label",
                label,
            ]
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            active.append((process, log, run, label))
        remaining = []
        for process, log, run, label in active:
            if process.poll() is None:
                remaining.append((process, log, run, label))
                continue
            log.close()
            marker = run / ("final_" + label + ".json")
            row = {"run": str(run), "label": label, "exit_code": process.returncode}
            (
                completed
                if marker.exists()
                and json.loads(marker.read_text(encoding="utf-8")).get("complete")
                else failed
            ).append(row)
        active = remaining
        atomic_json(
            folder / "status.json",
            {
                "pid": os.getpid(),
                "epoch": time.time(),
                "status": "evaluating" if active else "waiting_for_ready_checkpoints",
                "active": [
                    {"pid": p.pid, "run": str(r), "label": l} for p, _, r, l in active
                ],
                "pending": len(pending),
                "completed": completed,
                "failed": failed,
            },
        )
        if active or pending:
            time.sleep(2)
    atomic_json(
        folder / "summary.json",
        {
            "completed": completed,
            "failed": failed,
            "epoch": time.time(),
            "planned_jobs": len(runs) * 2,
            "complete": len(completed) == len(runs) * 2 and not failed,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--version", default="formal_v5")
    p.add_argument(
        "--queue",
        help="Evaluate registered supplementary 1M checkpoints as each becomes ready",
    )
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=3)
    main(p.parse_args())
