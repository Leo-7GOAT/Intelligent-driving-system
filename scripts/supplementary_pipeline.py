"""Registered secondary controls start only after the full primary training matrix."""

import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.research import atomic_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--queue", default="configs/supplementary_v5_queue.json")
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    root = Path(__file__).resolve().parents[1]
    deadline = json.loads((batch / "batch.json").read_text(encoding="utf-8"))[
        "deadline_epoch"
    ]
    required = [
        batch / "formal_v5" / f"{m}_seed{s}" / "validation_1000000.json"
        for s in (11, 22, 33, 44, 55)
        for m in ("H", "B1")
    ]
    while not all(p.exists() for p in required):
        atomic_json(
            batch / "supplementary_status.json",
            {"status": "waiting_for_primary_1M", "epoch": time.time()},
        )
        if time.time() > deadline - 180 or (batch / "STOP").exists():
            sys.exit(0)
        time.sleep(5)
    atomic_json(
        batch / "supplementary_status.json",
        {"status": "training", "epoch": time.time()},
    )
    evaluator = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/final_evaluation_queue.py"),
            "--batch",
            str(batch),
            "--queue",
            args.queue,
            "--source-root",
            args.source_root,
            "--workers",
            "1",
        ],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/parallel_research.py"),
            "--batch",
            str(batch),
            "--queue",
            args.queue,
            "--source-root",
            args.source_root,
            "--workers",
            "3",
        ],
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    atomic_json(
        batch / "supplementary_status.json",
        {"status": "evaluating_completed_1M_controls", "epoch": time.time()},
    )
    evaluator.wait()
    atomic_json(
        batch / "supplementary_status.json",
        {"status": "stopped_or_complete", "epoch": time.time()},
    )
