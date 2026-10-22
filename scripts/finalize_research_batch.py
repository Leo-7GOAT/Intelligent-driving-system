"""Write an evidence-based batch handoff after its fixed wall deadline; never extend training."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    batch = Path(args.batch).resolve()
    root = Path(__file__).resolve().parents[1]
    deadline = json.loads((batch / "batch.json").read_text(encoding="utf-8"))["deadline_epoch"]
    record = {"pid": os.getpid(), "deadline_epoch": deadline, "status": "waiting_for_budget_boundary"}
    path = batch / "finalizer_status.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    while time.time() < deadline + 20:
        time.sleep(min(5, max(.1, deadline + 20 - time.time())))
    results = []
    for name in ("publication_report.py", "failure_audit.py", "closed_loop_evidence.py"):
        with (batch / ("finalize_" + name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(root / "scripts" / name), "--batch", str(batch)],
                cwd=root, stdout=log, stderr=log,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=180,
            )
        results.append({"script": name, "exit_code": result.returncode})
    record.update(status="reports_written" if all(r["exit_code"] == 0 for r in results)
                  else "report_error_inspect_logs", results=results, finished_epoch=time.time())
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
