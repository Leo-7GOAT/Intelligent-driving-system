"""Bounded supervisor of its own diagnostic children, with no retry loops."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json, resources_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--adopt-from")
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    if len(protocol["jobs"]) > protocol["resource_limits"]["cpu_workers"]:
        raise RuntimeError("Job count exceeds the registered concurrent worker budget")
    mismatches = [name for name, digest in protocol["source_hashes"].items()
                  if not (batch / "source" / name).is_file()
                  or hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError("Frozen source mismatch: " + repr(mismatches))
    atomic_json(batch / "source_verified.json", {"epoch": time.time(), "files": len(protocol["source_hashes"]), "mismatches": []})
    children = []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    adoption = json.loads(Path(args.adopt_from).read_text(encoding="utf-8-sig")) if args.adopt_from else None
    with ProcessLease(batch / "supervisor.lock"):
        if (batch / "STOP").exists():
            raise RuntimeError("STOP present; inspect before a new window")
        for job in protocol["jobs"]:
            out = (batch / (job["id"] + ".stdout.log")).open("a")
            err = (batch / (job["id"] + ".stderr.log")).open("a")
            command = [sys.executable, str(batch / "source" / job.get("worker_script","scripts/development_cycle.py")), "--batch", str(batch), "--job", job["id"]]
            if adoption:
                from navstack.owned_process import OwnedProcess
                record=next(r for r in adoption if r["job"]==job["id"])
                if str(batch) not in record["command"] or job["id"] not in record["command"]:
                    raise RuntimeError("Adoption command does not match this batch/job")
                child=OwnedProcess(record["pid"],record["creation_epoch"],record["executable"])
            else:
                child = subprocess.Popen(command, stdout=out, stderr=err, creationflags=flags,
                                         env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            children.append((job["id"], child, out, err))
        stop_time = None
        try:
            while any(c.poll() is None for _, c, _, _ in children):
                now = time.time()
                ok, reason = resources_ok(batch)
                if shutil.disk_usage(batch).free < 12 * 1024**3:
                    ok, reason = False, "disk_below_12GiB"
                if now >= protocol["deadline_epoch"] - 120:
                    ok, reason = False, "diagnostic_window_drain"
                # GPU jobs are not launched; temperature remains a shared-machine
                # resource guard and never authorizes killing another project's PID.
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10, creationflags=flags)
                if gpu.returncode == 0 and int(gpu.stdout.split(",")[0]) >= 85:
                    ok, reason = False, "gpu_temperature_85C"
                rows = []
                for name, child, out, err in children:
                    heartbeat = batch / protocol.get("run_group","sanity_v6") / name / "heartbeat.json"
                    recent = max(heartbeat.stat().st_mtime if heartbeat.exists() else 0,
                                 Path(out.name).stat().st_mtime, Path(err.name).stat().st_mtime)
                    scope=batch/protocol.get("run_group","sanity_v6")/name
                    recent=max(recent,max((p.stat().st_mtime for p in scope.glob("validation_*_trace.jsonl.gz")),default=0))
                    evaluation_heartbeat=scope/"evaluation_heartbeat.json"
                    if evaluation_heartbeat.exists():recent=max(recent,evaluation_heartbeat.stat().st_mtime)
                    if child.poll() is None and now - recent > 300:
                        ok, reason = False, "worker_no_progress_300s"
                    rows.append({"job": name, "pid": child.pid, "exit_code": child.poll(), "last_evidence_epoch": recent})
                if not ok and not (batch / "STOP").exists():
                    (batch / "STOP").write_text(reason, encoding="utf-8")
                if (batch / "STOP").exists():
                    stop_time = stop_time or now
                atomic_json(batch / "supervisor.json", {"pid": os.getpid(), "epoch": now, "children": rows,
                            "stop_epoch": stop_time, "gpu": gpu.stdout.strip(), "deadline_epoch": protocol["deadline_epoch"]})
                if stop_time and (now - stop_time > 120 or now >= protocol["deadline_epoch"]):
                    break
                time.sleep(5)
        finally:
            for name, child, out, err in children:
                if child.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, creationflags=flags)
                    else:
                        child.terminate()
                    child.wait(timeout=20)
                out.close(); err.close()
            atomic_json(batch / "supervisor_final.json", {"epoch": time.time(),
                        "children": [{"job": n, "pid": c.pid, "exit_code": c.returncode} for n, c, _, _ in children]})


if __name__ == "__main__":
    main()
