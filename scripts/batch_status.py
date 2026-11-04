"""Compact checkpoint/PID/heartbeat evidence; queued work is not called executed work."""

import argparse, ctypes, json, os, time
from pathlib import Path


def read(path):
    if not Path(path).exists():
        return {}
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 2:
                return {"read_status": "temporarily_unavailable"}
            time.sleep(0.05)


def alive(pid, observed_epoch=None):
    if not pid or os.name != "nt":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        running = bool(
            kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259
        )
        if running and observed_epoch is not None:
            from ctypes.wintypes import FILETIME
            times = [FILETIME() for _ in range(4)]
            kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(FILETIME)] * 4
            if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                created = ((times[0].dwHighDateTime << 32) + times[0].dwLowDateTime) / 10_000_000 - 11644473600
                if created > observed_epoch + 1:
                    return False  # Windows has reused this historical PID for another process.
        return running
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    batch = Path(args.batch)
    now = time.time()
    spec = read(batch / "batch.json")
    rows = []
    groups = [
        (
            "formal_v5",
            [f"{m}_seed{s}" for s in (11, 22, 33, 44, 55) for m in ("H", "B1")],
        ),
        (
            "supplementary_v5",
            [
                f"{m}_seed{s}"
                for s in (11, 22, 33)
                for m in ("B2", "B1_hold10", "H_shared", "B1_shared")
            ],
        ),
        (
            "carla_formal_training_v5",
            [f"{m}_seed{s}" for s in (11, 22, 33, 44, 55) for m in ("H", "B1")],
        ),
        ("carla_v5_100k_source_finetune", ["H_seed11", "H_seed11_retry1", "B1_seed11"]),
        ("carla_v5_100k_source_finetune_warmup1", ["H_seed11", "B1_seed11"]),
    ]
    for version, names in groups:
        for name in names:
            run = batch / version / name
            latest = read(run / "latest.json")
            heartbeat = read(run / "heartbeat.json")
            failure = {}
            if (run / "failures.jsonl").exists():
                failure = json.loads((run / "failures.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            if not failure and (run / "infrastructure_failure_accounting.json").exists():
                failure = read(run / "infrastructure_failure_accounting.json")
            rows.append(
                {
                    "version": version,
                    "name": name,
                    "backend": "CARLA_0.9.15" if version.startswith("carla_") else "lightweight",
                    "saved_physics_steps": latest.get("physics_steps", 0),
                    "latest_observed_physics_steps": max(latest.get("physics_steps",0),heartbeat.get("physics_steps",0),failure.get("counts",{}).get("physics_steps",0)),
                    "infrastructure_failure_recorded": bool(failure),
                    "failure_counting_note": failure.get("counting_note", read(run / "infrastructure_failure_accounting.json").get("unknown_cost")),
                    "optimization_phases": latest.get("optimization_phases", 0),
                    "optimizer_steps": latest.get("optimizer_steps"),
                    "pid": heartbeat.get("pid"),
                    "process_alive": alive(heartbeat.get("pid"), heartbeat.get("epoch")),
                    "heartbeat_age_seconds": (
                        round(now - heartbeat["epoch"], 1)
                        if "epoch" in heartbeat
                        else None
                    ),
                    "completed_validation_budgets": [
                        b
                        for b in (100000, 300000, 1000000)
                        if (run / f"validation_{b}.json").exists()
                    ],
                    "final_primary_complete": read(run / "final_primary.json").get(
                        "complete", False
                    ) if not version.startswith("carla_") else None,
                    "last_save_reason": latest.get("reason", "infrastructure_failure_before_checkpoint" if failure else "not_started"),
                }
            )
    result = {
        "observed_epoch": now,
        "remaining_wall_seconds": max(0, spec["deadline_epoch"] - now),
        "stop": (
            (batch / "STOP").read_text(encoding="utf-8")
            if (batch / "STOP").exists()
            else None
        ),
        "runs": rows,
        "carla_pipeline": read(batch / "carla_formal_pipeline_v5/status.json"),
        "carla_development": read(batch / "carla_v5_100k_source_finetune_warmup1/status.json") or read(batch / "carla_v5_100k_source_finetune/status.json"),
        "final_evaluation": read(batch / "final_evaluation_queue_v5/status.json"),
        "supplementary": read(batch / "supplementary_status.json"),
    }
    result["carla_evaluations"] = []
    for folder in sorted((batch / "carla_formal_evaluation_v5").glob("*")):
        if not folder.is_dir():
            continue
        summary = read(folder / "evaluation_summary.json")
        records = set()
        episodes = folder / "episodes.jsonl"
        if episodes.exists():
            for line in episodes.read_text(encoding="utf-8").splitlines():
                try:
                    records.add(json.loads(line)["case_id"])
                except json.JSONDecodeError:
                    # An active writer may not yet have finished its last line.
                    pass
        result["carla_evaluations"].append({
            "name": folder.name, "recorded_conditions": len(records),
            "scheduled_conditions": summary.get("cases"),
            "complete": summary.get("complete", False) and summary.get("infrastructure_unresolved", 0) == 0,
            "schedule_closed": summary.get("complete", False),
            "infrastructure_unresolved": summary.get("infrastructure_unresolved"),
            "censored": summary.get("censored"),
        })
    result["carla_pipeline_process_alive"] = alive(result["carla_pipeline"].get("pid"), result["carla_pipeline"].get("epoch"))
    result["carla_development_process_alive"] = alive(result["carla_development"].get("pid"), result["carla_development"].get("epoch"))
    (batch / "analysis/live_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Remaining wall budget: {result['remaining_wall_seconds']/60:.1f} minutes; STOP={result['stop']}"
        )
        print(
            "|Version / run|Saved physics|Observed physics|Phases|PID alive|Validation budgets|Lightweight primary complete|"
        )
        print("|---|---:|---:|---:|---|---|---|")
        for r in rows:
            print(
                f"|{r['version']}/{r['name']}|{r['saved_physics_steps']}|{r['latest_observed_physics_steps']}|{r['optimization_phases']}|{r['pid']} / {r['process_alive']}|{r['completed_validation_budgets']}|{r['final_primary_complete']}|"
            )
        print(
            "CARLA formal:",
            result["carla_pipeline"].get("status"),
            "alive=",result["carla_pipeline_process_alive"],
            "CARLA development:",result["carla_development"].get("status"),
            result["carla_development"].get("method"),
            "alive=",result["carla_development_process_alive"],
            "Final evaluation:",
            result["final_evaluation"].get("status"),
            "Supplementary:",
            result["supplementary"].get("status"),
        )
        print("\nCARLA评估单列；上表None表示不适用轻量主测试，不能读成CARLA测试失败。")
        print("\n|CARLA evaluation|Recorded conditions|Scheduled conditions|Complete|Unresolved infrastructure|Censored|")
        print("|---|---:|---:|---|---:|---|")
        for row in result["carla_evaluations"]:
            print(f"|{row['name']}|{row['recorded_conditions']}|{row['scheduled_conditions']}|{row['complete']}|{row['infrastructure_unresolved']}|{row['censored']}|")
