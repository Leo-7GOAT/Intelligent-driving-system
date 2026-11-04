"""Read-only checks of frozen sources, manifests and checkpoint accounting."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(batch):
    batch = Path(batch).resolve()
    root = Path(__file__).resolve().parents[1]
    failures = []
    hashes = []
    source = batch / "source_formal_v5"
    groups = [
        (
            "frozen_training_source",
            source,
            read(source / "frozen_manifest.json")["hashes"],
        ),
        (
            "fixed_final_manifests",
            root,
            read(batch / "development_freeze.json")["manifest_hashes"],
        ),
        (
            "registered_carla_runtime",
            root,
            read(batch / "carla_formal_pipeline_v5/protocol.json")["code_sha256"],
        ),
    ]
    for label, base, expected in groups:
        for relative, sha in expected.items():
            path = base / relative
            actual = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            )
            ok = actual == sha
            hashes.append({"group": label, "file": str(path), "matches": ok})
            if not ok:
                failures.append(
                    {
                        "kind": "hash_mismatch",
                        "path": str(path),
                        "expected": sha,
                        "actual": actual,
                    }
                )
    runs = []
    for version in (
        "formal_v5",
        "supplementary_v5",
        "carla_v5_100k_source_finetune_warmup1",
        "carla_formal_training_v5",
    ):
        for marker in sorted((batch / version).glob("*/latest.json")):
            latest = read(marker)
            checkpoint = Path(latest["checkpoint"])
            saved = read(checkpoint / "training_summary.json")
            keys = (
                "physics_steps",
                "actions",
                "optimization_phases",
                "optimizer_steps",
            )
            ok = all(latest[k] == saved[k] for k in keys)
            if latest.get("backend") == "CARLA_0.9.15":
                ok = (
                    ok
                    and latest["physics_steps"]
                    == latest["controlled_physics_steps"]
                    + latest["reset_physics_steps"]
                )
            if not ok:
                failures.append(
                    {
                        "kind": "checkpoint_accounting_mismatch",
                        "run": str(marker.parent),
                    }
                )
            config = marker.parent / "resolved_config.json"
            if config.exists() and read(config)["env"]["deadline"] != 45:
                failures.append(
                    {"kind": "unequal_task_deadline", "run": str(marker.parent)}
                )
            runs.append(
                {
                    "run": str(marker.parent),
                    "checkpoint": str(checkpoint),
                    "accounting_consistent": ok,
                    **{k: saved[k] for k in keys},
                    "reload_exact": saved["reload_exact"],
                }
            )
    result = {
        "observed_epoch": time.time(),
        "checks_passed": not failures,
        "failures": failures,
        "hash_checks": hashes,
        "checkpoint_checks": runs,
        "scope": "Integrity and accounting only; does not assert completed experiments, physical realism beyond recorded evidence, or statistical superiority.",
    }
    path = batch / "analysis/batch_integrity.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    result = verify(parser.parse_args().batch)
    print(
        json.dumps(
            {
                "passed": result["checks_passed"],
                "hashes": len(result["hash_checks"]),
                "checkpoints": len(result["checkpoint_checks"]),
                "failures": result["failures"],
            },
            ensure_ascii=True,
        )
    )
    raise SystemExit(0 if result["checks_passed"] else 1)
