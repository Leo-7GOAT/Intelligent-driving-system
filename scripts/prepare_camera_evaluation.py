"""Prepare an immutable real-camera evaluation with explicit model provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-batch", required=True)
    parser.add_argument("--cases", default="configs/carla_recording_cases.json")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    calibration = Path(args.calibration_batch).resolve() / "calibration"
    if not (calibration / "calibration.json").is_file():
        raise ValueError("Measured calibration is required")
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Recording resource guard: " + str(reason))
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_carla_camera_evidence")
    batch.mkdir()
    (batch / "infrastructure").mkdir()
    shutil.copytree(calibration, batch / "calibration")
    hashes = {}
    for folder in ("navstack", "scripts", "configs"):
        for source in (ROOT / folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix in (".py", ".ps1", ".json"):
                rel = source.relative_to(ROOT)
                target = batch / "source" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes[rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(batch / "cases.json", cases)
    write_json(batch / "batch.json", {"created_epoch": time.time(), "deadline_epoch": time.time()+900,
        "checkpoint": str(checkpoint), "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest(),
        "seed": config["seed"], "source_hashes": hashes, "scope": "Illustrative development success/failure camera replay; not model selection or new confirmation",
        "capture": "Actual CARLA RGB sensor,640x360 at20Hz, fixed overhead camera aligned with landscape frame; orange ego/blue task obstacles are rendering-only labels; no policy image input",
        "resource_limits": {"wall_seconds": 900, "disk_min_GiB": 12, "ram_min_GiB": 4, "carla_owners": 1},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT / "runs/carla_recording_development.json", {"batch": str(batch)})
    print(batch)


if __name__ == "__main__":
    main()
