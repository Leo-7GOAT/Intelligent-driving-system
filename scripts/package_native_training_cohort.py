"""Export closed TRAIN-only native jobs with unchanged raw/model bytes.

The release includes every supplied job and its full registered new training
segment. Ancestor raw traces are not duplicated; inherited costs and recovery
uncertainties stay in the original protocol/proof files.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_native_continuation import (
    SUFFIXES, operational_guard_files, portable_metadata, read, scan_payload, sha,
)


def package(batches, output):
    output = Path(output)
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError("Preserve existing exports")
    files, transformations, jobs = {}, {}, []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
        for batch in map(lambda p: Path(p).resolve(), batches):
            p = read(batch / "protocol.json")
            method, = p["methods"]
            summary = read(batch / method / "summary.json")
            guard = read(batch / "guard_final.json")
            raw = read(batch / "training_only_raw_audit.json")
            engine = read(batch / "strict_delivery.json")
            proof = read(batch / method / "saved_learning_proof.json")
            if (read(batch / "summary.json")["status"] != "complete"
                    or guard["exit_code"] != 0 or guard.get("reason") is not None
                    or summary["stopped_reason"] is not None or summary["optimizer_steps"] <= 0
                    or read(batch / "infrastructure/server_stop.json")["status"] != "owned_empty_server_stopped"):
                raise ValueError("Require a normally closed trained segment: " + batch.name)
            if ((batch / method / "evaluation.json").exists()
                    or not engine["strict_delivery_gate_passed"]
                    or engine["counts"]["controls"] != summary["physics_steps"]
                    or raw["training_control_frames_verified"] != summary["physics_steps"]
                    or raw["protocol_sha256"] != sha((batch / "protocol.json").read_bytes())
                    or proof["summary"] != summary or not proof["RMS_exact"]):
                raise ValueError("TRAIN-only proof or frame accounting differs")
            for relative, expected in p["source_hashes"].items():
                if sha((batch / "source" / relative).read_bytes()) != expected:
                    raise ValueError("Frozen source changed: " + relative)
            selected = set(batch.glob("*.json"))
            protected = {batch / "protocol.json", batch / "summary.json"}
            for folder in ("source", "calibration", "resume", method):
                paths = {f for f in (batch / folder).rglob("*") if f.is_file()
                         and "__pycache__" not in f.parts and f.suffix in SUFFIXES}
                selected.update(paths)
                protected.update(f for f in paths if folder in ("source", "calibration", "resume")
                                 or f.name in ("config.json", "summary.json", "metadata.json"))
            selected.update(f for f in (batch / "infrastructure").iterdir() if f.is_file() and f.suffix in (".json", ".gz"))
            operational = operational_guard_files(batch)
            selected.update(operational); protected.update(operational)
            for path in sorted(selected):
                payload = original = path.read_bytes()
                changes = []
                if path.suffix == ".json":
                    value = portable_metadata(read(path), changes)
                    if changes:
                        if path in protected:
                            raise ValueError("Cannot redact a hash-bound input: " + path.name)
                        payload = json.dumps(value, indent=2).encode()
                scan_payload(path.name, payload)
                name = batch.name + "/" + path.relative_to(batch).as_posix()
                if name in files:
                    raise ValueError("Duplicate training job")
                archive.writestr(name, payload)
                files[name] = dict(bytes=len(payload), sha256=sha(payload))
                if changes:
                    transformations[name] = dict(original_sha256=sha(original), export_sha256=sha(payload), changes=changes)
            jobs.append(dict(batch=batch.name, method=method, controls=summary["physics_steps"],
                optimizer_steps=summary["optimizer_steps"], charged_controls_lower_bound=summary["charged_control_steps_lower_bound"],
                model_sha256=sha((batch / method / "model.zip").read_bytes()),
                protocol_sha256=sha((batch / "protocol.json").read_bytes())))
        for name, payload in {
            "LICENSE": (ROOT / "LICENSE").read_bytes(),
            "requirements.lock.txt": (ROOT / "requirements.lock.txt").read_bytes(),
            "README.txt": b"Closed native CARLA development TRAIN segments. No final acceptance claim.\nAll new controls, reset/engine streams, checkpoints, optimizer/RMS proofs, source and inherited metadata are retained. Ancestor raw traces are not duplicated.\nAfter extraction: python scripts/audit_native_training_only.py --batch EXTRACTED_BATCH --output NEW_RAW_AUDIT.json\nThen: python scripts/audit_native_control_delivery.py --batch EXTRACTED_BATCH --output NEW_ENGINE_AUDIT.json\nThese are offline evidence audits, zero new driving. No CARLA installation or credentials included.\n",
        }.items():
            scan_payload(name, payload); archive.writestr(name, payload)
            files[name] = dict(bytes=len(payload), sha256=sha(payload))
        manifest = dict(schema_version=1, scope="Complete specified native TRAIN segments; development only",
            repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            jobs=jobs, files=files, metadata_transformations=transformations,
            packager_sha256=sha(Path(__file__).read_bytes()))
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files) | {"manifest.json"}:
            raise ValueError("Archive inventory mismatch")
        for name, row in files.items():
            if sha(archive.read(name)) != row["sha256"]:
                raise ValueError("Archive hash mismatch")
    temporary.rename(output)
    result = dict(jobs=jobs, files=len(files), bytes=output.stat().st_size, sha256=sha(output.read_bytes()))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", nargs="+", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); package(a.batches, a.output)
