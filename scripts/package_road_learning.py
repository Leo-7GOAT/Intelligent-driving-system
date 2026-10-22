"""Immutable development asset: all native-road rollouts, models and failures."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


def package(batches, audit, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve previous assets")
    content = {}
    for batch in map(Path, batches):
        summary = json.loads((batch/"summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete" or {r["method"] for r in summary["methods"]} != {"H", "B1"}:
            raise ValueError("Require both methods, including every failed/unfinished episode")
        protocol = json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
        for name, digest in protocol["source_hashes"].items():
            if hashlib.sha256((batch/"source"/name).read_bytes()).hexdigest() != digest:
                raise ValueError("Source changed")
        paths = [batch/"protocol.json", batch/"summary.json"]
        for folder in ("source", "calibration", "analysis", "H", "B1"):
            paths.extend(p for p in (batch/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                         and p.name not in ("heartbeat.json", "worker.lock"))
        paths.extend((batch/"infrastructure").glob("*_requests.jsonl.gz"))
        paths.extend((batch/"infrastructure").glob("*_responses.jsonl.gz"))
        for path in paths:
            if path.suffix not in (".py", ".json", ".gz", ".jsonl", ".csv", ".png", ".zip", ".pkl"):
                raise ValueError("Unexpected evidence type: "+path.name)
            content[batch.name+"/"+path.relative_to(batch).as_posix()] = path.read_bytes()
    content["map_query_audit.json"] = Path(audit).read_bytes()
    root = Path(__file__).resolve().parents[1]
    content["LICENSE"] = (root/"LICENSE").read_bytes()
    content["README.txt"] = (
        "Native-road shared-raster PPO interface experiments. NOT convergence, full road truth, or hierarchy superiority.\n"
        "Both batches and all failed/unfinished episodes are retained; same seed, not independent policies.\n"
        "H/B1 models are trained from scratch with4447inputs and legacy actuator; old605models are incompatible.\n"
        "Road geometry is sampled Driving-lane cells, including narrow gaps; not physical asphalt segmentation.\n"
        "Rebuild with python scripts/analyze_native_road_learning.py --batch EXTRACTED_BATCH.\n"
        "Load using navstack.experiments.load_run(EXTRACTED_BATCH/METHOD); normalization stays frozen.\n"
        "No CARLA binaries, complete maps, environments, credentials or personal paths are included.\n"
    ).encode()
    patterns = [r"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]", r"/(?:home|Users)/[A-Za-z0-9_.-]+/",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}", r"\bgithub_pat_[A-Za-z0-9_]{40,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    def scan(name, payload):
        if name.endswith(".png"):
            return
        if name.endswith(".gz"):
            payload = gzip.decompress(payload)
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.namelist():
                    scan(member, archive.read(member))
            return
        text = payload.decode("utf-8", errors="replace")
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError("Private data candidate: "+name)
    for name, payload in content.items():
        scan(name, payload)
    manifest = {"scope":"Development only, no discarded failures", "files":{name:hashlib.sha256(data).hexdigest() for name,data in sorted(content.items())}}
    content["manifest.json"] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(json.dumps({"files":len(manifest["files"]), "bytes":output.stat().st_size,
        "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--batches", nargs="+", required=True)
    p.add_argument("--audit", required=True); p.add_argument("--output", required=True)
    a=p.parse_args(); package(a.batches, a.audit, a.output)
