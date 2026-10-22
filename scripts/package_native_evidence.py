"""Package all native-control cases/probes, without runtime paths or CARLA assets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile


def package(batches, output, extras=()):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing evidence archive")
    content = {}
    episode_count = 0
    for batch in map(Path, batches):
        summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or len(summary["episodes"]) != 4:
            raise ValueError("Require all four cases, not successful-case selection")
        episode_count += len(summary["episodes"])
        protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
        for name, digest in protocol["source_hashes"].items():
            if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen source mismatch")
        files = [batch / name for name in ("protocol.json", "summary.json")]
        if (batch/"probe_gap_audit.json").exists():
            files.append(batch/"probe_gap_audit.json")
        for folder in ("calibration", "source", "analysis", "planning_audit", "brake_probe", "throttle_probe", "throttle_probe_manual1"):
            if (batch / folder).exists():
                files.extend(p for p in (batch / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for episode in summary["episodes"]:
            case = batch / ("case_%02d" % episode["case_id"])
            files.extend(p for p in case.iterdir() if p.is_file())
        for path in files:
            if path.suffix not in (".py", ".json", ".jsonl", ".gz", ".png"):
                raise ValueError("Unexpected file type")
            content[batch.name + "/" + path.relative_to(batch).as_posix()] = path.read_bytes()
    for folder in map(Path, extras):
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in (".json", ".jsonl", ".gz", ".png"):
                raise ValueError("Unexpected extra evidence type")
            content[folder.name+"/"+path.relative_to(folder).as_posix()] = path.read_bytes()
    patterns = [r"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]",
                r"/(?:home|Users)/[A-Za-z0-9_.-]+/", r"\bgh[pousr]_[A-Za-z0-9]{30,}",
                r"\bgithub_pat_[A-Za-z0-9_]{40,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for name, payload in content.items():
        if name.endswith(".png"):
            continue
        decoded = gzip.decompress(payload) if name.endswith(".gz") else payload
        text = decoded.decode("utf-8")
        # JSON may encode path separators as escaped backslashes.
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError("Potential private path/credential in " + name)
    root = Path(__file__).resolve().parents[1]
    content["LICENSE"] = (root / "LICENSE").read_bytes()
    content["README.txt"] = (
        "Native-road development diagnostics, not PPO or research acceptance.\n"
        "Includes all " + str(episode_count) + " control episodes from the specified batches, severe brake spikes, "
        "and every raw brake/throttle probe present in those batches. No CARLA binaries/maps or credentials.\n"
        "From the repository checkout, run scripts/analyze_native_control.py --batch EXTRACTED_BATCH "
        "for each batch, then scripts/compare_native_control.py --baseline BASELINE --candidate CANDIDATE.\n"
        "All alternative actuator variants are experimental and remain disabled by default. "
        "Optional extra directories retain offline feasibility and frozen-policy counterfactual diagnoses; these are not formal evaluations. "
        "Each batch has original frozen Python source and calibration, not an installed environment.\n"
    ).encode()
    manifest = {"scope": "All development control cases and probes; no formal comparison or independent-seed claim",
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(content.items())}}
    content["manifest.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != set(content):
            raise ValueError("Archive inventory mismatch")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Archive content mismatch")
    print(json.dumps({"files": len(manifest["files"]), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extras", nargs="*", default=[])
    args = parser.parse_args()
    package(args.batches, args.output, args.extras)
