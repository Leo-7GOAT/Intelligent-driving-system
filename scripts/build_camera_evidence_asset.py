"""Package actual camera examples with their learned model and complete traces."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_metadata(value):
    if isinstance(value, dict):
        return {k: public_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_metadata(v) for v in value]
    if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
        return "origin_metadata/" + Path(value).name
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output).resolve()
    protocol = read(batch / "batch.json")
    runs = list((batch / "carla").glob("*/evaluation_summary.json"))
    if len(runs) != 1:
        raise ValueError("Expected exactly one evaluation run")
    run = runs[0].parent
    summary = read(runs[0])
    if summary["completed"] != summary["cases"] or summary["infrastructure_unresolved"]:
        raise ValueError("Camera evaluation has unresolved infrastructure")
    checkpoint = Path(protocol["checkpoint"])
    if hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest() != protocol["model_sha256"]:
        raise ValueError("Source model changed")
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(batch / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(batch / "videos", output / "videos", ignore=shutil.ignore_patterns("*.log"))
    (output / "model").mkdir()
    for name in ("model.zip", "normalization.pkl"):
        shutil.copy2(checkpoint / name, output / "model" / name)
    for name in ("config.json", "training_summary.json"):
        (output / "model" / name).write_text(json.dumps(public_metadata(read(checkpoint / name)), indent=2), encoding="utf-8")
    (output / "evidence").mkdir()
    for name in ("episodes.jsonl", "scenes.jsonl", "transitions.jsonl.gz"):
        shutil.copy2(run / name, output / "evidence" / name)
    for name in ("config.json", "evaluation_summary.json"):
        (output / "evidence" / name).write_text(json.dumps(public_metadata(read(run / name)), indent=2), encoding="utf-8")
    for episode in (run / "camera").iterdir():
        if not episode.is_dir():
            continue
        destination = output / "evidence" / "camera" / episode.name
        destination.mkdir(parents=True)
        for name in ("frames.jsonl", "world_ticks.jsonl", "duplicates.jsonl"):
            if (episode / name).exists():
                shutil.copy2(episode / name, destination / name)
    shutil.copytree(batch / "calibration", output / "calibration")
    shutil.copy2(batch / "cases.json", output / "cases.json")
    for name in ("LICENSE", "requirements.lock.txt", "requirements-video.txt"):
        shutil.copy2(ROOT / name, output / name)
    (output / "README.txt").write_text(
        "DEVELOPMENT CAMERA EXAMPLES, NOT RESEARCH ACCEPTANCE.\n"
        "Actual CARLA0.9.15 RGB sensor videos: orange ego, blue task obstacles. Colors and camera pose do not enter the policy.\n"
        "H seed11 uses the declared25k CARLA-adapted checkpoint after101923 lightweight physical steps.\n"
        "The full six-case development evaluation was5/6 at25k and4/6 at100k; regression is retained.\n"
        "These two inspected conditions illustrate success and failure, not independent confirmation or best-model selection.\n"
        "Includes the policy, frozen normalization/config/source, measured calibration, both full evaluation traces, world/camera frame manifests and encoded videos.\n"
        "Raw PNGs and training transition archives remain local; no CARLA binaries/runtime/map assets are included.\n"
        "Absolute local metadata paths are replaced by explicitly non-resolving origin_metadata identifiers. Weight bytes are unchanged.\n"
        "The camera world is reloaded after synchronous setup; raw progress may differ from earlier non-rendering development runs.\n"
        "Reproduce on an exclusively owned local CARLA0.9.15 server using source/scripts/carla_evaluate.py --record-camera with model/, cases.json and calibration/.\n",
        encoding="utf-8")
    private = re.compile(rb"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]")
    files = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        payloads = [path.read_bytes()]
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                payloads += [archive.read(n) for n in archive.namelist()]
                if "data" in archive.namelist():
                    for value in json.loads(archive.read("data")).values():
                        if isinstance(value, dict) and ":serialized:" in value:
                            payloads.append(base64.b64decode(value[":serialized:"]))
        if any(private.search(payload) for payload in payloads):
            raise ValueError("Private path found in " + str(path.relative_to(output)))
        files[path.relative_to(output).as_posix()] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    manifest = {"scope": "single-seed development camera examples", "origin_batch": batch.name,
                "model_sha256": protocol["model_sha256"], "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(json.dumps({"archive": archive, "sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(), "files": len(files)}))


if __name__ == "__main__":
    main()
