"""Package a bounded, traceable development checkpoint/evaluation release asset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(batch / "protocol.json")
    for name, digest in protocol["source_hashes"].items():
        assert hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() == digest, name
    shutil.copytree(batch / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    shutil.copy2(ROOT / "requirements.lock.txt", output / "requirements.lock.txt")
    models = []
    for label in ("H", "H0"):
        run = batch / protocol["run_group"] / (label + "_seed11")
        evaluation = read(run / "validation_200000.json")
        checkpoint = Path(evaluation["checkpoint"])
        assert evaluation["model_sha256"] == hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest()
        destination = output / "models" / label
        destination.mkdir(parents=True)
        for name in ("model.zip", "normalization.pkl", "config.json"):
            shutil.copy2(checkpoint / name, destination / name)
        training = read(checkpoint / "training_summary.json")
        training["checkpoint"] = "models/" + label
        (destination / "training_summary.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
        evidence = output / "evidence" / label
        evidence.mkdir(parents=True)
        evaluation["checkpoint"] = "models/" + label
        (evidence / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        for name in ("validation_200000_trace.jsonl.gz", "episodes.jsonl", "optimization.jsonl"):
            shutil.copy2(run / name, evidence / name)
        models.append({"label": label, "training_seed": 11, "physical_steps": training["physics_steps"],
                       "model_sha256": evaluation["model_sha256"], "completion": evaluation["safe_completion"]})
    (output / "cases.json").write_text(json.dumps(protocol["development_cases"], indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "DEVELOPMENT PRE-RELEASE, NOT RESEARCH ACCEPTANCE.\n"
        "H/H0 seed11 at approximately200k physical steps. All10 development cases per model included, including failures.\n"
        "Includes weights, frozen normalization/config/source, optimizer summaries, training episode summaries and complete200k evaluation traces.\n"
        "Does not include every training transition, CARLA binaries, or CARLA videos. No new final-test or five-seed claim.\n"
        "Original checkpoints are preserved locally. Only metadata checkpoint paths are made relative here.\n"
        "Model/normalization files use Python serialization; verify manifest hashes and only load this trusted release.\n",
        encoding="utf-8")
    # Serialized policy metadata and all plain files must not disclose a user's home.
    patterns = (b"C:\\Users\\", b"C:\\\\Users\\\\", b"C:/Users/", b"/home/")
    for path in output.rglob("*"):
        if not path.is_file():continue
        payloads = [path.read_bytes()]
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:payloads += [archive.read(n) for n in archive.namelist()]
        if any(pattern in data for data in payloads for pattern in patterns):
            raise ValueError("Private path found in asset: " + str(path.relative_to(output)))
    files = {str(path.relative_to(output)).replace("\\", "/"): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"scope": "single-seed development", "origin_batch": batch.name,
                "original_source_hashes": protocol["source_hashes"], "models": models, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(json.dumps({"archive": archive, "sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(), "files": len(files)}))


if __name__ == "__main__":
    main()
