"""Verify a camera evidence export and load its actual PPO on CPU.

This checks transport/reload integrity, not CARLA outcome reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asset = Path(args.asset).resolve()
    manifest = json.loads((asset / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (asset / name).resolve()
        if not path.is_relative_to(asset):
            raise ValueError("Manifest path escapes asset")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Hash mismatch: " + name)
    model_hash = hashlib.sha256((asset / "model/model.zip").read_bytes()).hexdigest()
    if model_hash != manifest["model_sha256"]:
        raise ValueError("Model provenance mismatch")
    videos = json.loads((asset / "videos/manifest.json").read_text(encoding="utf-8"))["videos"]
    episodes = [json.loads(line) for line in (asset / "evidence/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    for video in videos:
        row = next(e for e in episodes if e["case_id"] == video["case_id"])
        if video["model_sha256"] != model_hash or video["case"] != row["case"] or video["success"] != row["success"]:
            raise ValueError("Video/episode/model provenance mismatch")
        recording = row["recording"]
        frame_manifest = asset / "evidence" / recording["manifest"]
        if not frame_manifest.resolve().is_relative_to(asset):
            raise ValueError("Recording manifest escapes asset")
        if hashlib.sha256(frame_manifest.read_bytes()).hexdigest() != video["frame_manifest_sha256"]:
            raise ValueError("Camera manifest provenance mismatch")
    # Verification must not add machine-specific bytecode paths to the export.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(asset / "source"))
    import numpy as np
    from navstack.experiments import load_run
    model, normalizer = load_run(asset / "model")
    mean, variance = normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy()
    predictions = []
    try:
        for seed in (11, 7301, 7302):
            normalizer.seed(seed)
            obs = normalizer.reset()
            action, _ = model.predict(obs, deterministic=True)
            again, _ = model.predict(obs, deterministic=True)
            np.testing.assert_array_equal(action, again)
            if not model.action_space.contains(action[0]):
                raise ValueError("Invalid reloaded policy action")
            predictions.append({"seed": seed, "action": int(action[0])})
        np.testing.assert_array_equal(mean, normalizer.obs_rms.mean)
        np.testing.assert_array_equal(variance, normalizer.obs_rms.var)
    finally:
        normalizer.close()
    result = {"scope": "artifact integrity and CPU policy reload; not CARLA replay or independent confirmation",
              "files_verified": len(manifest["files"]), "model_sha256": model_hash,
              "normalization_frozen": True, "deterministic_predictions": predictions,
              "linked_videos": len(videos)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
