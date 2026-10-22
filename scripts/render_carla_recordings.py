"""Encode verified real CARLA RGB frames, retaining success/failure provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verified_frames(run, recording):
    manifest = run / recording["manifest"]
    if not manifest.resolve().is_relative_to(run.resolve()):
        raise ValueError("Recording manifest escapes evaluation run")
    frames = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    selected = sorted((r for r in frames if recording["begin_frame"] <= r["frame"] <= recording["terminal_frame"]), key=lambda r: r["frame"])
    expected = list(range(recording["begin_frame"], recording["terminal_frame"]+1))
    if [r["frame"] for r in selected] != expected:
        raise ValueError("Missing or duplicate actual camera frames; refuse synthetic interpolation")
    ticks = {r["frame"]: r["world_timestamp"] for line in (manifest.parent / "world_ticks.jsonl").read_text().splitlines()
             for r in [json.loads(line)]}
    for index, row in enumerate(selected):
        path = manifest.parent / row["file"]
        if not path.resolve().is_relative_to(manifest.parent.resolve()):
            raise ValueError("Frame path escapes recording")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Camera frame hash mismatch")
        if row["frame"] not in ticks or abs(row["timestamp"] - ticks[row["frame"]]) > .001:
            raise ValueError("Camera frame lacks a matching actual world snapshot")
        if index and abs(row["timestamp"] - selected[index-1]["timestamp"] - recording["sensor_tick"]) > .001:
            raise ValueError("Camera timestamps do not match fixed physical time")
    return manifest, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if not config.get("record_camera"):
        raise ValueError("This evaluation did not record a real camera")
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    results = []
    for row in rows:
        if row["status"] != "completed" or "recording" not in row:
            results.append({"case_id": row["case_id"], "status": "no_completed_recording"})
            continue
        recording = row["recording"]
        manifest, frames = verified_frames(run, recording)
        filename = "case_%02d_%s_%s.mp4" % (row["case_id"], row["case"]["scenario"], row["termination_reason"])
        fps = 1. / recording["sensor_tick"]
        command = [encoder, "-hide_banner", "-loglevel", "error", "-n", "-f", "image2pipe", "-vcodec", "png",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", filename]
        with (output / (filename + ".encoder.log")).open("wb") as log:
            process = subprocess.Popen(command, cwd=output, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                for frame in frames:
                    process.stdin.write((manifest.parent / frame["file"]).read_bytes())
                process.stdin.close()
                if process.wait(timeout=60):
                    raise RuntimeError("Video encoder failed; inspect encoder log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        results.append({"file": filename, "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                        "run_id": run.name, "seed": config["seed"], "model_sha256": config["model_sha256"],
                        "case_id": row["case_id"], "case": row["case"], "success": row["success"],
                        "termination_reason": row["termination_reason"], "frame_count": len(frames), "fps": fps,
                        "first_world_frame": frames[0]["frame"], "last_world_frame": frames[-1]["frame"],
                        "sensor_elapsed_seconds": frames[-1]["timestamp"]-frames[0]["timestamp"],
                        "video_duration_seconds": len(frames)/fps,
                        "frame_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "render_source": "actual CARLA RGB sensor; no trajectory reconstruction or invented frames"})
    (output / "manifest.json").write_text(json.dumps({"scope": "real evaluation recordings, not independent confirmation",
                    "videos": results}, indent=2), encoding="utf-8")
    print(json.dumps({"recordings": len(results), "output": str(output)}))


if __name__ == "__main__":
    main()
