import json
import pytest
from navstack.camera_recording import FrameRecorder


def test_frame_evidence_and_terminal_timeout(tmp_path):
    # Isolated callback test only; not evidence of a real CARLA camera run.
    class Image:
        frame, timestamp, width, height = 123, 6.15, 2, 1
        raw_data = bytes([3, 2, 1, 255, 10, 20, 30, 255])
    recorder = FrameRecorder(tmp_path, "test_episode")
    recorder.save(Image())
    recorder.finish(123, timeout=0)
    recorder.finish(123, timeout=0, timestamp=6.15)
    with pytest.raises(RuntimeError, match="timestamp"):
        recorder.finish(123, timeout=0, timestamp=6.2)
    row = json.loads(recorder.manifest.read_text())
    assert row["frame"] == 123 and row["timestamp"] == 6.15 and len(row["sha256"]) == 64
    from PIL import Image as PillowImage
    with PillowImage.open(recorder.directory / row["file"]) as png:
        assert png.size == (2, 1)
        assert png.getpixel((0, 0)) == (1, 2, 3, 255)
        assert png.getpixel((1, 0)) == (30, 20, 10, 255)
    with pytest.raises(RuntimeError, match="missing"):
        recorder.finish(124, timeout=0)
    recorder.save(Image())
    recorder.finish(123, timeout=0)
    assert len((recorder.directory / "duplicates.jsonl").read_text().splitlines()) == 1
    changed = Image()
    changed.raw_data = bytes([0] * 8)
    recorder.save(changed)
    with pytest.raises(RuntimeError, match="Conflicting duplicate"):
        recorder.finish(123, timeout=0)


def test_video_validation_rejects_missing_sensor_frames(tmp_path):
    from scripts.render_carla_recordings import verified_frames
    import hashlib
    folder = tmp_path / "camera"
    folder.mkdir()
    rows = []
    for frame in (10, 12):
        path = folder / (str(frame) + ".png")
        path.write_bytes(b"unit-test-frame")
        rows.append({"frame": frame, "timestamp": frame*.05, "file": path.name,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Missing"):
        verified_frames(tmp_path, {"manifest": "camera/frames.jsonl", "begin_frame": 10, "terminal_frame": 12, "sensor_tick": .05})


def test_camera_png_handles_unicode_directory(tmp_path):
    from navstack.camera_recording import write_bgra_png
    from PIL import Image
    folder = tmp_path / "中文录像"
    folder.mkdir()
    class Sensor:
        width, height = 1, 2
        raw_data = bytes([0, 0, 255, 255, 0, 255, 0, 255])
    target = folder / "真实帧.png"
    write_bgra_png(Sensor(), target)
    with Image.open(target) as png:
        assert png.getpixel((0, 0)) == (255, 0, 0, 255)
        assert png.getpixel((0, 1)) == (0, 255, 0, 255)
