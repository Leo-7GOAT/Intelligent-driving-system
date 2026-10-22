"""Isolated live RGB metadata probe; never counted as PPO evaluation."""
import argparse
import json
from pathlib import Path
import queue
import time
import carla


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20)
    world = client.get_world()
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("World is not empty")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    camera = None
    rows = []
    try:
        # Official deterministic setup: synchronous settings precede reload.
        world = client.reload_world(False)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", "640"), ("image_size_y", "360"), ("sensor_tick", "0")):
            bp.set_attribute(k, v)
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(z=25), carla.Rotation(pitch=-90)))
        frames = queue.Queue()
        camera.listen(frames.put)
        for tick in range(30):
            frame = world.tick(15)
            snapshot = world.get_snapshot()
            received = []
            until = time.monotonic() + 3
            matched = False
            while time.monotonic() < until:
                try:
                    image = frames.get(timeout=max(.001, until-time.monotonic()))
                except queue.Empty:
                    break
                received.append({"frame": image.frame, "timestamp": image.timestamp})
                if image.frame == frame and abs(image.timestamp-snapshot.timestamp.elapsed_seconds) < .001:
                    matched = True
                    break
            rows.append({"tick": tick, "world_frame": frame, "world_timestamp": snapshot.timestamp.elapsed_seconds,
                         "received": received, "matched": matched})
            output.write_text(json.dumps({"scope": "live camera metadata diagnostic", "rows": rows}, indent=2), encoding="utf-8")
            print(json.dumps(rows[-1]), flush=True)
            if not matched:
                break
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        world.tick(15)
        world.apply_settings(original)


if __name__ == "__main__":
    main()
