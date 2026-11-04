"""Optional CARLA RGB evidence capture; compatible with the isolated Python3.7.

Only sensor callbacks supply images. No vehicle state is reconstructed, no
missing frame is synthesized, and capture is never a policy observation.
"""
import hashlib
import json
from pathlib import Path
import threading
import time
import struct
import zlib


def write_bgra_png(image, target):
    """Encode sensor BGRA losslessly using Python's Unicode-safe file writer.

    CARLA's native save_to_disk failed to create files in this Windows Unicode
    workspace. PNG has one unfiltered RGBA scanline per sensor row; no resizing,
    interpolation or reconstruction is performed.
    """
    width, height = int(image.width), int(image.height)
    data = bytes(image.raw_data)
    if width <= 0 or height <= 0 or len(data) != width * height * 4:
        raise ValueError("Invalid BGRA sensor dimensions")
    rgba = bytearray(len(data))
    rgba[0::4], rgba[1::4], rgba[2::4], rgba[3::4] = data[2::4], data[1::4], data[0::4], data[3::4]
    scanlines = b"".join(b"\0" + rgba[start:start+width*4] for start in range(0, len(rgba), width*4))
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(scanlines, 3)) + chunk(b"IEND", b""))
    Path(target).write_bytes(png)


class FrameRecorder:
    def __init__(self, root, episode_id):
        self.directory = Path(root) / episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = self.directory / "frames.jsonl"
        self.lock = threading.Lock()
        self.last_frame = -1
        self.error = None
        self.seen = {}
        self.confirmed = set()

    def save(self, image):
        try:
            with self.lock:
                frame = int(image.frame)
                target = self.directory / ("frame_%010d.png" % frame)
                raw_digest = hashlib.sha256(bytes(image.raw_data)).hexdigest()
                identity = (float(image.timestamp), raw_digest)
                if frame in self.seen:
                    if self.seen[frame] != identity:
                        raise RuntimeError("Conflicting duplicate camera frame")
                    with (self.directory / "duplicates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"frame": frame, "timestamp": identity[0], "raw_sha256": raw_digest,
                                                 "status": "identical_callback_retained_as_audit_only"}) + "\n")
                    return
                write_bgra_png(image, target)
                row = {"frame": frame, "timestamp": float(image.timestamp),
                       "width": int(image.width), "height": int(image.height),
                       "file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                with self.manifest.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                self.last_frame = max(self.last_frame, frame)
                self.seen[frame] = identity
        except Exception as exc:
            self.error = repr(exc)

    def finish(self, terminal_frame, timeout=3., timestamp=None):
        deadline = time.monotonic() + timeout
        while terminal_frame not in self.seen and self.error is None and time.monotonic() < deadline:
            time.sleep(.01)
        if self.error:
            raise RuntimeError("Camera recording failed: " + self.error)
        if terminal_frame not in self.seen:
            raise RuntimeError("Terminal camera frame missing; recording incomplete")
        if timestamp is not None:
            if abs(self.seen[terminal_frame][0] - timestamp) > .001:
                raise RuntimeError("Camera timestamp does not match the actual world snapshot")
            if terminal_frame not in self.confirmed:
                with (self.directory / "world_ticks.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"frame": terminal_frame, "world_timestamp": timestamp}) + "\n")
                self.confirmed.add(terminal_frame)
