"""Decode every encoded camera frame and create an inspection contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    args = parser.parse_args()
    root = Path(args.videos).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    rows, thumbnails = [], []
    for record in manifest["videos"]:
        if "file" not in record:
            raise ValueError("Incomplete recording cannot pass verification")
        path = (root / record["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Video path escapes recording directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Encoded video hash mismatch")
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            if abs(metadata["fps"] - record["fps"]) > .001:
                raise ValueError("Decoded frame rate differs from sensor recording")
            selected = {0, record["frame_count"] // 2, record["frame_count"] - 1}
            pictures = []
            count = 0
            for index, data in enumerate(reader):
                count += 1
                if index in selected:
                    frame = Image.frombytes("RGB", metadata["size"], data)
                    frame.thumbnail((320, 180))
                    pictures.append((index, frame))
        finally:
            reader.close()
        if count != record["frame_count"] or len(pictures) != 3:
            raise ValueError("Decoded frame count differs from real sensor frames")
        thumbnails.append((record["file"], pictures))
        rows.append({"file": record["file"], "decoded_frames": count,
                     "fps": metadata["fps"], "sha256": record["sha256"]})
    sheet = Image.new("RGB", (960, 220 * len(thumbnails)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, pictures) in enumerate(thumbnails):
        draw.text((8, row * 220 + 4), name, fill="black")
        for column, (index, picture) in enumerate(pictures):
            sheet.paste(picture, (column * 320, row * 220 + 24))
            draw.text((column * 320 + 8, row * 220 + 205), "Frame %d" % index, fill="black")
    sheet.save(root / "contact_sheet.png")
    result = {"scope": "complete decoding/hash/fps verification; visual inspection is separate",
              "videos": rows}
    (root / "decode_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
