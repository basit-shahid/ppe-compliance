"""Make a labelled integration sequence from audited validation stills, not real motion."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ppe_compliance.detector import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/samples/phase5-validation-sequence.mp4"))
    args = parser.parse_args()
    metadata = args.output.with_suffix(".json")
    if args.output.exists() or metadata.exists():
        parser.error("Output or metadata already exists; choose another --output path")
    base = Path("data/processed/ppe-three-class/images/val")
    sources = [base/name for name in ("ultralytics-ppe-nov2026-705.jpg",
               "source-00f1fbfd9c8193a3c5a071512cf8c0a7.jpg", "source-150feca69395fa769adccdc8e157e935.jpg")]
    if any(not path.is_file() for path in sources):
        parser.error("Prepare the audited dataset first; required validation images are missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 6, (960,540))
    try:
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        for path in sources:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {path}")
            height, width = image.shape[:2]
            scale = min(960/width, 540/height)
            thumb = cv2.resize(image, (round(width*scale), round(height*scale)))
            frame = np.zeros((540,960,3), dtype=np.uint8)
            x, y = (960-thumb.shape[1])//2, (540-thumb.shape[0])//2
            frame[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
            for _ in range(10):
                writer.write(frame)
    finally:
        writer.release()
    metadata.write_text(json.dumps({"kind":"Synthetic integration sequence of validation stills; not natural motion or independent accuracy evidence",
                        "sources":[{"path":str(p),"sha256":file_sha256(p)} for p in sources],
                        "resolution":[960,540], "fps":6, "frames":30, "frames_per_scene":10,
                        "output_sha256":file_sha256(args.output)}, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
