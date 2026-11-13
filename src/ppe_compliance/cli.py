"""Command-line image inference with explicit, non-overwriting output files."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from ppe_compliance import __version__


def probability(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("confidence must be greater than 0 and at most 1")
    return number


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("image size must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="PPE project: object detection baseline.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    detect = commands.add_parser("detect", help="Detect objects in one local image")
    detect.add_argument("--source", type=Path, required=True)
    detect.add_argument("--weights", default="yolo11n.pt", help="Model path or downloadable YOLO weight name")
    detect.add_argument("--confidence", type=probability, default=0.25)
    detect.add_argument("--image-size", type=positive_integer, default=640)
    detect.add_argument("--device", default="cpu", help="cpu, 0 for a CUDA GPU, or mps")
    detect.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if not args.source.is_file():
        parser.error(f"source image does not exist: {args.source}")
    try:
        from ppe_compliance.detector import detect_image

        result, report = detect_image(
            args.source, args.weights, args.confidence, args.image_size, args.device
        )
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        image_path = run_dir / "annotated.jpg"
        result.save(filename=str(image_path))
        (run_dir / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Detection failed: {error}\n")
    print(report["note"])
    print(f"Detected {len(report['detections'])} objects")
    for item in report["detections"]:
        print(f"  {item['class_name']}: {item['confidence']:.3f}, box={item['bbox_xyxy']}")
    print(f"Annotated image: {image_path}")
    print(f"Detection JSON: {run_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
