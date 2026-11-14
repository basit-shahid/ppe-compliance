"""Image, recorded-video and webcam commands with non-overwriting outputs."""

import argparse
from datetime import datetime, timezone
import hashlib
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
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("camera index must be nonnegative")
    return number


def positive_float(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return number


def load_rules(path):
    from ppe_compliance.association import AssociationRules

    overrides = {}
    if path:
        import yaml

        try:
            overrides = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid rules YAML: {error}") from error
        if not isinstance(overrides, dict):
            raise ValueError("Rules YAML must contain a mapping")
    try:
        return AssociationRules(**overrides)
    except TypeError as error:
        raise ValueError(f"Invalid association rule: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="PPE image/video/webcam detection and worker-level evidence assessment.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    for name, description, default_weights in (
        ("detect", "Detect objects in one local image", "yolo11n.pt"),
        ("assess", "Match PPE to workers in one image with uncertainty-aware statuses", "models/best.pt"),
        ("video", "Assess a local recorded video and measure end-to-end processing speed", "models/best.pt"),
        ("webcam", "Assess a local camera; Q/Esc or Ctrl+C to stop", "models/best.pt"),
    ):
        command = commands.add_parser(name, help=description)
        if name == "webcam":
            command.add_argument("--camera", type=nonnegative_integer, default=0)
        else:
            command.add_argument("--source", type=Path, required=True)
        command.add_argument("--weights", default=default_weights, help="Model path or downloadable YOLO weight name")
        command.add_argument("--confidence", type=probability, default=0.25)
        command.add_argument("--image-size", type=positive_integer, default=640)
        command.add_argument("--device", default="cpu", help="cpu, 0 for a CUDA GPU, or mps")
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
        if name in {"assess", "video", "webcam"}:
            command.add_argument("--rules", type=Path, help="Optional YAML association-rule overrides")
        if name == "assess":
            command.add_argument("--visibility", type=Path, help="Optional human-reviewed visibility-template JSON")
        if name in {"video", "webcam"}:
            command.add_argument("--max-frames", type=positive_integer, help="Stop after this many processed frames")
            command.add_argument("--output-fps", type=positive_float, help="Playback FPS (default: source FPS for files; 10 for camera)")
            if name == "webcam":
                command.add_argument("--no-preview", dest="preview", action="store_false", default=True,
                                     help="Record without a window; stop with Ctrl+C or --max-frames")
            else:
                command.add_argument("--preview", action="store_true", help="Show a window; Q/Esc closes it")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command != "webcam" and not args.source.is_file():
        parser.error(f"source {'video' if args.command == 'video' else 'image'} does not exist: {args.source}")
    if args.command in {"video", "webcam"}:
        try:
            from ppe_compliance.video_pipeline import process_video

            source = args.camera if args.command == "webcam" else args.source
            print(f"Processing {source}; Ctrl+C stops cleanly. Preview: Q/Esc or close window.", flush=True)

            def progress(count, elapsed):
                print(f"Processed {count} frames ({count/elapsed:.2f} frames/s so far)", flush=True)

            summary = process_video(source, args.weights, args.confidence, args.image_size,
                                    args.device, args.output_dir, load_rules(args.rules), args.preview,
                                    args.max_frames, args.output_fps, on_progress=progress)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"{args.command.capitalize()} failed: {error}\n")
        print(f"{summary['status']}: {summary['stop_reason']}; {summary['processed_frames']} frames")
        print(f"End-to-end processing speed: {summary['end_to_end_processing_fps']:.2f} frames/s")
        print(f"Video, JSONL and summary: {summary['run_directory']}")
        print(summary["note"])
        return
    try:
        from ppe_compliance.detector import detect_image

        assessment = None
        if args.command == "assess":
            from ppe_compliance.compliance import assess_detections, visibility_template
            from ppe_compliance.rendering import draw_assessment

            rules = load_rules(args.rules)
            visibility = json.loads(args.visibility.read_text(encoding="utf-8")) if args.visibility else None
        result, report = detect_image(
            args.source, args.weights, args.confidence, args.image_size, args.device
        )
        if args.command == "assess":
            source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
            assessment = assess_detections(report, source_sha256, rules, visibility)
            annotated = draw_assessment(result.orig_img, assessment, report["detections"])
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        image_path = run_dir / "annotated.jpg"
        if assessment is None:
            result.save(filename=str(image_path))
        else:
            import cv2

            if not cv2.imwrite(str(image_path), annotated):
                raise OSError(f"Cannot write annotated image: {image_path}")
            (run_dir / "compliance.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
            template = visibility_template(assessment["workers"], source_sha256)
            (run_dir / "visibility-template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        (run_dir / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{args.command.capitalize()} failed: {error}\n")
    print(assessment["note"] if assessment is not None else report["note"])
    print(f"Detected {len(report['detections'])} objects")
    for item in report["detections"]:
        print(f"  {item['class_name']}: {item['confidence']:.3f}, box={item['bbox_xyxy']}")
    if assessment is not None:
        for worker in assessment["workers"]:
            print(f"  {worker['worker_id']}: {worker['status']} (helmet={worker['helmet']['state']}, vest={worker['vest']['state']})")
            print("    " + "; ".join(worker["reasons"]))
        print(f"Worker assessment: {run_dir / 'compliance.json'}")
        print(f"Optional human visibility template: {run_dir / 'visibility-template.json'}")
    print(f"Annotated image: {image_path}")
    print(f"Detection JSON: {run_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
