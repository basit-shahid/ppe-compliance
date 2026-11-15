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


def finite_float(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
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
            command.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto",
                                 help="Windows capture backend; auto tries DirectShow then Media Foundation")
            command.add_argument("--camera-exposure", type=finite_float, help="Optional driver-specific exposure; original value restored on exit")
            command.add_argument("--camera-brightness", type=finite_float, help="Optional driver-specific brightness; original value restored on exit")
            command.add_argument("--low-latency", action="store_true", help="Newest-frame capture and separate live/analyzed preview; recording off by default")
            command.add_argument("--record", action="store_true", help="With --low-latency, save analyzed frames and JSONL")
            command.add_argument("--stale-after", type=positive_float, default=.5, help="With --low-latency, hide results older than this many seconds")
            command.add_argument("--duration", type=positive_float, help="With --low-latency, stop after this wall-clock duration; shutdown may wait for inference")
            command.add_argument("--view", choices=("dashboard","split"), default="dashboard", help="Low-latency preview layout")
            command.add_argument("--focus", choices=("live","analyzed"), default="live", help="Dashboard's large pane; Tab switches focus")
            command.add_argument("--preview-width", type=positive_integer, default=1280, help="Dashboard width, 960 to 1920")
            command.add_argument("--capture-width", type=positive_integer, help="Low-latency: requested native camera width; provide height too")
            command.add_argument("--capture-height", type=positive_integer, help="Low-latency: requested native camera height")
            command.add_argument("--capture-fps", type=positive_float, help="Low-latency: request camera FPS; driver support varies")
            command.add_argument("--preview-fps", type=positive_float, default=60., help="Low-latency: maximum preview refresh rate")
        else:
            command.add_argument("--source", type=Path, required=True)
        if name == 'video':
            command.add_argument('--view', choices=('dashboard', 'frame'), default='dashboard',
                                 help='Recorded presentation: dashboard with worker scores or frames only')
            command.add_argument('--export-profile', choices=('standard', 'efficient'), default='standard',
                                 help='efficient: single H264 export, 720p dashboard or up to 1080p frames; source detection unchanged')
            command.add_argument('--processing-height', type=positive_integer,
                                 help='Recorded analysis cap, e.g. 1080 or 720 (16:9 bounds); default original resolution')
            command.add_argument('--detail', action='store_true', help='Show equipment boxes on analyzed frames')
            command.add_argument('--hide-scores', action='store_true', help='Hide model confidence labels')
            command.add_argument('--analysis-fps', type=positive_float, default=2.5,
                                 help='With --live-preview, maximum AI updates per second; default 2.5 leaves CPU capacity for playback')
        command.add_argument("--weights", default=default_weights, help="Model path or downloadable YOLO weight name")
        command.add_argument("--confidence", type=probability, default=0.25)
        command.add_argument("--image-size", type=positive_integer, default=416 if name == 'webcam' else 640)
        command.add_argument("--device", default="cpu", help="cpu, 0 for a CUDA GPU, or mps")
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
        if name in {"assess", "video", "webcam"}:
            command.add_argument("--rules", type=Path, help="Optional YAML association-rule overrides")
        if name == "assess":
            command.add_argument("--visibility", type=Path, help="Optional human-reviewed visibility-template JSON")
        if name in {"video", "webcam"}:
            command.add_argument("--track", action="store_true", help="Session-local worker IDs, confirmed status changes and events.jsonl")
            command.add_argument("--max-frames", type=positive_integer,
                                 help="Stop after this many source frames" if name=='video' else "Stop after this many processed frames")
            command.add_argument("--output-fps", type=positive_float, help="Playback FPS (default: source FPS for files; 10 for camera)")
            if name == "webcam":
                command.add_argument("--no-preview", dest="preview", action="store_false", default=True,
                                     help="Record without a window; stop with Ctrl+C or --max-frames")
            else:
                playback = command.add_mutually_exclusive_group()
                playback.add_argument("--preview", action="store_true",
                                      help="Show frames as inference completes; speed follows processing. Use --play for paced review")
                playback.add_argument("--play", action="store_true",
                                      help="After processing, play the completed scored video at its recorded FPS; Q/Esc closes it")
                playback.add_argument('--live-preview', action='store_true',
                                      help='Play source at its original rate with selected frames analyzed in a separate pane; default preview cap 540p')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command != "webcam" and not args.source.is_file():
        parser.error(f"source {'video' if args.command == 'video' else 'image'} does not exist: {args.source}")
    if args.command in {"video", "webcam"}:
        low_latency = args.command == "webcam" and args.low_latency
        sampled_review = args.command == 'video' and args.live_preview
        if sampled_review and (args.view != 'dashboard' or args.output_fps is not None):
            parser.error('--live-preview uses dashboard view and the original source frame rate')
        if args.command == 'webcam':
            if (args.capture_width is None)!=(args.capture_height is None):
                parser.error('Provide both --capture-width and --capture-height')
            if not low_latency and (args.record or args.duration is not None or args.stale_after != .5 or
                                    args.capture_width is not None or args.focus!='live' or
                                    args.view!='dashboard' or args.preview_width!=1280 or
                                    args.capture_fps is not None or args.preview_fps!=60.):
                parser.error('Live preview/recording/resolution options require --low-latency')
        try:
            from ppe_compliance.video_pipeline import process_video

            source = args.camera if args.command == "webcam" else args.source
            print(f"Processing {source}; Ctrl+C stops cleanly.", flush=True)
            if args.command == 'video' and args.preview:
                print('Processing preview follows inference speed. For source-paced finished playback, use --play instead.', flush=True)

            def progress(count, elapsed):
                print(f"Processed {count} frames ({count/elapsed:.2f} frames/s so far)", flush=True)

            if sampled_review:
                from ppe_compliance.recorded_preview import review_recorded_video
                print('Original-rate playback + selected-frame AI. Model warm-up and any large-video preparation run together.',flush=True)
                summary = review_recorded_video(source,args.weights,args.confidence,args.image_size,args.device,
                    args.output_dir,processing_height=args.processing_height or 540,max_frames=args.max_frames,
                    rules=load_rules(args.rules),tracking=args.track,detail=args.detail,show_scores=not args.hide_scores,
                    analysis_fps=args.analysis_fps,
                    on_progress=lambda shown,analyzed: print(f'Source frames played: {shown}; AI samples: {analyzed}',flush=True))
            elif low_latency:
                from ppe_compliance.live_pipeline import process_live
                from ppe_compliance.video_pipeline import local_camera_context

                summary = process_live(camera=source,weights=args.weights,confidence=args.confidence,
                    image_size=args.image_size,device=args.device,output_dir=args.output_dir,
                    rules=load_rules(args.rules),preview=args.preview,max_frames=args.max_frames,
                    duration=args.duration,record=args.record,output_fps=args.output_fps or 10.,
                    stale_after=args.stale_after,on_progress=progress,
                    tracking=args.track,view=args.view,focus=args.focus,preview_width=args.preview_width,
                    preview_fps=args.preview_fps,
                    capture_context_factory=lambda: local_camera_context(source,args.camera_backend,
                        args.camera_exposure,args.camera_brightness,args.capture_width,args.capture_height,args.capture_fps))
            else:
                summary = process_video(source, args.weights, args.confidence, args.image_size,
                                        args.device, args.output_dir, load_rules(args.rules), args.preview,
                                        args.max_frames, args.output_fps, on_progress=progress,
                                        camera_exposure=getattr(args, "camera_exposure", None),
                                        camera_brightness=getattr(args, "camera_brightness", None),
                                        camera_backend=getattr(args, "camera_backend", "auto"),tracking=args.track,
                                        view=args.view if args.command == 'video' else 'frame',
                                        export_profile=args.export_profile if args.command == 'video' else 'standard',
                                        processing_height=args.processing_height if args.command == 'video' else None,
                                        detail=args.detail if args.command=='video' else False,
                                        show_scores=not args.hide_scores if args.command=='video' else True)
            if args.command == 'video' and args.play:
                from ppe_compliance.playback import play_recorded_output
                print(f"Saved {summary['run_directory']}; opening finished playback at {summary['output_fps']:.2f} FPS.", flush=True)
                playback = play_recorded_output(summary)
                print(f"Playback {playback['status']}: {playback['displayed_frames']} frames shown.", flush=True)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"{args.command.capitalize()} failed: {error}\n")
        print(f"{summary['status']}: {summary['stop_reason']}; {summary['processed_frames']} assessed frames")
        if sampled_review:
            print(f"Source: {summary['source_frames_played']} frames; AI analyzed {summary['processed_frames']} selected frames.")
            print(f"Sampled evidence and summary: {summary['run_directory']}")
        elif low_latency:
            print(f"Capture reads: {summary['captured_frames']}; assessment rate: {summary['assessment_throughput_fps']:.2f}/s")
            print(f"Recording: {summary['recording_enabled']}; summary: {summary['run_directory']}")
        else:
            print(f"End-to-end processing speed: {summary['end_to_end_processing_fps']:.2f} frames/s")
            print(f"Video, JSONL and summary: {summary['run_directory']}")
        print(summary["note"])
        warnings = summary.get("camera", {}).get("restore_warnings", []) if low_latency else summary.get("camera_control_restore_warnings", [])
        for warning in warnings:
            print(f"Warning: {warning}")
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
