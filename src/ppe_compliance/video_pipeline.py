"""Sequential recorded-video/webcam assessment with bounded-memory JSONL output."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

from ppe_compliance.association import AssociationRules
from ppe_compliance.compliance import assess_detections
from ppe_compliance.detector import FrameDetector, file_sha256
from ppe_compliance.rendering import draw_assessment


def process_video(source, weights="models/best.pt", confidence=.25, image_size=640,
                  device="cpu", output_dir=Path("outputs"), rules=None, preview=False,
                  max_frames=None, output_fps=None, on_progress=None):
    """Process a local video Path or nonnegative camera index; return run summary.

    No frame skipping/tracking/manual visibility propagation. A constant-rate
    webcam recording is not guaranteed to have the same duration as wall time.
    Failures finalize a partial summary and release resources before raising.
    """
    import cv2

    if isinstance(source, bool):
        raise ValueError("Source must be a video path or nonnegative camera index")
    camera = type(source) is int
    if camera:
        if source < 0:
            raise ValueError("Camera index must be nonnegative")
        capture_source, source_name = source, f"camera:{source}"
    else:
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"Source video does not exist: {path}")
        capture_source, source_name = str(path), str(path)
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("Image size must be a positive integer")
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Confidence must be in (0, 1]")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    if output_fps is not None and (isinstance(output_fps, bool) or not math.isfinite(output_fps) or output_fps <= 0):
        raise ValueError("Output FPS must be positive and finite")
    rules = rules or AssociationRules()
    started = perf_counter()
    capture = writer = frame_log = None
    run_dir = None
    detector = None
    processed = 0
    totals = Counter()
    first_frame_seconds = None
    latency_sum, latency_min, latency_max = 0., math.inf, 0.
    first_output_seconds = model_load_seconds = None
    loop_start = None
    source_fps = advertised_frames = selected_fps = None
    source_hash = None
    resolution = encoded_resolution = None
    status, stop_reason, error_text, failure = "completed", None, None, None
    window = "PPE assessment - Q or Esc to stop"
    preview_open = False
    try:
        if camera and platform.system() == "Windows":
            capture = cv2.VideoCapture(capture_source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {source_name}; check the path, device index, permissions and camera availability")
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        fps_value = capture.get(cv2.CAP_PROP_FPS)
        if math.isfinite(fps_value) and fps_value > 0:
            source_fps = float(fps_value)
        count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not camera and math.isfinite(count_value) and count_value > 0:
            advertised_frames = int(round(count_value))
        # File playback preserves source timing by default. Camera output has an
        # explicit constant rate; capture/processing wall time is recorded separately.
        selected_fps = output_fps if output_fps is not None else (10.0 if camera else source_fps)
        if selected_fps is None:
            raise ValueError("Video FPS metadata unavailable; pass --output-fps explicitly")
        source_hash = None if camera else file_sha256(Path(capture_source))
        model_start = perf_counter()
        detector = FrameDetector(weights, confidence, image_size, device)
        model_load_seconds = perf_counter()-model_start
        if not {"person", "helmet", "vest"} <= {n.casefold() for n in detector.model_classes.values()}:
            raise ValueError("Video assessment requires PPE-trained person/helmet/vest weights")
        frame_log = (run_dir / "frames.jsonl").open("w", encoding="utf-8")
        if preview:
            try:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            except cv2.error as error:
                raise RuntimeError("Preview unavailable; omit --preview or use webcam --no-preview") from error
            preview_open = True
        loop_start = perf_counter()
        while True:
            frame_start = perf_counter()
            ok, frame = capture.read()
            decoded = perf_counter()
            if not ok:
                if processed == 0:
                    raise RuntimeError("Source opened but no readable video frames were returned")
                if camera:
                    raise RuntimeError("Camera stopped returning frames")
                if advertised_frames is not None and processed < advertised_frames:
                    raise RuntimeError(f"Video ended early or decoding failed ({processed}/{advertised_frames} advertised frames)")
                stop_reason = "end_of_stream" if advertised_frames is not None else "end_of_stream_or_decode_failure"
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Unsupported decoded frame format; expected three-channel BGR")
            height, width = frame.shape[:2]
            if resolution is None:
                resolution = (width, height)
                encoded_resolution = (width+width % 2, height+height % 2)
                # mp4v is widely available in OpenCV; verify writer availability.
                writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), selected_fps, encoded_resolution)
                if not writer.isOpened():
                    raise RuntimeError("Cannot open MP4 video writer; check codec and output-directory support")
            elif resolution != (width, height):
                raise ValueError("Frame dimensions changed; refusing a malformed output video")
            predict_start = perf_counter()
            _, raw = detector.detect(frame, f"{source_name}#frame={processed}")
            inferred = perf_counter()
            frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            assessment = assess_detections(raw, frame_hash, rules)
            assessed = perf_counter()
            annotated = draw_assessment(frame, assessment, raw["detections"])
            if encoded_resolution != resolution:
                annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2,
                                               cv2.BORDER_CONSTANT, value=(0,0,0))
            rendered = perf_counter()
            writer.write(annotated)
            encoded = perf_counter()
            stop_requested = False
            if preview:
                cv2.imshow(window, annotated)
                stop_requested = cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27)
                try:
                    visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 0  # Backend may throw after the user closes the window.
                if visible < 1:
                    stop_requested = True
            record = {"frame_index": processed,
                      "source_time_seconds": processed/source_fps if source_fps is not None and not camera else None,
                      "stream_elapsed_seconds": decoded-loop_start,
                      "frame_hash_format": "SHA-256 of decoded BGR bytes; dimensions in assessment",
                      "detections": raw["detections"], "assessment": assessment,
                      "timing_seconds": {"decode": decoded-frame_start, "frame_setup": predict_start-decoded,
                                         "inference": inferred-predict_start,
                                         "assessment_and_hash": assessed-inferred, "render": rendered-assessed,
                                         "encode_submission": encoded-rendered,
                                         "pipeline_before_report_write": perf_counter()-frame_start}}
            # Detection IDs in this streaming format refer to this row's detections.
            assessment["detection_id_scope"] = "zero-based index into this frames.jsonl row's detections"
            frame_log.write(json.dumps(record, separators=(",", ":"))+"\n")
            frame_log.flush()
            elapsed = perf_counter()-frame_start
            latency_sum += elapsed
            latency_min, latency_max = min(latency_min, elapsed), max(latency_max, elapsed)
            if first_frame_seconds is None:
                first_frame_seconds = elapsed
            totals.update({key: value for key, value in assessment["summary"].items() if key != "workers"})
            processed += 1
            if first_output_seconds is None:
                first_output_seconds = perf_counter()-started
            if on_progress is not None and (processed == 1 or processed % 30 == 0):
                on_progress(processed, perf_counter()-loop_start)
            if stop_requested:
                status, stop_reason = "stopped", "preview_stop"
                break
            if max_frames is not None and processed >= max_frames:
                status, stop_reason = "stopped", "max_frames"
                break
    except KeyboardInterrupt:
        status, stop_reason = "stopped", "keyboard_interrupt"
    except Exception as error:
        status, stop_reason, error_text, failure = "failed", "error", str(error), error
    finally:
        # Close resources independently even if a backend raises during cleanup.
        for resource in (writer, capture):
            if resource is not None:
                try:
                    resource.release()
                except Exception as error:
                    if failure is None:
                        failure, error_text, status, stop_reason = error, str(error), "failed", "cleanup_error"
        if frame_log is not None:
            try:
                frame_log.close()
            except Exception as error:
                if failure is None:
                    failure, error_text, status, stop_reason = error, str(error), "failed", "report_close_error"
        if preview_open:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    finished = perf_counter()
    # Final encoder flushing is included in throughput, not just write() calls.
    processing_seconds = finished-loop_start if loop_start is not None else 0.
    packages = {}
    for package in ("ultralytics", "torch", "opencv-python", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    summary = {"schema_version": 1, "status": status, "stop_reason": stop_reason, "error": error_text,
               "run_directory": str(run_dir) if run_dir else None,
               "source": source_name, "source_kind": "webcam" if camera else "recorded_video",
               "source_sha256": source_hash, "source_fps_metadata": source_fps,
               "source_frame_count_metadata": advertised_frames, "output_fps": selected_fps,
               "output_duration_seconds": processed/selected_fps if selected_fps else 0,
               "source_resolution": list(resolution) if resolution else None,
               "output_resolution": list(encoded_resolution) if encoded_resolution else None,
               "odd_dimension_policy": "pad right/bottom by at most one pixel for MP4; report coordinates stay in source space",
               "output_codec": "mp4v",
               "audio_preserved": False,
               "processed_frames": processed, "intentional_frame_skipping": False,
               "image_size": image_size, "confidence_threshold": confidence,
               "weights": str(weights), "weights_sha256": detector.weights_sha256 if detector else None,
               "model_classes": detector.model_classes if detector else None,
               "device": device, "rules": asdict(rules), "preview_enabled": preview,
               "platform": platform.platform(), "python": platform.python_version(), "packages": packages,
               "processing_wall_seconds": processing_seconds,
               "end_to_end_processing_fps": processed/processing_seconds if processing_seconds else 0.,
               "total_wall_seconds": finished-started, "model_load_seconds": model_load_seconds,
               "first_output_seconds_since_start": first_output_seconds,
               "frame_latency_seconds": {"mean": latency_sum/processed if processed else None,
                                         "min": latency_min if processed else None,
                                         "max": latency_max if processed else None,
                                         "first": first_frame_seconds},
               "frame_latency_scope": "read through JSONL flush; progress/stop handling and final resource flushing are included in wall-clock throughput instead",
               "worker_observations": dict(totals),
               "note": "Worker observations are per frame, not unique people. No tracking or automatic missing-PPE violations. Model-input size and source resolution affect speed and recall. Webcam driver buffering/drops are not measured; stream elapsed time is not a device capture timestamp. Constant-FPS webcam output may play faster/slower than wall time. File timestamps use nominal FPS, not variable-rate presentation timestamps. EOF and decode failure cannot always be distinguished from backend metadata."}
    if run_dir is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failure is not None:
        location = f"; partial results: {run_dir}" if run_dir else ""
        raise RuntimeError(f"{error_text}{location}") from failure
    return summary
