"""Sequential recorded-video/webcam assessment with bounded-memory JSONL output."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from ppe_compliance.association import AssociationRules
from ppe_compliance.compliance import assess_detections
from ppe_compliance.detector import FrameDetector, file_sha256
from ppe_compliance.rendering import draw_assessment, resize_frame
from ppe_compliance.tracking import WorkerTracker


@contextmanager
def local_camera_context(index, backend="auto", exposure=None, brightness=None, width=None, height=None, fps=None):
    """Local camera compatibility controls for the independent live capture owner.

    Kept with the existing local-only camera experiments; the low-latency engine
    itself accepts a context factory and does not depend on these settings.
    """
    import cv2

    if backend not in {"auto","dshow","msmf"}:
        raise ValueError("Unknown camera backend")
    if (width is None)!=(height is None) or any(type(v) is not int or v<=0 for v in (width,height) if v is not None):
        raise ValueError("Provide both positive capture width and height")
    if fps is not None and (isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0):
        raise ValueError("Capture FPS must be positive and finite")
    capture = None
    controls, restore, warnings = {}, [], []
    info = {"backend":backend,"controls":controls,"restore_warnings":warnings}
    try:
        if platform.system() == "Windows":
            flag = cv2.CAP_MSMF if backend == "msmf" else cv2.CAP_DSHOW
            capture = cv2.VideoCapture(index,flag)
            if backend == "auto" and not capture.isOpened():
                capture.release()
                flag = cv2.CAP_MSMF
                capture = cv2.VideoCapture(index,flag)
            info["backend"] = "msmf" if flag == cv2.CAP_MSMF else "dshow"
        else:
            if backend != "auto":
                raise ValueError("Explicit camera backends require Windows")
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        if width is not None:
            accepted_w=bool(capture.set(cv2.CAP_PROP_FRAME_WIDTH,width))
            accepted_h=bool(capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height))
            info['resolution_request']={'width':width,'height':height,'accepted':accepted_w and accepted_h,
                'driver_width':capture.get(cv2.CAP_PROP_FRAME_WIDTH),'driver_height':capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}
            if not accepted_w or not accepted_h:
                warnings.append('Requested source resolution was not accepted; inspect actual source dimensions')
        if fps is not None:
            accepted = bool(capture.set(cv2.CAP_PROP_FPS, fps))
            actual = capture.get(cv2.CAP_PROP_FPS)
            info['fps_request'] = {'requested': fps, 'accepted': accepted,
                                   'driver_fps': actual if math.isfinite(actual) else None}
            if not accepted or not math.isfinite(actual) or abs(actual-fps) > 1:
                warnings.append('Camera FPS request not confirmed; measured preview rate is reported separately')
        for name,value,prop in (("exposure",exposure,cv2.CAP_PROP_EXPOSURE),
                                ("brightness",brightness,cv2.CAP_PROP_BRIGHTNESS)):
            if value is None:
                continue
            if isinstance(value,bool) or not math.isfinite(value):
                raise ValueError(f"Camera {name} must be finite")
            old = capture.get(prop)
            if not math.isfinite(old):
                raise ValueError(f"Cannot save original camera {name}")
            restore.append((name,prop,old))
            controls[name] = {"original":old,"requested":value}
            if not capture.set(prop,value):
                raise ValueError(f"Camera driver rejected {name}")
            controls[name]["applied_readback"] = capture.get(prop)
        yield capture,info
    finally:
        if capture is not None:
            for name,prop,old in reversed(restore):
                try:
                    accepted = bool(capture.set(prop,old))
                    actual = capture.get(prop)
                    controls[name].update(restore_accepted=accepted,restored_readback=actual)
                    if not accepted or not math.isfinite(actual) or abs(actual-old)>.5:
                        warnings.append(f"Could not confirm {name} restoration")
                except Exception as error:
                    warnings.append(f"Could not restore {name}: {error}")
            capture.release()


def process_video(source, weights="models/best.pt", confidence=.25, image_size=640,
                  device="cpu", output_dir=Path("outputs"), rules=None, preview=False,
                  max_frames=None, output_fps=None, on_progress=None,
                  camera_exposure=None, camera_brightness=None, camera_backend="auto", tracking=False,
                  view="frame", detail=False, show_scores=True, export_profile="standard", processing_height=None):
    """Process a local video Path or nonnegative camera index; return run summary.

    No frame skipping or manual visibility propagation. Tracking is opt-in. A constant-rate
    webcam recording is not guaranteed to have the same duration as wall time.
    Failures finalize a partial summary and release resources before raising.
    """
    import cv2

    if view not in {"frame", "dashboard"} or type(detail) is not bool or type(show_scores) is not bool:
        raise ValueError('Invalid video presentation or score option')
    if export_profile not in {'standard', 'efficient'}:
        raise ValueError('Export profile must be standard or efficient')
    efficient = export_profile == 'efficient'
    if isinstance(source, bool):
        raise ValueError("Source must be a video path or nonnegative camera index")
    camera = type(source) is int
    if processing_height is not None:
        if type(processing_height) is not int or processing_height <= 0:
            raise ValueError('Processing height must be a positive integer or None for original resolution')
        if camera:
            raise ValueError('Processing height applies only to recorded video')
    if camera_backend not in {"auto", "dshow", "msmf"}:
        raise ValueError("Camera backend must be auto, dshow or msmf")
    if not camera and camera_backend != "auto":
        raise ValueError("Camera backend applies only to webcam sources")
    if camera_backend != "auto" and platform.system() != "Windows":
        raise ValueError("DirectShow and Media Foundation backends require Windows")
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
    for name, value in (("camera_exposure", camera_exposure), ("camera_brightness", camera_brightness)):
        if value is not None:
            if not camera:
                raise ValueError("Camera controls apply only to webcam sources")
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
    rules = rules or AssociationRules()
    if type(tracking) is not bool:
        raise ValueError("tracking must be a boolean flag")
    tracker=WorkerTracker() if tracking else None
    started = perf_counter()
    capture = writer = dashboard_writer = frame_log = event_log = None
    dashboard_renderer = None
    run_dir = None
    detector = None
    processed = 0
    rejected_detections = frames_with_rejections = 0
    totals = Counter()
    first_frame_seconds = None
    latency_sum, latency_min, latency_max = 0., math.inf, 0.
    first_output_seconds = model_load_seconds = None
    last_progress_at = None
    loop_start = None
    source_fps = advertised_frames = selected_fps = None
    source_hash = None
    resolution = encoded_resolution = None
    render_resolution = None
    annotation_resolution = None
    processing_resolution = None
    status, stop_reason, error_text, failure = "completed", None, None, None
    window = "PPE assessment - Q or Esc to stop"
    preview_open = False
    selected_backend = "default"
    camera_controls, restore_controls, restore_warnings = {}, [], []
    try:
        if camera and platform.system() == "Windows":
            selected_backend = "msmf" if camera_backend == "msmf" else "dshow"
            flag = cv2.CAP_MSMF if selected_backend == "msmf" else cv2.CAP_DSHOW
            capture = cv2.VideoCapture(capture_source, flag)
            if camera_backend == "auto" and not capture.isOpened():
                capture.release()
                selected_backend = "msmf"
                capture = cv2.VideoCapture(capture_source, cv2.CAP_MSMF)
        else:
            capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {source_name}; check the path, device index, permissions and camera availability")
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        for name, value, prop in (("exposure", camera_exposure, cv2.CAP_PROP_EXPOSURE),
                                  ("brightness", camera_brightness, cv2.CAP_PROP_BRIGHTNESS)):
            if value is None:
                continue
            original = capture.get(prop)
            if not math.isfinite(original):
                raise ValueError(f"Cannot save original camera {name}; refusing to change it")
            camera_controls[name] = {"original": original, "requested": value}
            restore_controls.append((name, prop, original))
            accepted = bool(capture.set(prop, value))
            camera_controls[name].update(accepted=accepted, applied_readback=capture.get(prop))
            if not accepted:
                raise ValueError(f"Camera driver rejected {name}={value}; this control is device-dependent")
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
        if view == 'dashboard':
            from ppe_compliance.preview_ui import DashboardRenderer

            dashboard_renderer = DashboardRenderer()
            if efficient:
                from ppe_compliance.video_export import H264Writer
                dashboard_writer = H264Writer(run_dir/'dashboard.mp4', selected_fps, (1280, 720))
            else:
                dashboard_writer = cv2.VideoWriter(str(run_dir/'dashboard.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'), selected_fps, (1280, 720))
                if not dashboard_writer.isOpened():
                    raise RuntimeError('Cannot open dashboard MP4 writer')
        if tracker:
            event_log=(run_dir/'events.jsonl').open('w',encoding='utf-8')
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
            resize_start = perf_counter()
            ratio = min(1., (processing_height*16/9)/width, processing_height/height) if processing_height else 1.
            processing_frame = (resize_frame(frame, (max(1, round(width*ratio)), max(1, round(height*ratio))))
                                if ratio < 1 else frame)
            resized = perf_counter()
            processing_resolution = (processing_frame.shape[1], processing_frame.shape[0])
            if resolution is None:
                resolution = (width, height)
                render_width, render_height = processing_resolution if efficient else resolution
                ratio = min(1., 1920/render_width, 1080/render_height) if efficient and view == 'frame' else 1.
                render_resolution = (max(1, round(render_width*ratio)), max(1, round(render_height*ratio)))
                encoded_resolution = tuple(v+v % 2 for v in render_resolution)
                if not efficient or view == 'frame':
                    if efficient:
                        from ppe_compliance.video_export import H264Writer
                        writer = H264Writer(run_dir/'annotated.mp4', selected_fps, encoded_resolution)
                    else:
                        writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), selected_fps, encoded_resolution)
                        if not writer.isOpened():
                            raise RuntimeError("Cannot open MP4 video writer; check codec and output-directory support")
            elif resolution != (width, height):
                raise ValueError("Frame dimensions changed; refusing a malformed output video")
            predict_start = perf_counter()
            _, raw = detector.detect(processing_frame, f"{source_name}#frame={processed}")
            inferred = perf_counter()
            # Bind evidence to the exact pixels passed to detection. With a
            # processing cap this is explicitly the resized frame, not 4K pixels.
            frame_hash = hashlib.sha256(memoryview(processing_frame) if processing_frame.flags.c_contiguous
                                        else processing_frame.tobytes()).hexdigest()
            hashed = perf_counter()
            assessment = assess_detections(raw, frame_hash, rules)
            if tracker:
                at=processed/source_fps if source_fps and not camera else decoded-loop_start
                assessment=tracker.update(assessment,at)
                for event in assessment.get('status_events',[]):
                    event_log.write(json.dumps(event)+'\n')
                event_log.flush()
            assessed = perf_counter()
            fit = ((944, 532) if view == 'dashboard' else (1920, 1080)) if efficient else None
            annotated = draw_assessment(processing_frame if efficient else frame, assessment, raw["detections"],
                                        detail=detail, show_scores=show_scores, fit=fit)
            annotation_resolution = (annotated.shape[1], annotated.shape[0])
            screen = annotated
            if dashboard_renderer:
                timestamp = processed/selected_fps
                packet = SimpleNamespace(sequence=processed, received_at=timestamp, frame=frame)
                result = SimpleNamespace(source=packet, annotated=annotated, assessment=assessment,
                                         recent_events=tuple(tracker.history) if tracker else ())
                # Recorded evidence is tied to the displayed source frame, not
                # wall-clock age. Slow offline processing never hides its boxes.
                screen = dashboard_renderer.render(packet, result, True, timestamp,
                    {'mode': 'recorded', 'playback_fps': selected_fps, 'show_scores': show_scores,
                     'source_time_seconds': processed/source_fps if source_fps and not camera else None,
                     'ai_rate': (processed+1)/max(perf_counter()-loop_start, .001)}, focus='analyzed')
            if writer is not None and encoded_resolution != render_resolution:
                annotated = cv2.copyMakeBorder(annotated, 0, render_resolution[1] % 2, 0, render_resolution[0] % 2,
                                               cv2.BORDER_CONSTANT, value=(0,0,0))
            rendered = perf_counter()
            if writer is not None:
                writer.write(annotated)
            if dashboard_writer is not None:
                dashboard_writer.write(screen)
            encoded = perf_counter()
            stop_requested = False
            if preview:
                cv2.imshow(window, screen)
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
                      "frame_hash_format": ("SHA-256 of decoded BGR bytes; dimensions in assessment" if processing_resolution == resolution
                                            else "SHA-256 of resized processing BGR bytes; dimensions in assessment"),
                      "source_coordinate_scale": [width/processing_resolution[0], height/processing_resolution[1]],
                      "detections": raw["detections"], "assessment": assessment,
                      "preparation_timing_seconds": {"analysis_resize": resized-resize_start},
                      "assessment_timing_seconds": {"decoded_frame_sha256": hashed-inferred,
                                                    "rules_and_tracking": assessed-hashed},
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
            rejected = (raw.get('detection_validation') or {}).get('rejected_count', 0)
            rejected_detections += rejected
            frames_with_rejections += int(rejected > 0)
            if first_output_seconds is None:
                first_output_seconds = perf_counter()-started
            progress_now = perf_counter()
            if on_progress is not None and (last_progress_at is None or progress_now-last_progress_at >= 1.):
                on_progress(processed, progress_now-loop_start)
                last_progress_at = progress_now
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
        if capture is not None:
            for name, prop, original in reversed(restore_controls):
                try:
                    accepted = bool(capture.set(prop, original))
                    actual = capture.get(prop)
                    camera_controls[name].update(restore_accepted=accepted, restored_readback=actual)
                    if not accepted or not math.isfinite(actual) or abs(actual-original) > .5:
                        restore_warnings.append(f"Camera {name} restoration could not be confirmed")
                except Exception as error:
                    camera_controls[name]["restore_error"] = str(error)
                    restore_warnings.append(f"Camera {name} restoration failed")
        # Close resources independently even if a backend raises during cleanup.
        for resource in (writer, dashboard_writer, capture):
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
        if event_log is not None:
            try:
                event_log.close()
            except Exception as error:
                if failure is None:
                    failure,error_text,status,stop_reason=error,str(error),'failed','event_close_error'
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
               "processing_height_limit": processing_height,
               "processing_resolution": list(processing_resolution) if processing_resolution else None,
               "frame_hash_scope": "decoded_source_bgr" if processing_resolution == resolution else "resized_processing_bgr",
               "source_coordinate_scale": [resolution[i]/processing_resolution[i] for i in (0, 1)] if resolution and processing_resolution else None,
               "output_resolution": [1280, 720] if efficient and view == 'dashboard' else (list(encoded_resolution) if encoded_resolution else None),
               "export_profile": export_profile,
               "annotation_resolution": list(annotation_resolution) if annotation_resolution else None,
               "encoder_settings": {"preset": "veryfast", "crf": 18, "threads": 2,
                                    "pixel_format": "yuv420p", "tune": "fastdecode", "b_frames": 0,
                                    "max_keyframe_interval_frames": max(1, round(selected_fps*2))} if efficient and selected_fps else None,
               "annotated_file": "annotated.mp4" if writer is not None else None,
               "browser_video_file": ('dashboard.mp4' if view == 'dashboard' else 'annotated.mp4') if efficient else None,
               "detection_coordinate_space": "processing-frame pixels; multiply by source_coordinate_scale to map to original source",
               "presentation_layout": view, "show_confidence_scores": show_scores,
               "equipment_detail_boxes": detail,
               "dashboard_file": "dashboard.mp4" if dashboard_writer is not None else None,
               "dashboard_resolution": [1280, 720] if dashboard_writer is not None else None,
               "odd_dimension_policy": "pad right/bottom by at most one pixel for MP4; report coordinates stay in processing space",
               "output_codec": "h264" if efficient else "mp4v",
               "audio_preserved": False,
               "processed_frames": processed, "intentional_frame_skipping": False,
               "detection_validation": {"rejected_count": rejected_detections,
                                         "frames_with_rejections": frames_with_rejections},
               "image_size": image_size, "confidence_threshold": confidence,
               "weights": str(weights), "weights_sha256": detector.weights_sha256 if detector else None,
               "model_classes": detector.model_classes if detector else None,
               "device": device, "rules": asdict(rules), "preview_enabled": preview,
               "platform": platform.platform(), "python": platform.python_version(), "packages": packages,
               "camera_controls": camera_controls, "camera_control_restore_warnings": restore_warnings,
               "capture_backend_requested": camera_backend, "capture_backend_selected": selected_backend,
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
               "tracking_enabled":tracking,"tracking":tracker.summary() if tracker else None,
               "note": "Worker observations are per frame, not unique people. Optional geometric track IDs are session-local, not verified identity. No automatic missing-PPE violations. Model-input size and source resolution affect speed and recall. Webcam driver buffering/drops are not measured; stream elapsed time is not a device capture timestamp. Constant-FPS webcam output may play faster/slower than wall time. File timestamps use nominal FPS, not variable-rate presentation timestamps. EOF and decode failure cannot always be distinguished from backend metadata."}
    if run_dir is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failure is not None:
        location = f"; partial results: {run_dir}" if run_dir else ""
        raise RuntimeError(f"{error_text}{location}") from failure
    return summary
