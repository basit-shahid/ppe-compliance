"""Latest-frame webcam preview and inference with bounded, single-slot handoffs.

The live pane NEVER receives boxes from another frame. The result pane displays
the exact analyzed frame and disappears when its application-receive age expires.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from threading import Condition, Event, Thread
from time import perf_counter, sleep
from uuid import uuid4

from ppe_compliance.association import AssociationRules
from ppe_compliance.compliance import assess_detections
from ppe_compliance.detector import FrameDetector
from ppe_compliance.rendering import draw_assessment
from ppe_compliance.tracking import WorkerTracker


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    received_at: float
    frame: object


@dataclass(frozen=True)
class ResultPacket:
    source: FramePacket
    completed_at: float
    annotated: object
    assessment: dict
    recent_events: tuple = ()


class LatestSlot:
    """One immutable packet reference, replaced instead of queued."""

    def __init__(self):
        self.condition = Condition()
        self.value = None

    def publish(self, value):
        with self.condition:
            self.value = value
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.value

    def next_frame(self, after, stop):
        with self.condition:
            self.condition.wait_for(lambda: stop.is_set() or
                                    (self.value is not None and self.value.sequence > after), timeout=.05)
            value = self.value
            return value if not stop.is_set() and value is not None and value.sequence > after else None


@contextmanager
def default_camera(index):
    import cv2

    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(index)
    try:
        yield capture, {"backend": "dshow" if platform.system() == "Windows" else "default"}
    finally:
        capture.release()


def recent_result(result, now, stale_after):
    return result is not None and 0 <= now-result.source.received_at <= stale_after


def compose_preview(frame, result, now, stale_after, loading=False):
    """Return independent live/result panes; never project old boxes on live pixels."""
    import cv2
    import numpy as np

    height, width = frame.frame.shape[:2]
    pane_width = min(480, width)
    pane_height = max(1, round(height*pane_width/width))
    live = cv2.resize(frame.frame, (pane_width, pane_height))
    fresh = recent_result(result, now, stale_after)
    right = cv2.resize(result.annotated, (pane_width, pane_height)) if fresh else np.zeros_like(live)
    body = cv2.hconcat([live, right])
    header = np.full((45,body.shape[1],3),35,dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, f"LIVE frame {frame.sequence} (no AI boxes)", (5,18), font, .42, (255,255,255), 1)
    if fresh:
        text = f"ANALYZED frame {result.source.sequence} | age {now-result.source.received_at:.2f}s"
    else:
        text = "AI loading / warming up" if loading else "No recent result; old boxes hidden"
    cv2.putText(header, text, (pane_width+5,18), font, .42, (255,255,255), 1)
    cv2.putText(header, "Q/Esc: stop | Results refer only to the analyzed pane", (5,37), font, .4, (255,255,255), 1)
    return cv2.vconcat([header,body]), fresh


def process_live(camera=0, weights="models/best.pt", confidence=.25, image_size=320,
                 device="cpu", output_dir=Path("outputs"), rules=None, preview=True,
                 max_frames=None, duration=None, record=False, output_fps=10.,
                 stale_after=.5, capture_context_factory=None, on_progress=None,
                 shutdown_timeout=30., tracking=False, view="dashboard", focus="live",
                 preview_width=1280, preview_fps=60., on_preview=None, stop_requested=None):
    """Run low-latency webcam processing. Capture and inference each own their I/O.

    Recording is opt-in and contains analyzed frames only, not the smooth pane.
    A timed-out native read/predict cannot safely be killed by Python; report it
    as a failed shutdown rather than releasing a camera concurrently with read().
    """
    import cv2

    if type(camera) is not int or camera < 0:
        raise ValueError("Camera index must be nonnegative")
    if any(value is not None and not callable(value) for value in (on_preview, stop_requested)):
        raise ValueError('Preview and stop hooks must be callable')
    if type(record) is not bool or type(preview) is not bool:
        raise ValueError("record and preview must be boolean flags")
    if type(tracking) is not bool or view not in {"dashboard","split"} or focus not in {"live","analyzed"}:
        raise ValueError("Invalid tracking or preview layout option")
    if type(preview_width) is not int or not 960<=preview_width<=1920:
        raise ValueError("Preview width must be an integer from 960 to 1920")
    for name, value in (("confidence",confidence),("output_fps",output_fps),
                        ("preview_fps",preview_fps),
                        ("stale_after",stale_after),("shutdown_timeout",shutdown_timeout)):
        if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if confidence > 1 or type(image_size) is not int or image_size <= 0:
        raise ValueError("Invalid confidence or model image size")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames must be positive")
    if duration is not None and (isinstance(duration,bool) or not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive and finite")
    rules = rules or AssociationRules()
    run_dir = Path(output_dir)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    stop = Event()
    frames, results = LatestSlot(), LatestSlot()
    tracker = WorkerTracker() if tracking else None
    state = {"captured":0, "processed":0, "capture_error":None, "inference_error":None,
             "model_ready":False, "limit_reached":False, "recorded":0, "camera":{},
             "latency_sum":0., "latency_min":math.inf, "latency_max":0., "queue_sum":0.,
             "first_latency":None, "inference_started":None, "inference_finished":None,
             "model_load_seconds":None, "weights_sha256":None, "worker_observations":{}}
    state.update(source_resolution=None,recording_resolution=None)
    state.update(status_events_written=0,tracking_skipped_stale_results=0)
    state.update(rejected_detections=0, frames_with_rejections=0)

    def acquire():
        try:
            context = capture_context_factory or (lambda: default_camera(camera))
            with context() as (capture, info):
                state["camera"] = info
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open camera {camera}")
                resolution = None
                while not stop.is_set():
                    ok, image = capture.read()
                    received = perf_counter()
                    if stop.is_set():
                        break
                    if not ok:
                        if stop.is_set():
                            break
                        raise RuntimeError("Camera stopped returning frames")
                    if image is None or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError("Expected a three-channel camera frame")
                    if resolution is None:
                        resolution = image.shape
                        state["source_resolution"] = [image.shape[1],image.shape[0]]
                    elif resolution != image.shape:
                        raise ValueError("Camera frame dimensions changed")
                    # Some backends reuse their decode buffer. Own immutable pixels
                    # before the next read so result hashes/boxes refer to one frame.
                    image = image.copy()
                    image.flags.writeable = False
                    packet = FramePacket(state["captured"], received, image)
                    frames.publish(packet)
                    state["captured"] += 1
        except Exception as error:
            state["capture_error"] = str(error) or type(error).__name__
            stop.set()

    def analyze():
        writer = log = event_log = None
        try:
            # A failed/busy camera should report immediately, without waiting
            # for an unnecessary cold model import to finish during shutdown.
            while frames.snapshot() is None and not stop.is_set():
                stop.wait(.02)
            if stop.is_set():
                return
            load_started = perf_counter()
            detector = FrameDetector(weights, confidence, image_size, device)
            state["model_load_seconds"] = perf_counter()-load_started
            state["weights_sha256"] = detector.weights_sha256
            if not {"person","helmet","vest"} <= {name.casefold() for name in detector.model_classes.values()}:
                raise ValueError("Live assessment requires PPE-trained person/helmet/vest weights")
            state["model_ready"] = True
            state["inference_started"] = perf_counter()
            if record:
                log = (run_dir/"frames.jsonl").open("w",encoding="utf-8")
            if tracker:
                event_log = (run_dir/"events.jsonl").open("w",encoding="utf-8")
            previous = -1
            while not stop.is_set():
                packet = frames.next_frame(previous, stop)
                if packet is None:
                    continue
                # Protect capture/preview pixels from downstream model mutation.
                image = packet.frame.copy()
                prediction_started = perf_counter()
                _, raw = detector.detect(image, f"camera:{camera}#capture={packet.sequence}")
                if stop.is_set():
                    break
                assessment = assess_detections(raw, hashlib.sha256(packet.frame.tobytes()).hexdigest(), rules)
                assessment["detection_id_scope"] = "zero-based index into this analyzed frame's detections"
                if tracker:
                    if perf_counter()-packet.received_at <= stale_after:
                        assessment = tracker.update(assessment,packet.received_at-started)
                    else:
                        assessment["tracking_update_skipped"] = "Source too old for current live tracking state"
                        state["tracking_skipped_stale_results"] += 1
                annotated = draw_assessment(packet.frame, assessment, raw["detections"])
                completed = perf_counter()
                # Do not publish a prediction completed after the user stopped.
                if stop.is_set():
                    break
                if event_log:
                    for event in assessment.get("status_events",[]):
                        event_log.write(json.dumps(event)+"\n")
                        state["status_events_written"] += 1
                    event_log.flush()
                result = ResultPacket(packet, completed, annotated, assessment,
                                      tuple(tracker.history) if tracker else ())
                results.publish(result)
                if record:
                    height, width = annotated.shape[:2]
                    if writer is None:
                        state["recording_resolution"] = [width+width%2,height+height%2]
                        writer = cv2.VideoWriter(str(run_dir/"annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 output_fps,(width+width%2,height+height%2))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot open MP4 writer")
                    if width%2 or height%2:
                        annotated = cv2.copyMakeBorder(annotated,0,height%2,0,width%2,cv2.BORDER_CONSTANT,value=(0,0,0))
                    writer.write(annotated)
                    log.write(json.dumps({"capture_sequence":packet.sequence,
                              "received_elapsed_seconds":packet.received_at-started,
                              "result_elapsed_seconds":completed-started,
                              "application_result_age_seconds":completed-packet.received_at,
                              "detections":raw["detections"], "assessment":assessment},separators=(",",":"))+"\n")
                    log.flush()
                    state["recorded"] += 1
                latency = completed-packet.received_at
                state["latency_sum"] += latency
                state["latency_min"] = min(state["latency_min"],latency)
                state["latency_max"] = max(state["latency_max"],latency)
                state["queue_sum"] += prediction_started-packet.received_at
                if state["first_latency"] is None:
                    state["first_latency"] = latency
                for key,value in assessment["summary"].items():
                    if key != "workers":
                        state["worker_observations"][key] = state["worker_observations"].get(key,0)+value
                state["processed"] += 1
                rejected = (raw.get('detection_validation') or {}).get('rejected_count', 0)
                state['rejected_detections'] += rejected
                state['frames_with_rejections'] += int(rejected > 0)
                previous = packet.sequence
                if on_progress and (state["processed"] == 1 or state["processed"]%30 == 0):
                    on_progress(state["processed"],completed-state["inference_started"])
                if max_frames is not None and state["processed"] >= max_frames:
                    state["limit_reached"] = True
                    break
        except Exception as error:
            state["inference_error"] = str(error) or type(error).__name__
            stop.set()
        finally:
            for resource in (writer,log,event_log):
                if resource is not None:
                    try:
                        resource.release() if resource is writer else resource.close()
                    except Exception as error:
                        state["inference_error"] = str(error) or type(error).__name__
                        stop.set()
            state["inference_finished"] = perf_counter()

    capture_thread = Thread(target=acquire,name="ppe-capture",daemon=True)
    inference_thread = Thread(target=analyze,name="ppe-inference",daemon=True)
    reason, error, preview_open = None, None, False
    shown_live = shown_results = ticks = 0
    preview_started = preview_stopped = None
    last_live = last_result = -1
    window = "PPE low latency | live vs analyzed | Q/Esc stop"
    resized_window = False
    dashboard_renderer = None
    last_render_key = None
    last_render_at = -math.inf
    fullscreen = False
    try:
        capture_thread.start()
        inference_thread.start()
        if preview or on_preview is not None:
            preview_started = perf_counter()
        if preview:
            try:
                cv2.namedWindow(window,cv2.WINDOW_NORMAL)
            except cv2.error as failure:
                raise RuntimeError("Preview unavailable; add --no-preview") from failure
            preview_open = True
            preview_started = perf_counter()
        while True:
            now = perf_counter()
            if stop_requested is not None and stop_requested():
                reason = 'browser_stop'
                break
            if state["capture_error"] or state["inference_error"]:
                raise RuntimeError(state["capture_error"] or state["inference_error"])
            if state["limit_reached"]:
                reason = "max_frames"
                break
            if duration is not None and now-started >= duration:
                reason = "duration"
                break
            packet = frames.snapshot()
            result = results.snapshot()
            fresh = recent_result(result,now,stale_after)
            render_key = (packet.sequence if packet else -1,
                          result.source.sequence if result else -1, fresh, focus,
                          int(now*5), state["model_ready"])
            if ((preview or on_preview is not None) and packet is not None and render_key != last_render_key
                    and now-last_render_at >= 1/preview_fps):
                if view == "dashboard":
                    from ppe_compliance.preview_ui import DashboardRenderer

                    if dashboard_renderer is None:dashboard_renderer=DashboardRenderer()
                    active = now-state["inference_started"] if state["inference_started"] else 0.
                    screen = dashboard_renderer.render(packet,result,fresh,now,
                        {"tracking":tracking,"recording":record,
                         "loading":not state["model_ready"] or result is None,
                         "live_rate":shown_live/(now-preview_started) if preview_started and now>preview_started else 0.,
                         "ai_rate":state["processed"]/active if active else 0.},
                        focus=focus,width=preview_width)
                else:
                    screen,fresh = compose_preview(packet,result,now,stale_after,loading=not state["model_ready"] or result is None)
                if on_preview is not None:
                    on_preview(screen, {'loading':not state['model_ready'] or result is None,
                        'source_sequence':packet.sequence, 'processed_frames':state['processed'],
                        'analysis_sequence':result.source.sequence if result is not None else None,
                        'fresh_result':fresh, 'source_kind':'camera'})
                if preview:
                    cv2.imshow(window,screen)
                last_render_key, last_render_at = render_key, perf_counter()
                if preview and not resized_window:
                    try:
                        cv2.resizeWindow(window,screen.shape[1],screen.shape[0])
                    except cv2.error:
                        pass
                    resized_window = True
                ticks += 1
                if packet.sequence != last_live:
                    shown_live += 1
                    last_live = packet.sequence
                if fresh and result.source.sequence != last_result:
                    shown_results += 1
                    last_result = result.source.sequence
            if preview:
                key = cv2.waitKey(1)&0xFF
                if key == 9:
                    focus = "analyzed" if focus == "live" else "live"
                if key in (ord('f'),ord('F')):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                try:
                    visible = cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 0
                if key in (ord("q"),ord("Q"),27) or visible < 1:
                    reason = "preview_stop"
                    break
            # waitKey handles UI events; avoid the old fixed 20 ms display delay.
            sleep(.01 if not preview else .001)
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as failure:
        error, reason = str(failure) or type(failure).__name__, "error"
    finally:
        stop.set()
        if preview_started is not None:
            preview_stopped = perf_counter()
        if preview_open:
            preview_stopped = perf_counter()
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
        deadline = perf_counter()+shutdown_timeout
        for worker in (capture_thread,inference_thread):
            if worker.ident is not None:
                worker.join(timeout=max(0.,deadline-perf_counter()))
    unfinished = [worker.name for worker in (capture_thread,inference_thread) if worker.is_alive()]
    error = error or state["capture_error"] or state["inference_error"]
    if unfinished:
        error = (f'{error}; ' if error else '') + "Native capture/inference did not stop within timeout; resource finalization is not confirmed: "+", ".join(unfinished)
    ended = perf_counter()
    count = state["processed"]
    active = ((state["inference_finished"] or ended)-state["inference_started"]) if state["inference_started"] else 0
    preview_seconds = preview_stopped-preview_started if preview_started is not None else 0.
    packages = {}
    for name in ("ultralytics","torch","opencv-python","numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    summary = {"schema_version":1,"mode":"latest-frame", "status":"failed" if error else "stopped",
        "stop_reason":"shutdown_timeout" if unfinished else reason,"error":error,"run_directory":str(run_dir),
        "source":f"camera:{camera}","image_size":image_size,"device":device,"confidence_threshold":confidence,
        "source_resolution":state["source_resolution"],"platform":platform.platform(),
        "python":platform.python_version(),"packages":packages,
        "weights":str(weights),"weights_sha256":state["weights_sha256"],"camera":state["camera"],
        "tracking_enabled":tracking,"tracking":tracker.summary() if tracker else None,
        "tracking_skipped_stale_results":state["tracking_skipped_stale_results"],
        "status_events_written":state["status_events_written"],
        "preview_layout":view,"preview_width":preview_width,"final_focus":focus,
        "preview_fps_limit":preview_fps,
        "recording_enabled":record,"recorded_frames":state["recorded"],"recording_playback_fps":output_fps if record else None,
        "recording_resolution":state["recording_resolution"],"audio_preserved":False,
        "captured_frames":state["captured"],"processed_frames":count,
        "detection_validation":{"rejected_count":state['rejected_detections'],
                                 "frames_with_rejections":state['frames_with_rejections']},
        "captured_but_not_processed":state["captured"]-count,"intentional_frame_skipping":True,
        "total_wall_seconds":ended-started,"inference_active_seconds":active,
        "assessment_throughput_fps":count/active if active else 0.,
        "preview_enabled":preview,"browser_preview_enabled":on_preview is not None,
        "preview_ticks":ticks,"unique_live_frames_displayed":shown_live,
        "distinct_recent_results_displayed":shown_results,"preview_active_seconds":preview_seconds,
        "live_preview_fps":shown_live/preview_seconds if preview_seconds else None,
        "model_load_seconds":state["model_load_seconds"],"stale_after_seconds":stale_after,
        "application_result_latency_seconds":{"mean":state["latency_sum"]/count if count else None,
            "min":state["latency_min"] if count else None,"max":state["latency_max"] if count else None,
            "first":state["first_latency"],
            "mean_after_first":(state["latency_sum"]-state["first_latency"])/(count-1) if count>1 else None},
        "application_queue_wait_mean_seconds":state["queue_sum"]/count if count else None,
        "worker_observations":state["worker_observations"],
        "note":"Live pane has no AI boxes. Analyzed view uses its exact source frame; current decisions hide after the age limit. Age starts at application read, not sensor exposure; driver/network buffering is unmeasured. Frame skipping is intentional. Recording contains analyzed frames only. Optional track IDs are geometric session labels, not verified identities. Uncertainty is never held green; no automatic missing-PPE violations."}
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if error:
        raise RuntimeError(f"{error}; results: {run_dir}")
    return summary
