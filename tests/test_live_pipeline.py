"""Thread handoff, stale-data safety and optional I/O; no hardware/model required."""

from contextlib import contextmanager, redirect_stdout
import io
import hashlib
import json
from pathlib import Path
from threading import Event
from time import perf_counter, sleep
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ppe_compliance.cli import main
from ppe_compliance.live_pipeline import (
    FramePacket, ResultPacket, LatestSlot, recent_result, compose_preview, process_live,
)
from ppe_compliance.video_pipeline import local_camera_context
from test_assessment import PERSON, HELMET, VEST, report


class PacedCamera:
    def __init__(self, interval=.004, fail_at=None):
        self.interval = interval
        self.index = 0
        self.fail_at = fail_at
        self.released = False
        self.controls = {cv2.CAP_PROP_EXPOSURE:-5.,cv2.CAP_PROP_BRIGHTNESS:0.}

    def isOpened(self):
        return True

    def read(self):
        sleep(self.interval)
        if self.index == self.fail_at:
            return False,None
        self.index += 1
        return True,np.full((640,640,3),self.index%200,dtype=np.uint8)

    def release(self):
        self.released = True

    def get(self, prop):
        return self.controls[prop]

    def set(self, prop, value):
        self.controls[prop] = value
        return True


class SlowDetector:
    model_classes = {0:"person",1:"helmet",2:"vest"}
    weights_sha256 = "b"*64

    def __init__(self, delay=.04):
        self.delay = delay
        self.sequences = []

    def detect(self, frame, source):
        self.sequences.append(int(source.rsplit("=",1)[1]))
        sleep(self.delay)
        raw = report([PERSON,HELMET,VEST])
        raw.update(source=source,device="cpu",weights_sha256=self.weights_sha256)
        return None,raw


class LatestSlotTests(unittest.TestCase):
    def test_newest_packet_replaces_older_packets(self):
        slot = LatestSlot()
        for index in range(100):
            slot.publish(FramePacket(index,1.,None))
        self.assertEqual(slot.snapshot().sequence,99)
        self.assertEqual(slot.next_frame(-1,Event()).sequence,99)
        self.assertIsNone(slot.next_frame(99,Event()))

    def test_stop_prevents_pending_packet_consumption(self):
        slot = LatestSlot();slot.publish(FramePacket(2,1.,None))
        stop=Event();stop.set()
        self.assertIsNone(slot.next_frame(-1,stop))

    def test_age_is_based_on_source_receive_not_prediction_completion(self):
        result=ResultPacket(FramePacket(1,10.,None),10.9,None,{})
        self.assertFalse(recent_result(result,10.91,.5))
        self.assertTrue(recent_result(result,10.91,1.))

    def test_stale_result_pixels_never_appear_on_live_or_result_pane(self):
        frame=FramePacket(10,10.,np.full((100,200,3),50,np.uint8))
        result=ResultPacket(FramePacket(1,9.,frame.frame),9.1,np.full_like(frame.frame,200),{})
        view,fresh=compose_preview(frame,result,10.,.5)
        self.assertFalse(fresh)
        self.assertTrue(np.all(view[45:,:200]==50))
        self.assertTrue(np.all(view[45:,200:]==0))

    def test_fresh_result_is_displayed_only_in_separate_analyzed_pane(self):
        frame=FramePacket(10,10.,np.full((100,200,3),50,np.uint8))
        result=ResultPacket(FramePacket(8,9.8,frame.frame),9.9,np.full_like(frame.frame,200),{})
        view,fresh=compose_preview(frame,result,10.,.5)
        self.assertTrue(fresh)
        self.assertTrue(np.all(view[45:,:200]==50))
        self.assertTrue(np.all(view[45:,200:]==200))


class LiveTests(unittest.TestCase):
    def test_unreadable_camera_reports_source_error_before_loading_model(self):
        self.camera.fail_at=0
        with self.assertRaisesRegex(RuntimeError,'Camera stopped returning frames'):
            self.run_live()
        self.constructor.assert_not_called()
        self.assertTrue(self.camera.released)

    def test_headless_browser_callback_has_frames_and_can_stop_without_gui(self):
        callbacks=[]
        def show(frame,info):
            callbacks.append((frame.copy(),dict(info)))
        with patch('cv2.imshow') as gui,patch('cv2.namedWindow') as window:
            result=process_live(output_dir=self.root/'browser',preview=False,view='split',
                capture_context_factory=self.capture,on_preview=show,
                stop_requested=lambda:len(callbacks)>=3,preview_fps=30,duration=2)
        self.assertEqual(result['stop_reason'],'browser_stop')
        self.assertGreaterEqual(len(callbacks),3)
        self.assertEqual(callbacks[0][1]['source_kind'],'camera')
        self.assertTrue(self.camera.released)
        gui.assert_not_called();window.assert_not_called()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.camera=PacedCamera()
        self.detector=SlowDetector()
        self.patch=patch("ppe_compliance.live_pipeline.FrameDetector",return_value=self.detector)
        self.constructor=self.patch.start();self.addCleanup(self.patch.stop)

    @contextmanager
    def capture(self):
        try:
            yield self.camera,{"backend":"test"}
        finally:
            self.camera.release()

    def run_live(self, **kwargs):
        defaults=dict(output_dir=self.root,preview=False,max_frames=3,
                      capture_context_factory=self.capture,shutdown_timeout=2.)
        defaults.update(kwargs)
        return process_live(**defaults)

    def test_skips_old_frames_without_duplicate_inference_or_model_reload(self):
        result=self.run_live()
        self.assertEqual(result["processed_frames"],3)
        self.assertGreater(result["captured_frames"],result["processed_frames"])
        self.assertEqual(self.detector.sequences,sorted(set(self.detector.sequences)))
        self.assertTrue(any(b-a>1 for a,b in zip(self.detector.sequences,self.detector.sequences[1:])))
        self.constructor.assert_called_once()
        self.assertTrue(self.camera.released)

    def test_preview_only_creates_summary_without_video_or_frame_log(self):
        result=self.run_live()
        self.assertEqual({p.name for p in Path(result["run_directory"]).iterdir()},{"summary.json"})
        self.assertFalse(result["recording_enabled"])

    def test_recording_contains_only_selected_frames_and_matching_jsonl(self):
        result=self.run_live(record=True)
        folder=Path(result["run_directory"])
        rows=[json.loads(line) for line in (folder/"frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),3)
        self.assertEqual([r["capture_sequence"] for r in rows],self.detector.sequences)
        cap=cv2.VideoCapture(str(folder/"annotated.mp4"));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
        cap.release();self.assertEqual(count,3)

    def test_tracking_events_are_saved_without_enabling_video_recording(self):
        self.detector.delay=.09
        result=self.run_live(tracking=True,max_frames=6)
        root=Path(result['run_directory'])
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual(result['tracking']['tracks_created'],1)
        self.assertEqual(len([e for e in events if e['type']=='status_changed']),1)
        self.assertFalse((root/'annotated.mp4').exists())
        self.assertFalse((root/'frames.jsonl').exists())

    def test_old_predictions_do_not_update_live_track_state(self):
        self.detector.delay=.06
        result=self.run_live(tracking=True,max_frames=2,stale_after=.01)
        self.assertEqual(result['tracking']['tracks_created'],0)
        self.assertEqual(result['tracking_skipped_stale_results'],2)

    def test_requested_capture_resolution_is_reported_and_camera_released(self):
        with patch('cv2.VideoCapture',return_value=self.camera), \
                patch('ppe_compliance.video_pipeline.platform.system',return_value='Windows'):
            with local_camera_context(0,width=1280,height=720) as (cap,info):
                self.assertTrue(info['resolution_request']['accepted'])
        self.assertTrue(self.camera.released)

    def test_reused_capture_buffer_cannot_change_analyzed_source_pixels(self):
        buffer=np.zeros((640,640,3),np.uint8)
        def reused_read():
            sleep(.004)
            self.camera.index+=1
            buffer.fill(self.camera.index%200)
            return True,buffer
        self.camera.read=reused_read
        original=self.detector.detect
        expected=[]
        def detect(frame,source):
            expected.append(hashlib.sha256(frame.tobytes()).hexdigest())
            return original(frame,source)
        self.detector.detect=detect
        result=self.run_live(record=True)
        rows=[json.loads(line) for line in (Path(result['run_directory'])/'frames.jsonl').read_text().splitlines()]
        self.assertEqual([row['assessment']['source_sha256'] for row in rows],expected)

    def test_duration_stops_even_without_completed_predictions(self):
        self.detector.delay=.15
        result=self.run_live(duration=.03,max_frames=None)
        self.assertEqual(result["stop_reason"],"duration")
        self.assertEqual(result["processed_frames"],0)
        self.assertTrue(self.camera.released)

    def test_inference_error_is_propagated_and_capture_released(self):
        self.detector.detect=lambda *args: (_ for _ in ()).throw(RuntimeError("broken prediction"))
        with self.assertRaisesRegex(RuntimeError,"broken prediction"):
            self.run_live()
        self.assertTrue(self.camera.released)

    def test_empty_error_message_still_stops_instead_of_hanging(self):
        self.detector.detect=lambda *args: (_ for _ in ()).throw(RuntimeError())
        with self.assertRaisesRegex(RuntimeError,"RuntimeError"):
            self.run_live()

    def test_capture_disconnect_is_propagated(self):
        self.camera.fail_at=2
        with self.assertRaisesRegex(RuntimeError,"stopped returning"):
            self.run_live()
        self.assertTrue(self.camera.released)

    def test_wrong_model_fails_and_releases_capture(self):
        self.detector.model_classes={0:"person",1:"bus"}
        with self.assertRaisesRegex(RuntimeError,"PPE-trained"):
            self.run_live()
        self.assertTrue(self.camera.released)

    def test_preview_remains_active_during_slow_model_initialization(self):
        self.camera.interval=.004
        displayed=Event()
        displayed_count=[0]
        def load(*args):
            displayed.wait(timeout=1)
            return self.detector
        def display(*args):
            displayed_count[0]+=1
            if displayed_count[0]>=2:
                displayed.set()
        with patch("ppe_compliance.live_pipeline.FrameDetector",side_effect=load), \
                patch("cv2.namedWindow"),patch("cv2.imshow",side_effect=display) as show, \
                patch("cv2.waitKey",side_effect=lambda *args: 27 if displayed.is_set() else -1), \
                patch("cv2.getWindowProperty",return_value=1),patch("cv2.destroyWindow"):
            result=self.run_live(preview=True,duration=1,max_frames=None)
        self.assertGreater(show.call_count,1)
        self.assertEqual(result["processed_frames"],0)
        self.assertGreater(result["unique_live_frames_displayed"],1)

    def test_preview_stop_closes_window_and_finishes_workers(self):
        with patch("cv2.namedWindow"),patch("cv2.imshow"),patch("cv2.waitKey",return_value=27), \
                patch("cv2.getWindowProperty",return_value=1),patch("cv2.destroyWindow") as close:
            result=self.run_live(preview=True)
        self.assertEqual(result["stop_reason"],"preview_stop")
        close.assert_called_once();self.assertTrue(self.camera.released)

    def test_headless_preview_error_stops_and_releases(self):
        with patch("cv2.namedWindow",side_effect=cv2.error("headless")), \
                self.assertRaisesRegex(RuntimeError,"no-preview"):
            self.run_live(preview=True)
        self.assertTrue(self.camera.released)

    def test_invalid_options_are_rejected_before_threads_start(self):
        for options in ({"stale_after":0},{"duration":float("nan")},{"max_frames":0},{"image_size":0}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.run_live(**options)
        self.constructor.assert_not_called()

    def test_local_camera_controls_restore_in_capture_owner_context(self):
        with patch("cv2.VideoCapture",return_value=self.camera), \
                patch("ppe_compliance.video_pipeline.platform.system",return_value="Windows"):
            with local_camera_context(0,exposure=-3,brightness=24) as (cap,info):
                self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE),-3)
        self.assertEqual(info["controls"]["exposure"]["restored_readback"],-5)
        self.assertTrue(self.camera.released)

    def test_blocked_native_capture_is_reported_not_unsafely_released(self):
        waiting, unblock, released = Event(),Event(),Event()
        camera=self.camera
        def blocked_read():
            waiting.set()
            # The finally block always unblocks this call. A wall-clock timeout
            # made this race with cold dependency imports on busy machines.
            unblock.wait()
            return True,np.zeros((640,640,3),np.uint8)
        camera.read=blocked_read
        @contextmanager
        def context():
            try:
                yield camera,{}
            finally:
                camera.release();released.set()
        try:
            with patch("cv2.namedWindow"),patch("cv2.imshow"), \
                    patch("cv2.waitKey",side_effect=lambda *args: 27 if waiting.is_set() else -1), \
                    patch("cv2.getWindowProperty",return_value=1),patch("cv2.destroyWindow"), \
                    self.assertRaisesRegex(RuntimeError,"did not stop"):
                self.run_live(capture_context_factory=context,preview=True,max_frames=None,shutdown_timeout=.02)
            self.assertTrue(waiting.is_set())
            self.assertFalse(camera.released)
            summary=json.loads(next(self.root.glob('*/summary.json')).read_text())
            self.assertEqual(summary['stop_reason'],'shutdown_timeout')
        finally:
            unblock.set()
            self.assertTrue(released.wait(timeout=1))

    def test_cli_routes_low_latency_flags_without_affecting_sequential_mode(self):
        fake={"status":"stopped","stop_reason":"max_frames","processed_frames":2,"captured_frames":10,
              "assessment_throughput_fps":5.,"recording_enabled":False,"run_directory":"test","note":"test"}
        with patch("sys.argv",["ppe","webcam","--low-latency","--image-size","320","--duration","1","--no-preview"]), \
                patch("ppe_compliance.live_pipeline.process_live",return_value=fake) as execute,redirect_stdout(io.StringIO()):
            main()
        options=execute.call_args.kwargs
        self.assertEqual(options["image_size"],320)
        self.assertFalse(options["record"])
        self.assertEqual(options["duration"],1)


if __name__ == "__main__":
    unittest.main()
