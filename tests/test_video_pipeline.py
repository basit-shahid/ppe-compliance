"""Video I/O, cleanup and timing contracts using deterministic fake predictions."""

from contextlib import redirect_stdout, redirect_stderr
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ppe_compliance.cli import main
from ppe_compliance.detector import FrameDetector
from ppe_compliance.video_pipeline import process_video
from test_assessment import PERSON, HELMET, VEST, report


class FakeDetector:
    model_classes = {0: "person", 1: "helmet", 2: "vest"}
    weights_sha256 = "b" * 64

    def __init__(self):
        self.calls = 0

    def detect(self, frame, source):
        self.calls += 1
        data = report([PERSON, HELMET, VEST])
        data.update(source=source, device="cpu", weights_sha256=self.weights_sha256,
                    image_width=frame.shape[1], image_height=frame.shape[0])
        return None, data


class FakeCapture:
    def __init__(self, frames, fps=12, count=None, opened=True, interrupt_at=None):
        self.frames = frames
        self.fps = fps
        self.count = len(frames) if count is None else count
        self.opened = opened
        self.interrupt_at = interrupt_at
        self.index = 0
        self.released = False
        self.controls = {cv2.CAP_PROP_EXPOSURE: -5., cv2.CAP_PROP_BRIGHTNESS: 0.}

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop in self.controls:
            return self.controls[prop]
        return self.fps if prop == cv2.CAP_PROP_FPS else self.count

    def set(self, prop, value):
        self.controls[prop] = float(value)
        return True

    def read(self):
        if self.index == self.interrupt_at:
            raise KeyboardInterrupt()
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.avi"
        self.frames = [np.full((640, 640, 3), 25+i*15, dtype=np.uint8) for i in range(5)]
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 12, (640,640))
        self.assertTrue(writer.isOpened())
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        self.output = self.root / "outputs"
        self.detector = FakeDetector()
        self.model_patch = patch("ppe_compliance.video_pipeline.FrameDetector", return_value=self.detector)
        self.constructor = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_video(self, **kwargs):
        return process_video(self.source, output_dir=self.output, **kwargs)

    def rows(self, summary):
        return [json.loads(line) for line in (Path(summary["run_directory"])/"frames.jsonl").read_text().splitlines()]

    def partial(self):
        return json.loads(next(self.output.glob("*/summary.json")).read_text())

    def test_real_video_round_trip_loads_model_once_and_streams_rows(self):
        result = self.run_video()
        self.constructor.assert_called_once()
        self.assertEqual(self.detector.calls, 5)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["processed_frames"], 5)
        self.assertEqual(result["output_fps"], 12)
        rows = self.rows(result)
        self.assertEqual([r["frame_index"] for r in rows], list(range(5)))
        self.assertAlmostEqual(rows[-1]["source_time_seconds"], 4/12)
        self.assertEqual(result["worker_observations"]["PPE detected"], 5)
        self.assertFalse(result["intentional_frame_skipping"])
        video = cv2.VideoCapture(str(Path(result["run_directory"])/"annotated.mp4"))
        decoded = []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            decoded.append(frame.shape)
        video.release()
        self.assertEqual(decoded, [(640,640,3)]*5)
        self.assertGreater(result["end_to_end_processing_fps"], 0)
        self.assertGreaterEqual(result["total_wall_seconds"], result["processing_wall_seconds"])

    def test_max_frames_is_explicit_and_does_not_overwrite_runs(self):
        first = self.run_video(max_frames=2)
        second = self.run_video(max_frames=1)
        self.assertEqual(first["stop_reason"], "max_frames")
        self.assertEqual(len(self.rows(first)), 2)
        self.assertNotEqual(first["run_directory"], second["run_directory"])

    def test_dashboard_export_preserves_exact_frame_count_timing_and_native_export(self):
        from ppe_compliance.preview_ui import DashboardRenderer

        renderer = DashboardRenderer()
        with patch.object(renderer, 'render', wraps=renderer.render) as render, \
                patch('ppe_compliance.preview_ui.DashboardRenderer', return_value=renderer):
            result = self.run_video(view='dashboard', tracking=True, max_frames=3, output_fps=6)
        self.assertEqual(result['source_resolution'], [640, 640])
        self.assertEqual(result['dashboard_resolution'], [1280, 720])
        self.assertTrue(result['show_confidence_scores'])
        for index, call in enumerate(render.call_args_list):
            packet, analyzed, fresh, now, stats = call.args
            self.assertEqual(packet.sequence, index)
            self.assertEqual(analyzed.source.sequence, index)
            self.assertTrue(fresh)
            self.assertEqual(stats['mode'], 'recorded')
            self.assertAlmostEqual(stats['source_time_seconds'], index/12)
        for name, shape in [('annotated.mp4', (640, 640, 3)), ('dashboard.mp4', (720, 1280, 3))]:
            capture = cv2.VideoCapture(str(Path(result['run_directory'])/name))
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 6)
            decoded = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(frame.shape)
            capture.release()
            self.assertEqual(decoded, [shape]*3)

    def test_invalid_presentation_is_rejected_before_loading_model(self):
        with self.assertRaises(ValueError):
            self.run_video(view='invalid')
        self.constructor.assert_not_called()

    def test_efficient_dashboard_is_single_h264_with_identical_source_evidence(self):
        standard = self.run_video(view='dashboard', tracking=True)
        efficient = self.run_video(view='dashboard', tracking=True, export_profile='efficient')
        run = Path(efficient['run_directory'])
        self.assertFalse((run/'annotated.mp4').exists())
        self.assertEqual(efficient['browser_video_file'], 'dashboard.mp4')
        self.assertEqual(efficient['output_codec'], 'h264')
        self.assertEqual(efficient['output_resolution'], [1280, 720])
        a, b = self.rows(standard), self.rows(efficient)
        for reference, current in zip(a, b):
            self.assertEqual(current['assessment'], reference['assessment'])
            self.assertEqual(current['detections'], reference['detections'])
        from ppe_compliance.benchmark import verify_export
        self.assertEqual(verify_export(run/'dashboard.mp4'), {
            'decoded_frames': 5, 'resolution': [1280, 720], 'playback_fps': 12})

    def test_efficient_frame_export_fits_1080p_and_keeps_full_resolution_hash(self):
        frames = [np.full((2160, 3840, 3), 80, np.uint8)]
        capture = FakeCapture(frames)
        with patch('cv2.VideoCapture', return_value=capture):
            result = self.run_video(export_profile='efficient')
        self.assertEqual(result['source_resolution'], [3840, 2160])
        self.assertEqual(result['output_resolution'], [1920, 1080])
        row = self.rows(result)[0]
        self.assertEqual(row['assessment']['source_sha256'], hashlib.sha256(frames[0].tobytes()).hexdigest())
        self.assertEqual(row['detections'][0]['bbox_xyxy'], PERSON['bbox_xyxy'])
        from ppe_compliance.benchmark import verify_export
        self.assertEqual(verify_export(Path(result['run_directory'])/'annotated.mp4')['resolution'], [1920, 1080])

    def test_encoder_finalization_failure_is_not_reported_as_success(self):
        from ppe_compliance.video_export import H264Writer
        close = H264Writer.release

        def fail_after_close(writer):
            close(writer)
            raise RuntimeError('encoder finalization failed')

        with patch.object(H264Writer, 'release', fail_after_close):
            with self.assertRaisesRegex(RuntimeError, 'encoder finalization failed'):
                self.run_video(view='dashboard', export_profile='efficient', max_frames=1)
        self.assertEqual(self.partial()['status'], 'failed')

    def test_processing_cap_keeps_every_frame_with_explicit_hash_and_coordinate_space(self):
        frames = [np.full((1280, 1280, 3), 50+i*10, np.uint8) for i in range(3)]
        capture = FakeCapture(frames, fps=25)
        with patch('cv2.VideoCapture', return_value=capture):
            result = self.run_video(view='dashboard', export_profile='efficient', processing_height=640)
        self.assertEqual(result['source_resolution'], [1280, 1280])
        self.assertEqual(result['processing_resolution'], [640, 640])
        self.assertEqual(result['source_coordinate_scale'], [2, 2])
        self.assertEqual(result['frame_hash_scope'], 'resized_processing_bgr')
        self.assertFalse(result['intentional_frame_skipping'])
        rows = self.rows(result)
        self.assertEqual([r['frame_index'] for r in rows], [0, 1, 2])
        for index, row in enumerate(rows):
            pixels = cv2.resize(frames[index], (640, 640), interpolation=cv2.INTER_AREA)
            self.assertEqual(row['assessment']['source_sha256'], hashlib.sha256(pixels.tobytes()).hexdigest())
            self.assertEqual(row['assessment']['image_width'], 640)
            self.assertEqual(row['assessment']['image_height'], 640)
            self.assertEqual(row['source_coordinate_scale'], [2, 2])
            self.assertAlmostEqual(row['source_time_seconds'], index/25)
            self.assertEqual(row['assessment']['workers'][0]['vest']['confidence'], .95)
        from ppe_compliance.benchmark import verify_export
        self.assertEqual(verify_export(Path(result['run_directory'])/'dashboard.mp4'), {
            'decoded_frames': 3, 'resolution': [1280, 720], 'playback_fps': 25})

    def test_processing_cap_does_not_upscale_small_sources_and_validates_options(self):
        result = self.run_video(max_frames=1, processing_height=1080)
        self.assertEqual(result['processing_resolution'], [640, 640])
        self.assertEqual(result['frame_hash_scope'], 'decoded_source_bgr')
        for value in (0, -1, True, 720.5):
            with self.assertRaises(ValueError):
                self.run_video(processing_height=value)

    def test_720p_processing_and_frame_export_have_matching_encoder_dimensions(self):
        capture = FakeCapture([np.full((2160, 3840, 3), 80, np.uint8)])
        with patch('cv2.VideoCapture', return_value=capture):
            result = self.run_video(export_profile='efficient', processing_height=720)
        self.assertEqual(result['processing_resolution'], [1280, 720])
        self.assertEqual(result['output_resolution'], [1280, 720])
        self.assertEqual(result['processed_frames'], 1)

    def test_output_fps_changes_playback_not_source_timestamps(self):
        result = self.run_video(max_frames=2, output_fps=6)
        self.assertEqual(result["output_fps"], 6)
        self.assertAlmostEqual(self.rows(result)[1]["source_time_seconds"], 1/12)

    def test_tracking_saves_only_transition_events_and_stable_ids(self):
        result=self.run_video(tracking=True)
        rows=self.rows(result)
        ids=[row['assessment']['workers'][0]['track_id'] for row in rows]
        self.assertEqual(set(ids),{'T001'})
        self.assertEqual(rows[-1]['assessment']['workers'][0]['status'],'PPE detected')
        events=[json.loads(line) for line in (Path(result['run_directory'])/'events.jsonl').read_text().splitlines()]
        self.assertEqual(len([e for e in events if e['type']=='status_changed']),1)
        self.assertEqual(len({e['event_id'] for e in events}),len(events))

    def test_camera_disconnect_preserves_partial_data_and_releases_device(self):
        cap = FakeCapture(self.frames[:2])
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "Camera stopped"):
            process_video(0, output_dir=self.output)
        self.assertTrue(cap.released)
        result = self.partial()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["processed_frames"], 2)
        self.assertEqual(result["output_fps"], 10)
        self.assertTrue(all(row["source_time_seconds"] is None for row in self.rows(result)))

    def test_camera_limit_stops_cleanly_without_waiting_for_disconnect(self):
        cap = FakeCapture(self.frames)
        with patch("cv2.VideoCapture", return_value=cap):
            result = process_video(0, max_frames=2, output_dir=self.output)
        self.assertEqual(result["stop_reason"], "max_frames")
        self.assertTrue(cap.released)
        self.assertEqual(result["source_kind"], "webcam")

    def test_camera_controls_apply_and_restore_on_stop(self):
        cap = FakeCapture(self.frames)
        with patch("cv2.VideoCapture", return_value=cap):
            result = process_video(0, max_frames=1, output_dir=self.output,
                                   camera_exposure=-3, camera_brightness=24)
        self.assertEqual(result["camera_controls"]["exposure"]["applied_readback"], -3)
        self.assertEqual(result["camera_controls"]["brightness"]["applied_readback"], 24)
        self.assertEqual(cap.controls[cv2.CAP_PROP_EXPOSURE], -5)
        self.assertEqual(cap.controls[cv2.CAP_PROP_BRIGHTNESS], 0)
        self.assertEqual(result["camera_control_restore_warnings"], [])
        self.assertTrue(cap.released)

    def test_rejected_camera_control_still_restores_and_releases(self):
        cap = FakeCapture(self.frames)
        original_set = cap.set
        cap.set = Mock(side_effect=lambda prop, value: False if value == -3 else original_set(prop, value))
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "driver rejected"):
            process_video(0, output_dir=self.output, camera_exposure=-3)
        self.assertTrue(cap.released)
        self.assertEqual(cap.controls[cv2.CAP_PROP_EXPOSURE], -5)
        self.constructor.assert_not_called()

    def test_camera_controls_are_not_applied_to_video_files(self):
        with self.assertRaisesRegex(ValueError, "only to webcam"):
            self.run_video(camera_exposure=-3)

    def test_windows_auto_falls_back_to_media_foundation(self):
        failed = FakeCapture([], opened=False)
        working = FakeCapture(self.frames)
        with patch("ppe_compliance.video_pipeline.platform.system", return_value="Windows"), \
                patch("cv2.VideoCapture", side_effect=[failed, working]) as factory:
            result = process_video(1, output_dir=self.output, max_frames=1)
        self.assertEqual(factory.call_args_list[0].args, (1,cv2.CAP_DSHOW))
        self.assertEqual(factory.call_args_list[1].args, (1,cv2.CAP_MSMF))
        self.assertTrue(failed.released and working.released)
        self.assertEqual(result["capture_backend_selected"], "msmf")

    def test_explicit_msmf_does_not_try_directshow(self):
        cap = FakeCapture(self.frames)
        with patch("ppe_compliance.video_pipeline.platform.system", return_value="Windows"), \
                patch("cv2.VideoCapture", return_value=cap) as factory:
            result = process_video(1, output_dir=self.output, max_frames=1, camera_backend="msmf")
        factory.assert_called_once_with(1,cv2.CAP_MSMF)
        self.assertEqual(result["capture_backend_requested"], "msmf")

    def test_explicit_failed_backend_does_not_silently_switch(self):
        cap = FakeCapture([], opened=False)
        with patch("ppe_compliance.video_pipeline.platform.system", return_value="Windows"), \
                patch("cv2.VideoCapture", return_value=cap) as factory, self.assertRaisesRegex(RuntimeError,"Cannot open"):
            process_video(1, output_dir=self.output, camera_backend="dshow")
        factory.assert_called_once_with(1,cv2.CAP_DSHOW)
        self.assertTrue(cap.released)

    def test_keyboard_interrupt_releases_capture_and_finalizes_summary(self):
        cap = FakeCapture(self.frames, interrupt_at=1)
        with patch("cv2.VideoCapture", return_value=cap):
            result = self.run_video()
        self.assertTrue(cap.released)
        self.assertEqual(result["stop_reason"], "keyboard_interrupt")
        self.assertEqual(result["processed_frames"], 1)

    def test_preview_keys_close_window_and_capture(self):
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                cap = FakeCapture(self.frames)
                with patch("cv2.VideoCapture", return_value=cap), patch("cv2.namedWindow"), patch("cv2.imshow"), \
                        patch("cv2.waitKey", return_value=key), patch("cv2.getWindowProperty", return_value=1), \
                        patch("cv2.destroyWindow") as close:
                    result = self.run_video(preview=True)
                self.assertEqual(result["stop_reason"], "preview_stop")
                self.assertTrue(cap.released)
                close.assert_called_once()

    def test_closed_preview_window_is_clean_stop_even_if_backend_throws(self):
        with patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1), \
                patch("cv2.getWindowProperty", side_effect=cv2.error("closed")), patch("cv2.destroyWindow"):
            result = self.run_video(preview=True)
        self.assertEqual(result["stop_reason"], "preview_stop")

    def test_headless_preview_failure_has_actionable_error(self):
        cap = FakeCapture(self.frames)
        with patch("cv2.VideoCapture", return_value=cap), patch("cv2.namedWindow", side_effect=cv2.error("headless")), \
                self.assertRaisesRegex(RuntimeError, "no-preview"):
            self.run_video(preview=True)
        self.assertTrue(cap.released)

    def test_unavailable_capture_releases_and_does_not_load_model(self):
        cap = FakeCapture([], opened=False)
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "Cannot open"):
            self.run_video()
        self.assertTrue(cap.released)
        self.constructor.assert_not_called()

    def test_writer_failure_releases_both_resources(self):
        cap = FakeCapture(self.frames)
        writer = Mock(); writer.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer), \
                self.assertRaisesRegex(RuntimeError, "writer"):
            self.run_video()
        self.assertTrue(cap.released)
        writer.release.assert_called_once()
        self.assertEqual(self.partial()["processed_frames"], 0)

    def test_no_readable_frames_is_failure(self):
        cap = FakeCapture([])
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "no readable"):
            self.run_video()
        self.assertTrue(cap.released)

    def test_early_end_is_not_silently_treated_as_complete(self):
        cap = FakeCapture(self.frames[:2], count=5)
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "ended early"):
            self.run_video()
        self.assertEqual(self.partial()["processed_frames"], 2)

    def test_unknown_fps_requires_explicit_playback_rate(self):
        cap = FakeCapture(self.frames, fps=0)
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "output-fps"):
            self.run_video()
        self.assertTrue(cap.released)

    def test_explicit_rate_allows_missing_fps_metadata(self):
        cap = FakeCapture(self.frames, fps=float("nan"), count=0)
        with patch("cv2.VideoCapture", return_value=cap):
            result = self.run_video(output_fps=15)
        self.assertEqual(result["stop_reason"], "end_of_stream_or_decode_failure")
        self.assertIsNone(self.rows(result)[0]["source_time_seconds"])

    def test_dimension_change_fails_before_writing_bad_frame(self):
        cap = FakeCapture([self.frames[0], self.frames[0][:300]])
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "dimensions changed"):
            self.run_video()
        self.assertTrue(cap.released)
        self.assertEqual(self.partial()["processed_frames"], 1)

    def test_odd_dimensions_are_padded_without_cropping_source(self):
        frame = self.frames[0][:639,:639]
        cap = FakeCapture([frame])
        with patch("cv2.VideoCapture", return_value=cap):
            result = self.run_video()
        self.assertEqual(result["source_resolution"], [639,639])
        self.assertEqual(result["output_resolution"], [640,640])
        self.assertEqual(self.rows(result)[0]["assessment"]["image_width"], 639)
        output = cv2.VideoCapture(str(Path(result["run_directory"])/"annotated.mp4"))
        ok, decoded = output.read()
        output.release()
        self.assertTrue(ok)
        self.assertEqual(decoded.shape[:2], (640,640))

    def test_prediction_failure_preserves_earlier_rows(self):
        original = self.detector.detect
        self.detector.detect = Mock(side_effect=[original(self.frames[0], "frame0"), RuntimeError("inference failed")])
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_video()
        self.assertEqual(len(self.rows(self.partial())), 1)

    def test_general_weights_are_rejected_before_frame_processing(self):
        self.detector.model_classes = {0:"person",1:"bus"}
        with self.assertRaisesRegex(RuntimeError, "PPE-trained"):
            self.run_video()
        self.assertEqual(self.detector.calls, 0)

    def test_bad_parameters_fail_before_model_load(self):
        for kwargs in ({"max_frames":0},{"output_fps":0},{"output_fps":float("nan")},
                       {"confidence":0},{"image_size":0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_video(**kwargs)
        self.constructor.assert_not_called()

    def test_frame_detector_reuses_loaded_model(self):
        result = SimpleNamespace(boxes=None, names={0:"person",1:"helmet",2:"vest"})
        model = SimpleNamespace(names=result.names, predict=Mock(return_value=[result]))
        constructor = Mock(return_value=model)
        with patch.dict(sys.modules, {"ultralytics":SimpleNamespace(YOLO=constructor)}):
            detector = FrameDetector("missing-test-only.pt")
            detector.detect(self.frames[0], "first")
            detector.detect(self.frames[1], "second")
        constructor.assert_called_once()
        self.assertEqual(model.predict.call_count, 2)

    def test_cli_video_dispatches_source_and_options(self):
        argv=["ppe","video","--source",str(self.source),"--max-frames","2","--output-dir",str(self.output)]
        with patch("sys.argv",argv),redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(self.partial()["processed_frames"],2)

    def test_cli_camera_accepts_index_zero_and_no_preview(self):
        argv=["ppe","webcam","--camera","0","--no-preview","--max-frames","1","--output-dir",str(self.output),
              "--camera-exposure","-3","--camera-brightness","24"]
        with patch("sys.argv",argv),patch("cv2.VideoCapture",return_value=FakeCapture(self.frames)), \
                redirect_stdout(io.StringIO()):
            main()
        self.assertFalse(self.partial()["preview_enabled"])
        self.assertEqual(self.partial()["camera_controls"]["brightness"]["requested"], 24)

    def test_cli_rejects_invalid_fps_and_camera(self):
        for argv in (["ppe","webcam","--output-fps","nan"],["ppe","webcam","--camera","-1"],
                     ["ppe","webcam","--camera-exposure","nan"]):
            with patch("sys.argv",argv),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                main()
            self.assertEqual(result.exception.code,2)


if __name__ == "__main__":
    unittest.main()
