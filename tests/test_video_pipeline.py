"""Video I/O, cleanup and timing contracts using deterministic fake predictions."""

from contextlib import redirect_stdout, redirect_stderr
import io
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

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.count

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

    def test_output_fps_changes_playback_not_source_timestamps(self):
        result = self.run_video(max_frames=2, output_fps=6)
        self.assertEqual(result["output_fps"], 6)
        self.assertAlmostEqual(self.rows(result)[1]["source_time_seconds"], 1/12)

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
        argv=["ppe","webcam","--camera","0","--no-preview","--max-frames","1","--output-dir",str(self.output)]
        with patch("sys.argv",argv),patch("cv2.VideoCapture",return_value=FakeCapture(self.frames)), \
                redirect_stdout(io.StringIO()):
            main()
        self.assertFalse(self.partial()["preview_enabled"])

    def test_cli_rejects_invalid_fps_and_camera(self):
        for argv in (["ppe","webcam","--output-fps","nan"],["ppe","webcam","--camera","-1"]):
            with patch("sys.argv",argv),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                main()
            self.assertEqual(result.exception.code,2)


if __name__ == "__main__":
    unittest.main()
