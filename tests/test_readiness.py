"""Regression checks for presentation audit findings; no trained model needed."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ppe_compliance.detector import FrameDetector, validate_inference_options
from ppe_compliance.interface import (assess_image_bytes, browser_video, camera_command,
                                      decode_image_bytes, prepare_browser_video, recorded_review_source)
from ppe_compliance.workspace import status_html
from test_assessment import PERSON, HELMET, VEST, report


class ReadinessTests(unittest.TestCase):
    def image(self):
        return cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1].tobytes()

    def test_public_image_api_rejects_invalid_settings_before_model_load(self):
        for confidence, size in [(True, 640), (float('nan'), 640), (float('inf'), 640),
                                 (0, 640), ('0.5', 640), (.5, False), (.5, 0), (.5, 3.5)]:
            with self.subTest(confidence=confidence, size=size), patch('ppe_compliance.detector.FrameDetector') as detector:
                with self.assertRaises(ValueError):
                    assess_image_bytes(self.image(), 'missing.pt', size, confidence)
                detector.assert_not_called()

    def test_detector_options_are_checked_before_yolo_import_or_load(self):
        with self.assertRaises(ValueError):
            FrameDetector('missing.pt', confidence=-.5)
        validate_inference_options(.25, 416)

    def test_oversized_image_is_rejected_before_opencv_allocation(self):
        data = self.image()
        with patch('ppe_compliance.interface.MAX_IMAGE_PIXELS', 100), patch('cv2.imdecode') as decode:
            with self.assertRaisesRegex(ValueError, 'megapixels'):
                decode_image_bytes(data)
            decode.assert_not_called()

    def test_image_byte_limit_checked_before_decode(self):
        with patch('ppe_compliance.interface.MAX_UPLOAD_BYTES', 2), patch('cv2.imdecode') as decode:
            with self.assertRaisesRegex(ValueError, '200 MB'):
                decode_image_bytes(b'too large')
            decode.assert_not_called()

    def test_corrupt_image_never_calls_auto_installing_pillow_opener(self):
        for data in [b'corrupt', b'\xff\xd8\xffcorrupt', b'\x89PNG\r\n\x1a\ncorrupt']:
            with self.subTest(data=data), patch('PIL.Image.open') as opener:
                with self.assertRaises(ValueError):
                    decode_image_bytes(data)
                opener.assert_not_called()

    def test_failed_image_export_does_not_leave_a_completed_run(self):
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            with patch('cv2.imwrite', return_value=False), self.assertRaises(OSError):
                assess_image_bytes(self.image(), 'fixture.pt', output_dir=folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_report_write_failure_removes_staged_image(self):
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')), self.assertRaisesRegex(OSError, 'disk full'):
                assess_image_bytes(self.image(), 'fixture.pt', output_dir=folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_unavailable_device_assertion_becomes_actionable_error(self):
        detector = FrameDetector.__new__(FrameDetector)
        detector.model = Mock()
        detector.model.predict.side_effect = AssertionError('CUDA unavailable')
        detector.confidence, detector.image_size, detector.device = .25, 416, '0'
        with self.assertRaisesRegex(ValueError, 'Try device cpu'):
            detector.detect(np.zeros((32, 32, 3), np.uint8), 'fixture')

    def test_camera_launcher_forwards_user_confidence(self):
        command = camera_command(Path('workspace with spaces'), 'python.exe', confidence=.7)
        self.assertEqual(command[command.index('--confidence')+1], '0.7')

    def test_direct_preview_requires_nonempty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = {'run_directory': folder, 'browser_video_file': 'dashboard.mp4'}
            for exists in [False, True]:
                if exists:
                    (Path(folder)/'dashboard.mp4').touch()
                with self.assertRaisesRegex(OSError, 'missing or empty'):
                    prepare_browser_video(summary)

    def test_legacy_conversion_checks_actual_encoder_output(self):
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.run', return_value=SimpleNamespace(returncode=0)):
            source = Path(folder)/'annotated.mp4'
            source.write_bytes(b'fixture')
            with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
                browser_video(source)

    def test_review_upload_limit_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.interface.MAX_UPLOAD_BYTES', 2):
            with self.assertRaisesRegex(ValueError, '200 MB'):
                recorded_review_source(b'large', '.mp4', folder)
            self.assertFalse((Path(folder)/'review-inputs').exists())

    def test_worker_status_html_escapes_untrusted_text(self):
        result = status_html('<script>bad</script>')
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)


if __name__ == '__main__':
    unittest.main()
