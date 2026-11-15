"""Quality and evaluation contracts; no camera or model download required."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ppe_compliance.evaluation import aggregate_scores, load_manifest, score_workers
from ppe_compliance.interface import assess_image_bytes, camera_command
from ppe_compliance.rendering import draw_assessment, worker_badge, worker_scores, confidence_label
from ppe_compliance.video_pipeline import local_camera_context
from test_assessment import PERSON, HELMET, VEST, assess, report
from test_live_pipeline import PacedCamera


class OverlayTests(unittest.TestCase):
    def test_processing_coordinates_scale_to_native_export_without_mutating_scores(self):
        from copy import deepcopy
        frame = np.full((1280, 1280, 3), 80, np.uint8)
        evidence = assess([PERSON, HELMET, VEST])
        reference = deepcopy(evidence)
        annotated = draw_assessment(frame, evidence, [PERSON, HELMET, VEST])
        self.assertFalse(np.all(annotated[1120, 600] == 80))
        np.testing.assert_array_equal(annotated[700:900, 300:500], np.full((200, 200, 3), 80, np.uint8))
        self.assertEqual(evidence, reference)

    def test_display_sized_overlay_scales_corners_and_preserves_reports(self):
        from copy import deepcopy
        frame = np.full((640, 640, 3), 80, np.uint8)
        evidence = assess([PERSON, HELMET, VEST])
        reference = deepcopy(evidence)
        annotated = draw_assessment(frame, evidence, [PERSON, HELMET, VEST], fit=(320, 320))
        self.assertEqual(annotated.shape, (320, 320, 3))
        self.assertFalse(np.all(annotated[280, 150] == 80))
        np.testing.assert_array_equal(annotated[160:220, 80:120], np.full((60, 40, 3), 80, np.uint8))
        self.assertEqual(evidence, reference)
        np.testing.assert_array_equal(frame, np.full_like(frame, 80))

    def test_scores_come_from_detection_evidence_and_can_be_hidden(self):
        worker = assess([PERSON, HELMET, VEST])['workers'][0]
        self.assertEqual(worker_scores(worker), ('95%', '95%', '95%'))
        self.assertIn('P95% H95% V95%', worker_badge(worker))
        self.assertEqual(worker_badge(worker, False), 'W1  PPE')
        missing = assess([PERSON])['workers'][0]
        self.assertEqual(worker_scores(missing), ('95%', '--', '--'))
        for invalid in (None, True, float('nan'), -1, 1.1):
            self.assertEqual(confidence_label(invalid), '--')

    def test_compact_overlay_preserves_interior_and_source_resolution(self):
        frame = np.full((640, 640, 3), 80, np.uint8)
        annotated = draw_assessment(frame, assess([PERSON, HELMET, VEST]), [PERSON, HELMET, VEST])
        self.assertEqual(annotated.shape, frame.shape)
        np.testing.assert_array_equal(frame, np.full_like(frame, 80))
        np.testing.assert_array_equal(annotated[160:350, 130:270], frame[160:350, 130:270])
        np.testing.assert_array_equal(annotated[-40:], frame[-40:])
        self.assertLess(np.count_nonzero(np.any(annotated != frame, axis=2)), frame.shape[0]*frame.shape[1]*.025)

    def test_dense_and_tiny_inputs_do_not_mutate_or_resize(self):
        for size in (8, 50, 640):
            frame = np.zeros((size, size, 3), np.uint8)
            data = assess([PERSON, HELMET, VEST])
            data['workers'] *= 15
            self.assertEqual(draw_assessment(frame, data, []).shape, frame.shape)


class InterfaceTests(unittest.TestCase):
    def test_invalid_upload_rejected_before_loading_detector(self):
        with patch('ppe_compliance.detector.FrameDetector') as detector:
            for data in (b'', b'not-an-image'):
                with self.assertRaisesRegex(ValueError, 'decoded'):
                    assess_image_bytes(data, 'missing.pt')
            detector.assert_not_called()

    def test_image_exports_lossless_pixels_and_matching_evidence(self):
        data = cv2.imencode('.png', np.full((640, 640, 3), 80, np.uint8))[1].tobytes()
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            result = assess_image_bytes(data, 'test.pt', output_dir=folder)
            run = Path(result['run_directory'])
            assessment = json.loads((run/'compliance.json').read_text())
            self.assertEqual(assessment['source_sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(cv2.imread(str(run/'annotated.png')).shape, (640, 640, 3))
            self.assertTrue((run/'visibility-template.json').is_file())

    def test_camera_command_keeps_paths_and_settings_as_separate_arguments(self):
        command = camera_command(Path('folder with spaces'), 'python.exe', profile='Detail', fps=60)
        self.assertIn('folder with spaces', command[command.index('--weights')+1])
        self.assertEqual(command[command.index('--image-size')+1], '640')
        self.assertEqual(command[command.index('--capture-fps')+1], '60')

    def test_fps_request_reports_driver_readback(self):
        camera = PacedCamera()
        with patch('cv2.VideoCapture', return_value=camera):
            with local_camera_context(0, fps=60) as (_, info):
                self.assertEqual(info['fps_request']['requested'], 60)
                self.assertEqual(info['fps_request']['driver_fps'], 60)
        self.assertTrue(camera.released)


class EvaluationTests(unittest.TestCase):
    def worker(self, status='PPE detected', x=0):
        return {'bbox_xyxy': [x, 0, x+50, 100], 'status': status}

    def test_missed_and_spurious_workers_are_not_dropped(self):
        outcomes = score_workers([self.worker(), self.worker(x=100)],
                                 [self.worker('review required'), self.worker(x=300)])
        scores = aggregate_scores([{'outcomes': outcomes}])
        self.assertEqual(scores['missed_workers'], 1)
        self.assertEqual(scores['spurious_workers'], 1)
        self.assertEqual(scores['review_rate'], .5)
        self.assertEqual(scores['correct_status_per_expected_worker'], 0)
        self.assertEqual(scores['per_status']['PPE detected']['recall'], 0)

    def test_matching_is_one_to_one_and_independent_of_status(self):
        outcomes = score_workers([self.worker()], [self.worker('review required'), self.worker()])
        self.assertEqual(sum(o['expected'] == 'spurious' for o in outcomes), 1)
        self.assertEqual(len(outcomes), 2)

    def test_zero_support_is_null_not_perfect_accuracy(self):
        scores = aggregate_scores([{'outcomes': []}])
        self.assertIsNone(scores['status_accuracy_on_matches'])
        self.assertIsNone(scores['worker_detection_recall'])

    def test_corrupt_or_overlapping_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'image.jpg').write_bytes(b'frozen-pixels')
            digest = hashlib.sha256(b'frozen-pixels').hexdigest()
            data = {'dataset_id': 'test', 'independent': True, 'annotation_notes': 'reviewed fixture',
                    'items': [{'image': 'image.jpg', 'sha256': digest, 'source_group': 'site-day', 'workers': []}]}
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(data))
            self.assertEqual(load_manifest(manifest)['dataset_id'], 'test')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_manifest(manifest, [root])
            data['items'][0]['sha256'] = 'wrong'
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_manifest(manifest)

    def test_bad_boxes_and_thresholds_rejected(self):
        for threshold in (0, -1, 2, float('nan'), True):
            with self.assertRaises(ValueError):
                score_workers([], [], threshold)
        bad = self.worker()
        bad['bbox_xyxy'][2] = -1
        with self.assertRaises(ValueError):
            score_workers([bad], [])


if __name__ == '__main__':
    unittest.main()
