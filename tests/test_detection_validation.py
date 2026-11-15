"""Malformed model predictions must not abort video or corrupt evidence indices."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from ppe_compliance.compliance import assess_detections
from ppe_compliance.detector import FrameDetector, validated_model_detections
from ppe_compliance.video_pipeline import process_video
from test_assessment import PERSON, HELMET, VEST, report

ROOT = Path(__file__).resolve().parents[1]


def controlled_detector(items):
    """Use actual Ultralytics Results slicing without loading or downloading weights."""
    detector = FrameDetector.__new__(FrameDetector)
    detector.weights = 'controlled.pt'
    detector.weights_sha256 = 'a'*64
    detector.confidence = .25
    detector.image_size = 640
    detector.device = 'cpu'
    detector.model_classes = {0: 'person', 1: 'helmet', 2: 'vest'}
    boxes = torch.tensor([[*item['bbox_xyxy'], item['confidence'], item['class_id']] for item in items])
    detector.model = SimpleNamespace(predict=lambda **kwargs: [Results(
        kwargs['source'], path=str(kwargs['source'].shape), names=detector.model_classes, boxes=boxes.clone())])
    return detector


class ValidationTests(unittest.TestCase):
    def test_invalid_geometry_and_scores_rejected_without_fabricating_boxes(self):
        cases = [([10, 10, 10, 30], .9), ([30, 10, 10, 40], .9),
                 ([10, 30, 40, 10], .9), ([0, 0, float('nan'), 30], .9),
                 ([0, 0, 30, float('inf')], .9), ([-1, 0, 30, 30], .9),
                 ([0, 0, 641, 30], .9), ([0, 0, 30, 30], float('nan')),
                 ([0, 0, 30, 30], 1.2)]
        for box, confidence in cases:
            invalid = {**HELMET, 'bbox_xyxy': box, 'confidence': confidence}
            accepted, validation = validated_model_detections([PERSON, invalid, VEST], 640, 640)
            self.assertEqual([d['model_detection_index'] for d in accepted], [0, 2])
            self.assertEqual(accepted[0]['bbox_xyxy'], PERSON['bbox_xyxy'])
            self.assertEqual(validation['rejected_count'], 1)
            self.assertEqual(validation['rejected_detections'][0]['model_detection_index'], 1)
            json.dumps(validation, allow_nan=False)

    def test_adapter_handles_bad_sixth_detection_and_keeps_result_report_aligned(self):
        items = [PERSON, HELMET, VEST, deepcopy(HELMET), deepcopy(VEST),
                 {**HELMET, 'bbox_xyxy': [640, 0, 640, 50]}]
        result, raw = controlled_detector(items).detect(np.zeros((640, 640, 3), np.uint8), 'fixture')
        self.assertEqual(len(result.boxes), 5)
        self.assertEqual(result.plot().shape, (640, 640, 3))
        self.assertEqual(len(raw['detections']), 5)
        self.assertEqual(raw['detection_validation']['rejected_detections'][0]['model_detection_index'], 5)
        assessment = assess_detections(raw, 'fixture-hash')
        self.assertEqual(assessment['detection_validation']['rejected_count'], 1)
        worker = assessment['workers'][0]
        self.assertEqual(raw['detections'][worker['person_detection_id']]['class_name'], 'person')
        self.assertEqual(raw['detections'][worker['helmet']['detection_id']]['class_name'], 'helmet')
        self.assertEqual(raw['detections'][worker['vest']['detection_id']]['class_name'], 'vest')

    def test_all_bad_results_become_empty_evidence_not_an_exception(self):
        result, raw = controlled_detector([{**PERSON, 'bbox_xyxy': [5, 5, 5, 5]}]).detect(
            np.zeros((640, 640, 3), np.uint8), 'fixture')
        self.assertEqual(len(result.boxes), 0)
        self.assertEqual(raw['detections'], [])
        self.assertEqual(assess_detections(raw, 'fixture')['summary']['workers'], 0)

    def test_rejected_helmet_does_not_confirm_ppe_and_association_stays_strict(self):
        invalid = {**HELMET, 'bbox_xyxy': [200, 80, 200, 120]}
        _, raw = controlled_detector([PERSON, invalid, VEST]).detect(
            np.zeros((640, 640, 3), np.uint8), 'fixture')
        assessment = assess_detections(raw, 'fixture')
        self.assertEqual(assessment['workers'][0]['status'], 'review required')
        with self.assertRaisesRegex(ValueError, 'Invalid bounding box'):
            assess_detections(report([PERSON, invalid, VEST]), 'fixture')

    def test_video_keeps_all_frames_and_audits_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 12, (640, 640))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((640, 640, 3), 70, np.uint8))
            writer.release()
            detector = controlled_detector([PERSON, HELMET, VEST, {**HELMET, 'bbox_xyxy': [640, 0, 640, 60]}])
            with patch('ppe_compliance.video_pipeline.FrameDetector', return_value=detector):
                summary = process_video(source, output_dir=root/'outputs', tracking=True)
            self.assertEqual(summary['processed_frames'], 3)
            self.assertEqual(summary['detection_validation'], {'rejected_count': 3, 'frames_with_rejections': 3})
            rows = [json.loads(line) for line in (Path(summary['run_directory'])/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row['assessment']['detection_validation']['rejected_count'], 1)
                self.assertEqual(len(row['detections']), 3)
            capture = cv2.VideoCapture(str(Path(summary['run_directory'])/'annotated.mp4'))
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            capture.release()
            self.assertEqual(decoded, 3)

    def test_existing_colab_can_be_repaired_without_replacing_weights_or_video(self):
        spec = importlib.util.spec_from_file_location('build_video_repair', ROOT/'scripts/build_video_colab.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = builder.build_repair(destination=root/'download')
            namespace = {'__name__': 'repair_test'}
            exec(compile(script.read_text(), str(script), 'exec'), namespace)
            project = root/'existing-project'
            (project/'src/ppe_compliance').mkdir(parents=True)
            (project/'pyproject.toml').write_text('fixture')
            (project/'src/ppe_compliance/detector.py').write_text('old source')
            (project/'best.pt').write_bytes(b'keep weights')
            (project/'source.mp4').write_bytes(b'keep video')
            namespace['apply_fix'](project)
            self.assertEqual((project/'best.pt').read_bytes(), b'keep weights')
            self.assertEqual((project/'source.mp4').read_bytes(), b'keep video')
            self.assertEqual((project/'src/ppe_compliance/detector.py').read_bytes(),
                             (ROOT/'src/ppe_compliance/detector.py').read_bytes())
            self.assertTrue((project/'colab-box-fix.json').is_file())


if __name__ == '__main__':
    unittest.main()
