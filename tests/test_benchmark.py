"""Benchmark timing/exports and portable Colab bundle contracts, without a GPU."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from ppe_compliance.benchmark import benchmark_video, summarize_timings
from test_video_pipeline import FakeDetector

ROOT = Path(__file__).resolve().parents[1]


def timings(seconds):
    return {'timing_seconds': {name: seconds for name in
            ('decode', 'frame_setup', 'inference', 'assessment_and_hash', 'render', 'encode_submission')}}


class TimingTests(unittest.TestCase):
    def test_cold_inference_excluded_only_from_steady_metrics(self):
        result = summarize_timings([timings(5), timings(.1), timings(.2)], warmup_frames=1)
        self.assertEqual(result['first_inference_ms'], 5000)
        self.assertEqual(result['steady_frames'], 2)
        self.assertAlmostEqual(result['mean_stage_ms']['inference'], 150)
        self.assertAlmostEqual(result['inference_only_fps'], 1/.15)

    def test_short_clip_has_no_invented_steady_rate(self):
        result = summarize_timings([timings(.1)], warmup_frames=5)
        self.assertEqual(result['steady_frames'], 0)
        self.assertIsNone(result['inference_only_fps'])
        self.assertIsNone(result['mean_stage_ms']['render'])


class BenchmarkTests(unittest.TestCase):
    def test_repeated_direct_exports_preserve_evidence_without_an_extra_transcode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, weights = root/'source.avi', root/'best.pt'
            weights.write_bytes(b'weights fixture')
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 12, (640, 640))
            self.assertTrue(writer.isOpened())
            for value in (60, 80):
                writer.write(np.full((640, 640, 3), value, np.uint8))
            writer.release()
            detector = FakeDetector()
            detector.weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            with patch('ppe_compliance.video_pipeline.FrameDetector', return_value=detector), \
                    patch('ppe_compliance.interface.browser_video', side_effect=AssertionError('Unexpected transcode')):
                result = benchmark_video(source, weights, root/'benchmark', devices=('cpu',),
                    views=('dashboard',), export_profiles=('efficient',), repeats=2,
                    max_frames=2, warmup_frames=1)
            self.assertTrue(result['evidence_matches_between_exports'])
            self.assertEqual(len(result['runs']), 2)
            for run in result['runs']:
                self.assertEqual(run['h264_conversion_seconds'], 0)
                self.assertEqual(run['verification']['decoded_frames'], 2)
                self.assertEqual(run['verification']['resolution'], [1280, 720])
                self.assertIn('decoded_frame_sha256', run['mean_assessment_detail_ms'])

    def test_identical_cpu_jobs_export_both_views_and_preserve_timing_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source.avi'
            weights = root/'best.pt'
            weights.write_bytes(b'controlled weights fixture')
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 12, (640, 640))
            self.assertTrue(writer.isOpened())
            for value in range(3):
                writer.write(np.full((640, 640, 3), 60+value, np.uint8))
            writer.release()
            detector = FakeDetector()
            detector.weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            with patch('ppe_compliance.video_pipeline.FrameDetector', return_value=detector):
                result = benchmark_video(source, weights, root/'benchmark', devices=('cpu',),
                    repeats=1, max_frames=3, warmup_frames=1, browser_preview=False)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual([r['frames'] for r in result['runs']], [3, 3])
            self.assertEqual([r['verification']['resolution'] for r in result['runs']], [[640, 640], [1280, 720]])
            for run in result['runs']:
                self.assertEqual(run['source_fps'], 12)
                self.assertEqual(run['output_fps'], 12)
                self.assertEqual(run['steady_frames'], 2)
                self.assertGreaterEqual(run['job_seconds'], run['model_load_seconds'])
                self.assertEqual(run['verification']['decoded_frames'], 3)
            persisted = json.loads((root/'benchmark/benchmark.json').read_text())
            self.assertEqual(persisted['status'], 'completed')
            with self.assertRaises(FileExistsError):
                benchmark_video(source, weights, root/'benchmark', devices=('cpu',), repeats=1)

    def test_missing_gpu_is_rejected_before_output_or_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'clip.mp4').write_bytes(b'fixture')
            (root/'best.pt').write_bytes(b'fixture')
            with patch('torch.cuda.is_available', return_value=False), \
                    patch('ppe_compliance.video_pipeline.process_video') as process:
                with self.assertRaisesRegex(ValueError, 'CUDA GPU'):
                    benchmark_video(root/'clip.mp4', root/'best.pt', root/'output')
                process.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_user_interrupt_stops_matrix_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'clip.mp4').write_bytes(b'fixture')
            (root/'best.pt').write_bytes(b'fixture')
            with patch('ppe_compliance.benchmark.process_video', return_value={'stop_reason': 'keyboard_interrupt'}) as process:
                with self.assertRaises(KeyboardInterrupt):
                    benchmark_video(root/'clip.mp4', root/'best.pt', root/'output', devices=('cpu',))
            process.assert_called_once()
            report = json.loads((root/'output/benchmark.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['runs'], [])


class BundleTests(unittest.TestCase):
    def test_bundle_pins_current_snapshot_and_excludes_unrelated_files(self):
        spec = importlib.util.spec_from_file_location('build_video_colab', ROOT/'scripts/build_video_colab.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'src/ppe_compliance').mkdir(parents=True)
            (root/'models').mkdir()
            for name, content in [('src/ppe_compliance/benchmark.py', 'pass'),
                                  ('src/ppe_compliance/__init__.py', ''),
                                  ('pyproject.toml', 'test'), ('README.md', 'test'),
                                  ('models/best.pt', 'controlled model fixture'),
                                  ('.env', 'must not be bundled'), ('personal.mp4', 'must not be bundled')]:
                (root/name).write_text(content)
            notebook, bundle = module.build(root)
            first_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
            data = json.loads(notebook.read_text())
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(cell['source'])
                    self.assertEqual(cell['outputs'], [])
            self.assertIn(first_hash, notebook.read_text())
            with zipfile.ZipFile(bundle) as archive:
                self.assertNotIn('.env', archive.namelist())
                self.assertNotIn('personal.mp4', archive.namelist())
                manifest = json.loads(archive.read('bundle-manifest.json'))
                for name, digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            # Rebuilding unchanged files preserves the matching notebook/bundle pair.
            module.build(root)
            self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(), first_hash)


if __name__ == '__main__':
    unittest.main()
