"""Rebuilt notebook bootstraps fixed source and resolves inputs without upload widgets."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from ppe_compliance.colab_setup import inspect_video, resolve_video, resolve_weights, normalize_selected_path
from test_video_pipeline import FakeCapture

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_colab_quoted_empty_form_values_mean_automatic_selection(self):
        for value in (None, '', ' ', '""', "''", '  ""  ', '\'""\''):
            with self.subTest(value=value):
                self.assertIsNone(normalize_selected_path(value))
        self.assertEqual(normalize_selected_path('"/content/a video.mp4"'), '/content/a video.mp4')
        self.assertEqual(normalize_selected_path("/content/worker's video.mp4"), "/content/worker's video.mp4")

    def test_exact_cell_three_error_value_resolves_existing_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b'selected model'
            with zipfile.ZipFile(root/'ppe-video-gpu-bundle.zip', 'w') as archive:
                archive.writestr('models/best.pt', data)
            result = resolve_weights(root/'project', root, hashlib.sha256(data).hexdigest(), '""')
            self.assertEqual(result.read_bytes(), data)

    def test_quoted_video_paths_and_empty_quoted_video_input_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'clip.mp4').write_bytes(b'video')
            self.assertEqual(resolve_video(root, '""'), root/'clip.mp4')
            self.assertEqual(resolve_video(root, '"clip.mp4"'), root/'clip.mp4')

    def test_old_bundle_supplies_weights_without_restoring_old_code(self):
        with tempfile.TemporaryDirectory() as folder:
            upload = Path(folder)
            project = upload/'ready-project'
            (project/'src/ppe_compliance').mkdir(parents=True)
            detector = project/'src/ppe_compliance/detector.py'
            detector.write_text('fixed source')
            data = b'selected trained weights'
            with zipfile.ZipFile(upload/'ppe-video-gpu-bundle.zip', 'w') as archive:
                archive.writestr('models/best.pt', data)
                archive.writestr('src/ppe_compliance/detector.py', 'old invalid source')
            weights = resolve_weights(project, upload, hashlib.sha256(data).hexdigest())
            self.assertEqual(weights.read_bytes(), data)
            self.assertEqual(detector.read_text(), 'fixed source')

    def test_cached_weights_reused_and_wrong_explicit_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            upload = Path(folder)
            cached = upload/'ppe-video-project/models/best.pt'
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b'original trained model')
            expected = hashlib.sha256(cached.read_bytes()).hexdigest()
            project = upload/'ready-project'
            result = resolve_weights(project, upload, expected)
            self.assertEqual(result.read_bytes(), cached.read_bytes())
            (upload/'wrong.pt').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'WEIGHTS_PATH'):
                resolve_weights(project, upload, expected, 'wrong.pt')

    def test_missing_model_has_files_panel_recovery_instructions(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Files.*Upload'):
                resolve_weights(Path(folder)/'project', folder, '0'*64)

    def test_duplicate_video_upload_is_unambiguous_but_different_clips_need_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'clip.mp4').write_bytes(b'original video')
            (root/'clip-copy.mp4').write_bytes(b'original video')
            self.assertEqual(resolve_video(root).read_bytes(), b'original video')
            (root/'other.mp4').write_bytes(b'another video')
            with self.assertRaisesRegex(ValueError, 'VIDEO_PATH'):
                resolve_video(root)
            self.assertEqual(resolve_video(root, 'other.mp4').name, 'other.mp4')

    def test_existing_runtime_video_reused_and_generated_previews_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cached = root/'ppe-video-input/source.mp4'
            cached.parent.mkdir()
            cached.write_bytes(b'original')
            (root/'preview.mp4').write_bytes(b'output')
            self.assertEqual(resolve_video(root), cached)

    def test_video_preflight_decodes_real_frame_and_checks_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'clip.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12, (320, 240))
            self.assertTrue(writer.isOpened())
            writer.write(np.full((240, 320, 3), 70, np.uint8))
            writer.release()
            self.assertEqual(inspect_video(path), {'fps': 12., 'resolution': [320, 240]})
            capture = FakeCapture([np.zeros((240, 320, 3), np.uint8)], fps=0)
            with patch('cv2.VideoCapture', return_value=capture), self.assertRaisesRegex(ValueError, 'FPS'):
                inspect_video(path)
            self.assertTrue(capture.released)


class ReadyNotebookTests(unittest.TestCase):
    def test_embedded_source_boots_over_stale_imports_and_contains_no_weights(self):
        spec = importlib.util.spec_from_file_location('ready_builder', ROOT/'scripts/build_video_colab.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root/'build-project'
            (project/'src/ppe_compliance').mkdir(parents=True)
            for module in (ROOT/'src/ppe_compliance').glob('*.py'):
                shutil.copy2(module, project/'src/ppe_compliance'/module.name)
            (project/'models').mkdir()
            (project/'models/best.pt').write_bytes(b'fixture weights')
            (project/'README.md').write_text('fixture')
            (project/'pyproject.toml').write_text('fixture')
            notebook, bundle = builder.build(project)
            document = json.loads(notebook.read_text())
            self.assertEqual(notebook.name, 'PPE_Video_GPU_Ready.ipynb')
            self.assertFalse(any('files.upload(' in cell['source'] for cell in document['cells']))
            script = root/'check_bootstrap.py'
            script.write_text('''
import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
notebook, target, old = map(Path, sys.argv[1:])
sys.path.insert(0, str(old))
import ppe_compliance
assert ppe_compliance.OLD_SENTINEL
document = json.loads(notebook.read_text())
cell = next(c['source'] for c in document['cells'] if c['cell_type']=='code' and 'SOURCE_PAYLOAD =' in c['source'])
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
context = dict(WORK=target, sys=sys, Path=Path, hashlib=hashlib, json=json, sha256=sha256)
exec(compile(cell, 'embedded_setup', 'exec'), context)
assert not getattr(context['ppe_compliance'], 'OLD_SENTINEL', False)
assert context['diagnostics']['rejected_count'] == 1
with zipfile.ZipFile(io.BytesIO(base64.b64decode(context['SOURCE_PAYLOAD']))) as archive:
    assert not any(name.endswith('.pt') for name in archive.namelist())
assert not context['PREFLIGHT_PASSED']
print('embedded-source-bootstrap-passed')
''', encoding='utf-8')
            old = root/'old-source'
            (old/'ppe_compliance').mkdir(parents=True)
            (old/'ppe_compliance/__init__.py').write_text('OLD_SENTINEL=True')
            result = subprocess.run([sys.executable, str(script), str(notebook), str(root/'restored'), str(old)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('embedded-source-bootstrap-passed', result.stdout)

            main = next(c['source'] for c in document['cells'] if '#@title 5.' in c['source'])
            with self.assertRaisesRegex(AssertionError, 'smoke test'):
                exec(compile(main, 'benchmark_guard', 'exec'), {})


if __name__ == '__main__':
    unittest.main()
