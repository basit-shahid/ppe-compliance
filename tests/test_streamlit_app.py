"""Optional UI interaction tests. Install .[ui] to run these."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

from ppe_compliance.interface import assess_image_bytes
from test_assessment import PERSON, HELMET, VEST, report


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Optional Streamlit UI is not installed')
class StreamlitTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        # Navigation and mocked service tests must run on fresh clones without
        # distributing actual trained weights. No unmocked inference is invoked.
        weights = Path(__file__).resolve().parents[1]/'models/best.pt'
        original = Path.is_file
        model_present = patch.object(Path, 'is_file', lambda path: True if path == weights else original(path))
        model_present.start()
        self.addCleanup(model_present.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/streamlit_app.py'), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def upload(self, data):
        if not hasattr(self.app.file_uploader[0], 'set_value'):
            self.skipTest('Upload interactions require newer Streamlit AppTest support')
        self.app.file_uploader[0].set_value(('sample.png', data, 'image/png')).run()

    def test_image_action_disabled_until_upload(self):
        self.assertTrue(self.app.button[0].disabled)

    def test_bad_file_displays_error_without_crashing(self):
        self.upload(b'corrupt-image')
        self.app.button[0].click().run(timeout=30)
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertNotIn('result', self.app.session_state)

    def test_result_and_downloads_survive_rerun_and_settings_hide_old_results(self):
        data = cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1].tobytes()
        self.upload(data)
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            result = assess_image_bytes(data, 'fixture.pt', output_dir=folder)
            with patch('ppe_compliance.interface.assess_image_bytes', return_value=result):
                self.app.button[0].click().run(timeout=30)
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.metric[0].value, '1')
            self.app.run()
            self.assertEqual(self.app.metric[0].value, '1')
            next(s for s in self.app.selectbox if s.label == 'Processing profile').select('Detail').run()
            self.assertEqual(len(self.app.metric), 0)
            self.assertIn('changed', self.app.info[0].value)

    def test_video_and_live_modes_open(self):
        self.app.radio[0].set_value('Video').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.file_uploader[0].label, 'Choose a video')
        next(r for r in self.app.radio if r.label=='Source').set_value('Live camera').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.button[0].label, 'Start live camera')
        self.assertEqual(next(r for r in self.app.radio if r.label=='Preview location').value,'In this browser')

    def test_project_guide_and_validation_navigation(self):
        next(r for r in self.app.radio if r.label == 'Workspace').set_value('Project guide').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, 'Inside the pipeline')
        self.assertEqual(len(self.app.get('download_button')), 4)
        next(r for r in self.app.radio if r.label == 'Workspace').set_value('Validation').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.metric[2].value, '90.08%')
        self.assertTrue(any('Independent final evaluation' in w.value for w in self.app.warning))

    def test_camera_launch_uses_confidence_setting(self):
        next(r for r in self.app.radio if r.label == 'Source').set_value('Live camera').run()
        next(r for r in self.app.radio if r.label == 'Preview location').set_value('Desktop window').run()
        next(s for s in self.app.slider if s.label == 'Detection confidence').set_value(.7).run()
        process = Mock()
        process.poll.return_value = None
        with patch('subprocess.Popen', return_value=process) as launch:
            self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--confidence')+1], '0.7')

    def test_deleted_image_export_does_not_crash_persistent_result(self):
        self.upload(cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1].tobytes())
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            result = assess_image_bytes(self.app.file_uploader[0].value.getvalue(), 'fixture.pt', output_dir=folder)
            with patch('ppe_compliance.interface.assess_image_bytes', return_value=result):
                self.app.button[0].click().run()
            (Path(result['run_directory'])/'annotated.png').unlink()
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any('moved or removed' in w.value for w in self.app.warning))

    def test_included_sample_enables_analysis_and_keeps_result_after_filter_change(self):
        next(c for c in self.app.checkbox if c.label == 'Use the included presentation sample').check().run()
        self.assertFalse(self.app.button[0].disabled)
        data = (Path(__file__).resolve().parents[1]/'app/assets/presentation-sample.jpg').read_bytes()
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            result = assess_image_bytes(data, 'fixture.pt', output_dir=folder)
            with patch('ppe_compliance.interface.assess_image_bytes', return_value=result) as assess:
                self.app.button[0].click().run()
                next(s for s in self.app.selectbox if s.label == 'Filter worker evidence').select('review required').run()
                self.assertEqual(assess.call_count, 1)
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.metric[0].value, '1')

    def test_new_upload_replaces_previously_selected_sample(self):
        next(c for c in self.app.checkbox if c.label == 'Use the included presentation sample').check().run()
        data = cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1].tobytes()
        self.upload(data)
        sample = next(c for c in self.app.checkbox if c.label == 'Use the included presentation sample')
        self.assertFalse(sample.value)
        self.assertTrue(sample.disabled)
        with tempfile.TemporaryDirectory() as folder, patch('ppe_compliance.detector.FrameDetector') as detector:
            detector.return_value.detect.return_value = (None, report([PERSON, HELMET, VEST]))
            result = assess_image_bytes(data, 'fixture.pt', output_dir=folder)
            with patch('ppe_compliance.interface.assess_image_bytes', return_value=result) as assess:
                self.app.button[0].click().run()
                self.assertEqual(assess.call_args.args[0], data)
            self.assertFalse(self.app.exception)

    def test_video_uses_monitor_dashboard_and_confidence_scores_by_default(self):
        self.app.radio[0].set_value('Video').run()
        next(r for r in self.app.radio if r.label=='Video review').set_value('Analyze every frame').run()
        if not hasattr(self.app.file_uploader[0], 'set_value'):
            self.skipTest('Upload interaction is unavailable in this Streamlit version')
        self.app.file_uploader[0].set_value(('sample.mp4', b'fixture', 'video/mp4')).run()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'dashboard.mp4').write_bytes(b'preview fixture')
            summary = {'run_directory': str(run), 'processed_frames': 3,
                       'end_to_end_processing_fps': 5, 'output_fps': 30, 'stop_reason': 'end_of_stream',
                       'presentation_layout': 'dashboard', 'browser_video_file': 'dashboard.mp4'}
            with patch('ppe_compliance.video_pipeline.process_video', return_value=summary) as process, \
                    patch('ppe_compliance.interface.browser_video') as convert:
                self.app.button[0].click().run(timeout=30)
            self.assertFalse(self.app.exception)
            self.assertEqual(process.call_args.kwargs['view'], 'dashboard')
            self.assertTrue(process.call_args.kwargs['show_scores'])
            self.assertEqual(process.call_args.kwargs['export_profile'], 'efficient')
            self.assertEqual(process.call_args.kwargs['processing_height'], 1080)
            convert.assert_not_called()

    def test_smooth_review_launches_a_persistent_source_and_separate_desktop_process(self):
        self.app.radio[0].set_value('Video').run()
        self.assertEqual(next(r for r in self.app.radio if r.label=='Video review').value,'Smooth live review')
        self.assertEqual(next(s for s in self.app.selectbox if s.label=='Video processing').value,'Preview · 540p')
        next(r for r in self.app.radio if r.label=='Preview location').set_value('Desktop window').run()
        if not hasattr(self.app.file_uploader[0],'set_value'):
            self.skipTest('Upload interaction unavailable')
        self.app.file_uploader[0].set_value(('sample.mp4',b'fixture','video/mp4')).run()
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'clip with spaces.mp4';source.write_bytes(b'fixture')
            process=Mock();process.poll.return_value=None
            with patch('ppe_compliance.interface.recorded_review_source',return_value=source), \
                    patch('subprocess.Popen',return_value=process) as launch, \
                    patch('ppe_compliance.video_pipeline.process_video') as full:
                self.app.button[0].click().run(timeout=30)
            self.assertFalse(self.app.exception)
            full.assert_not_called()
            command=launch.call_args.args[0]
            self.assertIn('--live-preview',command)
            self.assertIn(str(source),command)
            self.assertEqual(command[command.index('--processing-height')+1],'540')
            self.assertTrue(source.is_file())

    def test_browser_camera_start_and_stop_are_visible_and_forward_settings(self):
        from ppe_compliance.browser_preview import BrowserPreview, write_status
        next(r for r in self.app.radio if r.label=='Source').set_value('Live camera').run()
        with tempfile.TemporaryDirectory() as folder:
            process=Mock();process.poll.return_value=None
            job=BrowserPreview(folder,process,'camera')
            write_status(folder,status='running',message='Fixture preview running')
            with patch('ppe_compliance.browser_preview.start_browser_preview',return_value=job) as start:
                self.app.button[0].click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(start.call_args.kwargs['kind'],'camera')
            self.assertEqual(start.call_args.kwargs['confidence'],.25)
            self.assertTrue(next(b for b in self.app.button if b.label=='Start live camera').disabled)
            next(b for b in self.app.button if b.label=='Stop live preview').click().run()
            self.assertFalse(self.app.exception)
            self.assertTrue((Path(folder)/'stop').is_file())

    def test_browser_recorded_review_is_default_and_exposes_worker_errors(self):
        from ppe_compliance.browser_preview import BrowserPreview, write_status
        self.app.radio[0].set_value('Video').run()
        self.assertEqual(next(r for r in self.app.radio if r.label=='Preview location').value,'In this browser')
        if not hasattr(self.app.file_uploader[0],'set_value'):
            self.skipTest('Upload interactions unavailable')
        self.app.file_uploader[0].set_value(('sample.mp4',b'fixture','video/mp4')).run()
        with tempfile.TemporaryDirectory() as folder:
            process=Mock();process.poll.return_value=1
            job=BrowserPreview(folder,process,'video')
            write_status(folder,status='failed',message='fixture camera or decode failure')
            source=Path(folder)/'video.mp4';source.write_bytes(b'fixture')
            with patch('ppe_compliance.browser_preview.start_browser_preview',return_value=job) as start, \
                    patch('ppe_compliance.interface.recorded_review_source',return_value=source):
                self.app.button[0].click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(start.call_args.kwargs['kind'],'video')
            self.assertEqual(start.call_args.kwargs['processing_height'],540)
            self.assertTrue(any('decode failure' in e.value for e in self.app.error))


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Optional Streamlit UI is not installed')
class MissingModelTests(unittest.TestCase):
    def test_missing_weights_shows_recovery_and_keeps_project_guide_available(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        original = Path.is_file
        with patch.object(Path, 'is_file', lambda path: False if path == root/'models/best.pt' else original(path)):
            app = AppTest.from_file(str(root/'app/streamlit_app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('model is missing' in error.value for error in app.error))
            next(r for r in app.radio if r.label == 'Workspace').set_value('Project guide').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, 'Inside the pipeline')


if __name__ == '__main__':
    unittest.main()
