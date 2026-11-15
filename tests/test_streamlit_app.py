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
@unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/best.pt').is_file(),
                     'Local UI smoke checks require the configured model file')
class StreamlitTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/streamlit_app.py')).run(timeout=30)
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
            self.app.selectbox[0].select('Detail').run()
            self.assertEqual(len(self.app.metric), 0)
            self.assertIn('changed', self.app.info[0].value)

    def test_video_and_live_modes_open(self):
        self.app.radio[0].set_value('Video').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.file_uploader[0].label, 'Choose a video')
        next(r for r in self.app.radio if r.label=='Source').set_value('Live camera').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.button[0].label, 'Open camera monitor')

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


if __name__ == '__main__':
    unittest.main()
