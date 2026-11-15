"""Finished playback is paced independently from model inference."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ppe_compliance.cli import main
from ppe_compliance.playback import play_recorded_output
from test_video_pipeline import FakeCapture


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root/'dashboard.mp4').write_bytes(b'fixture')
        self.summary = {'run_directory':str(self.root),'dashboard_file':'dashboard.mp4'}
        self.capture = FakeCapture([np.full((16,16,3), n, np.uint8) for n in range(4)],fps=25)
        self.clock = 0.
        self.shown = []

    def wait(self, milliseconds):
        self.clock += milliseconds/1000
        return -1

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(patch('cv2.VideoCapture', return_value=self.capture))
        stack.enter_context(patch('cv2.namedWindow'))
        stack.enter_context(patch('cv2.resizeWindow'))
        stack.enter_context(patch('cv2.imshow', side_effect=lambda _,frame:self.shown.append(int(frame[0,0,0]))))
        stack.enter_context(patch('cv2.getWindowProperty', return_value=1))
        stack.enter_context(patch('cv2.waitKey', side_effect=self.wait))
        stack.enter_context(patch('cv2.destroyWindow'))
        stack.enter_context(patch('ppe_compliance.playback.perf_counter', side_effect=lambda:self.clock))
        return stack

    def test_frames_remain_in_order_and_last_frame_gets_its_full_period(self):
        with self.mocks(): result=play_recorded_output(self.summary)
        self.assertEqual(self.shown, [0,1,2,3])
        self.assertEqual(result['displayed_frames'],4)
        self.assertAlmostEqual(result['wall_seconds'],4/25,delta=.002)
        self.assertEqual(result['pacing_resets'],0)
        self.assertTrue(self.capture.released)

    def test_escape_stops_and_cleans_up_without_reprocessing(self):
        with self.mocks(), patch('cv2.waitKey', return_value=27):
            result=play_recorded_output(self.summary)
        self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['displayed_frames'],1)
        self.assertTrue(self.capture.released)

    def test_slow_decode_reanchors_without_skipping_source_frames(self):
        original_read=self.capture.read
        def slow_read():
            self.clock += .12
            return original_read()
        with self.mocks(),patch.object(self.capture,'read',side_effect=slow_read):
            result=play_recorded_output(self.summary)
        self.assertEqual(self.shown,[0,1,2,3])
        self.assertGreater(result['pacing_resets'],0)
        self.assertFalse(result['intentional_frame_skipping'])

    def test_failed_window_and_early_decode_release_the_capture(self):
        with self.mocks(),patch('cv2.namedWindow',side_effect=cv2.error('headless')):
            with self.assertRaisesRegex(RuntimeError,'media player'):
                play_recorded_output(self.summary)
        self.assertTrue(self.capture.released)
        self.capture=FakeCapture([np.zeros((16,16,3),np.uint8)],fps=25,count=4)
        with self.mocks(),self.assertRaisesRegex(RuntimeError,'ended early'):
            play_recorded_output(self.summary)
        self.assertTrue(self.capture.released)

    def test_cli_play_starts_after_export_and_disables_processing_preview(self):
        source=self.root/'source.mp4';source.write_bytes(b'fixture')
        summary={**self.summary,'status':'completed','stop_reason':'end_of_stream','processed_frames':4,
                 'output_fps':25,'end_to_end_processing_fps':2,'note':'fixture'}
        with patch.object(sys,'argv',['ppe','video','--source',str(source),'--play']), \
                patch('ppe_compliance.video_pipeline.process_video',return_value=summary) as process, \
                patch('ppe_compliance.playback.play_recorded_output',return_value={'status':'completed','displayed_frames':4}) as play, \
                redirect_stdout(io.StringIO()):
            main()
        self.assertFalse(process.call_args.args[7])
        play.assert_called_once_with(summary)


if __name__=='__main__':unittest.main()
