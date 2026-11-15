"""Recorded source cadence and selected-frame evidence remain independent."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from time import sleep
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

from ppe_compliance.cli import main
from ppe_compliance.recorded_preview import review_recorded_video, ScaledVideoReader, PrefetchedReader, prepare_playback_source
from ppe_compliance.preview_ui import DashboardRenderer
from test_assessment import PERSON, HELMET, VEST, report


class MemoryReader:
    def __init__(self, source, resolution, run):
        self.resolution = resolution
        self.index = 0
        self.released = False

    def read(self):
        if self.index == 30: return False,None
        width,height = self.resolution
        image = np.full((height,width,3),self.index,np.uint8)
        self.index += 1
        return True,image

    def release(self): self.released=True


class SlowModel:
    weights_sha256 = 'b'*64
    model_classes = {0:'person',1:'helmet',2:'vest'}

    def detect(self, frame, source):
        sleep(.16)
        frame[:] = 222  # Protect source/assessment pixels from detector mutation.
        raw = report([PERSON,HELMET,VEST])
        raw.update(source=source,weights_sha256=self.weights_sha256)
        return None,raw


class RecordedPreviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root/'source.avi'
        writer = cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*'MJPG'),25,(640,640))
        self.assertTrue(writer.isOpened())
        for n in range(30): writer.write(np.full((640,640,3),n,np.uint8))
        writer.release()
        self.reader = MemoryReader(self.source,(640,640),self.root)

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(patch('ppe_compliance.recorded_preview.ScaledVideoReader',return_value=self.reader))
        stack.enter_context(patch('ppe_compliance.recorded_preview.FrameDetector',return_value=SlowModel()))
        stack.enter_context(patch('torch.get_num_threads',return_value=2))
        stack.enter_context(patch('torch.set_num_threads'))
        return stack

    def run_review(self, **kwargs):
        return review_recorded_video(self.source,output_dir=self.root/'runs',preview=False,processing_height=720,**kwargs)

    def test_source_does_not_wait_for_each_prediction_and_sample_hashes_remain_exact(self):
        with self.mocks(): result=self.run_review()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['source_frames_played'],30)
        self.assertLess(result['playback_seconds'],3)  # Sequential predictions alone need 4.8 seconds.
        self.assertGreater(result['processed_frames'],1)
        self.assertLess(result['processed_frames'],30)
        self.assertFalse(result['intentional_source_frame_skipping'])
        self.assertTrue(result['intentional_analysis_sampling'])
        self.assertTrue(self.reader.released)
        rows=[json.loads(line) for line in (Path(result['run_directory'])/'frames.jsonl').read_text().splitlines()]
        indices=[row['frame_index'] for row in rows]
        self.assertEqual(indices,sorted(set(indices)))
        self.assertTrue(any(b-a>1 for a,b in zip(indices,indices[1:])))
        for row in rows:
            index=row['frame_index']
            self.assertAlmostEqual(row['source_time_seconds'],index/25)
            image=np.full((640,640,3),index,np.uint8)
            self.assertEqual(row['assessment']['source_sha256'],hashlib.sha256(memoryview(image)).hexdigest())
            self.assertEqual(row['source_sha256'],hashlib.sha256(self.source.read_bytes()).hexdigest())
            self.assertEqual(row['assessment']['workers'][0]['vest']['confidence'],.95)

    def test_separate_scored_pane_references_its_own_frame_not_the_moving_source(self):
        renderer=DashboardRenderer()
        calls=[]
        shown_pixels=[]
        def render(packet,result,fresh,now,stats,**options):
            calls.append((packet,result,dict(stats)))
            return renderer.render(packet,result,fresh,now,stats,**options)
        with self.mocks(),patch('ppe_compliance.recorded_preview.DashboardRenderer') as factory, \
                patch('cv2.namedWindow'),patch('cv2.resizeWindow'), \
                patch('cv2.imshow',side_effect=lambda _,pixels:shown_pixels.append(int(pixels[300,500,0]))), \
                patch('cv2.getWindowProperty',return_value=1),patch('cv2.destroyWindow'), \
                patch('cv2.waitKey',side_effect=lambda ms:(sleep(ms/1000) or -1)):
            factory.return_value.render.side_effect=render
            factory.return_value.update_source.side_effect=renderer.update_source
            result=review_recorded_video(self.source,output_dir=self.root/'runs',max_frames=15,processing_height=720)
        self.assertEqual(shown_pixels[-15:],list(range(15)))
        self.assertEqual(result['stop_reason'],'max_source_frames')
        scored=[call for call in calls if call[1] is not None]
        self.assertTrue(scored)
        self.assertTrue(any(packet.sequence>analysis.source.sequence for packet,analysis,_ in scored))
        for packet,analysis,stats in scored:
            self.assertEqual(int(packet.frame[0,0,0]),packet.sequence)
            self.assertAlmostEqual(stats['analysis_time_seconds'],analysis.source.sequence/25)
            self.assertEqual(analysis.assessment['source_sha256'],hashlib.sha256(memoryview(analysis.source.frame)).hexdigest())

    def test_reader_failure_and_model_failure_finalize_a_failed_summary(self):
        with self.mocks(),patch.object(self.reader,'read',side_effect=RuntimeError('decoder failed')):
            with self.assertRaisesRegex(RuntimeError,'decoder failed'):
                self.run_review()
        self.assertTrue(self.reader.released)
        failed=json.loads(next((self.root/'runs').glob('*/summary.json')).read_text())
        self.assertEqual(failed['status'],'failed')
        with patch('ppe_compliance.recorded_preview.FrameDetector',side_effect=ValueError('bad model')):
            with self.assertRaisesRegex(RuntimeError,'bad model'):
                self.run_review()

    def test_cli_routes_to_sampled_review_instead_of_full_video_processing(self):
        result={'status':'completed','stop_reason':'end_of_video','processed_frames':4,'source_frames_played':30,
                'run_directory':str(self.root),'note':'selected samples'}
        with patch.object(sys,'argv',['ppe','video','--source',str(self.source),'--live-preview','--track']), \
                patch('ppe_compliance.recorded_preview.review_recorded_video',return_value=result) as live, \
                patch('ppe_compliance.video_pipeline.process_video') as full,redirect_stdout(io.StringIO()):
            main()
        full.assert_not_called()
        self.assertEqual(live.call_args.kwargs['processing_height'],540)
        self.assertTrue(live.call_args.kwargs['tracking'])

    def test_real_decoder_retains_every_source_frame_and_closes(self):
        reader=ScaledVideoReader(self.source,(320,320),self.root)
        values=[]
        try:
            while True:
                ok,image=reader.read()
                if not ok:break
                self.assertEqual(image.shape,(320,320,3))
                values.append(float(image.mean()))
        finally: reader.release()
        self.assertEqual(len(values),30)
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))
        self.assertIsNone(reader.process)

    def test_playback_copy_retains_count_rate_and_is_reused_without_encoding_again(self):
        digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        target,info=prepare_playback_source(self.source,digest,(320,320),25,30,
                                          self.root/'cache-root',self.root,Event())
        self.assertFalse(info['cached'])
        self.assertEqual(info['frames'],30)
        self.assertEqual(info['fps'],25)
        self.assertEqual(info['sha256'],hashlib.sha256(target.read_bytes()).hexdigest())
        with patch('subprocess.Popen',side_effect=AssertionError('Unexpected re-encode')):
            second,cached=prepare_playback_source(self.source,digest,(320,320),25,30,
                                                self.root/'cache-root',self.root,Event())
        self.assertEqual(second,target)
        self.assertTrue(cached['cached'])

    def test_quit_during_warmup_stops_before_source_playback_and_joins_worker(self):
        with self.mocks(),patch('ppe_compliance.recorded_preview.ScaledVideoReader') as reader, \
                patch('cv2.namedWindow'),patch('cv2.resizeWindow'),patch('cv2.imshow'), \
                patch('cv2.waitKey',return_value=27),patch('cv2.getWindowProperty',return_value=1), \
                patch('cv2.destroyWindow'):
            result=review_recorded_video(self.source,output_dir=self.root/'runs',processing_height=720)
        self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['stop_reason'],'preview_stop')
        self.assertEqual(result['source_frames_played'],0)
        reader.assert_not_called()

    def test_prefetch_startup_failure_releases_decoder(self):
        with patch('ppe_compliance.recorded_preview.ScaledVideoReader',return_value=self.reader), \
                patch('ppe_compliance.recorded_preview.Thread.start',side_effect=RuntimeError('cannot start')):
            with self.assertRaisesRegex(RuntimeError,'cannot start'):
                PrefetchedReader(self.source,(640,640),self.root)
        self.assertTrue(self.reader.released)

    def test_large_odd_source_uses_even_playback_copy_dimensions(self):
        metadata=Mock()
        metadata.isOpened.return_value=True
        metadata.get.side_effect=lambda prop:25 if prop==cv2.CAP_PROP_FPS else 1
        metadata.read.return_value=(True,np.zeros((1081,1920,3),np.uint8))
        def prepare(source,digest,resolution,*args):
            self.assertEqual(resolution,(958,540))
            return source, {'cached':False}
        self.reader.resolution=(958,540)
        with self.mocks(),patch('cv2.VideoCapture',return_value=metadata), \
                patch('ppe_compliance.recorded_preview.prepare_playback_source',side_effect=prepare):
            result=review_recorded_video(self.source,output_dir=self.root/'runs',preview=False,max_frames=1)
        self.assertEqual(result['processing_resolution'],[958,540])
        self.assertTrue(metadata.release.called)


if __name__=='__main__':unittest.main()
