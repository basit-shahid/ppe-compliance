"""Browser handoff, lease expiry, cancellation and isolated worker failures."""

import json
import os
from pathlib import Path
import tempfile
from time import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ppe_compliance.browser_preview import (BrowserPreview, LEASE_SECONDS, atomic_write,
    run_worker, start_browser_preview, write_status)


class BrowserPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.process=Mock();self.process.poll.return_value=None

    def job(self,kind='video'):
        with patch('subprocess.Popen',return_value=self.process):
            return start_browser_preview(self.root,'python.exe',kind=kind,source=0 if kind=='camera' else 'clip.mp4',confidence=.7)

    def test_launcher_uses_isolated_worker_and_exact_settings(self):
        with patch('subprocess.Popen',return_value=self.process) as start:
            job=start_browser_preview(self.root,'python.exe',kind='camera',source=1,confidence=.7,backend='msmf')
        config=json.loads((job.folder/'config.json').read_text())
        self.assertEqual(config['source'],1)
        self.assertEqual(config['confidence'],.7)
        self.assertEqual(config['backend'],'msmf')
        self.assertIn('ppe_compliance.browser_preview',start.call_args.args[0])
        self.assertFalse(start.call_args.kwargs.get('shell',False))

    def test_atomic_latest_frame_and_stop_request(self):
        job=self.job()
        atomic_write(job.folder/'frame.jpg',b'frame1')
        atomic_write(job.folder/'frame.jpg',b'frame2')
        self.assertEqual(job.snapshot()['image'],b'frame2')
        job.stop();job.stop()
        self.assertTrue((job.folder/'stop').exists())
        self.assertEqual(job.snapshot()['status'],'stopping')
        self.assertFalse((job.folder/'frame.jpg.tmp').exists())

    def test_atomic_handoff_retries_a_transient_windows_reader_lock(self):
        target=self.root/'snapshot.json'
        original=Path.replace
        attempts=[]
        def replace(path,destination):
            attempts.append(destination)
            if len(attempts)<3:
                raise PermissionError('reader still holds the old file')
            return original(path,destination)
        with patch.object(Path,'replace',replace),patch('ppe_compliance.browser_preview.sleep'):
            atomic_write(target,b'complete snapshot')
        self.assertEqual(target.read_bytes(),b'complete snapshot')
        self.assertEqual(len(attempts),3)

    def test_unexpected_exit_is_visible_and_completed_status_is_preserved(self):
        job=self.job()
        self.process.poll.return_value=1
        self.assertEqual(job.snapshot()['status'],'failed')
        write_status(job.folder,status='completed',message='done')
        self.assertEqual(job.snapshot()['status'],'completed')

    def test_stop_has_bounded_process_fallback(self):
        job=self.job();job.stop()
        with patch('ppe_compliance.browser_preview.monotonic',return_value=job.stop_at+7):
            job.heartbeat()
        self.process.terminate.assert_called_once()

    def test_expired_browser_lease_cancels_without_opening_source(self):
        job=self.job('camera')
        stale=time()-LEASE_SECONDS-1
        os.utime(job.folder/'heartbeat',(stale,stale))
        with patch('ppe_compliance.live_pipeline.process_live') as live:
            run_worker(job.folder)
        live.assert_not_called()
        self.assertEqual(job.snapshot()['status'],'stopped')

    def test_worker_publishes_real_jpeg_without_opening_gui(self):
        job=self.job('camera')
        def live(**options):
            self.assertFalse(options['preview'])
            self.assertFalse(options['stop_requested']())
            options['on_preview'](np.full((100,160,3),80,np.uint8),{'loading':False,'source_sequence':3,'processed_frames':1})
            self.assertEqual(json.loads((job.folder/'status.json').read_text())['source_sequence'],3)
            return {'status':'stopped','run_directory':'fixture'}
        with patch('ppe_compliance.live_pipeline.process_live',side_effect=live), patch('cv2.imshow') as gui:
            run_worker(job.folder)
        gui.assert_not_called()
        decoded=cv2.imdecode(np.frombuffer((job.folder/'frame.jpg').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape,(100,160,3))
        self.assertEqual(job.snapshot()['status'],'stopped')

    def test_worker_error_retains_actionable_message(self):
        job=self.job('camera')
        with patch('ppe_compliance.live_pipeline.process_live',side_effect=RuntimeError('camera busy')):
            with self.assertRaisesRegex(RuntimeError,'camera busy'):
                run_worker(job.folder)
        state=job.snapshot()
        self.assertEqual(state['status'],'failed')
        self.assertIn('camera busy',state['message'])


if __name__=='__main__':
    unittest.main()
