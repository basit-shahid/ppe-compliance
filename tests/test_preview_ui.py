"""Preview layout/freshness contracts using generated pixels, no physical camera."""

import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np

from ppe_compliance.preview_ui import dashboard,visible_workers,DashboardRenderer
from ppe_compliance.live_pipeline import FramePacket,ResultPacket
from ppe_compliance.tracking import WorkerTracker
from test_tracking import observation


class DashboardTests(unittest.TestCase):
    def test_portable_fallback_font_respects_requested_size(self):
        from ppe_compliance.preview_ui import _font
        _font.cache_clear()
        with patch('ppe_compliance.preview_ui.Path.is_file', return_value=False):
            small, large = _font(12), _font(25)
        self.assertGreater(large.getlength('PPE Monitor'), small.getlength('PPE Monitor')*1.7)
        _font.cache_clear()

    def setUp(self):
        self.image=np.full((480,640,3),90,np.uint8)
        self.frame=FramePacket(10,10.,self.image)
        tracker=WorkerTracker()
        for at in (0,.1,.21):self.assessment=tracker.update(observation(),at)
        self.result=ResultPacket(FramePacket(9,9.9,self.image),9.95,self.image,self.assessment,tuple(tracker.history))

    def test_dashboard_native_size_and_scaled_size(self):
        for width in (960,1280,1600):
            view=dashboard(self.frame,self.result,True,10.,{'tracking':True},width=width)
            self.assertEqual(view.shape,(round(720*width/1280),width,3))
            self.assertEqual(view.dtype,np.uint8)

    def test_stale_green_is_not_exposed_as_current_worker_card(self):
        self.assertEqual(visible_workers(self.result,False),[])
        self.assertEqual(visible_workers(self.result,True)[0]['status'],'PPE detected')
        view=dashboard(self.frame,self.result,False,12.,{},focus='analyzed')
        self.assertEqual(view.shape,(720,1280,3))

    def test_dashboard_does_not_change_source_pixels(self):
        before=self.image.copy()
        for focus in ('live','analyzed'):
            dashboard(self.frame,self.result,True,10.,{},focus=focus)
        np.testing.assert_array_equal(self.image,before)

    def test_loading_without_predictions_has_no_worker_cards(self):
        self.assertEqual(visible_workers(None,False),[])
        image=dashboard(self.frame,None,False,10.,{'loading':True})
        self.assertEqual(image.shape,(720,1280,3))

    def test_cached_chrome_rebuilds_on_stale_transition_but_not_frame_counter(self):
        renderer=DashboardRenderer()
        renderer.render(self.frame,self.result,True,10.,{})
        newer=FramePacket(11,10.1,self.image)
        renderer.render(newer,self.result,True,10.1,{})
        self.assertEqual(renderer.build_count,1)
        renderer.render(newer,self.result,False,11.,{})
        self.assertEqual(renderer.build_count,2)

    def test_confidence_changes_refresh_cards_without_changing_status(self):
        renderer = DashboardRenderer()
        renderer.render(self.frame, self.result, True, 10., {})
        updated = deepcopy(self.assessment)
        updated['workers'][0]['person_confidence'] = .72
        result = ResultPacket(self.result.source, 10., self.image, updated)
        renderer.render(self.frame, result, True, 10., {})
        self.assertEqual(renderer.build_count, 2)

    def test_recorded_source_time_is_distinct_from_wall_clock_age(self):
        renderer = DashboardRenderer()
        screen = renderer.render(self.frame, self.result, True, 999.,
            {'mode': 'recorded', 'source_time_seconds': 1.5, 'playback_fps': 30}, focus='analyzed')
        self.assertEqual(screen.shape, (720, 1280, 3))
        self.assertEqual(len(visible_workers(self.result, True)), 1)


if __name__=='__main__':unittest.main()
