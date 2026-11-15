"""Identity/state continuity and conservative transitions without model inference."""

from copy import deepcopy
import unittest

from ppe_compliance.tracking import WorkerTracker, TrackingRules, PPE, REVIEW
from test_assessment import assess, PERSON, HELMET, VEST


def observation(status=PPE, boxes=None):
    value=assess([PERSON,HELMET,VEST])
    if boxes is not None:
        worker=value['workers'][0]
        value['workers']=[{**deepcopy(worker),'worker_id':f'W{i+1}','bbox_xyxy':box} for i,box in enumerate(boxes)]
    for worker in value['workers']:
        worker['status']=status
    return value


class TrackingTests(unittest.TestCase):
    def green(self,tracker):
        for at in (0,.1,.21):
            result=tracker.update(observation(),at)
        return result

    def test_three_consistent_observations_and_time_are_required_for_green(self):
        tracker=WorkerTracker()
        first=tracker.update(observation(),0)
        second=tracker.update(observation(),.1)
        third=tracker.update(observation(),.21)
        self.assertEqual(first['workers'][0]['status'],REVIEW)
        self.assertEqual(second['workers'][0]['status'],REVIEW)
        self.assertEqual(third['workers'][0]['status'],PPE)
        self.assertEqual({x['workers'][0]['track_id'] for x in (first,second,third)},{'T001'})

    def test_repeated_green_does_not_repeat_status_events(self):
        tracker=WorkerTracker();self.green(tracker)
        for index in range(1,10):
            result=tracker.update(observation(),.21+index*.1)
            self.assertEqual(result['status_events'],[])
        self.assertEqual(tracker.status_change_count,1)

    def test_uncertainty_removes_green_immediately(self):
        tracker=WorkerTracker();self.green(tracker)
        result=tracker.update(observation(REVIEW),.3)
        self.assertEqual(result['workers'][0]['status'],REVIEW)
        self.assertEqual(result['status_events'],[])

    def test_persistent_review_generates_one_transition(self):
        tracker=WorkerTracker();self.green(tracker)
        events=[]
        for at in (.3,.4,.51,.6,.7):
            events.extend(tracker.update(observation(REVIEW),at)['status_events'])
        changes=[e for e in events if e['type']=='status_changed']
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0]['previous_status'],PPE)
        self.assertEqual(changes[0]['status'],REVIEW)

    def test_transient_dropout_requires_green_reconfirmation_but_no_alert_spam(self):
        tracker=WorkerTracker();self.green(tracker)
        tracker.update(observation(REVIEW),.3)
        one=tracker.update(observation(),.4)
        two=tracker.update(observation(),.5)
        three=tracker.update(observation(),.61)
        self.assertEqual(one['workers'][0]['status'],REVIEW)
        self.assertEqual(two['workers'][0]['status'],REVIEW)
        self.assertEqual(three['workers'][0]['status'],PPE)
        self.assertEqual(tracker.status_change_count,1)

    def test_evidence_gap_expires_green_before_reconfirmation(self):
        tracker=WorkerTracker();self.green(tracker)
        result=tracker.update(observation(),.9)
        self.assertEqual(result['workers'][0]['status'],REVIEW)
        self.assertTrue(any(e['reason']=='PPE evidence expired' for e in result['status_events']))

    def test_expired_worker_is_not_reused_as_a_green_identity(self):
        tracker=WorkerTracker();self.green(tracker)
        result=tracker.update(observation(),2.)
        self.assertEqual(result['workers'][0]['track_id'],'T002')
        self.assertEqual(result['workers'][0]['status'],REVIEW)
        self.assertTrue(any(e['type']=='track_lost' for e in result['status_events']))

    def test_missing_people_produce_no_ghost_boxes(self):
        tracker=WorkerTracker();self.green(tracker)
        result=tracker.update(assess([]),.3)
        self.assertEqual(result['workers'],[])
        result=tracker.update(assess([]),2.)
        self.assertEqual(tracker.summary()['active_tracks'],0)

    def test_detection_order_changes_do_not_change_matching_track_ids(self):
        tracker=WorkerTracker()
        boxes=[[50,50,200,550],[350,50,500,550]]
        first=tracker.update(observation(boxes=boxes),0)
        second=tracker.update(observation(boxes=boxes[::-1]),.1)
        self.assertEqual([w['track_id'] for w in first['workers']],['T001','T002'])
        self.assertEqual([w['track_id'] for w in second['workers']],['T002','T001'])

    def test_ambiguous_crossing_never_inherits_another_green_state(self):
        tracker=WorkerTracker()
        for at in (0,.1,.21):
            previous=tracker.update(observation(boxes=[[50,60,250,560],[250,60,450,560]]),at)
        self.assertTrue(all(w['status']==PPE for w in previous['workers']))
        result=tracker.update(observation(boxes=[[145,60,345,560],[155,60,355,560]]),.31)
        self.assertTrue(all(w['status']==REVIEW for w in result['workers']))
        self.assertEqual(len({w['track_id'] for w in result['workers']}),2)

    def test_small_motion_keeps_id(self):
        tracker=WorkerTracker()
        ids=[]
        for index in range(10):
            result=tracker.update(observation(boxes=[[100+index*5,60,300+index*5,560]]),index*.1)
            ids.append(result['workers'][0]['track_id'])
        self.assertEqual(set(ids),{'T001'})

    def test_potential_violation_remains_a_concern_not_confirmed_absence(self):
        tracker=WorkerTracker();self.green(tracker)
        for at in (.3,.4,.51):
            result=tracker.update(observation('potential violation'),at)
        self.assertEqual(result['workers'][0]['status'],'potential violation')
        self.assertEqual(result['status_events'][0]['status'],'potential violation')

    def test_inputs_are_not_modified(self):
        value=observation();before=deepcopy(value)
        WorkerTracker().update(value,0)
        self.assertEqual(value,before)

    def test_invalid_rules_and_reverse_time_are_rejected(self):
        for value in ({'min_iou':0},{'confirmation_observations':True},{'confirmation_seconds':-1},
                      {'track_ttl_seconds':.1},{'ambiguity_margin':float('nan')}):
            with self.subTest(value=value),self.assertRaises(ValueError):TrackingRules(**value)
        tracker=WorkerTracker();tracker.update(observation(),1.)
        with self.assertRaises(ValueError):tracker.update(observation(),.5)


if __name__=='__main__':
    unittest.main()
