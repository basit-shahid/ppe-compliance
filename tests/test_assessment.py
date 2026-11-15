"""Meaningful rule cases independent of model predictions or network access."""

from copy import deepcopy
import unittest

from ppe_compliance.association import AssociationRules, associate_ppe
from ppe_compliance.compliance import assess_detections, visibility_template


def detection(kind, box, confidence=.95):
    return {"class_name": kind, "class_id": {"person": 0, "helmet": 1, "vest": 2}[kind],
            "bbox_xyxy": box, "confidence": confidence}


PERSON = detection("person", [100, 60, 300, 560])
HELMET = detection("helmet", [165, 70, 225, 125])
VEST = detection("vest", [135, 170, 265, 340])
SOURCE = "a" * 64


def report(items):
    return {"source": "example.jpg", "weights": "best.pt", "image_width": 640,
            "image_height": 640, "image_size": 640, "confidence_threshold": .25,
            "model_classes": {0: "person", 1: "helmet", 2: "vest"}, "detections": items,
            "note": "Object detections only."}


def assess(items, **kwargs):
    return assess_detections(report(items), SOURCE, **kwargs)


class AssociationTests(unittest.TestCase):
    def test_both_items_are_plausibly_matched(self):
        result = assess([PERSON, HELMET, VEST])
        worker = result["workers"][0]
        self.assertEqual(worker["status"], "PPE detected")
        self.assertEqual(worker["helmet"]["detection_id"], 1)
        self.assertEqual(worker["vest"]["detection_id"], 2)

    def test_bottom_cropped_colab_vest_matches_with_score_but_requires_review(self):
        data = report([
            detection('person', [338.5469, 169.67, 2178.50146, 2160], .9143),
            detection('helmet', [929.371, 218.2728, 1670.0706, 662.096], .9065),
            detection('vest', [514.5458, 971.35895, 2104.1565, 2160], .8236)])
        data.update(image_width=3840, image_height=2160, image_size=416)
        worker = assess_detections(data, SOURCE)['workers'][0]
        self.assertEqual(worker['vest']['state'], 'matched')
        self.assertEqual(worker['vest']['confidence'], .8236)
        self.assertEqual(worker['status'], 'review required')
        self.assertTrue(any('border' in reason for reason in worker['reasons']))

    def test_crop_exception_does_not_match_unclipped_held_or_bottom_strip_vests(self):
        person = detection('person', [100, 40, 400, 640])
        for box in ([150, 380, 350, 580], [150, 550, 350, 640], [20, 230, 180, 640]):
            worker = assess([person, detection('vest', box)])['workers'][0]
            self.assertEqual(worker['vest']['state'], 'not_detected')

    def test_cropped_vest_ownership_remains_ambiguous_for_overlapping_workers(self):
        result = assess([detection('person', [100, 40, 400, 640]),
                         detection('person', [110, 40, 410, 640]),
                         detection('vest', [150, 250, 350, 640])])
        self.assertTrue(all(w['vest']['state'] == 'ambiguous' for w in result['workers']))

    def test_held_helmet_at_waist_does_not_satisfy_head(self):
        held = detection("helmet", [165, 400, 225, 450])
        result = assess([PERSON, held, VEST])
        self.assertEqual(result["workers"][0]["helmet"]["state"], "not_detected")
        self.assertEqual(result["workers"][0]["status"], "review required")
        self.assertEqual(result["unmatched_equipment"][0]["detection_id"], 1)

    def test_missing_detection_is_not_automatic_violation(self):
        result = assess([PERSON, VEST])
        self.assertEqual(result["workers"][0]["status"], "review required")

    def test_low_confidence_ppe_is_uncertain(self):
        helmet = {**HELMET, "confidence": .35}
        worker = assess([PERSON, helmet, VEST])["workers"][0]
        self.assertEqual(worker["helmet"]["state"], "uncertain")
        self.assertEqual(worker["status"], "review required")

    def test_ambiguous_helmet_is_not_assigned_to_two_people(self):
        second = detection("person", [120, 60, 320, 560])
        result = assess([PERSON, second, HELMET])
        self.assertTrue(all(w["helmet"]["state"] == "ambiguous" for w in result["workers"]))
        self.assertTrue(all(w["helmet"]["detection_id"] is None for w in result["workers"]))

    def test_separate_workers_get_separate_equipment(self):
        second = detection("person", [330, 60, 530, 560])
        helmet2 = detection("helmet", [395, 70, 455, 125])
        vest2 = detection("vest", [365, 170, 495, 340])
        result = assess([second, helmet2, VEST, PERSON, vest2, HELMET])
        self.assertEqual([w["bbox_xyxy"][0] for w in result["workers"]], [100, 330])
        self.assertEqual(result["summary"]["PPE detected"], 2)
        ids = [w[kind]["detection_id"] for w in result["workers"] for kind in ("helmet", "vest")]
        self.assertEqual(len(set(ids)), 4)

    def test_equipment_with_no_workers_remains_unmatched(self):
        result = assess([HELMET])
        self.assertEqual(result["summary"]["workers"], 0)
        self.assertEqual(len(result["unmatched_equipment"]), 1)
        self.assertIn("No detected workers does not establish", result["note"])

    def test_small_worker_is_reviewed(self):
        small = detection("person", [100, 100, 120, 150])
        worker = assess([small])["workers"][0]
        self.assertTrue(any("too small" in reason for reason in worker["reasons"]))

    def test_scale_gate_uses_model_input_not_original_resolution(self):
        original = report([PERSON])
        original.update(image_width=2560, image_height=2560)
        worker = assess_detections(original, SOURCE)["workers"][0]
        self.assertFalse(any("too small" in reason for reason in worker["reasons"]))
        original["image_size"] = 320
        worker = assess_detections(original, SOURCE)["workers"][0]
        self.assertTrue(any("too small" in reason for reason in worker["reasons"]))

    def test_low_person_confidence_forces_review(self):
        worker = assess([{**PERSON, "confidence": .4}, HELMET, VEST])["workers"][0]
        self.assertEqual(worker["status"], "review required")

    def test_border_truncation_forces_review(self):
        person = detection("person", [100, 0, 300, 560])
        worker = assess([person, HELMET, VEST])["workers"][0]
        self.assertTrue(any("border" in reason for reason in worker["reasons"]))

    def test_duplicate_candidate_does_not_overwrite_best_match(self):
        extra = detection("helmet", [160, 70, 220, 125], .8)
        result = assess([PERSON, HELMET, VEST, extra])
        self.assertEqual(result["workers"][0]["helmet"]["detection_id"], 1)
        self.assertEqual(result["unmatched_equipment"][0]["detection_id"], 3)

    def test_large_equipment_box_is_not_a_valid_head_match(self):
        huge = detection("helmet", [100, 60, 300, 250])
        self.assertEqual(assess([PERSON, huge])["workers"][0]["helmet"]["state"], "not_detected")

    def test_general_model_is_rejected(self):
        original = report([PERSON]); original["model_classes"] = {0: "person", 1: "bus"}
        with self.assertRaisesRegex(ValueError, "PPE-trained"):
            assess_detections(original, SOURCE)

    def test_class_names_are_case_insensitive(self):
        original = report([{**PERSON, "class_name": "Person"}, HELMET, VEST])
        original["model_classes"][0] = "Person"
        self.assertEqual(assess_detections(original, SOURCE)["summary"]["PPE detected"], 1)

    def test_invalid_rules_are_rejected(self):
        for overrides in ({"min_confidence": 0}, {"ownership_margin": -1},
                          {"min_person_height": float("nan")}, {"min_confidence": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AssociationRules(**overrides)

    def test_invalid_detections_are_rejected(self):
        for bad in ({**PERSON, "bbox_xyxy": [300, 60, 100, 560]},
                    {**PERSON, "bbox_xyxy": [-1, 60, 300, 560]},
                    {**PERSON, "confidence": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                associate_ppe([bad], 640, 640)

    def test_empty_detections_have_zero_worker_summary(self):
        self.assertEqual(assess([])["summary"], {"workers": 0, "PPE detected": 0,
                                                "potential violation": 0, "review required": 0})

    def test_inputs_not_mutated(self):
        original = report([deepcopy(PERSON), deepcopy(HELMET)])
        before = deepcopy(original)
        assess_detections(original, SOURCE)
        self.assertEqual(original, before)


class HumanVisibilityTests(unittest.TestCase):
    def template(self, items):
        return visibility_template(assess(items)["workers"], SOURCE)

    def test_visible_region_allows_reviewable_potential_violation(self):
        items = [PERSON, VEST]
        template = self.template(items)
        template["workers"]["W1"]["head_visible"] = True
        worker = assess(items, visibility=template)["workers"][0]
        self.assertEqual(worker["status"], "potential violation")
        self.assertIn("absence is not proven", worker["reasons"][0])

    def test_visibility_does_not_override_low_confidence(self):
        items = [{**PERSON, "confidence": .4}, VEST]
        template = self.template(items)
        template["workers"]["W1"]["head_visible"] = True
        self.assertEqual(assess(items, visibility=template)["workers"][0]["status"], "review required")

    def test_explicit_invisible_region_forces_review_even_with_matches(self):
        items = [PERSON, HELMET, VEST]
        template = self.template(items)
        template["workers"]["W1"]["head_visible"] = False
        self.assertEqual(assess(items, visibility=template)["workers"][0]["status"], "review required")

    def test_partial_visibility_cannot_confirm_two_missing_items(self):
        template = self.template([PERSON])
        template["workers"]["W1"]["head_visible"] = True
        self.assertEqual(assess([PERSON], visibility=template)["workers"][0]["status"], "review required")

    def test_template_with_unknown_visibility_is_conservative(self):
        items = [PERSON, VEST]
        self.assertEqual(assess(items, visibility=self.template(items))["workers"][0]["status"], "review required")

    def test_other_image_hash_rejected(self):
        template = self.template([PERSON]); template["source_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            assess([PERSON], visibility=template)

    def test_stale_box_rejected(self):
        template = self.template([PERSON]); template["workers"]["W1"]["bbox_xyxy"] = [0, 0, 50, 50]
        with self.assertRaisesRegex(ValueError, "does not match"):
            assess([PERSON], visibility=template)

    def test_unknown_worker_rejected(self):
        template = self.template([PERSON]); template["workers"]["W99"] = template["workers"].pop("W1")
        with self.assertRaisesRegex(ValueError, "unknown worker"):
            assess([PERSON], visibility=template)

    def test_visibility_is_not_truthy_string_or_integer(self):
        for value in ("true", 1):
            template = self.template([PERSON]); template["workers"]["W1"]["head_visible"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "true, false or null"):
                assess([PERSON], visibility=template)


if __name__ == "__main__":
    unittest.main()
