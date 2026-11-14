"""CLI file/output integration with deterministic detections, not model accuracy."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ppe_compliance.cli import main
from test_assessment import PERSON, HELMET, VEST, report


class CliAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.jpg"
        self.frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.source), self.frame))
        self.output = self.root / "output"

    def run_cli(self, command="assess", extra=(), detections=None, model_classes=None):
        data = report(detections if detections is not None else [PERSON, HELMET, VEST])
        data["source"] = str(self.source)
        if model_classes is not None:
            data["model_classes"] = model_classes
        result = SimpleNamespace(orig_img=self.frame, save=lambda filename: cv2.imwrite(filename, self.frame))
        argv = ["ppe", command, "--source", str(self.source), "--output-dir", str(self.output), *extra]
        with patch("sys.argv", argv), patch("ppe_compliance.detector.detect_image", return_value=(result, data)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main()

    def test_assessment_outputs_json_image_and_visibility_template(self):
        self.run_cli()
        run = next(self.output.iterdir())
        self.assertEqual({p.name for p in run.iterdir()},
                         {"annotated.jpg", "detections.json", "compliance.json", "visibility-template.json"})
        result = json.loads((run / "compliance.json").read_text())
        self.assertEqual(result["summary"]["PPE detected"], 1)
        self.assertEqual(cv2.imread(str(run / "annotated.jpg")).shape, self.frame.shape)

    def test_visibility_template_round_trip_and_unique_runs(self):
        self.run_cli(detections=[PERSON, VEST])
        first = next(self.output.iterdir())
        template_path = first / "visibility-template.json"
        template = json.loads(template_path.read_text())
        template["workers"]["W1"]["head_visible"] = True
        template_path.write_text(json.dumps(template))
        self.run_cli(extra=["--visibility", str(template_path)], detections=[PERSON, VEST])
        self.assertEqual(len(list(self.output.iterdir())), 2)
        second = next(p for p in self.output.iterdir() if p != first)
        result = json.loads((second / "compliance.json").read_text())
        self.assertEqual(result["summary"]["potential violation"], 1)

    def test_detect_command_keeps_original_two_output_files(self):
        self.run_cli(command="detect")
        run = next(self.output.iterdir())
        self.assertEqual({p.name for p in run.iterdir()}, {"annotated.jpg", "detections.json"})

    def test_general_weights_rejected_without_partial_outputs(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli(model_classes={0: "person", 1: "bus"})
        self.assertEqual(error.exception.code, 1)
        self.assertFalse(self.output.exists())

    def test_invalid_rules_rejected_without_partial_outputs(self):
        rules = self.root / "rules.yaml"; rules.write_text("min_confidence: 2")
        with self.assertRaises(SystemExit):
            self.run_cli(extra=["--rules", str(rules)])
        self.assertFalse(self.output.exists())

    def test_unknown_rule_rejected(self):
        rules = self.root / "rules.yaml"; rules.write_text("made_up_rule: true")
        with self.assertRaises(SystemExit):
            self.run_cli(extra=["--rules", str(rules)])

    def test_invalid_visibility_rejected_without_partial_outputs(self):
        visibility = self.root / "visibility.json"; visibility.write_text('{"source_sha256":"wrong","workers":{}}')
        with self.assertRaises(SystemExit):
            self.run_cli(extra=["--visibility", str(visibility)])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
