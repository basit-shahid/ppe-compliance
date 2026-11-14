# Phase 4: worker-level image evidence

The image-assessment baseline is implemented and checked. The new `ppe assess`
command matches visible helmets/vests to workers, emits conservative statuses and
reasons, draws an overlay and saves structured output. This is geometry-based
evidence, not verified wearing-state recognition or independent compliance accuracy.

## Run an assessment

From `D:\ppe-compliance` in PowerShell:

```powershell
.\.venv\Scripts\ppe.exe assess --source "D:\path\to\photo.jpg" --weights models/best.pt --device cpu
```

`assess` defaults to `models/best.pt`; it rejects models lacking all three class
names `person`, `helmet`, `vest` (case insensitive). General COCO weights are
unsuitable. The existing `detect` command still generates object detections only.

Each assessment creates a unique directory with:

- `annotated.jpg`: item boxes and per-worker status labels.
- `detections.json`: raw model outputs, class mapping, requested image size and
  local weight hash when available.
- `compliance.json`: worker/item matches, status/reasons, quality flags, rules,
  unmatched equipment, summary, input hash and model provenance.
- `visibility-template.json`: optional input for manual visibility review.

Worker IDs W1, W2, etc. are frame-local and ordered by person boxes. They are not
tracking IDs. Equipment detection IDs index the companion raw detections list.
No detected people means **no detected workers**, not a safe or empty scene.

## Association algorithm

The modules are independent of the detector/UI. `association.py` operates on boxes;
`compliance.py` applies the decision policy; `rendering.py` draws the returned
records. CLI integration loads the model and writes the artifacts.

For person box `(left, top, right, bottom)` with width W and height H:

- Helmet search region: horizontal `[left-0.05W, right+0.05W]`, vertical
  `[top-0.08H, top+0.32H]`.
- Vest search region: horizontal `[left-0.02W, right+0.02W]`, vertical
  `[top+0.16H, top+0.72H]`.
- An item's centre must be inside the corresponding region; at least 65% of its
  box must overlap that region. Relative box-size gates reject implausible sizes.
- Candidate scores combine region coverage (80%) and horizontal alignment (20%).
  They are deterministic geometry scores, not calibrated probabilities.
- Score gaps of 0.15 or less between leading owners produce ambiguity; shared
  equipment is not silently assigned to both workers.
- Multiple unambiguous detections for one worker/item select the best candidate;
  additional candidates remain in the unmatched-equipment report.

Default confidence gate is 0.5, separate from detection's default 0.25. Small or
low-confidence equipment remains uncertain rather than a confirmed match.
Person quality gates require at least 80 pixels height and 20 pixels width at
the requested model-input scale. They also flag person confidence below 0.5,
person IoU overlap of at least 0.35, and image-border proximity within two input
pixels. These gates are conservative heuristics, not a visibility or pose model.

## Status policy

| Status | Meaning |
| --- | --- |
| PPE detected | Both items have plausible, unambiguous region matches and no quality/visibility warning. Wearing state and overall safety are not proven. |
| Review required | Small, low-confidence, overlapping or truncated person; ambiguous/uncertain items; or an unmatched item whose body-region visibility is unknown. |
| Potential violation | An item is not confirmed in a manually reviewed visible body region, with no other quality/uncertainty flag. This is a reviewable concern, not proof of absence. |

**There is no automatic potential-violation decision from a missing detection.**
Our three-class model has no bare-head/bare-torso or pose/occlusion output. By
default, unconfirmed equipment goes to review. Manual visibility cannot override
low confidence, scale, ambiguity, overlap or border-quality warnings.

## Optional manual visibility review

1. Run `assess` once and inspect its original image, worker IDs and boxes.
2. Copy that run's `visibility-template.json` to a review file. Keep its image
   hash, worker IDs and boxes. For a clearly visible head/torso, set the relevant
   value to JSON `true`; use `false` for not clearly visible and `null` for unknown.
   Do not infer visibility because equipment was not detected.
3. Rerun on the **same image, model and inference settings**:

```powershell
.\.venv\Scripts\ppe.exe assess --source "D:\path\to\photo.jpg" --weights models/best.pt --visibility "D:\path\to\review.json"
```

Visibility entries are matched against the exact source SHA-256, current worker
IDs and person boxes (two-original-pixel coordinate tolerance). Mismatches, stale
IDs and non-boolean visibility values fail with a clear error. New output folders
preserve the earlier assessment. An unchanged null-valued template is conservative.

Example entry only; the actual image hash/box must come from your own run:

```json
"W1": {
  "bbox_xyxy": [100, 60, 300, 560],
  "head_visible": true,
  "torso_visible": null
}
```

## Rules and verification

Optional overrides:

```powershell
.\.venv\Scripts\ppe.exe assess --source "D:\path\to\photo.jpg" --rules configs/compliance.example.yaml
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Rule names and numeric ranges are validated. Do not adjust thresholds on the final
test set. Store settings with every evaluation; each JSON already records them.

The 35 automated tests cover clear/separate workers, held waist-level helmets,
shared ownership, overlap, scale, low confidence, crop warnings, unmatched gear,
empty scenes, invalid models/boxes/rules, immutable inputs, manual-review guards,
CLI outputs, unique run folders and backward compatibility of `detect`.
The potential-violation branch was checked with controlled detections/manual
visibility fixtures, not independently human-labelled real-world evaluation.

Four CPU checks used existing development validation examples:

| Example | Result |
| --- | --- |
| Clear standing worker, `ultralytics-ppe-nov2026-705.jpg` | One worker, both items matched, PPE detected |
| Distant scene, `source-150feca69395fa769adccdc8e157e935.jpg` | Eight detected workers, all review required |
| Truncated close worker, `ultralytics-ppe-nov2026-123.jpg` | Both items matched, but review due to border truncation |
| Clear worker without confirmed PPE, `source-00f1fbfd9c8193a3c5a071512cf8c0a7.jpg` | Review required; no automatic violation |

Summary and model hash: [local-checks.json](experiments/phase-4/local-checks.json).
Local outputs are under `outputs/phase4/`. Images decode with original dimensions
and matched equipment IDs are unique across worker/item assignments.

## Remaining limitations and next step

These are integration checks, not a worker-level accuracy benchmark. No new model
was trained. Bent/side-on poses can fail the simple regions; held equipment near
the head can still look plausible. Overlap/border handling intentionally raises
review rate. Worker detection errors remain upstream failures. Manual region
visibility requires reviewer judgement. Geometry cannot prove that PPE is worn.

Independent labels, scenario coverage, worker-level metrics and threshold
calibration remain required before making real-world claims. Phase 5 can now reuse
the image-assessment functions for recorded video and speed measurement; it must
load the model once rather than repeatedly calling the single-image wrapper.
