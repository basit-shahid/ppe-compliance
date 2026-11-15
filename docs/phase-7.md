# Phase 7: evaluation and submission workflow

The evaluation runner and submission materials are prepared. Independent final
accuracy results remain pending because this repository has no separate reviewed
real-camera test set. Development integration clips are not final test evidence.

## Freeze the test set

Collect independent footage/images with collection provenance and permission.
Split by site/camera/session before sampling; adjacent frames from one scene
must not span training, validation and test. Include overlap, distance, poor
lighting, truncation, occlusion, held equipment and target-free backgrounds.
Do not tune model or rule thresholds using this final set.

Copy `configs/evaluation.example.json` to a new manifest. For each image, record
its relative path, SHA256, source group and every worker's pixel `bbox_xyxy` and
human-reviewed expected status. For backgrounds use `workers: []`. Set
`independent: true` only after a reviewer confirms independence. Record the
reviewer, annotation protocol, disagreements and provenance in annotation_notes.

Use the same three statuses as runtime: PPE detected, review required and
potential violation. Expected labels must reflect reviewed visible PPE evidence;
unclear equipment is review required. Predictions receive no ground-truth
visibility hints, so missing equipment normally produces abstention/review.
Report this limitation instead of treating abstentions as correct violations.

Get an image hash in PowerShell:

```powershell
(Get-FileHash data/evaluation/site-a/photo.jpg -Algorithm SHA256).Hash.ToLower()
```

## Evaluate frozen worker outcomes

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_workers.py --manifest configs/evaluation-final.json --output outputs/evaluation-final-001 --weights models/best.pt --image-size 640 --confidence 0.25 --exclude-dir data/processed/ppe-three-class/images/train --exclude-dir data/processed/ppe-three-class/images/val
```

The command verifies image hashes, duplicate images and exact pixel overlap with
the supplied development directories before loading the model. Hash matching
does not detect near-duplicate/adjacent scenes; manual scene-split review remains
necessary. Outputs refuse to overwrite previous evaluation directories.

Outputs include the manifest snapshot, model/manifest hashes, configuration,
per-image evidence and lossless overlays, plus overall and per-source-group
metrics. Greedy descending-IoU, one-to-one matches (default IoU 0.5) are independent
of status. Missed and spurious workers stay in denominators. Report detection
precision/recall, status precision/recall/F1, review rate, accuracy on matches,
correct statuses per expected worker, and the confusion table. Unsupported
metrics are null rather than invented perfect scores. This is worker-level
assessment evaluation, not object-detection mAP.

For per-class detection mAP, create separately audited YOLO-format test labels
and a dataset YAML with a real `test` split, then use the existing Ultralytics CLI:

```powershell
.\.venv\Scripts\yolo.exe detect val model=models/best.pt data=configs/evaluation-detection.yaml split=test imgsz=640 device=cpu project=outputs/evaluation-detection name=final-001
```

Save the complete output with per-class precision/recall, mAP50, mAP50-95 and
confusion matrix. Do not substitute validation metrics for this final result.

## Performance and rehearsal

Use natural-motion clips and at least a multi-minute camera run per chosen
profile. Record model load/first result separately, actual capture dimensions,
live display FPS, assessment FPS, result age and observed failures. Keep the
same hardware, source and duration when comparing profiles. A short renderer
microbenchmark is not sustained camera performance.

Submission pack: current README and phase guides, training evidence, frozen
evaluation results, failure-case overlays, [presentation outline](submission-slides.md),
and a recorded backup demonstration. The comprehensive PDF predates these
interface changes and remains labeled as a historical snapshot.

Rehearse: start the app, analyze one normal image and one difficult example,
open an exported video, demonstrate the native monitor and its stop control,
and explain uncertainty, skipped live frames, review rate and dataset limits.
