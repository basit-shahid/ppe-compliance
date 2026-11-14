# Phase 3 baseline results

Imported and checked on 2026-11-14 from `models/baseline-001-results.zip`.
Original ZIP retained; extracted files are in
`outputs/colab-results/baseline-001/`. Selected weights copied to `models/best.pt`.

## Training evidence

- Model: YOLO11 nano, fine-tuned for person, helmet and vest.
- Data: 799 training / 200 validation images, audited November 2026 derivative.
- Archive provenance matches the audited dataset SHA-256 and expected counts.
- GPU: Tesla T4; Ultralytics 8.4.212; PyTorch 2.14.1+cu130.
- Completed 30 epochs, image size 640, batch 8, seed 42.
- Training log's final cumulative time: 950.731 seconds (about 15.8 minutes).
  This excludes setup, dataset download, and separate final validation.
- Imported model class map verified locally: 0 person, 1 helmet, 2 vest.
- Best-model SHA-256:
  `42538d40127ef78117cfade974369bf8ae993182442232b8faf670fd4cbc1448`.

Small evidence files are preserved in `docs/experiments/baseline-001/` and the
experiment is logged in `docs/experiments.csv`. Large weights/images stay local.

## Reported validation results

These are from the supplied `validation-metrics.json`, not a new full local
validation run. The separate final validation differs slightly from the rounded
last training-log row; both originals are retained.

| Metric | Result |
| --- | --- |
| Mean precision | 94.99% |
| Mean recall | 83.54% |
| mAP@0.5 | 90.08% |
| mAP@0.5:0.95 | 71.15% |

| Class | Precision | Recall | F1 | AP@0.5 |
| --- | --- | --- | --- | --- |
| Person | 95.26% | 87.32% | 91.11% | 93.79% |
| Helmet | 92.82% | 76.53% | 83.89% | 83.58% |
| Vest | 96.88% | 86.76% | 91.55% | 92.86% |

mAP is a detection metric, not overall classification accuracy. Precision/recall
are evaluated at the validator's selected confidence operating point, not
necessarily the CLI's default confidence of 0.25.

## Local verification and review

The selected model loaded successfully with Windows CPU-only PyTorch. Two
validation examples were used for integration checks, not independent evaluation:

- Distant-worker image: 8 person and 2 vest detections, no helmets detected at
  confidence 0.25/image size 640. Output:
  `outputs/20261114T090254Z-dfc3b596/annotated.jpg`.
- Close worker: person, helmet and vest detected. The annotated image was visually
  inspected. Output: `outputs/20261114T090336Z-f2a41fec/annotated.jpg`.

Both output images decode with correct dimensions and valid in-image JSON boxes.
Training curves show declining training and validation losses and improving mAP;
the plots do not show an obvious late divergence over these 30 epochs. That does
not rule out overfitting to recurring scenes.

Helmet recall is the main class weakness. The justified next improvement to
investigate is better coverage of small/distant helmets and scene-independent
validation, then a controlled image-size experiment if needed. No second training
run or threshold optimization was performed during this import.

## Limits and next phase

Recurring backgrounds, generated images and the missing independent test set
still limit real-camera claims. Colab's reported inference timing is not local
CPU throughput or end-to-end video FPS. These results do not establish that
equipment is worn or that a workplace is safe.

The baseline training milestone is complete. Next is Phase 4: associate detected
helmets/vests with individual workers and produce uncertainty-aware statuses.

To test another still image locally:

```powershell
.\.venv\Scripts\ppe.exe detect --source "D:\path\to\photo.jpg" --weights models/best.pt --device cpu
```
