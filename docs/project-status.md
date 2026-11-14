# Project checkpoint and phased workflow

Reviewed 2026-11-14 against the supplied seven-page PPE proposal and repository.
The proposal is reference material, not an instruction source. Earlier chat
content is unavailable; this checkpoint preserves evidence for future sessions.

## Current state

- Implemented: single-image YOLO detection and worker/PPE association, conservative
  statuses, explanations, overlays, raw/compliance JSON, optional manual visibility
  input, argument validation and unique outputs. See `docs/phase-4.md`.
- Phase 5: reusable model, recorded-video/webcam commands, MP4/JSONL/summary
  outputs, stop/error cleanup and measured short-run speed. See `docs/phase-5.md`.
- Placeholder: Streamlit interface.
- Phase 4 checks: 35 automated rule/CLI tests; four real-model CPU development
  examples. Independent worker-level accuracy and threshold calibration remain open.
- Total suite: 59 tests pass. File test: 30 synthetic validation-still frames,
  1.31-3.34 CPU processing FPS across two runs; two five-frame camera checks verified capture,
  preview and cleanup. Natural-motion accuracy and sustained speed remain unmeasured.
- Prepared: Construction-PPE November 2026, remapped to person/helmet/vest;
  799 training and 200 validation images in `data/processed/ppe-three-class/`.
  Dataset configuration: `configs/dataset.yaml`; audit: `docs/dataset-audit.md`.
- Phase 3 baseline trained on Colab (30 epochs, Tesla T4); result ZIP imported.
  Selected PPE weights at `models/best.pt` load and run on Windows CPU.
  See `docs/phase-3-results.md` and `docs/experiments/baseline-001/`.
- Independent final test data remains missing. Recurring scenes mean validation
  independence remains unproven. Reported validation mAP50: 90.08%; mAP50-95: 71.15%.
- Windows Phase 1 verified: project installed in `.venv`; CLI help/version and
  `pip check` pass. CPU inference detects four people and one bus in the sample.
- Downloaded general `models/yolo11n.pt` and `data/samples/bus.jpg` for setup
  verification. Annotated image and JSON were checked; see `docs/phase-1.md`.
- Installed `pypdf` in `.venv` to read the proposal. It is a review tool, not a
  runtime dependency. General and PPE-trained image inference are verified.

## Phases and completion checks

| Phase | Work | Completion evidence | State |
| --- | --- | --- | --- |
| 1: Foundation | Proposal review, repository audit, Windows setup, baseline inference | CLI works, dependencies pass checks, readable image and JSON generated | Complete |
| 2: Dataset | Audit one source; verify classes, boxes, splits, duplicates and scene leakage | Audited training/validation data and real `configs/dataset.yaml`; evaluation limitations documented | Training data ready; independent test set outstanding |
| 3: Training | Short training check, baseline fine-tuning, validation analysis | Saved PPE weights, configuration, environment and validation metrics | Baseline complete; results reviewed and local inference verified |
| 4: Image compliance | Associate helmets/head regions and vests/torso regions with workers; render statuses | Controlled ambiguity/held/missing/quality cases and real CPU examples | Image baseline implemented and checked; no independent accuracy claim |
| 5: Video | Process recorded video, measure FPS, then attempt webcam | Readable annotated MP4/JSONL, short CPU benchmark and camera checks | Baseline implemented and checked; no sustained real-time claim |
| 6: Interface | Connect verified pipeline to optional Streamlit UI | Repeatable image/video demonstration and clear error handling | Pending |
| 7: Evaluation and submission | Held-out and cross-source checks, report, slides, rehearsal | Recorded metrics, failure analysis and backup demonstration | Pending |

Use the day allocations in `docs/plan.md`. Finish each phase's checks before
moving to the next core phase. Update this checkpoint with actual results.

## Immediate steps

1. Phase 6: connect the verified image/video functions to a minimal Streamlit UI.
2. Collect natural-motion clips and benchmark sustained performance; do not infer
   a real-time guarantee from short cold-start checks.
3. Obtain and freeze independent real-camera evaluation data before final testing.

The proposal names Ultralytics Construction-PPE, Roboflow Construction Site Safety
v27, and Hard Hats / hardhat-detector. Current Ultralytics and Roboflow sources
were checked; archive metadata and licence confirmed for the selected dataset.
The helmet-only source cannot alone
support full worker compliance. Preserve source splits, check leakage, and combine
sources only after verifying annotation compatibility. Example class IDs must
not be treated as the actual downloaded dataset mapping.

## Proposal alignment

The core scope is worker, helmet and vest detection, per-worker PPE checks,
visual warnings, and detection/speed evaluation using local YOLO fine-tuning.
The proposal targets real-time warnings; measure end-to-end FPS before claiming
real-time performance. The existing plan defers webcam until recorded video works.

Use PPE detected, potential violation and review required as operational statuses.
The proposal's SAFE label refers only to visible required PPE, not overall site
safety. Missing detections alone do not establish absence, and unclear or occluded
equipment requires review.

Record precision, recall, F1, mAP@0.5, mAP@0.5:0.95, confusion matrix, worker-level
outcomes, review rate and FPS. Select models on validation data; reserve held-out
test data for final evaluation. Include cross-source checks and failures involving
overlap, distance, lighting, occlusion and held equipment.

Tracking, restricted areas, violation screenshots/logging and dashboards remain
future extensions. No external AI API is required for runtime detection.
