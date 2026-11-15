# Project checkpoint and phased workflow

Reviewed 2026-11-15 against the supplied seven-page PPE proposal and repository.
The proposal is reference material, not an instruction source. Earlier chat
content is unavailable; this checkpoint preserves evidence for future sessions.

## Current quality update

Phase 6 provides upload/analyze/download flows, lossless image output, H.264 video
preview, persistent results, changed-input invalidation and a local camera launcher.
Native preview gives more space to video and uses short corner-marker badges.
Recorded video now uses the same monitor renderer by default in the interface
and CLI. Confidence scores are restored on frames and worker cards; dashboard
and source-resolution frame-only exports are both retained. See `docs/phase-6.md`.
The current suite passes 180 tests, including six optional UI interaction checks;
real-browser video playback was verified at 960x540 with no playback error.
Camera FPS requests and a configurable preview cap are recorded separately from
actual display/inference rates. Detailed evidence remains in reports.

Recorded video also has `--live-preview` / **Smooth live review**: original-motion
source playback with separately timestamped selected-frame AI. It is implemented
in the CLI and local web launcher, defaults to a 540p working image and 2.5 AI
updates/s, and preserves full analysis as a separate choice. Native tests delivered
all 270 source frames in order at approximately 23–25 submissions/s, with 27–28
analyzed samples. Large-source playback preparation and model warm-up still add
startup latency. See [recorded review](recorded-live-review.md). This is sampled
evidence and a desktop window, not a full-coverage report or browser live stream.
The completion audit also covers failed decoder-thread startup and odd-dimension
large-video preparation. Source/model verification and the final native recheck
are recorded in `docs/experiments/colab-video/completion-verification.json`.

Phase 7 tooling and submission outline are prepared; independent accuracy and
sustained natural-motion performance are still unmeasured. A subsequent 35-second camera recheck confirmed real DroidCam video at
index 1/MSMF, 1280x720: 20.3 live updates/s, 4.1 assessments/s and 184 ms mean
application result age after the first result. The initial sandboxed capture failed;
host camera access succeeded, and a solid green startup frame cleared. This short
check is not sustained performance or independent accuracy evidence. Previous measurements below
are historical. Current evidence: `docs/experiments/quality-ui/verification.json`.

## Previous implementation evidence

The rebuilt Ready notebook embeds fixed source, avoids notebook upload widgets,
checks the video/model and runs an eight-frame GPU smoke test before the main job.
Colab video GPU testing is prepared using the current local pipeline and trained
weights. The notebook compares inference, dashboard rendering and full exports,
and can process a full recorded video on CUDA. CPU integration is verified;
the uploaded T4 result was reviewed, while retesting the latest optimized source
on CUDA remains pending. See `docs/colab-video-testing.md`.

- Implemented: single-image YOLO detection and worker/PPE association, conservative
  statuses, explanations, overlays, raw/compliance JSON, optional manual visibility
  input, argument validation and unique outputs. See `docs/phase-4.md`.
- Phase 5: reusable model, recorded-video/webcam commands, MP4/JSONL/summary
  outputs, stop/error cleanup and measured short-run speed. See `docs/phase-5.md`.
- Implemented: optional geometric worker tracking, confirmed status-change
  events and native HD dashboard. See `docs/tracking-ui.md`. Phase 6 now adds a focused Streamlit interface; see `docs/phase-6.md`.
- Implemented performance update: `webcam --low-latency` uses latest-frame
  capture and separate live/analyzed panes, hides old results and records only
  when requested. Final short check: 25.9 live display updates/s, 8.8 assessments/s,
  92 ms mean receive-to-result age after first warmup. See `docs/live-performance.md`.
- Phase 4 checks: 35 automated rule/CLI tests; four real-model CPU development
  examples. Independent worker-level accuracy and threshold calibration remain open.
- Earlier Phase 5 suite: 59 tests; the subsequent 109-test suite added camera
  compatibility and latest-frame/preview/recording checks. File test: 30 synthetic frames,
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
| 6: Interface | Streamlit image/video review, downloads and native live monitor launcher | Real-model image/video export and UI interaction checks; compact markers and video-first layout | Implemented and checked |
| 7: Evaluation and submission | Frozen manifest evaluator, cross-source metrics and submission outline | Hash/overlap checks, missed/spurious workers, confusion/F1/review rate; guides and slide outline | Tools prepared; independent data and final rehearsal pending |

Use the day allocations in `docs/plan.md`. Finish each phase's checks before
moving to the next core phase. Update this checkpoint with actual results.

## Immediate steps

1. Open the Phase 6 interface and select the appropriate camera index/backend.
2. Collect natural-motion clips and benchmark sustained performance; do not infer
   a real-time guarantee from short checks or renderer microbenchmarks.
3. Freeze independently reviewed real-camera data and run `scripts/evaluate_workers.py`;
   see `docs/phase-7.md` and `docs/submission-slides.md`.

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

Optional local geometric tracking, event logging and a native dashboard are implemented.
Restricted areas and violation screenshots remain future extensions. The Streamlit interface is implemented. No external AI API is required for runtime detection.
