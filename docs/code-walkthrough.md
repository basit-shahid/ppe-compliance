# Code walkthrough and viva preparation

Read [the pipeline](pipeline.md) first. This guide identifies code you should be
able to explain, with examples tied to the implementation rather than memorized
model descriptions.

## 1. Read the code in this order

| Order | File and entry point | What to explain |
| --- | --- | --- |
| 1 | `detector.py`: `FrameDetector.__init__`, `detect` | Load weights once for a detector, call prediction, build validated JSON-compatible detections |
| 2 | `association.py`: `AssociationRules`, `_candidate_score`, `associate_ppe` | Bounding-box geometry, uncertainty and equipment ownership |
| 3 | `compliance.py`: `assess_detections` | How warning priority determines the worker status |
| 4 | `rendering.py`: `draw_assessment`, `worker_scores` | Copy source pixels, scale coordinates for display, use actual matched scores |
| 5 | `tracking.py`: `WorkerTracker.update` | IoU matching, session IDs, evidence expiry and debounce |
| 6 | `video_pipeline.py`: `process_video` | Decode → infer → assess → encode loop; streaming reports; cleanup |
| 7 | `live_pipeline.py`: `LatestSlot`, `process_live` | Separate capture/inference/display and newest-frame replacement |
| 8 | `recorded_preview.py`: `review_recorded_video` | Independent source playback and selected-frame evidence |
| 9 | `interface.py` and `app/streamlit_app.py` | Validation, atomic image exports, UI actions, fingerprints, downloads and subprocess launchers |
| 10 | `evaluation.py`: `score_workers`, `aggregate_scores`, `evaluate_manifest` | Matched/missed/spurious workers and honest denominators |

## 2. The central data structures

A detection has `class_id`, `class_name`, `confidence` and `bbox_xyxy`.
`model_detection_index` preserves the original prediction index after validation;
downstream `detection_id` indexes the accepted report list.

A worker record has an ID, a person box/score, a helmet evidence object, a vest
evidence object, quality issues, a status and reasons. Equipment states include
`matched`, `not_detected`, `uncertain` and `ambiguous`.

An assessment combines those workers with unmatched equipment, aggregate counts,
rule values and provenance. Image IDs are ordered frame-local `W1`, `W2`, etc.
Tracking replaces the display ID with session IDs such as `T001` while retaining
the original `frame_worker_id`.

The output is a dictionary tree because it serializes directly into JSON. Model
tensors are converted into Python integers/floats/lists before reporting. The
same report supports CLI output, image overlays, worker cards and test assertions.

## 3. Trace one concrete worker

Assume a 640×640 image and the following high-confidence detections:

```python
person = [100, 60, 300, 560]
helmet = [160, 65, 235, 120]
vest   = [125, 180, 280, 390]
```

The person is 200 px wide and 500 px tall. Its helmet candidate region is
`[90, 20, 310, 220]`; its vest region is `[96, 140, 304, 420]`. The equipment centers
and area coverage fit these regions. With adequate scores/size, no competing
worker and no border/overlap warnings, both items match and the status is
`PPE detected`.

Move the helmet down to waist level. The model might still correctly detect a
helmet object, but it no longer passes the head-region ownership test. The worker
now defaults to review because head visibility is unknown. This explains why
counting helmets and people alone cannot establish per-worker compliance.

## 4. Association code to understand

`area()` computes positive rectangle area. `intersection()` constructs an overlap
rectangle. `iou()` divides intersection area by union area.

For equipment placement we use **intersection / equipment area**, not IoU with the
entire person: a small helmet should occupy only a small part of the worker box.
For worker overlap and tracking, IoU is appropriate because two person boxes are
being compared.

`_candidate_score()` checks region center, coverage and relative dimensions before
returning a score. `associate_ppe()` sorts candidates, marks ambiguous ownership,
applies evidence strength gates, and selects the best remaining match. It records
extra or rejected equipment rather than silently overwriting evidence.

The rules are a frozen dataclass. Invalid, non-finite or out-of-range overrides
raise `ValueError`. Geometry is deterministic, making controlled cases testable
without running a neural network.

## 5. The status decision in plain Python

The following is explanatory pseudocode; read `assess_detections()` for the exact
implementation and diagnostic reasons:

```python
if quality_or_visibility_warnings:
    status = "review required"
elif an_item_is_missing_in_a_human_reviewed_visible_region:
    status = "potential violation"
else:
    status = "PPE detected"
```

Missing matches with unknown visibility add a warning before this decision.
That detail prevents an absence claim based solely on a detector miss. A human
visibility file is bound to source hash and box geometry so stale labels cannot
be applied to a different image or different worker ordering.

## 6. UI execution and state

Streamlit reruns the script whenever a widget changes. Expensive inference occurs
only inside the Analyze button branch. Results persist in `st.session_state` with
a fingerprint containing source bytes hash, mode, profile, confidence, device and
render/export settings.

If the current fingerprint differs, the old result is hidden and the user is told
to analyze again. Changing a worker evidence filter does not rerun inference.
Content hashes are cached by upload ID; when no stable ID is available, bytes are
hashed again. The included sample has a stable known identity.

The default camera and smooth-review buttons use `browser_preview.py` to launch
an isolated headless worker with `Popen` and a JSON configuration. Atomic JPEG and
status snapshots feed a periodically refreshed Streamlit fragment. UI code never
calls Streamlit from inference threads. A stop file and heartbeat lease control
worker lifetime; errors and logs appear directly in the preview panel. Settings
apply when the user restarts the preview. Optional Desktop window mode uses the
original native launch commands. No shell interpolation is used.

`workspace.py` contains presentation components. It imports Streamlit lazily, so
pure presentation helpers and core modules remain importable without the UI extra.
`workspace.css` controls the local frontend; `preview_ui.py` renders the native
monitor. Changing style does not change assessment rules.

## 7. Video loop and cleanup

`process_video()` validates options and opens the source. It creates a unique run
directory, reads source metadata, loads one model, then loops over frames. It
hashes the exact processing pixels, assesses, optionally tracks, renders, writes
video and flushes one JSONL row. Counters update only for processed frames.

The `finally` block releases each writer/capture independently and closes logs.
A failed encoder finalization changes the run status to failed. A camera
disconnect or an early video ending preserves previous evidence and a failed
summary rather than claiming full coverage. Keyboard cancellation is recorded as
a stopped run. EOF metadata can be imperfect; the report states that limitation.

`H264Writer` pipes BGR frames to FFmpeg, validates frame dimensions and dtype, and
checks process completion. MP4 requires even dimensions with the chosen pixel
format, so odd sizes receive at most one padding pixel; detections keep their
original coordinate space. Legacy conversion also pads odd dimensions and checks
that a real nonempty output file exists.

## 8. Why live display uses concurrency

Sequential inference can stall display. The live path gives capture and inference
their own workers and communicates through a latest-value slot. `LatestSlot`
replaces old unpublished frames instead of creating a growing queue. Frame pixels
are copied where a camera backend might reuse its buffer.

A result packet contains its own source packet, annotated pixels and assessment.
The moving source pane and latest analyzed pane are separate. Result-age checks
hide stale evidence; source timestamps are distinguished from wall-clock times.
This is intentionally sampled inference. It improves responsiveness without
claiming that every displayed frame was analyzed.

## 9. Tracking is conservative

`WorkerTracker.update()` expires stale tracks, scores candidate overlaps, rejects
ambiguous identities, and updates or creates tracks. A consistent green candidate
needs both the observation-count and time gates. A missing/uncertain current item
removes displayed green immediately, even while the confirmed event state is
debounced. This prevents an old positive state from masking a current dropout.

The tracker can switch IDs at crossings. It does not count unique workers across
a whole site or recognize a person returning later. Repeated video observations
are not independent people.

## 10. What the tests prove

Unit fixtures provide known boxes to verify ambiguity, held equipment, cropping,
quality gates and visibility policy. Integration tests use real temporary files,
fake predictions and camera backends to verify output integrity, timing and
cleanup. AppTest checks navigation, invalid uploads, retained results, setting
changes and launch arguments. Browser QA checks rendering and a real-model sample.

`scripts/verify_presentation.py` runs the suite, dependency checks, actual local
weights on the sample and an eight-frame synthetic video. It decodes the output
and checks frame count, dimensions, FPS, JSONL rows and source timestamps.
It never opens a physical camera or downloads weights. Passing these checks is
evidence of software integration, not an independent accuracy measurement.

## 11. Common examiner questions

**What is original in your project if you use YOLO?**
YOLO provides the detector. The project audits/remaps data, fine-tunes the model,
implements equipment ownership and conservative status rules, builds tracking and
video orchestration, and provides evidence exports, evaluation and the interface.
Do not claim to have invented the YOLO architecture.

**Why YOLO11 nano?**
It is the small baseline selected for local-resource constraints. The repository
contains one recorded fine-tuning baseline, not a controlled comparison proving
it is the best architecture.

**Why is the confidence threshold 0.25 but an item is uncertain at 0.40?**
0.25 retains detection candidates; the evidence policy requires at least 0.50.
Candidate visibility and accepted evidence strength are separate decisions.

**Is 90.08% the accuracy of your system?**
No. It is saved validation mAP at IoU 0.5 for object detection. Worker status
accuracy needs separately labeled people/statuses and independent scenes.

**Why does a worker with both boxes sometimes show amber?**
Low confidence, insufficient size, border truncation, overlap, ambiguity, explicit
visibility warnings or tracking confirmation can still require review.

**Can the UI show red automatically when there is no helmet?**
No. Missing detection with unknown head visibility defaults to review. The CLI
supports explicit human visibility input for a reviewable potential violation.

**Why not use the original dataset's negative classes?**
This project explicitly prepares a three-class object detector. Absence is not
learned as a fourth runtime class; the conservative policy handles uncertainty.

**Does a 30 FPS video mean 30 FPS inference?**
No. Playback FPS is media timing. Processing FPS and live display rate are
separately measured. The sampled live path intentionally skips unselected frames.

**What happens if two workers overlap?**
Overlap triggers review and close equipment ownership scores become ambiguous.
Tracking also rejects ambiguous identity inheritance. It cannot guarantee IDs.

**How do you reproduce a result?**
Keep source bytes, source/model hashes, rules, confidence, input size, device and
environment. The JSON reports preserve these. A seeded training run improves
repeatability but is not a guarantee of bit-identical results on all hardware.

**What remains unfinished scientifically?**
Independent human-reviewed real-camera evaluation, sustained deployment
benchmarks and controlled threshold/model comparisons. Software readiness does
not replace those measurements.

**What would you improve next?**
Collect independent scenes, review missed small helmets and crowded/occluded
workers, then run controlled changes against frozen data. Add an appearance-based
tracker or learned wearing-state model only with suitable labels and evaluation.
