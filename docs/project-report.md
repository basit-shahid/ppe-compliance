# Real-Time PPE Compliance and Safety Violation Detection

Comprehensive project review and implementation roadmap

Status date: 14 November 2026. Prepared for a three-member semester project team.
Names can be assigned to the suggested roles; responsibilities below are proposals,
not claims about who performed work already completed.

## 1. Project purpose and current position

### The problem we are addressing
Construction and industrial environments require workers to wear appropriate protective equipment. Reviewing every camera frame manually is time-consuming. This project investigates a local computer-vision prototype that locates workers, helmets and high-visibility clothing, then assists a reviewer by displaying an interpretable PPE status for each visible worker.

The project does not determine whether an entire workplace is safe. It examines two visible PPE requirements: a helmet and a vest or equivalent high-visibility clothing. It cannot verify equipment certification, correct fit, hidden equipment or all other hazards. Its output should support human review rather than act as an autonomous safety enforcement system.

### What exists now
The project has a working Windows environment, a checked three-class dataset, a completed GPU fine-tuning run, selected model weights and verified still-image inference. The trained model identifies **person**, **helmet** and **vest** and exports an annotated image plus structured JSON detections. It runs locally without an external AI API.

| Milestone | Verified position |
| --- | --- |
| Phase 1: foundation | Complete: installation, CLI, error handling and baseline image output |
| Phase 2: data preparation | Training/validation data ready; independent test data outstanding |
| Phase 3: model training | Baseline complete: 30 epochs on Colab; weights checked locally |
| Phase 4: worker compliance | Planned: association, uncertainty rules and worker status |
| Phase 5: video/live stream | Planned project integration and performance measurement |
| Phase 6: demonstration UI | Streamlit placeholder exists; integration is planned |
| Phase 7: final evaluation | Planned independent evaluation, report and presentation |

### The main transition ahead
A detector answers, "Where are the people and equipment?" The next phase must answer, "Which equipment belongs to which worker, and is the visual evidence clear enough to report a status?" A helmet held in a hand may be correctly detected but must not be interpreted as a worn helmet. This is why completing model training does not complete the compliance system.

The available proposal is a seven-page slide-style PDF, **PPE_AI_Project_Proposal.pdf**. No separate editable PowerPoint file was supplied. Alignment in this report refers to that presentation PDF and the inspected repository evidence. [1-5]

## 2. Alignment with the proposal presentation

The implementation follows the proposal's central idea: pretrained YOLO plus project-specific fine-tuning, followed by per-worker rules and visible warnings. It also retains the proposal's separation between development/evaluation and runtime processing.

| Proposal page | Commitment | Current evidence and remaining work |
| --- | --- | --- |
| 1: project identity | Real-time PPE compliance using deep learning | Same semester scope. Real-time performance remains a measurement target, not an achieved claim. |
| 2: problem and idea | Camera input, AI detection and automatic status | Image detection works. Per-worker status and integrated camera input remain planned. |
| 3: objectives | Detect workers and PPE; associate items; show warnings | Detection baseline complete. Phases 4-5 deliver association and overlays. Extension readiness does not require implementing tracking now. |
| 4: datasets | Compatible labels, duplicate checks, held-out and cross-source evaluation | One compatible dataset audited and remapped; one cross-split variant excluded. Independent final test and cross-source check remain open. |
| 5: technology | Python, YOLO/PyTorch, OpenCV, analysis, Colab, optional Streamlit | Actual model trained with YOLO/PyTorch on Colab; local CPU inference works. OpenCV is used. Streamlit integration remains optional planned work. |
| 6: methodology | Separate training pipeline and image/video/webcam runtime | Training pipeline completed for the baseline. Image inference works; worker rules and video integration are next. |
| 7: expected output/evaluation | Worker status, precision/recall/F1, mAP, confusion matrix, FPS | Development detection metrics exist. Worker-level evaluation, independent test data and measured end-to-end FPS remain pending. |

### Changes that must be explained in the final presentation
**Dataset version:** the proposal describes an older 1,416-image, 11-class Construction-PPE release. The archive actually used is the November 2026, 1,000-image, 12-class release. Its included licence states CC BY 4.0 for images and annotations. Dataset licensing must be distinguished from software licensing.

**No supplied test split:** the current release contains training and validation only. We did not rename validation images as a test set or invent independent test results. Independent real-camera evaluation must still be added to satisfy the proposal's full evaluation plan.

**Status wording:** the proposal illustrates SAFE / PPE VIOLATION and also acknowledges uncertainty. The planned UI uses **PPE detected**, **potential violation** and **review required**. This preserves the intended visual warning function while avoiding a claim of general safety or proven absence from a missed detection.

**Webcam priority:** the proposal includes webcam input. The repository plan prioritizes recorded video first and treats webcam as a stretch milestone. For full input-mode alignment, integrate and demonstrate webcam after recorded video, or explicitly state this scope limitation at submission. A built-in YOLO webcam command alone does not complete the project's own video pipeline. [1-4]

## 3. Architecture: how the phases connect

### Development pipeline
Dataset source -> annotation audit -> class conversion -> preserved train/validation split -> pretrained YOLO fine-tuning -> validation analysis -> selected weights and provenance.

Phases 1-3 establish this pipeline. Phase 7 adds independent final evaluation. A new model should be adopted only with recorded data version, configuration and evidence; a weight filename alone is insufficient to reproduce an experiment.

### Intended runtime pipeline
Image/video/camera -> frame decoding -> YOLO detections -> person/PPE association -> uncertainty-aware rules -> per-worker overlay -> annotated output and structured report.

The detection portion works for still images today. The association and rule stages are placeholders. Phase 4 connects them for images; Phase 5 repeats the verified pipeline over frames; Phase 6 exposes it through a simple interface.

| Component | Responsibility | Position |
| --- | --- | --- |
| src/ppe_compliance/detector.py | Decode image, load model, predict boxes, return detections | Implemented |
| src/ppe_compliance/cli.py | Parse options, reject invalid input, create unique output folders | Implemented for still images |
| association.py | Link visible helmets/vests to candidate workers | Placeholder |
| compliance.py | Convert association evidence into worker status and reasons | Placeholder |
| video_pipeline.py | Decode/process/write frames and measure speed | Placeholder |
| app/streamlit_app.py | Upload, run and display the verified pipeline | Placeholder |
| configs/ and scripts/ | Dataset mapping, exclusions and reproducible preparation/Colab packaging | Available |
| data/, models/, outputs/ | Local data, selected weights and generated artifacts | Large artifacts excluded from Git |

### Information passed between modules
Detection records currently contain class ID/name, confidence and a pixel box `[left, top, right, bottom]`. A future worker record should add a frame-local worker ID, person box, helmet/vest evidence, ambiguity flags, final status and a short explanation. These additional fields are a proposed interface, not an implemented feature.

Keep the inference model loaded while processing video rather than recreating it for every frame. Keep UI logic separate from inference and rules so CLI, video and Streamlit share one implementation. A frame-local label such as W1 is not a persistent identity across frames. [2,5]

## 4. Phase 1: foundation and environment

### Purpose and starting input
The first phase answers whether the project can run reliably on the available machine before introducing dataset and training complexity. Inputs were the proposal, repository scaffold and a Windows laptop. The phase established a repeatable installation and a known-good general-object inference path.

### Work actually completed
The repository and proposal were reviewed, implemented modules were distinguished from placeholders, and phase checkpoints were recorded. The project was installed in editable mode in `.venv`, including Ultralytics, OpenCV, NumPy and PyYAML. CLI help/version and dependency consistency were checked.

General pretrained `yolo11n.pt` weights and the public `bus.jpg` sample were downloaded. Running image inference produced four people and one bus. The output image was readable, dimensions were preserved, and JSON boxes were finite, ordered and within the original image. Missing files, corrupt images, invalid confidence and invalid image size produced nonzero exit codes and clear messages.

### Hardware and compute decision
The inspected laptop is an HP ProBook 440 G4 with an Intel i5-7200U, two CPU cores/four threads, approximately 8 GB RAM and Intel HD Graphics 620. The local PyTorch build is CPU-only and reported no CUDA. This supports development and still-image inference, but GPU training was moved to Colab for practicality. No local real-time throughput claim has been measured.

The recorded baseline used Python 3.13.3 and Ultralytics 8.4.212. Resolved versions were saved in `docs/environment-windows.txt`; this is an environment snapshot, not a portable lockfile. Subsequent PDF/report tooling installations are separate from that original Phase 1 snapshot.

### Deliverables and acceptance
Deliverables are the working CLI, sample outputs, error-check evidence, environment snapshot and `docs/phase-1.md`. The acceptance gate was a reproducible command that generated a readable image and valid JSON with consistent dependencies. That gate passed.

### Contribution to the final project
This phase supplies the common image-input/output path and error handling used by later phases. If training later fails or a prediction looks wrong, the team can distinguish a model/data problem from a broken installation. The general model only verified the pipeline; it did not prove PPE detection accuracy. [2,5]

## 5. Phase 2: dataset selection and audit

### Why this source was selected
The current Ultralytics Construction-PPE archive was chosen as a practical first baseline because it includes all three project targets, provides YOLO labels and has a direct public download. It is suitable for a semester baseline, not established as universally best. The proposal's helmet-only source cannot alone support full worker/helmet/vest analysis.

The November 2026 archive contains 800 training and 200 validation images, with 14,863 boxes across 12 classes. Its metadata describes collected and generated imagery with human and AI annotation. Both the original archive and extracted data were retained. The SHA-256 pins the exact downloaded snapshot because the release URL can change.

### Preparation actually performed
The data was **already annotated by the publisher**. We did not manually draw every box. We checked its existing annotations and made a separate task-specific copy. Source class IDs were remapped: Person 5 -> project person 0; helmet 0 -> project helmet 1; vest 2 -> project vest 2. Other annotation classes were omitted from the prepared copy; original labels remain intact.

All 1,000 images were decoded and annotation rows checked for five finite fields, valid integer IDs, positive normalized sizes and boxes within image bounds. No missing label files or invalid images were found. Exact decoded-image duplicate checks found none. A difference-hash search flagged one cross-split near-duplicate; visual review confirmed nearly the same doorway composition with changed worker/PPE. Its training variant was excluded without moving images between splits.

| Prepared split | Images | Person boxes | Helmet boxes | Vest boxes |
| --- | --- | --- | --- | --- |
| Training | 799 | 1,961 | 1,281 | 1,476 |
| Validation | 200 | 481 | 294 | 340 |
| Independent test | Not available | - | - | - |

Eight training images have no remaining target labels. Visual inspection found forklifts without people, so they were retained as background examples. Sample annotation sheets were reviewed. Ultralytics' loader checked both prepared splits with zero corrupt entries and loaded sample tensors successfully.

### Outputs, limits and contribution
Outputs include the prepared dataset, `configs/dataset.yaml`, the preparation script, manifest, audit, attribution and reviewed exclusion. The training/validation preparation gate passed. Scene independence and exhaustive semantic annotation completeness were not established. Recurring backgrounds remain, and the independent final test set is outstanding. Phase 2 made training possible while exposing the evaluation gap that Phase 7 must close. [3]

## 6. Understanding images, labels and splits

### Why gallery images have no drawn boxes
YOLO training uses clean images and separate text files. For example, `images/train/example.jpg` is paired with `labels/train/example.txt`. The image remains unchanged; the text contains one row per annotated object. Coloured boxes in our previews are temporary visualizations of those coordinates.

```text
class_id  center_x  center_y  width  height
0         0.50      0.50      0.40   0.80
```

This illustrative row means a person box centred halfway across and down the image, with width 40% and height 80% of the image. It is an explanatory example, not a copied training record. Normalized coordinates let the loader transform boxes consistently when resizing and augmenting images.

For image width W and height H, the pixel corners are `(center_x - width/2)*W`, `(center_y - height/2)*H`, `(center_x + width/2)*W`, `(center_y + height/2)*H`. The runtime JSON instead reports pixel corner coordinates directly. Confusing these two formats would produce incorrect overlays and associations.

### What each split does
**Training:** model weights change while learning from these examples. **Validation:** measures development progress, selects checkpoints and supports justified tuning. **Independent test:** evaluates the frozen final choice; it must not guide thresholds, hyperparameters or model selection.

The current dataset supplies only the first two. A random collection of frames from one video can also leak scene information: neighbouring frames should be kept in the same group rather than divided across splits. A separate file name or a different PPE colour does not necessarily imply an independent scene.

### What annotation does not tell us
The person, helmet and vest boxes are separate. They do not explicitly label ownership, wearing state or persistent worker identity. A helmet can be near a worker without being worn. The source vest class includes high-visibility jackets and shirts as well as vests. These definitions must be reflected in the report and demo, rather than silently promising narrower clothing recognition.

Annotation checks establish structural usability and sampled visual consistency, not perfect labels. For later evaluation, reviewers will also need worker-level reference outcomes and an explicit unclear/occluded category. That additional ground truth supports testing the rules introduced in Phase 4. [3,5]

## 7. Phase 3: what Colab actually trained

### Learning objective
Fine-tuning adapts a pretrained YOLO11 nano detector to our three target classes. The model sees a training image, predicts object boxes/classes, compares predictions with the supplied annotations and updates its weights to reduce detection losses. The selected model remains an object detector; no worker-association logic is learned by the current training setup.

### Remote workflow executed
The project supplied a notebook and a small project bundle containing preparation logic, exclusions, attribution and pretrained weights. Colab downloaded the dataset directly, verified its hash, reproduced the audited derivative and checked class/instance counts. This avoided uploading the full image collection from the laptop.

The workflow included a short training check followed by a separate baseline initialized from general pretrained weights. The imported evidence confirms the baseline completed 30 epochs on a Tesla T4. Configuration: image size 640, batch 8, seed 42, patience 10 and no image cache. The final CSV records 950.731 cumulative training seconds, about 15.8 minutes, excluding setup, download and separate final validation.

The notebook saved checkpoints and experiment metadata to Drive and exported a results ZIP. The user returned that ZIP. Locally, we extracted it, verified data provenance and class names, installed `models/best.pt`, reviewed curves and metrics, and ran two CPU integration checks.

| Artifact | Interpretation |
| --- | --- |
| weights/best.pt | Checkpoint selected by validation fitness; selected local model |
| weights/last.pt | Final-epoch checkpoint; optimizer state may be stripped after finalized training |
| results.csv / results.png | Per-epoch losses, metrics and visual training history |
| validation-metrics.json | Separately reported validation aggregate and per-class measurements |
| confusion matrix and curves | Diagnostic views of misses, false predictions and class performance |
| args.yaml / provenance / environment | Configuration, dataset hash, hardware and package evidence |

### Acceptance and contribution
The baseline gate passed: selected PPE weights, recorded configuration/results and successful local inference exist. A second training experiment has not been completed. The next justified model improvement is better coverage of distant/small helmets and more independent development scenes before treating a longer training run as a solution. Phase 3 supplies the detector consumed by every remaining phase. [4,6]

## 8. Phase 3 results and their interpretation

The following values come from the returned final validation JSON. They were reviewed locally but the full validation set was not rerun on the laptop. The separate validation differs slightly from rounded final-epoch CSV values; both source records are retained.

| Metric | Reported value | Interpretation |
| --- | --- | --- |
| Mean precision | 94.99% | A large proportion of reported detections match the labelled objects at the evaluation setting. |
| Mean recall | 83.54% | A noticeable proportion of labelled objects are still missed. |
| mAP@0.5 | 90.08% | Mean average precision using an IoU matching threshold of 0.5. |
| mAP@0.5:0.95 | 71.15% | Mean AP across IoU thresholds from 0.5 to 0.95; stricter box matching lowers the score. |

| Class | Precision | Recall | F1 | AP@0.5 |
| --- | --- | --- | --- | --- |
| Person | 95.26% | 87.32% | 91.11% | 93.79% |
| Helmet | 92.82% | 76.53% | 83.89% | 83.58% |
| Vest | 96.88% | 86.76% | 91.55% | 92.86% |

IoU is intersection area divided by union area between two boxes. Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is their harmonic mean. Average precision summarizes a precision-recall curve, and mAP averages AP across classes. These are detection measurements, not a statement that the whole system is "90% accurate."

Precision/recall depend on the evaluation operating point and matching rules. The CLI's default confidence 0.25 is not necessarily the operating point used to summarize those metrics. Model confidence is a score, not a verified probability of correctness.

The observed curves show declining train/validation losses and improving mAP over 30 epochs, with no obvious late divergence in those plots. That does not rule out learning recurring scenes. Helmet recall is the weakest class and deserves focused failure analysis. These measurements do not yet evaluate PPE ownership, violation warnings, independent footage or local video speed. [4]

## 9. Visual evidence and local checks

![Selected PPE model running on a validation image locally: person, helmet and vest boxes. This verifies integration, not independent accuracy or worker-level compliance.](assets/phase-3-local-detection.jpg)

The close-worker check produced three detections at confidence 0.25: person approximately 0.955, helmet 0.948 and vest 0.832. The image was visually inspected and JSON boxes checked against image dimensions.

An earlier distant-worker example produced eight person detections and two vest detections, with no helmet detections at image size 640. This is a concrete warning that small equipment may be missed. We did not hide that result or interpret the absence of helmet boxes as proof of non-compliance.

![Returned 30-epoch learning curves. Losses decrease and validation mAP improves; recurring-scene limitations still apply.](assets/phase-3-training-curves.png)

The selected model SHA-256 is recorded in the evidence appendix. The source ZIP is retained in `models/`; extracted results are under `outputs/colab-results/baseline-001/`. Compact result/configuration evidence is also retained in `docs/experiments/baseline-001/`. [4]

## 10. Phase 4: worker association and image decisions

### Purpose, inputs and proposed implementation
Phase 4 turns detections into explainable per-worker results. Inputs are the original image, model detections and documented rule settings. Its outputs will be worker records, status overlays and JSON explanations. The `association.py` and `compliance.py` modules are currently placeholders; everything below is a proposed design to implement and validate.

First normalize detection records, preserve the model's actual class names, and identify candidate people. For each person, derive approximate head and torso search regions from the person's box. Use containment, region overlap, relative location and equipment size to score candidate matches. A box inside the person's full rectangle alone is insufficient, especially when the helmet is held near the waist.

Second enforce consistent ownership: an equipment detection should not satisfy two workers simultaneously. If two candidate workers have similar match scores, preserve ambiguity rather than forcing a confident assignment. Neighbouring bodies, bent poses and partial people must be represented explicitly. Initial geometry is a transparent baseline, not a proven pose estimator.

Third apply decision rules with reasons. Thresholds should be configurable and selected on development examples. They must not be chosen after inspecting the final test set. A low-confidence or distant worker may need review even when candidate equipment is nearby.

| Planned status | Evidence policy |
| --- | --- |
| PPE detected | Both required items have plausible, unambiguous region matches. This reports visible PPE only. |
| Potential violation | A required item is not confirmed on an assessable worker; show which item needs checking. This is a reviewable alert, not proven absence. |
| Review required | Occlusion, scale, confidence, pose or competing matches make the evidence insufficient. |

### Handling visible absence responsibly
The three-class model does not directly predict a bare head or bare torso. Geometry cannot reliably determine body-region visibility by itself. The team must define a conservative assessability policy, test it against human labels and use review required when it cannot be established. Missing detections should never silently become a confirmed violation. Using dedicated absence labels or pose estimation would be separate, justified extensions rather than an existing capability.

### Deliverables and acceptance gate
Implement pure association/rule functions, a proposed worker-result schema, annotated output and CLI integration. Verify scenes covering both PPE items, one unconfirmed item, held equipment, two overlapping workers, small workers and occlusion. Test that one helmet cannot satisfy two people and that uncertain cases retain reasons. Passing those checks establishes the complete still-image workflow needed by video and UI work. [1,2]

## 11. Phase 4 worked examples and verification

These examples describe intended behaviour. They are not outputs already generated by the current code.

| Scenario | Desired reasoning and result |
| --- | --- |
| Single clear worker with both items | Head/torso matches are unambiguous; PPE detected. |
| Helmet held in hand | Position fails the head-region check; do not count it as worn. Review or potential violation depends on assessability. |
| Two overlapping workers, one helmet | Do not give the same helmet to both. Competing ownership should trigger review. |
| Upper body hidden behind machinery | Equipment may be present but invisible; review required. |
| Tiny worker in the distance | Treat low resolution as uncertainty; absence of equipment boxes is weak evidence. |
| Helmet confirmed, vest not confirmed | Explain the item-level result; potential violation only under the documented assessability policy. |
| No person detected | Report no detected workers, not an empty scene certified safe. |

### Suggested JSON design
```json
{
  "worker_id": "W1",
  "helmet": {"state": "matched", "detection_id": 4},
  "vest": {"state": "uncertain", "detection_id": null},
  "status": "review required",
  "reasons": ["torso visibility not established"]
}
```

This proposed record makes decisions inspectable. The implementation should also include the person box, frame index when applicable, thresholds/configuration and a list of unmatched equipment. IDs in this example are illustrative and frame-local.

### Verification strategy
Use controlled detection-box fixtures to test geometry and ambiguity without running the network, then consented labelled images for end-to-end behaviour. Two reviewers should agree on reference worker-level states or document disagreement. Cases with missing persons must be tracked separately so compliance metrics are not computed only on easy detected workers.

The first image milestone should pass before video smoothing or tracking is introduced. Otherwise temporal behaviour can hide mistakes in the basic rule logic. Keep example outputs and failed cases so the final presentation can explain both strengths and boundaries of the prototype.

## 12. Phase 5: recorded video and live input

### How processing will work
Load the selected model once. Open the input with OpenCV, decode frames, run the Phase 4 image pipeline, draw worker/equipment/status overlays and write an annotated output video. Track processed-frame counts and timing, and release capture/writer resources on normal completion and error paths. Preserve aspect ratio and ensure the chosen output codec creates a playable file.

Start with short recorded clips because they are repeatable and can be compared between code versions. Include a clip with clear workers and one with distance, overlap or occlusion. Log resolution, input rate, processed frames, model image size, confidence and hardware. If frame skipping is used, disclose it and avoid implying that every source frame was checked.

### Live camera integration
After recorded-video stability, add webcam input, camera-unavailable handling, a stop control and predictable cleanup. Camera index 0 often selects the default device but is not guaranteed. Network CCTV/RTSP input would require a known stream source, connectivity and separate testing; it has not been configured here.

Ultralytics' built-in webcam command can be used now for exploratory object detection. It does not implement our worker rules or prove the custom video pipeline is complete. The current CLI supports still images only, and no verified live-camera run is recorded in project evidence.

### Performance measurement and trade-offs
Measure wall-clock end-to-end throughput: processed frames divided by elapsed processing time, with clearly stated inclusion of decode, inference, association, drawing and writing. Also report per-frame latency and first-frame/model-load cost separately. Colab model inference milliseconds are not local CPU video FPS.

The laptop's CPU and integrated graphics may require lower resolution, a smaller model input or fewer processed frames. Benchmark the trade-off rather than promising real-time speed. Reducing image size can worsen small-helmet recall, already a weakness. Define an acceptable project demo rate against the target camera and hardware before deciding whether "real-time" is justified.

### Contribution and completion
Deliver a playable annotated video, structured frame/worker results, a measured benchmark and a stable stop/error path. If webcam remains deferred, state that explicitly against the proposal. Temporal persistence rules, if added, should be documented and tested; persistent identities and worker tracking remain future scope. This phase demonstrates how the still-image intelligence operates over time.

## 13. Phase 6: demonstration interface

### Intended user journey
The optional Streamlit interface will let a user select an image or short video, choose an installed model and supported settings, start processing, inspect output and download the results. Interface code should call the same verified pipeline used by the CLI. The current app is only a placeholder and does not perform detection.

The main view should show the original and annotated result, a worker-status summary, item-level reasons and a clear indication when review is required. A legend should explain that PPE detected means visible required equipment, not overall safety. Raw class IDs or internal geometry settings need not appear in the main user flow; useful settings and detailed reports can be secondary controls.

### Implementation approach
Cache model loading where appropriate, validate file type and decoding, use unique temporary/output directories and handle failures without replacing earlier results. Avoid loading unlimited video files into RAM on the 8 GB laptop. Show processing progress for recorded video and make export paths predictable.

Use consistent status colours and text so colour is not the only way to understand a result. Do not expose example or fabricated numbers as live results. The UI must reflect actual returned worker records, including no detections and ambiguous evidence. If live camera is offered, verify its behaviour in the intended local browser/runtime rather than assuming a video-upload widget provides webcam access.

### Acceptance checks and contribution
Check valid image/video uploads, corrupt and unsupported files, repeated runs, missing weights, empty detections and partial failures. Confirm output downloads contain the same results displayed on screen. Complete a rehearsal on the actual presentation laptop and keep a recorded demonstration as backup.

This phase converts the technical pipeline into a repeatable demonstration that an instructor can understand without entering CLI commands. It does not replace missing evaluation or fix incorrect worker association. Integration should happen after those core behaviours work, and feature freeze should precede final measurements and slides.

## 14. Phase 7: independent evaluation and submission

### Evaluation design still needed
Collect consented real-camera examples distinct from the recurring training/development scenes. Include clear PPE, visible unconfirmed equipment, difficult lighting, distant workers, bent poses, held helmets, overlap and partial occlusion. Record source, permission, capture conditions and grouping. Annotate both object boxes and worker-level reference states, including an explicit unclear category.

Separate any additional development examples from the final evaluation set. Agree on scenario coverage and sample size with the instructor before final evaluation; no fixed sample count is claimed in this report. Freeze the final model, rule settings and evaluation procedure before examining test results. If the system is changed after seeing a failure, those examples become development evidence and fresh held-out evidence is needed for a clean final claim.

### Measurements to report
| Layer | Evidence required |
| --- | --- |
| Detection | Per-class precision, recall, F1, mAP@0.5, mAP@0.5:0.95, confusion matrix and object counts |
| Worker-level decisions | Status confusion table, item ownership errors, false alerts, missed concerns and review rate |
| Coverage | Missed workers, unclear cases, excluded samples and scene/worker totals |
| Runtime | End-to-end FPS/latency, resolution, input size, frame policy and hardware |
| Generalization | Independent real-camera results and a separate compatible cross-source check where feasible |

Define how detections are matched to ground truth and how review cases affect each denominator. Report review rate alongside accuracy on decided cases; otherwise a system can appear accurate by refusing difficult examples. Count false alerts and missed concerns separately because they affect usability differently. A confusion-matrix column normalized to background is not a universal false-positive percentage.

### Final deliverables and acceptance
Package the report, updated proposal/presentation slides, experiment log, model provenance, setup/run instructions, selected weights locally and demonstration media. Include a failure-analysis section and show at least one uncertainty case in the demo. Provide a backup recorded video and results screenshots in case the webcam, GPU or network is unavailable.

Phase 7 makes the project defensible: it connects the original objectives to measured behaviour, identifies unresolved limitations and supports a reproducible presentation. Final completion requires independent evidence and a working integrated demonstration, not just a high validation mAP value.

## 15. Suggested responsibilities for three members

Phases 1-3 are already completed to the extent documented. The following allocation begins with remaining work; it does not retrospectively assign credit. Replace Member A/B/C with the actual team names after agreeing responsibilities.

### Member A: detection and evidence lead
Own model loading/configuration and detection diagnostics. Lead independent object-box annotation guidance, model provenance and per-class evaluation. Investigate helmet failures and propose a controlled model/input-size experiment only when evidence justifies it. Support geometry fixtures with realistic box distributions. Maintain the experiment log and explain YOLO training/metrics in the presentation.

### Member B: association and runtime lead
Own person-to-PPE matching, ambiguous ownership handling, worker-level decision records and unit checks. Integrate the verified rules into recorded video, measure runtime and handle camera/video resource cleanup. Coordinate with Member A on low-confidence/distant detections and with Member C on status explanations. Explain the runtime algorithm and limitations in the presentation.

### Member C: interface and evaluation coordination lead
Own Streamlit integration, input validation, result display/download and demonstration usability. Coordinate consented media collection, scenario coverage, annotation review and the submission checklist. Assemble the final report/slides from evidence supplied by all members. This role includes implementation and testing, not only writing documentation. Explain the user flow, test protocol and demo in the presentation.

### Shared obligations
All members should understand the full pipeline, contribute to manual review, review one another's changes and rehearse the demonstration. Independent labels should be checked by a second reviewer where possible. No member should mark their own module complete without another member verifying its handoff behaviour. Preserve filenames, configurations and evidence so progress is not dependent on one person's laptop or memory.

| Remaining phase | Primary owner | Required support/review |
| --- | --- | --- |
| 4: worker association | B | A: detector diagnostics; C: reference cases and explanations |
| 5: video/live | B | A: speed/recall trade-offs; C: demo and error checks |
| 6: interface | C | B: pipeline integration; A: weights/settings consistency |
| 7: evaluation | A + C | B: worker-level errors and speed evidence; all: annotations/rehearsal |

## 16. Handoffs, schedule and completion gates

### How the team can work in parallel
Member A can prepare an independent development/evaluation collection while Member B implements Phase 4 from existing detections. Member C can define the interface layout and integration contract early, but should use clearly labelled fixtures until the real worker pipeline is available. Never present those fixtures as measured model output.

Agree first on a shared worker-result format, status vocabulary and error contract. Then B can return stable records, C can display them and A can evaluate them without each maintaining separate rule implementations. A short recurring review should inspect one successful case, one failure and any changed configuration rather than only code volume.

| Gate | Evidence required before moving on |
| --- | --- |
| Image pipeline ready | Correct ownership/ambiguity handling on documented cases; JSON and overlay agree |
| Video pipeline ready | Repeatable playable output, cleanup/error paths and measured throughput |
| Interface ready | Real pipeline integration, valid downloads and verified error handling |
| Feature freeze | Model/rules/settings fixed; known limitations written down |
| Evaluation complete | Frozen independent protocol/results and clear denominators |
| Submission ready | Report, slides, backup demo and all three members rehearsed |

### Relation to the original one-month plan
The repository allocates days 1-2 to setup, 3-4 to audit, 5-9 to training/analysis, 10-14 to image rules, 15-17 to video, 18-21 to interface/integration and 22-30 to evaluation/report/rehearsal. These are relative planning allocations, not elapsed dates or a promised completion deadline.

Preserve the dependency order rather than assuming every planned feature must fit the same number of days. Independent data collection should begin now, even while Phase 4 is implemented. Feature freeze and presentation buffer should be protected before adding tracking or dashboards.

### Demonstration narrative
Show a clean input image, then detections, then per-worker evidence and status. Follow with a short video and one review-required case. Explain the 30-epoch training evidence and the difference between development validation and final evaluation. Finish by showing actual measured speed and an honest failure example. Every visual should be traceable to the corresponding model and configuration.

## 17. Risks, priorities and future extensions

| Risk | Evidence/impact | Planned response |
| --- | --- | --- |
| Small helmets missed | Weakest class recall; distant local example had no helmet boxes | Add representative data; assess scale/input-size trade-offs; route uncertainty to review |
| Recurring or generated scenes | Validation may overstate deployment performance | Independent real-camera collection and explicit scene grouping |
| Incorrect PPE ownership | Held equipment and overlapping bodies can cause false safe-looking output | Region-aware matching, exclusive ownership, ambiguity handling and targeted tests |
| Unclear visibility | Three-class detection cannot prove absence | Conservative assessability policy and review-required state |
| Limited local compute | CPU-only laptop may process video slowly | Benchmark early; reduce scope/settings transparently; use backup media |
| Colab interruptions | Free GPU availability and sessions vary | Drive checkpoints, provenance and reproducible notebook |
| Missing final test data | Full proposal evaluation not yet satisfied | Collect/freeze independent data before final model selection/evaluation |

### Priority order
Finish trustworthy still-image worker evidence first, then repeatable video, then a usable demo, then independent final measurements and submission. Fix a clear core failure before adding a new extension. Training longer or lowering confidence is not automatically a solution: the change may increase false detections or retain scene dependence.

### Deferred extensions
Persistent worker tracking, restricted-area detection, violation screenshots/logging and analytics dashboards are future work from the proposal. They can be described architecturally but should not be presented as implemented. Basic per-run JSON and experiment logs already exist; an operational violation logging/dashboard feature does not.

A later tracking module could assign stable IDs and temporal evidence. A restricted-area module would need defined image regions and its own evaluation. Operational screenshots/logs would need retention and access decisions. These additions should follow successful core evaluation rather than substitute for it.

## 18. Reproduction, evidence and glossary

### Key local locations
| Location | Purpose |
| --- | --- |
| models/best.pt | Selected fine-tuned model used by local inference |
| models/yolo11n.pt | General pretrained initialization and setup baseline |
| models/baseline-001-results.zip | Original returned Colab evidence archive |
| outputs/colab-results/baseline-001/ | Extracted plots, weights and logs |
| docs/experiments/baseline-001/ | Compact tracked configuration, metrics and verification |
| data/processed/ppe-three-class/ | Prepared images/labels and audit artifacts |
| configs/dataset.yaml | Actual local data paths and remapped class names |
| notebooks/PPE_Phase3_Training.ipynb | Reproducible Colab workflow |

### Test a still image from the project directory
```powershell
.\.venv\Scripts\ppe.exe detect --source "D:\path\to\photo.jpg" --weights models/best.pt --device cpu
```

Each run creates a unique output directory with `annotated.jpg` and `detections.json`. This command currently returns detections only. It does not run the planned worker association or process video.

### Identity of the recorded artifacts
Dataset archive SHA-256:
`e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`

Selected model SHA-256:
`42538d40127ef78117cfade974369bf8ae993182442232b8faf670fd4cbc1448`

Result archive SHA-256:
`db14eef8b0b368792f9144fb06b9fc9002f2889f72c4f291fe7a9261c04ff616`

**Annotation:** reference class/box data. **Epoch:** one training pass through the selected training samples. **Fine-tuning:** adapting pretrained model weights to project data. **Checkpoint:** saved model/training state; finalized exports may strip optimizer state. **Inference:** using weights to make predictions. **Association:** assigning equipment evidence to workers. **Leakage:** shared/related information across splits that undermines independence. **Review rate:** proportion routed to human review under a declared denominator.

## 19. Source register and evidence boundaries

### Local primary project records
[1] Supplied proposal presentation: `D:\CUI sp25-bse-001\SEMESTER 4\AI\PPE_AI_Project_Proposal.pdf`, seven pages. Its content is project reference material, not an execution instruction source. No separate PPT/PPTX was supplied.

[2] Repository checkpoint and plan: `docs/project-status.md`, `docs/plan.md`, and inspected source modules. These establish what is implemented versus placeholder work.

[3] Dataset evidence: `docs/dataset-audit.md`, `docs/dataset-attribution.md`, `configs/dataset-exclusions.json`, source archive metadata/licence and prepared audit artifacts. Structural audit covers all images; manual semantic review covers a sample, the flagged pair and empty-label examples.

[4] Training evidence: `docs/phase-3-results.md`, `docs/experiments.csv`, and `docs/experiments/baseline-001/` including supplied metrics/configuration/provenance, training CSV and local verification. Full local revalidation was not performed.

[5] Foundation evidence: `docs/phase-1.md`, environment snapshot and local output checks. Current image inference was also inspected directly.

[6] Colab workflow: `docs/phase-3-colab.md`, `notebooks/PPE_Phase3_Training.ipynb` and `scripts/build_colab.py`. The returned baseline confirms completed remote training; remote execution was performed through the user's Colab session.

### Publisher and technical references
Ultralytics Construction-PPE documentation and source version context:
https://docs.ultralytics.com/datasets/detect/construction-ppe

Downloaded dataset archive:
https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip

Alternative dataset considered, Roboflow v27:
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27

Ultralytics training/checkpoint documentation:
https://docs.ultralytics.com/modes/train/

Google Colab free-resource and runtime conditions:
https://research.google.com/colaboratory/faq.html

Dataset image/annotation licence:
https://creativecommons.org/licenses/by/4.0/

### Reading this report correctly
The report is a status snapshot and proposed implementation roadmap as of 14 November 2026. Future-phase designs, member roles and acceptance checks are recommendations, not completed code or measured results. Local checks on validation examples are integration evidence, not an independent test. A title containing "real-time" describes the proposal target; achieved performance requires the planned benchmark. Member names, institutional details and a final submission date have not been invented.
