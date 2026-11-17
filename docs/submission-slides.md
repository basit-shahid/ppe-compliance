# PPE Vision presentation slides and speaker notes

Use 10 slides for an approximately 8–10 minute presentation. Insert team names,
supervisor and institution yourself. Do not present pending work as measured.
The matching PDF is `PPE_Presentation.pdf`, built by `scripts/build_slides.py`.

## 1. PPE Vision

- Worker PPE detection and evidence review.
- Person, helmet and high-visibility vest detection.
- Local inference, understandable evidence, human review.

Speaker note: “Our project combines a fine-tuned detector with worker ownership
rules, uncertainty handling and practical image/video review.” State the project
scope before showing results; it does not assess every workplace hazard.

## 2. The problem and our scope

- Equipment counts alone cannot explain which worker has which item.
- Crowding, distance and occlusion create uncertain evidence.
- We assist a reviewer with per-worker status and reasons.

Speaker note: A helmet held at waist level can be a correct object detection and
still fail the worker's head-region check. Explain this example in one sentence.

## 3. Data and model training

- Audited Construction-PPE derivative: 799 train / 200 validation images.
- Three prepared labels: person, helmet, vest.
- YOLO11 nano; 30 epochs; image size 640; batch 8; seed 42; Colab T4.
- Independent test scenes remain pending.

Speaker note: Credit Ultralytics and distinguish our data preparation, fine-tuning
and application logic from the pretrained YOLO architecture. Mention recurring
scenes and that the included demonstration sample belongs to validation.

## 4. From pixels to a worker report

- Validate and decode the source.
- Detect boxes; reject invalid model outputs.
- Associate helmets with heads and vests with torsos.
- Apply uncertainty rules; optionally track; render and export.

Speaker note: Open the Project guide page or point to the pipeline diagram.
Each step has a separate module and can be tested independently.

## 5. Three explainable outcomes

- PPE detected: both plausible matches, no quality warning.
- Review required: weak, missing, small, cropped or ambiguous evidence.
- Potential violation: missing item in a human-reviewed visible region.
- Missing detections alone default to review.

Speaker note: The detector threshold retains candidates; evidence acceptance is
stricter. Detecting an object does not prove wearing state. The UI does not
silently infer region visibility; human visibility input is supported in the CLI.

## 6. The demonstration

- Inspect the included attributed sample with the actual model.
- Compare original/annotated views and read worker reasons.
- Change settings; rerun; download the evidence JSON.
- Show recorded output and the pipeline/validation pages.

Speaker note: Follow docs/presentation-guide.md. Have an already exported MP4
available. Amber on the border-cropped sample is an intentional conservative
decision, even when helmet and vest match.

## 7. Video, camera and session tracking

- Every-frame analysis exports scored video and streaming JSONL.
- Smooth recorded/camera review separates source motion from sampled AI.
- Geometric session IDs and repeated evidence confirmation reduce flicker.
- Display FPS, processing FPS and playback FPS measure different things.

Speaker note: Do not claim smooth display means inference on every frame.
Tracking is local IoU association and can switch IDs at crossings. UI video
defaults to 300 frames; limit 0 requests the whole source.

## 8. Recorded validation results

- Precision: 94.99%; recall: 83.54%.
- mAP at IoU 0.5: 90.08%; mAP at IoU 0.5:0.95: 71.15%.
- Helmet recall is weakest: 76.53%.
- These are development detection metrics, not final compliance accuracy.

Speaker note: Explain precision as false-positive control, recall as missed-object
coverage, and mAP as detection precision/recall aggregated over classes and IoU
criteria. Worker status accuracy needs separately reviewed worker labels.

## 9. Reliability and verification

- Automated checks cover geometry, uncertainty, tracking and failure cleanup.
- UI checks cover navigation, uploads, retained results and launch settings.
- Actual CPU weights pass image and H.264 video integrity smoke checks.
- Oversized/corrupt inputs and missing outputs produce actionable errors.

Speaker note: Show the dated verification JSON. Software tests establish behavior
and integration; they do not establish independent real-world accuracy. Physical
camera and sustained performance need a separate host rehearsal.

## 10. Limits, next work and conclusion

- Independent real-camera worker evaluation is still required.
- Improve small-helmet, occlusion and scene coverage with controlled experiments.
- Benchmark sustained performance on target hardware.
- Deliverable: an explainable, locally runnable PPE review prototype.

Speaker note: Finish with what works and what is measured. Invite questions about
the detection/ownership distinction, status priority, video timing and code flow.
Prepare with docs/code-walkthrough.md rather than memorizing unsupported claims.
