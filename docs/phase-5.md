# Phase 5: recorded video and local webcam

The project now has `ppe video` and `ppe webcam` commands. Both reuse the same
frame detector, Phase 4 association policy and overlays. The model is loaded once
per stream, and frame results are written incrementally to JSONL. Recorded-video
and bounded physical-camera checks passed. Natural-motion PPE accuracy and sustained
performance on deployment footage remain unmeasured.

## Recorded video

Run in PowerShell from `D:\ppe-compliance`:

```powershell
.\.venv\Scripts\ppe.exe video --source "D:\path\to\clip.mp4" --weights models/best.pt --device cpu
```

By default this runs without a display window and processes every readable frame
to completion. Add `--preview` for a window, `--max-frames 100` for a bounded check,
or `--image-size 416` to explore a smaller model input. Smaller inputs can miss
small equipment; compare failures as well as speed before choosing settings.

To reproduce the integration sample after preparing the audited dataset:

```powershell
.\.venv\Scripts\python.exe scripts/make_video_sample.py
.\.venv\Scripts\ppe.exe video --source data/samples/phase5-validation-sequence.mp4 --device cpu
```

The generator refuses to overwrite existing output/metadata. Use `--output` with
a new filename for another copy. It creates 30 frames at 960x540/6 FPS from three
validation stills, ten frames per scene. This is a synthetic integration sequence,
not natural worker motion or independent evaluation. Dataset attribution applies.

## Webcam

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --weights models/best.pt --device cpu --image-size 416
```

A preview window opens. Press **Q**, **Esc**, close the window or use **Ctrl+C**
in the terminal to stop. Model loading/first prediction can take several seconds.
The terminal prints initial progress and updates every 30 frames.

For a bounded test or a system without a preview display:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --no-preview --max-frames 30 --image-size 416
```

Camera 0 is the default; use another nonnegative index if needed. Close other apps
using the camera and check Windows camera permissions if it cannot open. Windows
uses DirectShow; other systems use OpenCV's default capture backend.
If a preview backend is unavailable, omit `--preview` for video or use webcam
`--no-preview`. Ctrl+C/frame limits can stop a headless run. No external AI API or
network-camera URL is used. RTSP, reconnect logic and multi-camera capture are not
implemented.

## Output files

Each run creates a new directory under `outputs/`:

| File | Contents |
| --- | --- |
| `annotated.mp4` | Silent MP4 with equipment/worker/status overlays, encoded using `mp4v` |
| `frames.jsonl` | One JSON object per completed frame: raw detections, assessment, index, timing and source-time fields |
| `summary.json` | Stop/failure status, frame count, configuration/provenance, playback settings and processing performance |

JSONL is streamed and flushed per frame, without storing a whole video's records
or frames in memory. Odd source dimensions are padded on the right/bottom by at
most one pixel for MP4 encoding. JSON boxes remain in the original source space.
Small workers use compact overlay labels (`PPE`, `review`, `check`) and best-effort
label collision avoidance; full item states and reasons remain in JSONL.
Input audio is not preserved. OpenCV readback is verified; browser codec support
for `mp4v` has not been established and may need handling during Phase 6.

Frame indices start at zero. Equipment detection IDs refer to that row's raw
`detections` list. The assessment's input hash describes decoded BGR bytes, while
the stream summary hashes the original input video file when applicable. Webcam
frames have elapsed processing-loop time, not verified device-capture timestamps.
File times use frame index / nominal FPS; variable-rate presentation times are
not supported.

**Default processing has no tracking:** W1 is frame-local. The unpublished optional
`--track` enhancement adds geometric session IDs and status-change events; see
[tracking/UI guide](tracking-ui.md). Summary
worker observations count repeated per-frame statuses, not distinct people.

## Rules, visibility and stop handling

The same optional `--rules configs/compliance.example.yaml` overrides apply.
Shared/uncertain equipment, small/low-confidence workers and other quality flags
go to review. Unconfirmed equipment also defaults to review. Manual visibility
files are supported for still images only; the pipeline does not propagate a
visibility judgement or a previous frame's status into later frames.

Normal EOF, frame limit, preview stop and keyboard interrupt are recorded distinctly.
Failures save a partial summary when a run directory exists, retain completed
JSONL rows, release capture/writer resources and exit nonzero. A file ending before
its advertised frame count is reported as an early-end/decode failure rather than
silently marked complete. Without reliable metadata, EOF and decode failure may
be indistinguishable. Hard process termination is not a clean stop and cannot be
guaranteed to finalize an MP4. Partial failed outputs require review.

## Understanding playback and speed

Recorded output defaults to input FPS. Offline processing can be slower than the
source video and still produce the original nominal playback duration. Overriding
`--output-fps` changes output playback speed, not the number of frames assessed.

Webcam output defaults to **10 FPS playback**, because camera-rate metadata may
be missing or unrelated to actual processed cadence. Slow processing can therefore
produce a recording that plays faster than the wall-clock session. Set an explicit
`--output-fps` if desired; keep frame elapsed-time records for interpretation. The
camera driver can buffer/drop frames independently; that behaviour is not measured.

`end_to_end_processing_fps` is processed frames divided by processing-loop wall time,
including read/decode, prediction, assessment, hashing, drawing, encoding/reporting,
enabled preview and final resource flushing. Model initialization/source hashing
are excluded from that rate but included in total wall time. First prediction
warmup is included. Separate first-output, model-load and frame-latency fields show
cold-start effects. Final summary serialization is outside the processing timer.
Per-frame latency ends after JSONL flush; progress/stop checks and final flushing
are included in the aggregate wall-clock rate instead.

## Actual local checks

Hardware: existing HP ProBook/i5-7200U, approximately 8 GB RAM, CPU-only PyTorch.
These are short integration measurements, not sustained benchmark claims:

| Check | Frames / resolution | Model input | Processing FPS | Result |
| --- | --- | --- | --- | --- |
| Synthetic validation-still sequence | 30 / 960x540 | 640 | 3.34 | Completed EOF; 30 output frames and JSONL rows |
| Same sequence, final compact layout | 30 / 960x540 | 640 | 1.31 | Completed EOF; 30 output frames and JSONL rows |
| Physical camera 0, no preview | 5 / 640x480 | 416 | 0.82 | Frame limit; camera released |
| Physical camera 0, preview | 5 / 640x480 | 416 | 0.81 | Preview opened; frame limit closed it and released camera |

The initial file check took 8.99 seconds in the processing loop, 15.68 seconds
including setup, with 5.12 seconds first-frame latency. The final-layout run took
22.85 seconds processing and 32.46 seconds total, with 13.42 seconds first-frame
latency. Both original summaries are retained rather than selecting only the
faster result. These short cold-start rates varied and are not steady-state rates.
The five-frame camera checks were
also dominated by cold first predictions, so their averages are not steady-state
camera rates. Camera frames had no detected workers and do not establish PPE
performance. Source/playback FPS is not the achieved processing FPS. A real-time
target has not been shown to be met.

All four saved MP4s were opened and decoded back; frame counts, JSONL indices and
dimensions agreed. The second physical-camera run also demonstrated that the
first released the device. Q/Esc/window-close and Ctrl+C paths were exercised via
controlled backend tests; physical preview was stopped with a frame limit.

Evidence: [Phase 5 verification](experiments/phase-5/verification.json), original
summary files in that folder, and local media under `outputs/phase5*` (ignored by
Git). Camera media has not been copied to versioned documentation.

## Automated verification and next phase

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All **59 tests** pass: the 35 image-rule/CLI tests plus 24 video/reuse/CLI cases.
Video cases cover one-time loading, real codec round-trip, streaming records,
timestamps/rate overrides, repeated runs, limits, early EOF, unknown metadata,
camera disconnects, keyboard/preview stops, headless preview errors, writer/model
failures, dimension changes, even padding and cleanup.

The still-image command was also rechecked after the reusable-detector refactor:
the existing user image still produced a person/helmet match and vest-related
review result. No new model was trained, and no worker-level video accuracy was
measured. The next phase is the Streamlit demonstration interface. Natural-motion
clips, independent labels, sustained hardware testing and final evaluation remain
required before deployment or general accuracy/performance claims.
