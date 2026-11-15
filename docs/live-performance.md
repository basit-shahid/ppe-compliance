# Low-latency webcam mode

This local enhancement separates live display from AI assessment. It does not
make the detector run on every camera frame and does not claim perfectly
synchronized results. Recorded-video processing remains sequential/full-frame.

## Start a responsive preview

From the project directory:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --low-latency --image-size 320 --device cpu
```

Use the working camera index on your machine. Existing local camera-backend and
brightness options can still be supplied if needed; no device-specific values
are imposed by the low-latency engine. Stop with Q, Esc, window close or Ctrl+C.
`--duration 20` requests a bounded 20-second session; shutdown can additionally
wait for an in-progress native read or prediction to finish.

The preview has two panes:

- **LIVE:** latest available camera frame, with no AI boxes. It continues updating
  while the model loads or performs an expensive prediction.
- **ANALYZED:** the exact frame associated with a recent result, with its capture
  sequence and application-receive age. Boxes are never projected onto a different,
  newer live frame. Results older than `--stale-after 0.5` seconds are hidden.

The age limit is based on when OpenCV returned the frame to this application, not
the camera sensor's exposure time. Phone transmission and driver buffering are
outside this measurement. A blank analyzed pane during model startup/slow inference
means no sufficiently recent result is available, not that the scene is safe.
Increasing the age limit can show older analyzed results for longer, but does not
improve latency or accuracy. The age label remains visible.

## Recording is optional

Live mode saves only `summary.json` by default: no per-frame JSON or video.
Optional `--track` also writes status/track events to `events.jsonl`.
For an analyzed-frame recording:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --low-latency --image-size 320 --record
```

`--record` adds `annotated.mp4` and `frames.jsonl`. They contain selected analyzed
frames only, not every live-preview frame. The JSONL records original capture
sequence numbers, receive/result elapsed times, raw detections and worker evidence.
Sequence gaps are intentional. MP4 uses constant playback FPS (`--output-fps`,
default 10) and does not preserve wall-clock pacing or audio. Odd dimensions are
padded by one pixel as needed; JSON coordinates remain in source space.

`--max-frames 30` limits completed AI assessments rather than camera reads. A
headless check can use `--no-preview --duration 20`. These options do not alter
`ppe video`, and existing webcam commands without `--low-latency` retain the
sequential record-every-processed-frame behaviour. `--record`, `--duration` and
non-default `--stale-after` require low-latency mode.

## Implementation and shutdown

One capture thread owns the camera and continually replaces a single latest-frame
slot. Frames are copied into owned, read-only arrays because some drivers reuse
decode buffers. One inference thread loads the model once, selects the newest
not-yet-analyzed sequence, and publishes a single latest result. There are no
unbounded frame/result queues. The main thread handles OpenCV display and stop
controls. Inference uses its own pixel copy; frame hashes and boxes refer to the
same source packet.

Recording happens only in the inference owner when enabled. Camera-control cleanup
also happens in the camera owner. On stop, the preview closes immediately, then
workers are allowed to finish in-flight operations. A blocked native call cannot
be safely killed by Python. A bounded shutdown timeout is reported as failure,
with finalization unconfirmed; the application does not release a camera from
another thread while that camera may still be reading. Hard process termination
can leave incomplete recordings.

## Measurements and interpretation

The summary separates:

- Camera read count and AI-assessed count; their difference includes frames
  replaced while busy/initializing or not selected before shutdown.
- Unique application capture sequences displayed and live display-update rate.
  These are not guaranteed unique sensor frames.
- AI assessment throughput, including first prediction warmup but excluding
  model construction, and separate model-load time.
- Receive-to-result age: mean/min/max/first and mean after the first result.
- Mean application queue wait, recording state and stale-result threshold.

The first 20-second physical-camera check at model input 320 recorded 540 capture
reads, 109 assessments, about 24.4 live display updates/s, and 8.1 assessments/s.
After the first cold prediction, mean application receive-to-result age was about
95 ms. Recording was off; the first prediction took about 5.1 seconds. These
describe a short local run, not a controlled comparison with the earlier size-416
or size-640 measurements and not a sustained or independent PPE benchmark.

The final 20-second check, after buffer-safety and timing refinements, recorded
552 capture reads and 128 assessments at 640x480 source/input size 320. It measured
25.9 display updates/s during the preview's active interval, 8.8 assessments/s,
and 92 ms mean application receive-to-result age after the first prediction.
Mean application queue wait was 17 ms. The first cold result took 4.8 seconds;
model loading was separately excluded from assessment throughput. The initial
run's display rate used whole-run wall time; the final metric uses preview-active
time, so those rates are not a controlled before/after comparison.

An additional five-assessment physical-camera recording produced five decodable
MP4 frames and five matching JSONL rows with strictly increasing capture sequences.
No camera media was copied into documentation. Summaries and readback evidence:
[live-performance evidence](experiments/live-performance/verification.json).

A smaller input can miss small helmets. Latest-frame capture reduces application
queueing, not neural-network compute time; faster display must not be reported
as 25-30 FPS AI inference. All existing ambiguity/visibility rules still apply.
Without `--track`, worker IDs remain frame-local. Optional tracking and the new
dashboard are described in [tracking/UI guide](tracking-ui.md); automatic
missing-PPE violations have not been added. The measurements above predate the
HD dashboard and should not be attributed to its more expensive rendering.

## Verification and repository state

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The original low-latency check used 109 tests, including slot replacement, no duplicate inference,
stale-pane removal, no old boxes on the live pane, preview during model loading,
recording round-trip, buffer reuse protection, errors/stops and blocked-call
shutdown reporting. Actual camera preview and optional recording were also checked.

Source, tests and compact verification reports are part of the repository update.
Device-specific configuration, weights and camera/video media remain local.
Current details are in `docs/project-status.md`;
the earlier comprehensive PDF remains a historical Phase 3 snapshot.

## Recorded-video review

Recorded clips now have a separate source-paced review command:

```powershell
.\.venv\Scripts\ppe.exe video --source data/samples/my-test-video.mp4 --image-size 416 --device cpu --track --live-preview
```

The source pane retains original motion while AI scores selected frames in its own
timestamped pane. The local web app can launch this desktop window too. Defaults,
startup preparation, measured 23–25 FPS source submissions and sampled-report
limitations are documented in [recorded live review](recorded-live-review.md).
