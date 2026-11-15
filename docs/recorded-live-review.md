# Recorded video: original motion with selected-frame AI

## Completion and scope

Implemented for terminal use and the local Streamlit launcher. The app's
**Video → Smooth live review → Open desktop review** starts a desktop window on
the computer running the app. It is not an in-browser live stream. **Analyze
every frame** retains the previous full-processing/browser-export workflow.

```powershell
cd D:\ppe-compliance
.\.venv\Scripts\ppe.exe video --source "data/samples/my-test-video.mp4" --weights models/best.pt --image-size 416 --device cpu --track --live-preview
```

| Option | Behaviour |
| --- | --- |
| `--live-preview` | Source-paced video with a separate, sampled AI pane |
| `--preview` | Frame-by-frame processing window; motion follows inference speed |
| `--play` | Analyze every frame first, then play the completed scored export |

These options are mutually exclusive. Q/Esc closes the review; Tab swaps the
large pane. `--max-frames` limits source frames played in live review. The default
540p working image fits the video pane; 720p/1080p can retain more detail but use
more CPU. `--analysis-fps` defaults to 2.5 and caps AI updates rather than source
playback FPS. A slow source decoder or GUI can still reduce the delivered rate.

## Data and motion guarantees

- The original file is unchanged. Source frames are submitted in order; no
  intermediate frames are generated and no source frames are intentionally skipped.
- A bounded three-frame decode queue overlaps decode and drawing. AI has its own
  one-frame slot and always takes a recent displayed frame, avoiding a backlog.
- The moving source pane has no added PPE overlays. The scored pane retains the
  exact analyzed image and its source time. Its scores do not describe a newer
  frame in the moving pane. UI status refresh is slower than source-pane updates.
- `frames.jsonl` contains only selected observations. Each row records its source
  frame index/time, original-file SHA256, processing-pixel SHA256, dimensions,
  coordinate scale, detections and assessment. It is not an all-frame report.
- Source time uses nominal FPS, not variable-rate presentation timestamps.
  The desktop review does not play audio or export a fully scored movie.

## Startup and caching

The model loads and warms before the recording starts, so a short clip is not
consumed while the model initializes. Large inputs above Full HD are prepared
as smaller H.264 playback files alongside model warm-up. Preparation checks frame
count, FPS and dimensions; the cache is bound to original-file and copy SHA256.
Cached files are reused. The interface shows model/video readiness during startup.

This preparation decodes/resizes the large recording before its first review;
it does not run PPE inference across the entire recording. It still adds first-use
latency. In the recorded tests, preparation took about 18 seconds alongside model
warm-up; a cache hit took under one second. Observed warm-up varied around
16–31 seconds on this laptop. Playback then ran while selected-frame inference
continued. The output/input caches remain under ignored `outputs/` directories.

## Verification and limitations

The 3840x2160, 25 FPS, 270-frame development clip was reviewed at model input 416
on CPU. A normal full run submitted all 270 frames in 10.887 seconds (24.80 FPS),
with 27 AI observations. An instrumented acceptance run submitted all 270 in
11.594 seconds (23.29 FPS), with 28 AI observations. These are short native GUI
submission measurements, not OS compositor counters or a sustained real-time
guarantee. Load, window rendering and working resolution affect the outcome.

Acceptance checks verified every source index reached the renderer in order,
checked that the moving pane pixels had no added AI overlays at three positions,
replayed and SHA-verified every analyzed sample, and matched every displayed PPE
confidence to its detection. The new mode does not have an independent accuracy
evaluation or a new GPU benchmark. Unobserved frames may contain events missed
by sampled inference; use full analysis when per-frame coverage is required.

Evidence: [native review acceptance](experiments/colab-video/recorded-live-review-verification.json).
The completion audit passed 180 tests; its current-source native recheck is in
[completion verification](experiments/colab-video/completion-verification.json).
Automated tests additionally cover original-rate pacing independent of inference,
source/report pixel ownership, cache reuse, decoder errors, startup/cancellation,
CLI dispatch and the web launcher. See `tests/test_recorded_preview.py` and
`tests/test_streamlit_app.py`.
