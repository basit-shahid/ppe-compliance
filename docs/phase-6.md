# Phase 6: focused local interface and video quality

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501. The service binds to this computer by default.
Image and Video modes accept uploads, analyze only when requested, and preserve
results through download/rerun actions. Changing a file or processing settings
hides the previous results until analysis is repeated. Images export lossless PNG,
worker evidence, raw detections and a manual visibility template. Worker reasons
live in an expandable table rather than over the picture. Corrupt files and
missing models produce readable errors.

Video defaults to a 300-frame demonstration limit. Set it to 0 for a full export.
The default Monitor dashboard presentation uses the same renderer as the native
camera monitor: a large analyzed frame, synchronized source-frame inset, worker
statuses and confidence scores. Its header uses playback FPS and source-video
time rather than live-camera age. Slow offline analysis does not expire evidence
belonging to that displayed frame. Video options also offers Video only.

Every selected source frame is analyzed. `annotated.mp4` retains native dimensions;
`dashboard.mp4` is a separate 1280x720 presentation at the same playback FPS and
frame count. Processing may take longer than playback on CPU. A separate H.264
CRF 18 derivative (`preview.mp4`) provides browser playback of the chosen view.
All exports and JSONL remain downloadable. Audio is omitted. Conversion cannot
restore source detail or remove existing compression. Equipment detail boxes
are available for images and recorded videos.

Confidence scores are visible by default on frames and in worker cards:
P = person, H = matched helmet, V = matched vest. These are the detector's
confidence values, not a compliance/safety probability. `--` means there is no
matched score; it never substitutes zero for missing/ambiguous evidence. Advanced
settings can hide scores or show equipment boxes. Reanalyze earlier uploaded
videos to generate the new presentation; existing MP4 files are unchanged.

The CLI defaults to the dashboard for recorded video. Use `video --view frame`
for annotations only. Python `process_video` keeps `view="frame"` as its default
for compatibility; the interface explicitly requests `view="dashboard"`.

Live camera mode opens the native monitor on this computer. It keeps camera
pixels moving while assessing selected frames in another thread. It shows a
large primary video, a small alternate view, and concise worker evidence.
Tab swaps focus, F toggles full screen, Q/Esc stops. The browser can be closed
while the native monitor runs; stop capture in its own window. Startup errors
are written to `outputs/camera-launch.log`.

An equivalent balanced desktop command:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --low-latency --track --image-size 416 --capture-width 1280 --capture-height 720 --capture-fps 30
```

Use `--camera 1 --camera-backend msmf` for the previously configured phone camera
only when its Windows client is already streaming. Camera FPS and dimensions are
requests; the summary includes driver acceptance/readback and actual received
dimensions. The monitor separately shows measured live display FPS and AI rate.
`--preview-fps 60` caps refresh without introducing a fixed 20 ms sleep on every
iteration. Unchanged panes and typography are cached. It redraws on new source
frames/results, freshness changes and periodic metric updates.

Profiles: Fast = input 320, Balanced = 416, Detail = 640. Source resolution is
independent of the model's internal input size. HD capture keeps more actual
detail, while a smaller model input lowers neural compute cost and can miss
small helmets. The CLI webcam default is now balanced 416. No upscaling or
increased export FPS can manufacture camera detail or additional AI evidence.

All assessment overlays now default to thin corner markers and short status
badges with model confidence. Dense crowds omit colliding badges rather than obscure workers.
Equipment boxes and longer item states are opt-in for image review.
Full evidence and coordinates remain in reports. Current decisions expire after
the existing age limit, and boxes stay on their exact analyzed source frame.

Optional `--record` still records analyzed frames, with constant playback FPS
(default 10). It is not a recording of every smooth live frame. Use the live
pane for motion and the analyzed pane for current evidence.

Implementation references: [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
and [OpenCV capture properties](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html).

Verification evidence is recorded in `docs/experiments/quality-ui/verification.json`.

The later recorded-video update adds **Smooth live review** as the Video default:
it launches a desktop window with original-motion playback and selected-frame AI
in a separate timestamped pane. Choose **Analyze every frame** for the browser
export workflow. See [recorded live review](recorded-live-review.md) for startup,
sampling limits and the measured source submission rates.
UI tests are optional when Streamlit is not installed. Upload interaction tests
need a Streamlit release providing AppTest file-uploader interactions.
