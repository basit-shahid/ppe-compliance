# PPE video GPU notebook: rebuilt workflow

Use `notebooks/PPE_Video_GPU_Ready.ipynb`. The earlier
`PPE_Video_GPU_Benchmark.ipynb` link is regenerated with the same rebuilt workflow
for compatibility, but the Ready filename makes it easier to select the new copy.

The notebook carries the current Python source, including the invalid-box fix.
It does not embed model weights or personal video. No repair script is needed.
It does not use `files.upload()`, avoiding the upload-widget failure observed in
the earlier session.

## Files to use

- `notebooks/PPE_Video_GPU_Ready.ipynb`
- `outputs/colab-video/ppe-video-gpu-bundle.zip` (about 5 MB, includes `models/best.pt`)
- Your original recorded video, not an already annotated export.

Rebuild after changes to source, README or selected weights:

```powershell
.\.venv\Scripts\python.exe scripts/build_video_colab.py
```

The builder also maintains the older repair file for compatibility. Fresh Ready
notebooks do not need it. The bundle contains source and weights, but the Ready
notebook reads only the selected model from it. Older source inside an uploaded
bundle cannot replace the fixed code embedded in the notebook.

## Run in Colab

1. Upload/open `PPE_Video_GPU_Ready.ipynb` at https://colab.research.google.com/.
2. Choose Runtime > Change runtime type > GPU (T4 if assigned).
3. Open **Files**, using the folder icon on the left. Upload
   `ppe-video-gpu-bundle.zip` and the original video; wait for both uploads to finish.
   Alternatively upload the trained `models/best.pt` instead of the ZIP.
4. Run the cells from the top. There is no notebook upload prompt and no repair
   file to select. Existing verified uploads in the same runtime can be reused.
5. Leave WEIGHTS_PATH and VIDEO_PATH form fields completely blank for automatic
   selection; do not type quotation marks to represent an empty field. Quoted
   empty fields and quoted pasted paths are normalized automatically. If several
   different videos exist, use Files > right-click file > Copy path and put the
   chosen path into VIDEO_PATH. WEIGHTS_PATH accepts a best.pt file path; leave it
   blank when using the ZIP.
6. Wait for **GPU smoke test PASSED** and the scored dashboard preview. Then run
   the main test, inspect the table/video, and download the result ZIP.

Defaults favor a quick GPU demonstration: input 416, tracking/scores on, dashboard,
single direct H.264 export, first 300 readable frames, one repetition, no frame skipping.
The default PROCESSING_HEIGHT=1080 limits analysis to 1920x1080; 720 uses fewer pixels,
and 0 keeps original-resolution analysis. Smaller processing frames can lose
small PPE detail. Drawing fits the dashboard pane before overlays. Detection
coordinates and pixel SHA256 bind the exact processing frame; the original
video-file SHA256, source frame index and per-axis coordinate scale are also
retained. A resized-frame hash is explicitly labeled. Efficient dashboard mode
does not also save a native 4K annotated copy. For the full video,
set RUN_FULL_VIDEO=True in the optional cell. Audio is omitted.

Google controls GPU availability and runtime duration. Download results before
disconnecting; Colab storage is temporary. The laptop webcam is not directly
attached to this remote runtime. See [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Checks before the main run

- CUDA availability is checked before the dependency install.
- Dependency setup keeps one OpenCV distribution, version <5, and a scalable
  Pillow font fallback. Use a fresh runtime when changing OpenCV versions.
- `PREFERRED_DEVICE=auto` uses CUDA when present and otherwise runs explicitly
  labeled CPU diagnostics. Set it to `0` to require a GPU, or `cpu` to test locally.
- The embedded source ZIP and each source file are verified by SHA256.
- A fresh source directory and cleared module cache prevent old notebook code
  from being imported. The notebook exercises invalid-box rejection immediately.
- Trained weights must match the selected model hash. Reusing a cached/older ZIP
  is allowed only when its weights match; its Python code is ignored.
- Duplicate copies of the same video are deduplicated by hash. Different videos
  require explicit selection rather than an arbitrary choice.
- The first video frame must decode, with valid source FPS metadata.
- An eight-frame GPU smoke test checks inference, association, scored dashboard,
  H.264 conversion and output readback. The main run requires a successful smoke
  test for the current source/model/video/profile.

Malformed model boxes are rejected and audited before strict association.
Coordinates and scores are not fabricated, and source frames are retained.
Discarded model indices, coordinates and reasons appear in frame JSON; summary
counts and the comparison table expose the rejected detections. Strict checks
for invalid external association inputs remain enabled.

## Optional controlled comparison

Set COMPARE_CPU=True to compare the Colab GPU and CPU with identical settings.
This takes longer; GPU runs first. Set COMPARE_VIEWS=True to compare native
frame-only export and dashboard export. Set COMPARE_EXPORTS=True to compare
`standard` (native drawing, MP4V exports and extra H.264 conversion) with
`efficient` (display-sized drawing and one direct H.264 output). Choose REPEATS=2
for a more stable short comparison. Device/export order reverses on alternating
repetitions. Export jobs on the same device must have matching detection and
assessment evidence. This compares export changes; hashing and association fixes
are shared by both profiles.

To compare processing resolutions, run cells 3–5 again with PROCESSING_HEIGHT
set to 0, 1080 or 720. Each setting requires a fresh smoke test. Keep source/model,
image size, frame range and export settings fixed. Pixel hashes differ between
resolutions by design; compare frame indices/timestamps and normalized detection
geometry, counts and associations. Treat original-resolution results as a
reference, not independent ground truth. Do not compare rates across different
Colab hosts as a controlled resolution improvement.

The detailed assessment timer separates decoded-frame SHA256 from rules/tracking.
`H264 seconds` measures the additional conversion only and is zero for direct
H.264 exports, whose encoding/finalization time remains included in pipeline FPS.

| Metric | Meaning |
| --- | --- |
| Steady inference FPS | Detector call after excluding five warmup frames; not full video throughput |
| Pipeline FPS | Decode, inference, association, drawing, MP4 encoding and finalization; excludes model construction |
| Complete job FPS | Model construction, pipeline exports and H.264 conversion; excludes upload/download and readback verification |
| Playback FPS | Original nominal video frame rate, independent of AI speed |

The GPU smoke test initializes CUDA before the main test, so results describe an
already initialized runtime. Cold first-frame and model-load times for each job
are still recorded. Colab's CPU differs from the laptop; its CPU/GPU ratio is not
a measurement of laptop speedup. Drawing and ordinary encoding can remain
CPU-bound even when inference uses CUDA. These are short performance/integration
checks, not independent accuracy evaluation or sustained real-time guarantees.

The benchmark verifies the model/video hashes and decoded frame range across
jobs. Stopping a run records an interrupted status and retains partial evidence.
Rerunning the main cell uses a new output folder, preserving earlier results.
The download cell also packages partial benchmark evidence after a failure.

## Local verification

Run the full tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Checks include malformed-box recovery, real video readback, model/video input
selection, source-bootstrap execution over stale imports, absent upload widgets,
preflight guards, cancellation, and generated notebook syntax. Local tests use
CPU or controlled predictions; actual CUDA smoke/benchmark execution remains a
check to perform in the user's Colab runtime.

## Execute the notebook locally, not just its component tests

Install the optional notebook dependencies, then execute the same generated
notebook in a local Jupyter kernel with the real trained model:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe scripts/check_video_notebook.py --source data/samples/my-test-video.mp4 --output outputs/notebook-check-001 --max-frames 300 --quoted-empty-paths
```

The runner changes configuration only: CPU device, existing local dependencies,
frame limit, and optionally the exact quoted-empty form values from the reported
cell 3 error. It does not simulate CUDA or mock predictions. Video and weights
are copied into an isolated runtime folder, an actual Jupyter kernel executes
all eight code cells, and `executed.ipynb` plus `execution-report.json` preserve
outputs and the first failure. Kernel processes are shut down afterward.

A 2026-11-15 run completed all eight cells and all 270 frames of the then-current
3840x2160, 25 FPS clip using real CPU inference, dashboard export and H.264
readback. Its exact source hash and test configuration are recorded in
`docs/experiments/colab-video/notebook-execution.json`. This verifies notebook
logic and CPU processing, not Colab CUDA performance or independent accuracy.

Two separate reported software problems were reproduced locally: literal `""`
form input was treated as a filename, and an earlier 1080p video had a zero-height
helmet prediction at frame 209. Current input normalization handles the first;
detector output validation handles the second while retaining source frames.
