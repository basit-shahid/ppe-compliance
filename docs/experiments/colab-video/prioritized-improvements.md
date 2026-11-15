# Prioritized video improvements

Implemented 15 November 2026 in response to the uploaded T4 result.

This records the first export-focused improvement. The subsequent
[recorded-video lag follow-up](recorded-video-lag.md) adds selectable processing
resolution, new playback encoding and explicit processing-coordinate/hash metadata.

## Behavior

- `efficient` exports one browser-ready H.264 video: 1280x720 dashboard or
  aspect-preserving video up to 1920x1080. It uses CRF 18, the veryfast preset,
  two encoder threads, and checks encoder exit/finalization errors.
- Dashboard annotations are drawn at pane size instead of copying, drawing on,
  and encoding a second native 4K annotated video. The original input is retained.
  Large evenly divisible frames first use integer area reduction before the final
  pane resize, reducing arbitrary-ratio 4K filtering while keeping area filtering.
- Detection still receives each original decoded frame with model input 416 in
  the comparison. Report boxes stay in original source coordinates. Frames are
  neither skipped nor interpolated; source nominal FPS is preserved.
- Decoded-frame SHA256 uses a contiguous buffer without a `tobytes()` copy.
  The digest is identical to the previous full decoded-BGR digest. Noncontiguous
  arrays retain a copying fallback. Hashing and rules/tracking have separate timers.
- Dashboard text uses cached small text tiles; changing scores no longer requires
  converting the whole canvas through Pillow. The portable font honors its size.
- A bottom-cropped vest can match a bottom-cropped worker when it occupies a
  substantial, contained torso region and reaches the same boundary. Existing
  uncertainty, confidence, size and ambiguous-ownership checks still apply.
  Border-cropped workers remain Review; this does not establish wearing state.

The Streamlit app defaults to efficient export. **Keep a native resolution
export** selects the previous native export behavior. Python/CLI defaults remain
standard for compatibility; use `--export-profile efficient` explicitly in CLI.
The rebuilt Ready notebook defaults to efficient export and exposes
`COMPARE_EXPORTS=True` for a comparison. Choose `REPEATS=2` to reverse export order
on the second repetition. Every export job checks source/weights, decoded frame
range, frame count and readback. Same-device detection/assessment evidence is
checked for equality across export jobs.

Both export profiles share the new hashing, text cache and association code.
Their comparison measures the export/render-size changes together, rather than
isolating every improvement or comparing the entire old release with the new one.
Reducing the frames supplied to inference remains a separate accuracy experiment;
it is deliberately unnecessary for these export improvements.

## Verification

The test suite checks real H.264 readback, frame/FPS preservation, unchanged
source hashes/detections/assessment, coordinate scaling without report mutation,
encoder finalization failure, font sizing, cropped vest scores, held/outside
equipment rejection, ambiguous ownership and the app's direct-video path.

All **161 tests passed**. Replaying the uploaded 270-frame Colab detection records
with the updated association rule yields **264 vest matches with original scores**,
six uncertain vest observations, and 276 observations with no matched vest.
All 546 worker observations remain Review. This is a rule replay on saved
detections, not a fresh inference or independent wearing-state evaluation;
see [association replay](association-replay.json).

The rebuilt notebook also completed **all eight code cells** in a real local CPU
Jupyter kernel with the selected model, including the quoted-empty path regression.
After the eight-frame smoke check, its main run processed the complete 270-frame
4K clip. Independent readback decoded 270 distinct output thumbnails at 1280x720,
25 FPS, lasting 10.8 seconds; source indices and decoded-frame hashes were unique
and contiguous. No cell failed. The generated bundle and notebook pipeline source
hashes match the current workspace. Local requirements check passed, and the
updated Streamlit interface returned HTTP 200 at `http://127.0.0.1:8501`.
See [notebook execution](notebook-final-execution.json) and
[final verification](prioritized-verification.json). This notebook check reuses
local dependencies; the Colab dependency-install and CUDA paths still require a
fresh GPU runtime check.

The real-model CPU comparison used the first 90 frames of the same 4K/25 FPS
clip, image size 416, two repetitions and reversed export order. Source frames,
model, detections and worker assessments matched across all four jobs. Every
output decoded 90 frames at 1280x720 and 25 FPS.

| Export / repetition | Pipeline FPS | Complete-job FPS | Job seconds | Extra H.264 conversion seconds |
| --- | ---: | ---: | ---: | ---: |
| Standard / 1 | 1.18 | 1.12 | 80.15 | 2.82 |
| Efficient / 1 | 2.10 | 2.08 | 43.23 | 0 |
| Efficient / 2 | 1.99 | 1.94 | 46.40 | 0 |
| Standard / 2 | 1.42 | 1.34 | 67.20 | 2.62 |

Pooling frames and job durations gives **1.22 FPS standard vs 2.01 FPS efficient**,
about 64% higher complete-job throughput in this local CPU check. Rates varied
between repetitions, including inference time; this is a short diagnostic, not
a sustained real-time guarantee. See [full comparison evidence](cpu-export-comparison.json)
for settings, source/code hashes, detailed stage times and readback checks.

No new T4 performance measurement is available locally. The previous GPU result
is a different host/version configuration and cannot be used as a controlled
speedup baseline for local CPU measurements. Both profiles still hash every
original 4K frame; this remains a substantial CPU cost.

## Next Colab check

1. Open the rebuilt `notebooks/PPE_Video_GPU_Ready.ipynb` in a fresh GPU runtime.
2. Upload the original video and matching model ZIP using the Files panel.
3. Run cells 1–4; check the scored smoke preview.
4. In cell 5 set `COMPARE_EXPORTS=True`, `REPEATS=2`, keeping input size 416,
   CPU/view comparisons off, and the same source clip.
5. Run the remaining cells and download the results ZIP. Compare complete-job
   FPS and confirm evidence equality, source FPS and decoded frame counts.

The local suite does not verify installation on an actual Colab runtime or
independent detection/compliance accuracy. Those remain separate checks.
