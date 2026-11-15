# Review of the uploaded Colab GPU result

Reviewed 15 November 2026. Archive:
`outputs/ppe-video-gpu-results-20261115T124202Z-f2242e67.zip`.

The run completed on **Tesla T4**, using the selected trained model, image size
416, tracking/scores on and dashboard presentation. All 270 source frames were
processed. Independent local readback decoded all 270 H.264 frames at 1280x720,
25 FPS, lasting 10.8 seconds. Small-frame fingerprints differed across all 270
outputs. No malformed predictions were rejected in this clip. This is integration
and performance evidence, not independent accuracy evaluation.

## What the timings show

| Measurement | Result |
| --- | ---: |
| Source resolution | 3840x2160 |
| Source/playback rate | 25 FPS |
| Warm detector-call throughput | 51.61 FPS |
| Complete video-loop throughput | 3.69 FPS |
| Complete job throughput | 3.13 FPS |
| Job duration | 86.38 seconds |
| Video duration | 10.8 seconds |
| Additional H.264 conversion | 12.89 seconds |

| Warm stage | Mean ms/frame | Share of reported stage time |
| --- | ---: | ---: |
| Decode | 12.69 | 4.8% |
| Setup | 0.02 | <0.1% |
| Detector call | 19.38 | 7.3% |
| Assessment and decoded-frame hashing | 88.82 | 33.3% |
| Annotations and dashboard rendering | 73.39 | 27.5% |
| MP4 encoding submissions | 72.72 | 27.2% |

Inference is faster than the input frame rate. Approximately 93% of the reported
warm stage time is outside inference. Faster model inference alone cannot solve
this run's processing delay. There is one GPU-only repetition; the archive does
not contain a controlled Colab CPU/GPU comparison or resolution ablation.

The same video/model hashes and eight recorded pipeline source hashes match the
previous local CPU notebook run. However, the hosts, operating systems and
package versions differ, so their rates are not a controlled GPU-only speedup.

## Changes to prioritize

1. **Reduce per-frame 4K work.** Benchmark an aspect-preserving 1080p working
   copy with model input still 416 and a 720p dashboard. Retain the original
   separately, preserve frame count and source FPS, and record processing/source
   coordinate spaces explicitly. Compare detection/association outcomes before
   adopting lower working resolution as the default.
2. **Produce one browser-ready video.** Current dashboard mode writes a full 4K
   annotated MP4, a dashboard MP4 and then a separate H.264 derivative. Make the
   native 4K export optional and encode the selected dashboard directly to H.264
   once. Benchmark encoder settings and verify the resulting file. See
   [FFmpeg encoder documentation](https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb).
3. **Profile hashing separately from association.** A decoded 4K BGR frame is
   24,883,200 bytes; 270 such frames total 6.72 GB of pixels to hash, in addition
   to allocating the current `tobytes()` copies. A local CPU micro-check preserved
   identical SHA256 using a contiguous array buffer, with mean copied hashing
   152.7 ms versus buffer hashing 120.7 ms; association alone was about 0.20 ms.
   These local timings are not measurements of the Colab host. Use a zero-copy
   buffer where supported and retain the same digest semantics. If a faster
   preview mode makes decoded-pixel hashes optional, label that policy explicitly
   and retain source-file SHA256 plus frame index; never label another identifier
   as a decoded-frame SHA256.
4. **Avoid rebuilding full dashboard typography unnecessarily.** Rounded score
   changes currently invalidate the whole chrome cache. Cache static text and
   redraw only changed worker rows/metrics. Preserve exact frame/result alignment.
5. **Repeat on the same Colab host.** Use the same clip/model/profile, two or three
   repetitions and one change at a time. Compare complete-job FPS as well as
   model-only FPS. Measure sustained natural-motion processing before claiming
   25 FPS end-to-end. Model/TensorRT optimization is lower priority given this
   measured time distribution.

A practical first candidate is GPU inference at size 416, 1080p working frames,
720p dashboard and one H.264 output. Its speed and small-PPE performance remain
unmeasured; no expected FPS improvement is asserted here.

## Quality findings independent of speed

- All 546 worker observations were review required. They are repeated frame-level
  observations, not 546 people.
- There were 540 helmet detections and **270 vest detections**. Every vest was
  unmatched with `no plausible head/torso region`, so the dashboard reported no
  vest match even when a vest was detected.
- At frame 135, the first worker extends to the bottom image border. Its vest
  box spans y=971..2160, while the fixed torso window ends near y=1603. Only
  about 53% of the vest lies in that window, below the 65% coverage requirement.
  This illustrates the cropped-person association problem. Add focused cases
  for truncated workers and improve item matching without removing the existing
  review requirement for uncertain visibility or forcing green decisions.
- The inspected Colab dashboard uses very small text. The font fallback calls
  `ImageFont.load_default()` without the requested size when configured system
  fonts are unavailable. Honor requested font size or provide a portable font.
- Colab imported OpenCV 5.0.0.93 while the project specifies OpenCV <5; its
  environment lists standard, headless and contrib distributions. Normalize the
  supported OpenCV environment before controlled comparisons. This is a
  reproducibility issue, not a measured explanation for the observed slowdown.

Raw metadata and local verification are retained under
`outputs/result-review/colab-run/`. User media remains in ignored output folders.

## Implementation follow-up

The [prioritized implementation](prioritized-improvements.md) now provides
display-sized drawing, a single direct H.264 export, identical zero-copy frame
hashes, cached typography, scalable fallback text and cropped-worker vest matching.
Inference continues to receive original frames, avoiding a source-resolution
accuracy change while export performance is measured. The rebuilt notebook has
an export comparison switch for retesting on the same GPU.

Follow-up verification: 161 tests passed, and the rebuilt notebook completed all
eight cells with the real model and all 270 frames locally. A two-repetition CPU
export comparison measured pooled complete-job throughput of 1.22 FPS standard
versus 2.01 FPS efficient, with identical detection/assessment evidence. These are
local CPU measurements; a new T4 comparison remains to be run.
