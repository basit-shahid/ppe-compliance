# Recorded-video lag follow-up

Implemented and verified 15 November 2026.

The interface now defaults to **Balanced · 1080p** analysis rather than handling
all 4K pixels throughout inference, hashing and rendering. It preserves aspect
ratio within 1920x1080 and never enlarges smaller inputs. **Lower resolution ·
720p** and **Original resolution** remain available under Video options. Every
source frame is processed, with source nominal FPS retained in the export.

The original upload is unchanged. Raw boxes and decoded-pixel hashes describe
the exact processing frame. `processing_resolution`, `frame_hash_scope` and
`source_coordinate_scale` make the coordinate/hash policy explicit; original
video-file SHA256, source frame indices and timestamps are retained. Native-size
exports scale boxes back to the source canvas. Selecting native export alone
does not restore detail excluded by a lower analysis resolution; use Original
resolution analysis when that detail matters.

H.264 export now uses `fastdecode`, no B frames, and a maximum two-second keyframe
interval, retaining CRF 18 and yuv420p. These settings reduce decoder work and
the distance needed for seeking. Progress updates arrive about once a second
while processing rather than waiting for 30 analyzed frames.

## Same-host comparison

The first 60 frames of the original 3840x2160/25 FPS clip were processed at model
input 416, confidence .25, CPU, tracking/scores on, efficient dashboard export.
Two repetitions reversed resolution order. Every output decoded all 60 frames
at 1280x720 and 25 FPS.

| Analysis resolution | Job FPS, repetition 1 | Job FPS, repetition 2 | Pooled complete-job FPS |
| --- | ---: | ---: | ---: |
| Original 4K | 1.47 | 2.43 | 1.84 |
| Up to 1080p | 2.95 | 2.71 | 2.82 |
| Up to 720p | 3.10 | 2.50 | 2.77 |

The measured pooled complete-job rate was about 54% higher at 1080p. The first
original-resolution job includes cold model/import startup; host load and
inference times also varied. These numbers describe the complete short test and
are not an isolated causal estimate or a sustained real-time guarantee. Both
pipeline and whole-job times are retained in the
[comparison evidence](processing-resolution-comparison.json).

The hash timer fell from about 136–138 ms/frame at 4K to 33–34 ms at 1080p and
14–16 ms at 720p. Additional resizing costs explain why fewer pixels did not
consistently give the best whole-job rate. 1080p remains the default for its detail
and measured balance.

Each run produced 120 person, 120 helmet and 60 vest detections. Relative to
original-resolution inference, every 1080p detection paired at IoU >= .5, with
mean normalized IoU above .995 for each class. All 120 paired worker observations
per run retained the same equipment states and review statuses. Confidence values
changed slightly and remain the actual model outputs. Original-resolution
inference is a reference, not independent ground truth; these large foreground
workers do not establish performance for distant or small PPE.

## Full notebook and playback checks

- **165 tests passed**, including resized-frame hash binding, coordinate mapping,
  native rendering, 720p encoder dimensions and unchanged frame pacing.
- The rebuilt notebook completed **all eight cells** with the real model and
  the entire 270-frame clip in local CPU mode, using 1080p processing.
- Independent FFmpeg readback found 270 frames, timestamped at exact 40 ms
  intervals, lasting 10.8 seconds. Output thumbnails were distinct across all
  270 frames. The encoder produced six I frames and 264 P frames, with keyframes
  at frame indices 0, 50, 100, 150, 200 and 250.
- Reports identified 1920x1080 analysis of a 3840x2160 source, resized-pixel hash
  scope, and per-axis source coordinate scale 2. Bundle and notebook source hashes
  matched the current workspace. The full run measured 3.87 pipeline FPS on CPU;
  it is a separate functional run, not another controlled comparison repetition.
- Local notebook execution reused installed dependencies. A fresh Colab install
  and new T4 timing are still unverified locally.

See [cadence/verification](recorded-lag-verification.json) and
[notebook execution](recorded-lag-notebook-execution.json).

Refresh the local app, upload the original video, choose Balanced · 1080p, leave
native export off for the lighter path, and analyze again. Earlier saved outputs
keep their previous encoding and processing settings. Use 0 for the frame limit
to export the entire clip. Faster analysis and smooth saved playback are separate
measurements; the saved playback follows the source nominal frame rate.
