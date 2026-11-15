# Finished-player motion investigation

Checked 15 November 2026 after the user clarified that the unsatisfactory motion
was in the app's finished video player. Processing throughput alone cannot test
that experience. The user chose to retain original 25 FPS motion after viewing a
short 25-versus-60-FPS comparison. Interpolation is not enabled in the app, CLI or
normal exports.

## Source versus export

All 270 exported frames aligned with their correct source frame, with no source
index offset and no near-static repeated-motion pairs under the measured luma
threshold. Motion was compared on image pixels excluding badges and dashboard
chrome. The source/export optical-flow magnitude series correlated at 0.99356;
mean flow-vector difference was 0.0314 pixels on the reduced analysis grid.

This supports correct exported motion/cadence, not a claim that the user's
browser playback is satisfactory. See [motion evidence](finished-player-motion-test.json).

## Actual app browser test

An isolated visible Brave window used the real Streamlit upload, processing and
finished-player flow with the original 4K clip. Two 10.8-second playbacks reported:

- Zero dropped or corrupted frames.
- No waiting/stalled events.
- Consecutive presented-frame counters and 40 ms media-time intervals.
- Wall durations 10.8125 and 10.8057 seconds.

Frame presentation callbacks followed the display refresh cadence, with observed
wall intervals up to about 66.7 ms. This is distinct from 40 ms media timestamps;
nominal FPS alone does not establish perfectly even physical presentation.
The first test had the player partially below the viewport; the second had it
fully visible. An earlier headless/offscreen attempt was unsuitable for assessing
visible motion and is not treated as evidence of a player defect.

The visible test did **not reproduce the user's stutter**. An isolated profile
does not reproduce every extension, browser setting, window state or concurrent
load in another session. See [browser evidence](finished-player-browser-test.json).

## Playback checks in the actual user's session

The app now provides **Check playback smoothness → Copy playback test** below
the finished player. It reads the local browser's video-frame counters and reports
delivery FPS, dropped frames, buffered seconds, player visibility and tab state.
It does not send these counters to an external service. Initial partial sampling
intervals and paused/seeking playback are excluded from the FPS calculation.

The component was tested with the actual Streamlit video widget and a visible
Brave player. Stable 25 FPS delivery, zero dropped frames and 100% player
visibility were observed. Source-paced video was preserved. The counters can
help distinguish local decode/presentation loss, buffering or offscreen throttling
from the recording's own motion. See [component browser test](playback-monitor-browser-test.json).

## Terminal mode

The latest pre-investigation terminal run used `--preview`, original 4K analysis
and standard export. It processed only 23 frames in 17.18 seconds before its
preview was closed. Such a processing window necessarily advances at inference
speed and is not a source-paced finished player.

New **`--play`** first finishes processing, then opens the scored output with a
monotonic playback schedule at its encoded FPS. It retains every frame in order,
handles Q/Esc/window close, holds the last frame for its period, and releases
resources on failure. A slow decoder reanchors timing rather than skipping
evidence or rushing through overdue frames. `--preview` remains available for
observing inference progress; it is mutually exclusive with `--play`.

A real native-window check submitted all 270 original frames at nominal 25 FPS
in 10.844 seconds with zero pacing resets. This is an OpenCV submission/timing
check, not a measurement of the OS compositor's displayed frames. See
[terminal test](terminal-playback-test.json). The timing/order, slow-decoder,
stop/failure cleanup and CLI wiring have automated coverage.
