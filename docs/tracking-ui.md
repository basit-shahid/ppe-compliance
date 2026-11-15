# Local worker tracking and dashboard

This unpublished enhancement adds session worker IDs, status-change events and a
native camera dashboard. It does not require Streamlit or an external AI service.

## Run it

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --low-latency --track --image-size 320 --focus analyzed
```

For the local DroidCam setup, replace `--camera 0` with
`--camera 1 --camera-backend msmf`. The phone must already show footage in the
Windows client; these options cannot repair an upstream green/placeholder stream.

Press Tab to swap the large live/analyzed panes; Q or Esc stops the session.
The live pane contains current camera pixels; boxes remain on the exact analyzed
frame. Current cards disappear when that result exceeds the 0.5-second age limit.

Optional native HD capture: add `--capture-width 1280 --capture-height 720`.
The driver must support the request; the summary records requested and actual
resolution. HD increases CPU work. `--preview-width 960` makes a smaller dashboard;
`--view split` selects the simpler preview. Capture resolution and model input
size are separate: size 320 favors speed but can miss small helmets. Increasing
model input size can improve detail at a speed cost. Lighting and camera focus
still determine the available image detail.

For a recorded test video:

```powershell
.\.venv\Scripts\ppe.exe video --source "D:\path\to\test.mp4" --track --weights models/best.pt --device cpu
```

## What changes

- One worker box and badge replace repeated equipment boxes when tracking is on.
- Geometric overlap links consecutive observations to IDs such as T001.
- Green requires three consistent observations spanning at least 0.2 seconds.
  Uncertain current evidence immediately displays amber; green is never latched
  permanently. Confirmed changes generate events once rather than every frame.
- Evidence gaps over 0.5 seconds clear previous green state. IDs retire after
  1.2 seconds without an observation. No boxes are drawn for unseen workers.
- Ambiguous matches start fresh review tracks. Crossing, fast motion, cuts and
  occlusion can change IDs; these are session labels, not verified identities.
- Missing detections alone remain review required. Red potential-violation status
  still depends on the existing reviewed-visibility rules; green means plausible
  visible PPE matches, not proof of wearing or general workplace safety.

With `--track`, each run writes `events.jsonl` for track starts, confirmed changes
and losses, plus the summary. Low-latency mode records no video by default;
`--record` adds analyzed-frame MP4 and per-frame JSONL. Event history is explicitly
separate from current worker decisions. Recorded-video processing retains MP4 and
per-frame JSONL output. Summary worker observations are not unique-person counts.

## Verification

The original tracking/UI checkpoint passed 109 tests. Controlled checks cover stable IDs, ambiguous
matches, immediate amber on uncertainty, evidence expiry, event deduplication,
stale-result hiding, rendering caches and video/live integration.

A 30-frame development video produced 30 readable MP4 frames and 30 JSONL rows.
The first worker remained T001 across its first ten frames and generated exactly
one transition to green. This clip consists of validation stills; it does not
establish natural-motion tracking accuracy.

A 20-second physical-camera check delivered actual 1280x720 frames. After caching
unchanged dashboard panels, it measured 19.2 live display updates/s and 4.8 AI
assessments/s, with 137 ms mean application receive-to-result age after the first
result. Cold model loading took 6.3 seconds and the first result took 6.1 seconds.
The earlier uncached HD check measured 9.8 display updates/s and 3.4 assessments/s.
These short runs are not controlled sustained benchmarks. No workers were observed
in the HD camera check, so it validates capture/display rather than PPE accuracy.
Camera/phone buffering before application receive is not measured.

The earlier simple 640x480 preview reached 25.9 display updates/s and 8.8 AI
assessments/s in a separate short run. Higher source resolution and the dashboard
cost CPU time; display speed must not be reported as detector speed.

Compact evidence is in [tracking/UI verification](experiments/tracking-ui/verification.json).
Reusable source, tests and compact evidence are included in the repository update;
camera media and device-specific settings remain local.
