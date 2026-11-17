# Image and video pipeline tests

Run from the repository root after installing the application dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use standard-library `unittest`. Rule fixtures do not run YOLO or require
weights/network access. CLI integration tests use controlled detections and real
temporary image/JSON files. They verify logic and file behaviour, not model accuracy.

The original Phase 5 suite had 59 tests; the pre-finalization suite passed 180.
The expanded suite additionally covers presentation audit findings and the new UI.
See `docs/experiments/final-readiness/verification.json` for the latest recorded count.
Coverage includes camera compatibility, playback, recorded sampling and low-latency behaviour. Video checks use temporary files and
fake predictions/camera backends to test streaming, rate/timing metadata, resource
cleanup and stop/failure paths without requiring a physical camera or GUI.

`test_live_pipeline.py` checks single-slot handoffs, separate preview pixels,
result-age expiry, responsive display during initialization, optional recording,
reused camera buffers, errors and bounded shutdown. No network/weights are needed.

Quality checks in `test_quality.py` cover source-pixel preservation, dense compact
overlays, lossless image/report exports, requested capture FPS, manifest hashes,
exact overlap rejection, one-to-one worker matches and missed/spurious denominators.
Optional `test_streamlit_app.py` checks upload errors, persistent results,
changed-settings invalidation and source-mode navigation. Install `.[ui]` first;
uploader interactions require a recent AppTest implementation.

`test_readiness.py` checks oversized/corrupt images before expensive decoding,
offline-only format readers, invalid public inference arguments, atomic image
write failures, unavailable-device errors, camera confidence forwarding and
missing/empty video previews. UI tests also cover the included sample, evidence
filtering, missing model recovery, documentation pages and deleted output files.
These mocked UI checks do not require actual trained weights.

`test_browser_preview.py` covers isolated worker configuration, atomic snapshots,
Windows reader-lock retries, stop/lease expiry, process failure reporting and real
JPEG handoff without a GUI. Pipeline tests verify browser callbacks and external
Stop during camera/video operation and warmup. UI checks verify embedded previews
are the default, readable errors, and Start/Stop controls.

For a combined suite and actual-weight CPU smoke check:

```powershell
.\.venv\Scripts\python.exe scripts/verify_presentation.py
```

The separate optional `scripts/check_workspace_browser.py` checks an already
running local app in a dedicated Chromium session. See `docs/presentation-guide.md`.

Recorded presentation checks additionally verify matching dashboard/native frame
counts and FPS, source-time synchronization, confidence-card cache refresh,
missing-score formatting and default dashboard selection in the interface.

`test_benchmark.py` verifies warmup accounting, complete CPU exports/readback,
missing-GPU rejection and the matching Colab source/model ZIP. Generated notebook
cells are parsed and bundled files are hashed; these checks do not run a Colab GPU.

`test_detection_validation.py` covers degenerate/non-finite boxes, aligned filtered
Results/JSON indices, conservative missing-item decisions, frame retention and
rejection auditing, plus repair of an existing Colab session without replacing
its model or video. Strict association validation remains enabled.

`test_colab_setup.py` checks cached weights, model-only extraction from old bundles,
duplicate/ambiguous video inputs, first-frame/FPS inspection, embedded-source boot
over stale imports and the smoke-test guard. Benchmark cancellation is also tested.

`test_recorded_preview.py` checks source pacing independent of slow inference,
selected-frame hashes/timestamps, no model mutation of source pixels, cache reuse,
decoder errors, warm-up cancellation, failed prefetch startup, odd source sizing
and CLI routing. The UI tests cover both full analysis and the desktop-review
launcher. Native GUI and real-model measurements are recorded separately in
`docs/recorded-live-review.md`.
