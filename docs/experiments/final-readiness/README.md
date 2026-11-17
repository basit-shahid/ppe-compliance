# Presentation readiness evidence — 16 November 2026

- `verification.json`: 212 automated tests, dependency consistency, actual-weight
  CPU image inference, eight-frame H.264 video readback and invalid-image checks.
  Includes selected model hash and final source/test/UI hashes.
- `browser-verification.json`: dedicated Chromium session checking real sample
  inference, source comparison, worker evidence, documentation/validation pages,
  video/camera controls and a 390px viewport. Optional video-playback details are
  included when the short-clip browser check was run.
- `cli-verification.json`: actual process exit behavior for help, bad confidence,
  bad image size and unavailable CUDA on this CPU-only machine.

The full automated log, image reports, synthetic clip and scored movie remain
under `outputs/presentation-final-verified/`. Browser screenshots are in the local
browser-check output directory; reviewed images are versioned under `docs/assets/`.

The clip repeats a development validation still. These checks establish software
behavior and integration, not independent accuracy or sustained performance.
The later `live-browser-verification.json` records short actual video and physical
camera preview checks, including changing images, AI evidence and Stop controls.
The unit tests use camera fixtures; the separate host check verified camera 1/MSMF.
See [final audit](../../final-audit.md).
