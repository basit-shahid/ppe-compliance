# Finalization audit

Reviewed **16 November 2026**. This audit concerns presentation software readiness
and documentation. Independent scientific evaluation is a separate pending task.

## 1. Scope reviewed

Reviewed the detector adapter, geometry/visibility policy, rendering, session
tracking, sequential video exports, latest-frame camera pipeline, sampled recorded
review, CLI/UI adapters, frozen-manifest evaluation, training artifacts, existing
tests and submission documentation.

The existing application was functional; all 180 original tests passed before
changes. The main report and PDF still described an earlier Phase 3 milestone,
and the frontend had limited presentation structure.

## 2. Findings and resolutions

| Finding | Resolution | Evidence |
| --- | --- | --- |
| Camera launch ignored the UI confidence setting | Forward `--confidence` to the child process | Command and UI regression checks |
| Image upload size only considered compressed bytes | Inspect supported format headers; cap at 24 megapixels before OpenCV allocation | Oversize rejection before `imdecode` |
| Ultralytics patches Pillow to install HEIF support for unidentified images | Use explicit JPEG/PNG/WebP/BMP readers; bad uploads never call patched `Image.open` | Corrupt-signature regression cases |
| Direct image API could receive invalid inference values | Validate finite probability and positive integer size before model loading | Invalid public API matrix |
| Unavailable device assertions could escape the UI | Convert model prediction assertion to actionable device/settings error; UI also handles assertion failures | Mocked CUDA rejection |
| Failed image writes could leave incomplete completed-looking runs | Stage and atomically publish the image/report group | Image and JSON write-failure checks |
| Browser preview trusted filenames or encoder exit code alone | Require nonempty output files and handle moved/missing exports | Missing/empty preview and conversion checks |
| Legacy H.264 conversion could reject odd dimensions | Pad right/bottom to even sizes | Encoder configuration reviewed; existing video padding checks retained |
| UI eagerly buffered every download | Load one selected export per rerun | New evidence export selector |
| Main PDF/report no longer reflected implemented code | Current report, pipeline walkthrough, viva guide and rehearsal material | Generated PDF and source documentation |
| Frontend did not support a structured presentation | PPE Vision styling, comparison/evidence tabs, guide/validation pages, attributed sample | AppTest and browser screenshots |

## 3. Verification

The machine-readable evidence is in
[`experiments/final-readiness/verification.json`](experiments/final-readiness/verification.json).
The repeatable runner is `scripts/verify_presentation.py`; full logs and large
artifacts remain under its generated local `outputs/` directory.

Checks include dependency consistency, automated unit/integration/UI tests,
actual CPU inference using selected weights, lossless image dimensions, all image
JSON reports, H.264 readback, exact video/report frame counts, source timestamps,
playback FPS and corrupt-input rejection. The browser utility separately checks
actual rendering and navigation.

The browser-preview update's recorded suite passes **212 tests** with no skips. Actual CLI checks
also verify normal help, invalid confidence/image-size exits and unavailable CUDA
recovery without an uncaught traceback. The snapshot includes hashes of runtime
source, tests, UI assets and verification scripts. Browser evidence is recorded
separately in `browser-verification.json`; CLI evidence is in `cli-verification.json`.

The checked model SHA-256 matches the saved baseline:
`42538d40127ef78117cfade974369bf8ae993182442232b8faf670fd4cbc1448`.

The video smoke input repeats the included validation still for eight frames.
That establishes export/integration correctness only. Its processing rate is a
short CPU measurement and must not be generalized to real-time deployment.

## 4. Scope limits that remain

| Item | Current position | Required next evidence |
| --- | --- | --- |
| Independent accuracy | Pending independent images and human worker labels | Frozen real-camera manifest; no train/validation scene overlap |
| Sustained speed | Historical short checks and current smoke test only | Multi-minute natural-motion runs on presentation/deployment hardware |
| Current CUDA pipeline | Local machine has CPU-only PyTorch | Rerun current source in a configured GPU environment |
| Physical camera | Short actual browser check passed on camera 1/MSMF at 1280x720, including AI and Stop | Repeat on the presentation setup; this is not a sustained benchmark |
| Wearing-state correctness | Geometry provides plausible ownership only | Dedicated labels/model and independent wearing-state evaluation |
| Identity continuity | Geometric session IDs can switch | Appearance tracking and evaluated ID continuity if required |
| Remote hosting | Browser dashboard uses the app host's camera; optional native windows remain | A remote visitor's own camera needs a separate browser-camera transport |
| Distribution rights | Dataset/sample attribution retained | Review software/model/dataset terms for the intended distribution |

These are disclosed boundaries, not fabricated completed results. The repository
can be demonstrated and its code explained while those research measurements
remain outstanding.

The recorded-video Stop control was rechecked in the actual browser on
17 November 2026: playback stopped after 22 of 1,096 source frames with three AI
samples and `preview_stop`, before the clip ended. The separate physical camera
check used camera 1/MSMF and ended with `browser_stop`. Details are in
`experiments/final-readiness/live-browser-verification.json`.

## 5. Maintaining the final state

1. Keep the selected weights and baseline provenance together.
2. Rerun `verify_presentation.py` after changes to inference, rules, video or UI.
3. Record new checks with date, source/model hashes, configuration and hardware.
4. Regenerate the report when changing claims or behavior.
5. Preserve development validation metrics as validation; add independent results
   in a separate experiment when suitable data becomes available.
