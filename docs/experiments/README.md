# Experiment evidence

`../experiments.csv` indexes recorded experiments. `baseline-001/` preserves:

- `args.yaml`: original Colab training configuration, including historical paths.
- `results.csv`: original per-epoch training measurements.
- `validation-metrics.json`: separately reported final validation metrics.
- `provenance.json`: hardware, versions, class mapping and input hashes.
- `dataset-audit.json`: preparation audit returned from Colab.
- `environment-colab.txt`: historical installed package snapshot.
- `local-verification.json`: weight/archive fingerprints and CPU integration checks.

These are original compact evidence files, not current machine configuration.
Weights, full exports and routine images remain in ignored local artifact folders.
Interpretation and caveats are in [the training review](../phase-3-results.md).

`phase-4/local-checks.json` records the selected model hash and four image-rule
integration checks on development examples. It is not another training run or an
independent worker-level evaluation. See [Phase 4](../phase-4.md).

`phase-5/` preserves original summaries for the synthetic recorded sequence and
two bounded camera checks, fixture provenance, and verification metadata. Camera
media remains ignored/local. Rates describe short runs with cold-start effects,
not sustained deployment speed. See [Phase 5](../phase-5.md).

`live-performance/` contains local short-run preview and optional recording
summaries, plus readback verification. Preview-update rate, assessment rate and
application result age are distinct metrics; these are not independent PPE
accuracy or a controlled A/B performance benchmark. Camera media stays local.

`tracking-ui/` contains local tracked-video readback and short HD dashboard checks.
No natural-motion accuracy or sustained performance claim is made.

- [Interface and quality verification](quality-ui/verification.json): UI interactions, real-model exports, browser H.264 playback, renderer microbenchmark and unavailable camera check (2026-11-15).

- [Colab video preparation](colab-video/verification.json): verified current source/model bundle, generated notebook, CPU exports and timing checks. Colab GPU execution remains pending.

- [Uploaded T4 result review](colab-video/gpu-results-review.md): 51.6 FPS detector calls versus 3.69 FPS complete processing; hash/render/export bottlenecks and cropped-worker vest association findings.

- [Prioritized video improvements](colab-video/prioritized-improvements.md): single H.264 export, display-sized drawing, unchanged source evidence and targeted cropped-worker association fixes, with a same-device comparison workflow.

- [Recorded-video lag follow-up](colab-video/recorded-video-lag.md): 1080p analysis mode, easier H.264 decoding/seeking, resolution comparison and verified 270-frame playback cadence.

- [Finished-player motion investigation](colab-video/finished-player-motion.md): source/export optical-flow alignment, actual visible-browser delivery tests, in-app frame counters and paced terminal review.

- [Recorded live review](../recorded-live-review.md): native source-paced playback with selected-frame AI; full 270-frame ordering, unmodified source-pane pixels and sample-hash verification.
