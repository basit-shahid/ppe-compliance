# Project documentation

## Start here

- [Pipeline and architecture](pipeline.md): complete runtime flow, geometry, rules,
  video modes, timing and output contracts.
- [Code walkthrough and viva](code-walkthrough.md): important modules, traced example
  and examiner questions with implementation-specific answers.
- [Presentation guide](presentation-guide.md): setup, six-minute demo, backup and troubleshooting.
- [Finalization audit](final-audit.md): reviewed findings, fixes, test evidence and remaining limits.
- [Presentation PDF](PPE_Presentation.pdf) and [editable slide content](submission-slides.md).
- [Project checkpoint](project-status.md): current milestone and next steps.
- [Comprehensive report (PDF)](PPE_Project_Comprehensive_Report.pdf): full project,
  current implementation, dataset/training evidence, worker/video pipelines,
  reliability, verification and reproduction. Updated 16 November 2026.
- [Report source](project-report.md): editable Markdown used to generate the PDF.
- [Historical Phase 3 report](archive/phase-3-project-report.md): earlier milestone;
  its planned-work statements have been superseded by the current report.
- [Phase plan](plan.md): current gates and original relative schedule.

## Completed work

- [Phase 1 verification](phase-1.md): environment, baseline and error checks.
- [Dataset audit](dataset-audit.md): preparation, actual mapping, counts and limitations.
- [Dataset attribution](dataset-attribution.md): source, licence and modifications.
- [Phase 3 results](phase-3-results.md): training review and local model checks.
- [Phase 4 image assessment](phase-4.md): association, statuses, review workflow,
  tests and CPU integration evidence.
- [Phase 5 video/webcam](phase-5.md): commands, playback/timing semantics, short
  CPU benchmarks, capture/preview checks and limitations.
- [Local live-performance enhancement](live-performance.md): latest-frame capture,
  separate preview/results, optional recording and observed latency. Not pushed yet.
- [Local tracking and dashboard](tracking-ui.md): session IDs, status-change events,
  clearer preview and HD capture options.
- [Phase 6 interface and quality](phase-6.md): upload/review/export, camera controls and compact overlays.
- [Recorded live review](recorded-live-review.md): original-motion video with separately timestamped selected-frame AI, startup caching and coverage limits.
- [Phase 7 evaluation and submission](phase-7.md): frozen manifest runner, final metrics and rehearsal.
- [Submission slides](submission-slides.md): presentation outline with pending evidence clearly marked.
- [Experiment log](experiments.csv) and [baseline evidence](experiments/baseline-001/).

## Reproduction

- [Colab training guide](phase-3-colab.md).
- [Colab video GPU testing](colab-video-testing.md): compare CPU/GPU inference,
  rendering and complete exports using the current source and trained weights.
- [Scripts](../scripts/README.md) for data preparation, notebook packaging and report generation.
- Environment snapshots: [Windows CPU](environment-windows.txt),
  [historical Linux CPU](environment-cpu.txt),
  [Colab GPU](experiments/baseline-001/environment-colab.txt).

Snapshots and original experiment configs can contain historical paths. They are
evidence, not portable installation lockfiles. Large data, weights, exported ZIPs
and routine generated outputs stay in ignored local artifact directories.
Reviewed report/README figures are in [assets](assets/README.md).
