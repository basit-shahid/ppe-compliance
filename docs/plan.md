# One-month implementation plan

## Current gates

- Complete: Phase 1 environment and baseline inference.
- Prepared: Phase 2 training/validation data; independent test data outstanding.
- Complete baseline: Phase 3 training, validation review and local PPE inference.
- Implemented baseline: Phase 4 image association, conservative statuses and tests.
- Implemented baseline: Phase 5 video/webcam, JSONL output, cleanup and short-run benchmarks.
- Next: Phase 6 interface; then Phase 7 independent evaluation/submission.

See `project-status.md` for evidence and `PPE_Project_Comprehensive_Report.pdf`
for phase details and suggested team roles. The allocations below are the
original relative schedule, not elapsed dates or a completion promise.

## Original day allocations

- Days 1–2: environment setup and one-image pretrained inference.
- Days 3–4: audit one dataset and verify class IDs and splits.
- Days 5–7: short training check, baseline training, and buffer.
- Days 8–9: validation analysis and one justified improvement.
- Days 10–14: association, uncertainty rules, complete image pipeline, and buffer.
- Days 15–17: recorded video and end-to-end speed measurements.
- Days 18–21: minimal interface, integration checks, buffer, and feature freeze.
- Days 22–24: held-out detection and compliance evaluation.
- Days 25–28: report, setup instructions, slides, and backup demonstration.
- Days 29–30: rehearsal and important fixes.

Webcam is optional. Tracking, restricted zones, and dashboards are deferred.
Never tune against the final held-out test set.
