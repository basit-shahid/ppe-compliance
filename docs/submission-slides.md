# Submission presentation outline

1. **Problem and scope:** assist reviewers with visible helmet/vest evidence;
   three classes, local inference and human review.
2. **Data:** audited source, mapping, 799/200 development split, provenance,
   recurring-scene limitations and independent test-set status.
3. **Training:** YOLO11 nano, Colab T4, 30 epochs; validation precision 94.99%,
   recall 83.54%, mAP50 90.08%, mAP50-95 71.15%. Label all as validation results.
4. **Worker evidence:** head/torso association, ambiguity rules, review defaults,
   session tracking and expiry. Show a held-helmet or occlusion failure example.
5. **Demonstration:** focused image/video UI, compact markers, worker report,
   native live monitor and downloads. Include a local recorded backup.
6. **Performance:** show measured display rate separately from assessment rate,
   source/model resolution, hardware and duration. Include cold-start latency.
7. **Final evaluation:** insert frozen independent results once available;
   missed/spurious workers, status confusion, review rate and cross-source cases.
   Until then, explicitly mark this section pending independent data.
8. **Limits and next work:** small-item recall, occlusion, scene independence,
   geometric ID switches, driver buffering and CPU throughput. Explain that
   detected equipment is not proof of wearing state or general workplace safety.
