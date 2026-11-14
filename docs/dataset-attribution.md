# Dataset attribution

Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026.

- Source: https://docs.ultralytics.com/datasets/detect/construction-ppe
- Archive: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
- Publisher-stated image/annotation licence: Creative Commons Attribution 4.0
  International, https://creativecommons.org/licenses/by/4.0/
- Retrieved: 2026-11-14. See `dataset-audit.md` for the downloaded archive hash
  and observed contents; the release URL can change over time.

Project modifications: retain person, helmet and vest labels; remap their IDs to
0, 1 and 2; omit other classes; remove exact decoded-image duplicates from the
prepared copy; exclude images with invalid or missing annotations. The original
download is retained without edits. Images without target annotations remain
negative examples, subject to the source annotation completeness limitations.

Retain this credit, source and licence links, and change description when sharing
derived data or examples. Dataset terms are separate from Ultralytics software
terms. No licence for the project itself is selected by this attribution file.
