# Local data

Keep raw downloads, prepared YOLO datasets, and consented demo media here.
Large files and personal media are excluded from Git.

Before training, record the source URL, version, licence, attribution, class mapping,
annotation quality, and split strategy in docs/dataset-audit.md.
Check duplicates and keep related video frames in the same split.

Prepared baseline: `processed/ppe-three-class/` (799 train, 200 validation).
Raw archive and original 12-class data are retained under `raw/`.
See `docs/dataset-audit.md` for conversion, exclusions, counts, and limitations.
No independent test set is present. Keep the dataset attribution when sharing.
