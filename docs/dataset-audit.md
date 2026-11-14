# Dataset audit: Construction-PPE

Audited 2026-11-14. Training/validation preparation is complete. An independent
final test set is still outstanding; scene independence is not established.

## Selection

Selected **Ultralytics Construction-PPE, November 2026 release**, as a manageable
first fine-tuning dataset covering all three project targets. It provides direct
YOLO annotations and a public download without account credentials.

This is a suitable semester baseline, not a claim that it is universally best.
The current release differs from the proposal's legacy 1,416-image/11-class data.

Alternatives checked:

- Roboflow Construction Site Safety v27: 2,801 exported images, including five
  augmented outputs per training example; 2,605 train, 114 validation, 82 test.
  Exported count is not an independent-scene count. Stretch resizing and baked-in
  augmentations make it less straightforward as the first clean baseline.
- The proposal's Hard Hats source lacks the full person/helmet/vest task coverage
  and cannot alone support this project's worker-level compliance pipeline.

Sources:
https://docs.ultralytics.com/datasets/detect/construction-ppe
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27

## Provenance and licence

- Archive: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
- Downloaded bytes: 647,666,037.
- SHA-256: `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`.
- Original archive: `data/raw/construction-ppe.zip`.
- Unmodified extracted source: `data/raw/construction-ppe/`.
- Confirmed from archive `README.md`, `LICENSE.txt`, and `data.yaml`:
  November 2026 release; images/annotations under CC BY 4.0.
- Credit and modification statement: `docs/dataset-attribution.md`.

The release URL is mutable; use the archive hash to identify this exact snapshot.

## Classes and conversion

Source IDs: 0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 Person,
6 no_helmet, 7 no_goggle, 8 no_gloves, 9 no_boots, 10 forklift, 11 no_vest.

| Target class | Source ID | Prepared ID |
| --- | --- | --- |
| person | 5 | 0 |
| helmet | 0 | 1 |
| vest | 2 | 2 |

Other annotations are omitted in the prepared copy; originals remain available.
The source vest class includes high-visibility vests, jackets and shirts.
Person boxes and equipment boxes are separate: they do not supply worker IDs or
prove equipment is worn. Association remains Phase 4 work. Missing-PPE classes
are not used in this three-class baseline, and no missing-PPE labels are invented.

## Observed counts

Raw: 800 training images, 200 validation images, 14,863 boxes over all 12 classes.

Prepared: `data/processed/ppe-three-class/`.

| Split | Images | Person boxes | Helmet boxes | Vest boxes |
| --- | --- | --- | --- | --- |
| Train | 799 | 1,961 | 1,281 | 1,476 |
| Validation | 200 | 481 | 294 | 340 |
| Test | Not supplied | — | — | — |

Eight training images have empty target labels. All eight were visually checked:
they show forklifts without people and are retained as background examples.
No source image was reassigned between splits.

## Checks and manual review

- All 1,000 images decode; all source rows have five finite YOLO fields,
  known integer class IDs, valid positive sizes, and boxes within image bounds.
- No missing label files or invalid images were found.
- SHA-256 of encoded files and decoded pixels recorded per prepared image.
  No exact decoded-image duplicates were found.
- Cross-split 64-bit difference-hash search at Hamming distance <=4 flagged
  one pair. Visual review confirmed near-identical doorway/background composition
  with changed worker/PPE. Its training variant was excluded; validation retained.
- Reviewed exclusion: `configs/dataset-exclusions.json`. The excluded prepared
  image/label are under `audit/quarantine/`; original raw files remain unchanged.
- Reviewed 12 evenly spaced annotated training images, 12 validation images,
  the flagged pair and all eight empty-label images. Sampled target boxes appear
  consistent with people and visible gear; this is not exhaustive semantic review.
- The preparation script was checked using controlled fixtures for ID conversion,
  invalid class rejection, duplicate removal, raw preservation and overwrite refusal.
- Ultralytics' actual YOLODataset loader scanned both prepared splits with zero
  corrupt entries, recognized the eight backgrounds, and successfully loaded a
  3-by-640-by-640 sample tensor from each split. No training was performed.

Audit artifacts under `data/processed/ppe-three-class/audit/`:
`report.json`, `manifest.csv`, `train-contact-sheet.jpg`, `val-contact-sheet.jpg`,
`review-cases.jpg`, and `loader-check.json`. Contact sheets were generated before
the single reviewed exclusion; the manifest and report reflect the final copy.

## Limitations and evaluation gate

The publisher reports both collected and generated imagery, with human and AI
annotation. Visual previews show recurring environments across train/validation;
the removed pair does not resolve all possible scene dependence. Hash checks
cannot detect all related scenes or prove annotation completeness. Development
metrics may be optimistic and must not be presented as independent deployment
performance. Synthetic visual artifacts and domain shift are additional risks.

No separate test set is supplied or fabricated. Obtain consented independent
real-camera images/video, annotate person/helmet/vest and worker-level outcomes,
and keep related video frames together. Freeze that evaluation set before model
selection and do not tune on it. This requirement remains open before final
evaluation; training on the prepared baseline can proceed in Phase 3.

## Reproduce preparation

Extract the recorded archive into `data/raw/`, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --source data/raw/construction-ppe --exclusions configs/dataset-exclusions.json --destination data/processed/ppe-three-class
```

The destination must not already exist. Use a new destination for another run;
the script refuses to overwrite existing prepared data. Training configuration:
`configs/dataset.yaml`. Its absolute root is specific to this checkout; update it
when moving to Colab or another computer. No training was started in Phase 2.
