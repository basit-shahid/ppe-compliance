# Phase 3: Google Colab training

Status: baseline-001 was completed by the user on Colab and its ZIP imported on
2026-11-14. See `docs/phase-3-results.md`. The instructions below remain available
for reproducing training. Sign-in, GPU allocation and Drive authorization happen
in the user's browser.

## Start training

1. Open https://colab.research.google.com/ and sign in.
2. Choose **File > Upload notebook**, then select
   `D:\ppe-compliance\notebooks\PPE_Phase3_Training.ipynb`.
3. Choose **Runtime > Change runtime type** and select a GPU hardware accelerator
   available on the free tier. Do not select a paid option.
4. Choose **Runtime > Run all**, or run each cell from the top.
5. At the file-upload prompt select
   `D:\ppe-compliance\outputs\colab\ppe-colab-project.zip` (about 5 MB).
6. Authorize the Google Drive mount when Google prompts you. The notebook writes
   its outputs into `My Drive/PPE_Compliance/runs/baseline-001`.
7. Let the short training check finish, then the baseline and validation. The
   final cell downloads `baseline-001-results.zip` and keeps a copy on Drive.

GPU availability and free runtime length are not guaranteed. If the GPU assertion
fails, check the runtime selection; if no free GPU is available, retry later.
No local CPU training or paid cloud service is started by these project files.

## What the notebook runs

- Installs Ultralytics 8.4.212 while keeping Colab's CUDA-enabled PyTorch.
- Verifies the small bundle and its individual file checksums.
- Downloads the original archive directly to Colab and checks our audited SHA-256.
  A changed upstream release stops execution for review.
- Reproduces the same three-class conversion and reviewed exclusion, checking
  799 training images, 200 validation images, and target instance counts.
- Records GPU, Python/package versions, data provenance and attribution on Drive.
- Runs a two-epoch training check on 10% of the training images.
- Starts a separate baseline from general pretrained YOLO11 nano weights:
  30 requested epochs, image size 640, batch 8, seed 42, patience 10.
  Early stopping can end the run before 30 epochs.
- Saves best/last weights and per-epoch checkpoints to Drive.
- Records validation precision, recall, F1, mAP metrics and diagnostic plots.
- Exports best/last weights, metrics, environment and provenance in the result ZIP.
  Per-epoch backup weights remain on Drive but are omitted from the download.

The smoke model is not used as the initial model for the full baseline.
There is no independent test set. Results are development validation measurements;
recurring scenes and generated imagery limit conclusions about real cameras.
Reported model timings are not end-to-end video FPS.

## If Colab disconnects

Reconnect with a GPU and rerun the notebook from the top. Re-upload the same small
bundle and authorize Drive. Dataset files on the temporary runtime may need to be
downloaded again. Keep the same `RUN_NAME` to resume from `weights/last.pt`.
At least one completed epoch/checkpoint is required to resume. A completed run is
not trained again. Use a new run name for a new experiment.

If GPU memory runs out, start a new run name with `BATCH = 4` in the baseline
cell (and change the smoke cell's batch if necessary). A resumed checkpoint keeps
its training settings. Do not delete saved checkpoints just to rerun a cell.

## Bring the model back to Windows

Extract the downloaded results ZIP. Copy its `weights/best.pt` to
`D:\ppe-compliance\models\best.pt`. Preserve the other results for review, then:

```powershell
.\.venv\Scripts\ppe.exe detect --source "D:\path\to\photo.jpg" --weights models/best.pt --device cpu
```

Worker-level compliance remains unimplemented until Phase 4. Record the actual
training results in `docs/experiments.csv` and review failures before another run.

## Local verification performed

All notebook Python cells parsed. Bundle member hashes were verified. Stubbed
training checks covered fresh runs, interrupted-checkpoint resume, completed and
finalized runs, and refusal to overwrite partial runs without checkpoints.
These checks do not establish GPU training success; actual Colab execution is pending.

Rebuild both matching upload files after changing bundled code/configuration:

```powershell
.\.venv\Scripts\python.exe scripts/build_colab.py
```

References: https://docs.ultralytics.com/modes/train/ and
https://research.google.com/colaboratory/faq.html.
