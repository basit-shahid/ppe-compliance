# PPE Compliance Detection
 semester prototype based on the PPE project proposal.

## Scope

Detect people, helmets, and safety vests in images and recorded videos; associate
PPE with each person; display **PPE detected**, **potential violation**, or
**review required**. Detection does not establish overall workplace safety.

## Current status

Repository scaffold only. The CLI help works; detection, training integration,
association, compliance decisions, and video processing are not implemented yet.
Module files document their planned responsibilities.

## Setup

Use Python 3.10 or newer. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
ppe --help
```

Optional interface and analysis packages:

```bash
python -m pip install -e ".[ui,analysis]"
streamlit run app/streamlit_app.py
```

Installing dependencies and downloading model weights require internet access.
GPU training is recommended; pretrained image inference can start on a CPU.
Record the resolved environment after setup with `python -m pip freeze`.

## First implementation milestone

Implement one-image inference in `src/ppe_compliance/detector.py`, then add a CLI
command accepting an image and weights path. Confirm boxes, class names,
confidence scores, and saved output. General pretrained YOLO weights are a setup
baseline, not a trained PPE model.

Next audit one dataset using `docs/dataset-audit.md`. Copy
`configs/dataset.example.yaml` to `configs/dataset.yaml` and correct its paths
and class mapping. After verifying annotations, the planned training command is:

```bash
yolo detect train cfg=configs/training.example.yaml
```

This command performs real training and may download weights. It has not been run
as part of scaffolding. Start with a short run before committing GPU time.

## Structure

- `src/ppe_compliance/`: detector, association, compliance, video, and CLI modules.
- `configs/`: example dataset and training configuration.
- `app/`: optional Streamlit interface.
- `data/`, `models/`, `outputs/`: local artifacts excluded from version control.
- `notebooks/`: exploratory and GPU-training notebooks.
- `docs/`: dataset audit, experiment log, and one-month plan.

## Evaluation

Measure per-class precision/recall, mAP@0.5, mAP@0.5:0.95, worker-level compliance
results, review rate, and end-to-end speed on stated hardware. Keep test data
separate from model selection. Include overlap, occlusion, held helmets, distant
workers, and poor lighting in the failure analysis.

## Data and model permissions

Verify dataset, pretrained-weight, and dependency terms before redistribution.
Record attribution and use consented demonstration footage. No project licence
has been selected yet.
