# PPE Compliance Detection
 semester prototype based on the PPE project proposal.

## Scope

Detect people, helmets, and safety vests in images and recorded videos; associate
PPE with each person; display **PPE detected**, **potential violation**, or
**review required**. Detection does not establish overall workplace safety.

## Current status

Single-image YOLO inference is implemented, with annotated JPEG and JSON output.
PPE fine-tuning, association, compliance decisions, and video processing remain
future milestones. General pretrained weights do not establish PPE compliance.

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
The verified Linux/Python 3.12 CPU environment is recorded in
`docs/environment-cpu.txt` as a reference, not a cross-platform lockfile.
For a CPU-only PyTorch installation, run this before `pip install -e .`:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

If your environment restricts home-directory writes, set `YOLO_CONFIG_DIR` to a
writable local directory, for example `.ultralytics`, and `MPLCONFIGDIR` to a
writable cache directory.

## Run image detection

```bash
ppe detect --source path/to/image.jpg --weights yolo11n.pt
```

The first run downloads general pretrained weights if absent. To keep weights
under `models/`, download or place them there and pass `--weights models/yolo11n.pt`.
This model detects general objects, including people; it is not PPE-trained.
Each run creates a unique directory under `outputs/` containing `annotated.jpg`
and `detections.json`. Input images and past results are not overwritten.

```bash
ppe detect --source path/to/image.jpg --weights models/best.pt --confidence 0.35 --image-size 640 --device cpu
ppe detect --help
```

Only local still images are supported in this milestone. Invalid input returns a
nonzero exit code. JSON boxes use original-image pixel coordinates
`[left, top, right, bottom]`; class names come from the loaded weights.
Confidence scores are model scores, not verified probabilities of correctness.

For a public setup-only sample (not PPE evaluation footage):

```bash
mkdir -p data/samples
curl -fL https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg -o data/samples/bus.jpg
ppe detect --source data/samples/bus.jpg
```

The sample is from the Ultralytics assets repository; it and model weights are
not redistributed in this repository. Review upstream terms for reuse.

## Next milestone: dataset audit and fine-tuning

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

## Verified baseline

The CPU smoke run on the upstream `bus.jpg` sample produced five detections:
four people and one bus. The annotated image was readable and JSON boxes were
checked against original image dimensions. Missing files, corrupt images, and
invalid confidence/image-size arguments were also checked. Dependency consistency
passed `pip check`. This is a setup verification, not a PPE accuracy evaluation.
