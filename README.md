# PPE Compliance Detection

A semester AI project using a fine-tuned YOLO11 nano model to detect workers,
helmets and high-visibility safety clothing. Development follows seven phases
covering data, training, worker-level PPE checks, video and final evaluation.

**Current milestone:** PPE training and local still-image inference are complete.
Association, compliance decisions, project video processing and the interface
are planned. Object detection alone does not establish workplace safety.

[Comprehensive report (PDF)](docs/PPE_Project_Comprehensive_Report.pdf) ·
[Current checkpoint](docs/project-status.md) · [Documentation index](docs/README.md)

## What works

- Detect `person`, `helmet` and `vest` using selected PPE weights.
- Export annotated JPEG and JSON into a unique directory for every image run.
- Reject missing/corrupt images and invalid confidence/image-size values.
- Reproduce dataset preparation and Colab GPU training.
- Review recorded configuration, metrics, provenance and failure analysis.

The source `vest` label includes high-visibility jackets and shirts. Worker-level
statuses are not implemented yet; detecting a helmet does not prove it is worn.

![Local PPE detection example](docs/assets/phase-3-local-detection.jpg)

*Validation image with local predictions: integration evidence, not an independent
accuracy result. [Attribution and changes](docs/dataset-attribution.md).*

## Baseline results

Trained on a Colab Tesla T4 for 30 epochs, image size 640, batch 8 and seed 42,
using **799 training and 200 validation images**.

| Validation metric | Reported result |
| --- | ---: |
| Mean precision | 94.99% |
| Mean recall | 83.54% |
| mAP@0.5 | 90.08% |
| mAP@0.5:0.95 | 71.15% |

These are development measurements, not overall real-world accuracy. Helmet recall
is weakest (76.53%). Recurring/generated scenes limit validation independence.
An independent real-camera test set and end-to-end video benchmarks are pending.
See [training review](docs/phase-3-results.md), [experiment log](docs/experiments.csv)
and [baseline evidence](docs/experiments/baseline-001/).

## Setup

Python 3.10+ is required for the application; artifact-building scripts require
Python 3.11+. The verified Windows environment uses Python 3.13.

### Windows PowerShell

```powershell
git clone https://github.com/basit-shahid/ppe-compliance.git
cd ppe-compliance
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ppe.exe --help
.\.venv\Scripts\python.exe -m pip check
```

These executable paths work without virtual-environment activation. For CPU-only
PyTorch, install it before installing the project:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ppe --help
```

Downloads require internet access. Recorded environments:
[Windows CPU](docs/environment-windows.txt), [historical Linux CPU](docs/environment-cpu.txt),
[Colab GPU](docs/experiments/baseline-001/environment-colab.txt).
These are historical snapshots, not portable lockfiles. Keep Colab's CUDA PyTorch;
do not install the Windows CPU snapshot into a GPU runtime.

## Run image detection

Place the selected PPE weights at `models/best.pt`, then run:

```powershell
.\.venv\Scripts\ppe.exe detect --source "D:\path\to\photo.jpg" --weights models/best.pt --device cpu
```

**Weights are not committed to Git.** They exist on the development machine.
On a fresh clone, reproduce training and copy exported `weights/best.pt` to
`models/best.pt`; see [Colab instructions](docs/phase-3-colab.md).

```powershell
.\.venv\Scripts\ppe.exe detect --source "D:\path\to\photo.jpg" --weights models/best.pt --confidence 0.35 --image-size 640 --device cpu
.\.venv\Scripts\ppe.exe detect --help
```

Every run creates `outputs/<unique-run>/annotated.jpg` and `detections.json`.
JSON includes class names, confidence and original-image pixel boxes
`[left, top, right, bottom]`. Confidence is a model score, not a calibrated probability.
Only local still images are supported by the project CLI at this milestone.

For a general-object setup check, use `--weights yolo11n.pt`; Ultralytics downloads
these weights if absent. General weights are not our fine-tuned PPE detector.
The recorded [public bus sample](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg)
baseline detected four people and one bus.

## Dataset preparation

The selected source is **Ultralytics Construction-PPE, November 2026**. Its existing
12-class annotations were checked and converted to three project classes:

| Class | Source ID | Prepared ID |
| --- | ---: | ---: |
| Person | 5 | 0 |
| Helmet | 0 | 1 |
| Vest | 2 | 2 |

The original 800/200 split was preserved except for one reviewed training exclusion,
leaving 799/200 images. Eight target-free backgrounds were retained. No independent
test split is supplied. The [audit](docs/dataset-audit.md) records the source/hash,
checks, exclusions and limitations; [attribution](docs/dataset-attribution.md) records terms.

After downloading and extracting the audited archive into `data/raw/`:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --source data/raw/construction-ppe --exclusions configs/dataset-exclusions.json --destination data/processed/ppe-three-class
Copy-Item data/processed/ppe-three-class/dataset.yaml configs/dataset.yaml
```

Preparation refuses to overwrite an existing destination. Use a new directory for
another run. **`configs/dataset.yaml` is local and ignored by Git** because it
contains machine-specific paths. Generate it as above; `dataset.example.yaml` is
only a template. Class IDs must match the actual labels.

## Train on free Colab

Follow the [Colab guide](docs/phase-3-colab.md).

1. Put general initialization weights at `models/yolo11n.pt` if absent:
   [official download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt).
2. Build the matching upload files with `python scripts/build_colab.py` (Python 3.11+).
3. Upload `notebooks/PPE_Phase3_Training.ipynb` to Colab and choose a free GPU runtime.
4. Run cells in order; upload `outputs/colab/ppe-colab-project.zip` when prompted
   and authorize Google Drive for checkpoints.
5. Download the results and copy `weights/best.pt` into `models/best.pt`.

The notebook downloads/verifies the dataset, reproduces preparation, runs a short
check, trains a separate baseline and exports validation results. Use the matching
notebook and ZIP generated together: the notebook checks its bundle hash.
Free GPU access/runtime duration depend on availability. No paid service or external
AI API is required by this workflow. Dataset images download directly to Colab.

For a separately configured GPU environment, `yolo detect train cfg=configs/training.example.yaml`
performs real training. Confirm local paths and weights first; this is not a CPU
laptop setup-verification command.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| 1 | Environment and baseline image pipeline | Complete |
| 2 | Dataset audit and preparation | Train/validation ready; independent test pending |
| 3 | PPE training and validation analysis | Baseline complete; local model verified |
| 4 | Worker/PPE association and image statuses | Next |
| 5 | Recorded video, speed measurement and webcam integration | Planned |
| 6 | Optional Streamlit demonstration interface | Planned |
| 7 | Independent evaluation, report and presentation | Planned |

Planned statuses: **PPE detected**, **potential violation**, **review required**.
Missing detections alone do not prove missing equipment. Tracking, restricted zones,
operational violation logging and dashboards remain future extensions.
The Streamlit app is currently a placeholder.

See the [phase plan](docs/plan.md) and [project report](docs/PPE_Project_Comprehensive_Report.pdf)
for acceptance gates, implementation details and suggested three-member roles.

## Repository structure

```text
ppe-compliance/
|-- src/ppe_compliance/    # CLI, detector and planned association/rule/video modules
|-- configs/              # Shared examples and reviewed exclusions; local YAML ignored
|-- scripts/              # Data preparation, Colab packaging and PDF generation
|-- notebooks/            # Colab training notebook
|-- app/                  # Planned Streamlit interface
|-- docs/                 # Guides, report, audit, plan and compact experiment evidence
|   |-- assets/           # Reviewed documentation figures
|   `-- experiments/      # Recorded metrics, configuration and provenance
|-- data/                 # Local raw/prepared data; ignored except README
|-- models/               # Local weights/returned archives; ignored except README
|-- outputs/              # Generated predictions/packages/reviews; ignored except README
`-- pyproject.toml        # Application dependencies and CLI entry point
```

[Script guide](scripts/README.md) · [Documentation guide](docs/README.md)

## Verification and evaluation

Verified: CLI help/version, `pip check`, general/PPE CPU inference, readable output
images and valid JSON boxes, invalid inputs, dataset loader, preparation fixtures,
notebook syntax/resume branches, and PDF structure/layout. No committed automated
test suite or independently verified project video pipeline exists yet.

Keep final test data out of tuning. Evaluate per-class detection, worker-level
outcomes, review rate, missed workers and end-to-end speed. Include distant workers,
occlusion, overlap, held helmets and poor lighting.

## Attribution and permissions

The selected dataset's images and annotations are CC BY 4.0 according to its
included licence. Credit the Ultralytics team and retain source/licence links and
changes in [dataset-attribution.md](docs/dataset-attribution.md). The displayed
example is a modified dataset image with prediction overlays. Software/model
terms are separate from dataset terms. No licence has been selected for this
project's own code. Use consented demonstration footage.
