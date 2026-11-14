# Project scripts

Run from the repository root with the project environment. Application dependencies
are declared in `pyproject.toml`. Dataset preparation preserves the source files.

## Prepare the dataset

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --source data/raw/construction-ppe --exclusions configs/dataset-exclusions.json --destination data/processed/ppe-three-class
Copy-Item data/processed/ppe-three-class/dataset.yaml configs/dataset.yaml
```

Download/extract the audited source first; see `docs/dataset-audit.md`. The script
refuses to overwrite an existing destination. The generated local YAML is ignored.

## Create a video integration sample

```powershell
.\.venv\Scripts\python.exe scripts/make_video_sample.py
```

Requires the audited prepared validation images. Generates a 30-frame sequence
of validation stills at 960x540/6 FPS, with provenance JSON. It is not natural-motion
footage or independent accuracy evidence. Existing outputs are not overwritten;
use `--output` for another location. See `docs/phase-5.md`.

## Build Colab upload artifacts

Python 3.11+; requires local `models/yolo11n.pt` initialization weights.

```powershell
.\.venv\Scripts\python.exe scripts/build_colab.py
```

Creates the notebook in `notebooks/` and matching upload copies under
`outputs/colab/`. The notebook embeds the bundle checksum, so upload a pair built
together. It downloads the dataset directly to Colab. See `docs/phase-3-colab.md`.

## Build the comprehensive PDF

The PDF is committed for reading without extra packages. To regenerate it:

```powershell
.\.venv\Scripts\python.exe -m pip install reportlab
.\.venv\Scripts\python.exe scripts/build_project_report.py
```

Inputs are `docs/project-report.md` and reviewed figures in `docs/assets/`.
Output is `docs/PPE_Project_Comprehensive_Report.pdf`. The build embeds Segoe UI
when available and otherwise uses standard PDF fonts. Report tooling is separate
from runtime application dependencies.
