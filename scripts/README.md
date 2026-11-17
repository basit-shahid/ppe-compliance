# Project scripts

Run from the repository root with the project environment. Application dependencies
are declared in `pyproject.toml`. Dataset preparation preserves the source files.

## Final presentation checks

```powershell
.\.venv\Scripts\python.exe scripts/verify_presentation.py
```

Runs dependency checks, automated tests and actual CPU image/video inference with
local weights. Decodes exports and checks JSON, dimensions, frame counts, FPS and
source timestamps. Creates a new `outputs/presentation-*` evidence directory;
never opens a physical camera or downloads model weights. `--skip-tests` reruns
only the smoke checks. `--output` must be a new directory.

`check_workspace_browser.py` optionally checks an already-running local app using
a dedicated Chromium DevTools session (`.[qa]`). See
[presentation guide](../docs/presentation-guide.md) for setup and limits.

`build_slides.py` generates the 10-slide `docs/PPE_Presentation.pdf` from editable
`docs/submission-slides.md`; install `.[docs]` for report generation tooling.

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
.\.venv\Scripts\python.exe -m pip install -e ".[docs]"
.\.venv\Scripts\python.exe scripts/build_project_report.py
.\.venv\Scripts\python.exe scripts/build_slides.py
```

Inputs are `docs/project-report.md` and reviewed figures in `docs/assets/`.
Output is `docs/PPE_Project_Comprehensive_Report.pdf`. The build embeds Segoe UI
when available and otherwise uses standard PDF fonts. Report tooling is separate
from runtime application dependencies.

## Final worker evaluation

`evaluate_workers.py` runs the reviewed frozen manifest with exact duplicate and
training/validation overlap checks. It exports worker precision/recall/F1,
missed/spurious counts, review rate and status confusion. See
[Phase 7](../docs/phase-7.md) and `configs/evaluation.example.json`. Independent
evaluation images/labels must be supplied; development integration fixtures do
not establish final accuracy.

## Colab video GPU test

`build_video_colab.py` builds `notebooks/PPE_Video_GPU_Ready.ipynb` and
`outputs/colab-video/ppe-video-gpu-bundle.zip` from the current local pipeline
and `models/best.pt`. The notebook compares CPU/GPU inference and complete video
export rates on the same clip; see [video testing guide](../docs/colab-video-testing.md).

`check_video_notebook.py` executes the generated notebook in an actual local CPU
Jupyter kernel using real weights and video. It saves the executed notebook and
a first-error report; install `.[notebook]` first. Use `--quoted-empty-paths` to
exercise the Colab cell 3 form-input regression.
