# Training notebooks

Use notebooks for dataset inspection and GPU training experiments.
Keep reusable inference and decision logic in src/ppe_compliance.
Clear sensitive data and large outputs before committing notebooks.

Phase 3 notebook: `PPE_Phase3_Training.ipynb`.
Upload it to Google Colab, then upload `outputs/colab/ppe-colab-project.zip` when
prompted. The notebook downloads and reproduces the audited dataset remotely.
Follow `docs/phase-3-colab.md` for GPU setup, Drive checkpoints and local model use.
Baseline GPU training results are recorded in `docs/phase-3-results.md`.

Uploaded-video GPU notebook: `PPE_Video_GPU_Ready.ipynb`. Fixed source is embedded;
use the Colab Files panel for model/video inputs. An eight-frame GPU smoke test
precedes the main run. Single-pass H.264 dashboard export is the default;
PROCESSING_HEIGHT=1080 reduces per-frame analysis work; use 720 for speed or 0
for original-resolution detail. COMPARE_EXPORTS compares export paths at the same
processing resolution. CPU comparison is optional. It benchmarks the
current scored video/dashboard pipeline on Colab CPU versus GPU and optionally
processes the full video on CUDA. Build its matching source/model ZIP using
`python scripts/build_video_colab.py`; see `docs/colab-video-testing.md`.
Notebook preparation and CPU integration are checked locally; this new GPU
benchmark must be executed in the user's Colab session.
