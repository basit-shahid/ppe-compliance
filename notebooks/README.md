# Training notebooks

Use notebooks for dataset inspection and GPU training experiments.
Keep reusable inference and decision logic in src/ppe_compliance.
Clear sensitive data and large outputs before committing notebooks.

Phase 3 notebook: `PPE_Phase3_Training.ipynb`.
Upload it to Google Colab, then upload `outputs/colab/ppe-colab-project.zip` when
prompted. The notebook downloads and reproduces the audited dataset remotely.
Follow `docs/phase-3-colab.md` for GPU setup, Drive checkpoints and local model use.
Notebook preparation is verified locally; actual GPU training is pending.
