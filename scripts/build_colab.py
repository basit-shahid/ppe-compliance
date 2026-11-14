"""Build the uploadable Colab notebook and a small, hashed project bundle."""

import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "colab"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    members = ["scripts/prepare_dataset.py", "configs/dataset-exclusions.json",
               "docs/dataset-attribution.md", "docs/dataset-audit.md", "models/yolo11n.pt"]
    missing = [name for name in members if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing bundle inputs: " + ", ".join(missing) +
                         ". See scripts/README.md for prerequisites.")
    bundle = OUTPUT / "ppe-colab-project.zip"
    manifest = {name: sha256(ROOT / name) for name in members}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(ROOT / name, name)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2))
    bundle_hash = sha256(bundle)
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"})

    def code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": dedent(source).strip() + "\n"})

    markdown('''
    # PPE project: Phase 3 GPU training

    Run in **Google Colab**, using **Runtime → Change runtime type → GPU**.
    Run cells in order. Upload `ppe-colab-project.zip` when prompted (about 5 MB).
    No payment or API key is required by this notebook. Free GPU access is subject
    to availability. The notebook stops if no CUDA GPU is assigned.

    The dataset is downloaded directly to Colab, verified against our audited
    SHA-256, and prepared using the same script and reviewed exclusion as Windows.
    You do not need to upload the 648 MB dataset from your laptop.

    Training results/checkpoints go to your Google Drive after you authorize the
    mount. Images stay on Colab's temporary disk. Reconnect by rerunning setup,
    upload, download and preparation cells with the same RUN_NAME; the baseline
    cell resumes an interrupted saved checkpoint. Work since the last saved epoch
    can be lost. Do not enable paid compute for this notebook.

    **Evaluation limitation:** 799 train / 200 validation images, no independent
    test set. Recurring and generated scenes limit generalization claims. These
    are development metrics, not final workplace accuracy or safety guarantees.
    ''')
    code('''
    import os, sys, subprocess
    os.environ["YOLO_CONFIG_DIR"] = "/content/ppe-yolo-config"
    os.environ["MPLCONFIGDIR"] = "/content/ppe-matplotlib"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.4.212", "PyYAML>=6,<7"])
    # Keep Colab's CUDA-enabled PyTorch; do not install the Windows CPU snapshot.
    import torch
    assert torch.cuda.is_available(), "No GPU assigned. Choose a GPU runtime and reconnect before continuing."
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    ''')
    markdown("## Upload the small project bundle\nSelect `outputs/colab/ppe-colab-project.zip` from your PC.")
    code('''
    from pathlib import Path
    import hashlib, json, zipfile
    from google.colab import files

    WORK = Path("/content/ppe-project")
    WORK.mkdir(parents=True, exist_ok=True)
    BUNDLE = Path("/content/ppe-colab-project.zip")
    if not BUNDLE.is_file():
        uploaded = files.upload()
        assert "ppe-colab-project.zip" in uploaded, "Upload ppe-colab-project.zip."
        BUNDLE.write_bytes(uploaded["ppe-colab-project.zip"])
        del uploaded
    ''')
    code(f'''
    def sha256(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    assert sha256(BUNDLE) == "{bundle_hash}", "Bundle differs from this notebook. Use the matching pair of files."
    def safe_extract(archive, target):
        target = target.resolve()
        with zipfile.ZipFile(archive) as handle:
            for item in handle.infolist():
                assert (target / item.filename).resolve().is_relative_to(target), "Unsafe archive path"
            handle.extractall(target)
    safe_extract(BUNDLE, WORK)
    for name, expected in json.loads((WORK / "bundle-manifest.json").read_text()).items():
        assert sha256(WORK / name) == expected, name
    print("Project bundle verified.")
    ''')
    markdown("## Mount Drive and choose the run\nUse the same RUN_NAME to resume. Use a new name for a new experiment. Drive authorization is handled by Google.")
    code('''
    from google.colab import drive
    drive.mount("/content/drive")
    RUN_NAME = "baseline-001"
    assert RUN_NAME and all(c.isalnum() or c in "-_" for c in RUN_NAME)
    DRIVE_ROOT = Path("/content/drive/MyDrive/PPE_Compliance")
    RUN_DIR = DRIVE_ROOT / "runs" / RUN_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("Persistent output:", RUN_DIR)
    ''')
    markdown("## Download and reproduce the audited dataset\nThe archive is about 648 MB. A checksum mismatch stops the notebook instead of silently training on a changed release.")
    code('''
    import urllib.request
    RAW = WORK / "data/raw"
    RAW.mkdir(parents=True, exist_ok=True)
    ARCHIVE = RAW / "construction-ppe.zip"
    EXPECTED_ARCHIVE_HASH = "e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232"
    if not ARCHIVE.is_file():
        temporary = ARCHIVE.with_suffix(".partial")
        urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip", temporary)
        assert sha256(temporary) == EXPECTED_ARCHIVE_HASH, "Source archive changed; audit the new version before using it."
        temporary.replace(ARCHIVE)
    assert sha256(ARCHIVE) == EXPECTED_ARCHIVE_HASH, "Source archive checksum mismatch."
    safe_extract(ARCHIVE, RAW)
    PREPARED = WORK / "data/processed/ppe-three-class"
    if not (PREPARED / "dataset.yaml").is_file():
        subprocess.check_call([sys.executable, str(WORK / "scripts/prepare_dataset.py"),
            "--source", str(RAW / "construction-ppe"),
            "--destination", str(PREPARED),
            "--exclusions", str(WORK / "configs/dataset-exclusions.json")])
    report = json.loads((PREPARED / "audit/report.json").read_text())
    assert report["prepared_counts"]["train"]["images"] == 799
    assert report["prepared_counts"]["val"]["images"] == 200
    expected_counts = {"train": {"person": 1961, "helmet": 1281, "vest": 1476},
                       "val": {"person": 481, "helmet": 294, "vest": 340}}
    for split, counts in expected_counts.items():
        assert report["prepared_counts"][split]["instances"] == counts
    DATA_YAML = str(PREPARED / "dataset.yaml")
    print(json.dumps(report["prepared_counts"], indent=2))
    ''')
    code('''
    import shutil, platform
    import ultralytics
    import yaml
    assert yaml.safe_load(Path(DATA_YAML).read_text())["names"] == {0:"person",1:"helmet",2:"vest"}
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (RUN_DIR / "environment-colab.txt").write_text(environment)
    provenance = {"archive_sha256": EXPECTED_ARCHIVE_HASH, "bundle_sha256": sha256(BUNDLE),
        "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0), "classes": {0:"person",1:"helmet",2:"vest"},
        "evaluation": "Validation only; independent real-camera test data remains outstanding."}
    (RUN_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    shutil.copy2(PREPARED / "audit/report.json", RUN_DIR / "dataset-audit.json")
    shutil.copy2(PREPARED / "dataset.yaml", RUN_DIR / "dataset.yaml")
    shutil.copy2(WORK / "docs/dataset-attribution.md", RUN_DIR / "dataset-attribution.md")
    ''')
    markdown("## Short training check\nTwo epochs on 10% of training data verify the training path. This is not the baseline and its metrics are not an accuracy claim.")
    code('''
    from ultralytics import YOLO
    SMOKE = WORK / "smoke"
    if not (SMOKE / "weights/best.pt").is_file():
        smoke = YOLO(str(WORK / "models/yolo11n.pt"))
        smoke.train(data=DATA_YAML, epochs=2, fraction=0.1, imgsz=640, batch=8,
            device=0, workers=2, cache=False, seed=42, deterministic=True,
            project=str(WORK), name="smoke", exist_ok=True, plots=False,
            save=True, amp=True)
        del smoke
        torch.cuda.empty_cache()
    assert (SMOKE / "weights/best.pt").is_file()
    print("Short training check passed.")
    ''')
    markdown("## Baseline fine-tuning\n30 epochs on all training images, batch 8, seed 42. Best/last checkpoints and per-epoch backups are saved to Drive. If GPU memory runs out, reconnect and reduce BATCH to 4 before starting a fresh run name.")
    code('''
    EPOCHS = 30
    BATCH = 8
    LAST = RUN_DIR / "weights/last.pt"
    COMPLETE = RUN_DIR / "training-complete.json"
    if COMPLETE.is_file():
        print("This run is already complete; proceeding to validation.")
    elif LAST.is_file():
        model = YOLO(str(LAST))
        if model.ckpt.get("epoch", -1) == -1:
            print("Finalized checkpoint found; proceeding to validation.")
        else:
            model.train(resume=True, data=DATA_YAML, device=0)
        COMPLETE.write_text(json.dumps({"status":"training completed or finalized checkpoint found"}))
    else:
        if (RUN_DIR / "results.csv").exists():
            raise RuntimeError("Partial run without last.pt. Use a new RUN_NAME or restore a valid saved checkpoint.")
        model = YOLO(str(WORK / "models/yolo11n.pt"))
        model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=640, batch=BATCH,
            device=0, workers=2, cache=False, seed=42, deterministic=True,
            project=str(RUN_DIR.parent), name=RUN_NAME, exist_ok=True,
            save=True, save_period=1, plots=True, patience=10, amp=True)
        COMPLETE.write_text(json.dumps({"status":"training completed", "requested_epochs":EPOCHS}))
    assert (RUN_DIR / "weights/best.pt").is_file(), "No selected model found."
    ''')
    markdown("## Validate and inspect results\nThis uses the development validation split. It is not a held-out final test. Inspect per-class results, plots, and mistakes before deciding whether another experiment is justified.")
    code('''
    best = YOLO(str(RUN_DIR / "weights/best.pt"))
    metrics = best.val(data=DATA_YAML, split="val", imgsz=640, batch=BATCH,
        device=0, workers=2, plots=True, project=str(RUN_DIR), name="validation", exist_ok=True)
    scores = {key: float(value) for key, value in metrics.results_dict.items()}
    per_class = []
    for i, class_id in enumerate(metrics.box.ap_class_index):
        p, r = float(metrics.box.p[i]), float(metrics.box.r[i])
        per_class.append({"class": best.names[int(class_id)], "precision":p, "recall":r,
            "f1": 2*p*r/(p+r) if p+r else 0.0, "mAP50":float(metrics.box.ap50[i]),
            "mAP50_95":float(metrics.box.ap[i])})
    result = {"split":"validation", "aggregate":scores, "per_class":per_class,
        "model_timing_ms_per_image":metrics.speed,
        "limitation":"Model timing is not end-to-end video FPS. No independent final test set."}
    (RUN_DIR / "validation-metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    from IPython.display import display, Image
    for plot in (RUN_DIR / "results.png", RUN_DIR / "validation/confusion_matrix.png"):
        if plot.is_file():
            display(Image(filename=str(plot), width=850))
    ''')
    markdown("## Download results for the Windows project\nDownload this ZIP and keep the Drive copy. Put `weights/best.pt` into your local project's `models/best.pt`. Preserve the report and provenance files alongside the model.")
    code('''
    export = DRIVE_ROOT / (RUN_NAME + "-results.zip")
    with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(RUN_DIR.rglob("*")):
            if path.is_file() and not path.name.startswith("epoch"):
                handle.write(path, path.relative_to(RUN_DIR))
    print("Saved:", export)
    files.download(str(export))
    ''')
    markdown('''
    ## After downloading

    Place the selected weights at `D:\\ppe-compliance\\models\\best.pt`, then:

    ```powershell
    .\\.venv\\Scripts\\ppe.exe detect --source "D:\\path\\to\\photo.jpg" --weights models/best.pt --device cpu
    ```

    Save validation results to the project experiment log before choosing further
    changes. Do not claim PPE compliance yet: person/equipment association is Phase 4.

    References: [Ultralytics training](https://docs.ultralytics.com/modes/train/),
    [Colab FAQ](https://research.google.com/colaboratory/faq.html).
    ''')
    notebook = {"cells": cells, "metadata": {"colab": {"name":"PPE_Phase3_Training.ipynb"},
        "accelerator":"GPU", "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}}, "nbformat":4, "nbformat_minor":5}
    for index, cell in enumerate(cells):
        cell["id"] = f"ppe-cell-{index:02d}"
    target = ROOT / "notebooks/PPE_Phase3_Training.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil_target = OUTPUT / target.name
    shutil_target.write_bytes(target.read_bytes())
    print(f"Notebook: {target}")
    print(f"Bundle: {bundle} ({bundle.stat().st_size:,} bytes)")
    print(f"Bundle SHA256: {bundle_hash}")


if __name__ == "__main__":
    build()
