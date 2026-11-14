# Phase 1: foundation complete

Verified on 2026-11-14 in `D:\ppe-compliance` on Windows 10.

## Completed work

1. Read the supplied proposal and reviewed the directory and implementation.
2. Recorded project phases and identified the unimplemented modules.
3. Installed the project in editable mode and its required dependencies in `.venv`.
4. Downloaded a public setup sample and general YOLO11 nano weights.
5. Verified CLI help, version `0.1.0`, and `pip check` (no broken requirements).
6. Ran CPU image inference and inspected its annotated JPEG and JSON.
7. Checked invalid inputs and saved the resolved environment.

## Environment

- Python: 3.13.3
- Ultralytics: 8.4.212
- PyTorch: 2.15.0+cpu
- CUDA available to this environment: no
- Full installed package snapshot: `docs/environment-windows.txt`.

The snapshot includes editable local paths and the PDF review tool `pypdf`;
it records this environment rather than providing a portable lockfile.
Optional Streamlit packages are deferred until the interface phase.

## Baseline result

Command, from the repository root in PowerShell:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
$env:MPLCONFIGDIR = "$PWD\.ultralytics\matplotlib"
.\.venv\Scripts\ppe.exe detect --source data/samples/bus.jpg --weights models/yolo11n.pt --device cpu
```

This run produced **four person detections and one bus detection** on an
810-by-1080-pixel image, using confidence 0.25 and inference image size 640.

- Image: `outputs/20261114T080350Z-43abd8c6/annotated.jpg`
- JSON: `outputs/20261114T080350Z-43abd8c6/detections.json`
- Validation summary: `outputs/phase1-checks/summary.json`
- Error-check evidence: `outputs/phase1-checks/negative-checks.json`

The saved image decodes, retains the source dimensions, and visibly contains
the predicted boxes. All JSON coordinates are finite, correctly ordered, and
inside the original image; confidence scores are within [0, 1].

## Error checks

| Input | Result |
| --- | --- |
| Missing source image | Exit 2; clear source-file error |
| Confidence 0 | Exit 2; rejected argument |
| Image size 0 | Exit 2; rejected argument |
| Corrupt JPEG | Exit 1; clear image-decoding error |

## Download origins and scope

- Sample: https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
- Weights: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

These artifacts and generated results remain local and are excluded from Git.
General object detection confirms that the pipeline works; it provides no PPE
accuracy or compliance result. No PPE training, video test, or FPS benchmark was
performed in Phase 1. Next: audit one PPE dataset in Phase 2.
