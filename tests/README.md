# Image and video pipeline tests

Run from the repository root after installing the application dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use standard-library `unittest`. Rule fixtures do not run YOLO or require
weights/network access. CLI integration tests use controlled detections and real
temporary image/JSON files. They verify logic and file behaviour, not model accuracy.

The full suite has 59 tests. Video checks use real temporary AVI/MP4 files and
fake predictions/camera backends to test streaming, rate/timing metadata, resource
cleanup and stop/failure paths without requiring a physical camera or GUI.
