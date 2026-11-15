"""Find already-uploaded Colab inputs without the notebook upload widget."""

from pathlib import Path
import zipfile

from ppe_compliance.detector import file_sha256

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}


def normalize_selected_path(selected):
    """Colab form fields may contain literal quotes pasted from Python examples."""
    if selected is None:
        return None
    value = str(selected).strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value or None


def resolve_weights(project, upload_root, expected_sha256, selected=None):
    """Reuse verified weights or read only the model from a companion bundle.

    Source code in an older bundle is deliberately not extracted. The notebook
    carries its own fixed source snapshot independently of the uploaded model.
    """
    project, upload_root = Path(project).resolve(), Path(upload_root).resolve()
    selected = normalize_selected_path(selected)
    destination = project/'models/best.pt'
    candidates = ([Path(selected)] if selected else
                  [destination, upload_root/'best.pt', upload_root/'ppe-video-project/models/best.pt'])
    mismatched = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = upload_root/candidate
        if candidate.is_file():
            if file_sha256(candidate) != expected_sha256:
                mismatched.append(str(candidate))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != destination.resolve():
                destination.write_bytes(candidate.read_bytes())
            return destination
    if selected:
        if mismatched:
            raise ValueError('WEIGHTS_PATH points to a different model (checksum mismatch): '+str(selected)+
                             '. Select the project best.pt; leave the field blank when using the model ZIP.')
        raise ValueError('WEIGHTS_PATH file does not exist: '+str(selected)+
                         '. Leave the form field blank for automatic model selection.')
    for archive_path in sorted(upload_root.glob('ppe-video-gpu-bundle*.zip')):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                member = archive.getinfo('models/best.pt')
                if member.file_size > 50*1024*1024:
                    continue
                data = archive.read(member)
            import hashlib
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                mismatched.append(str(archive_path))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return destination
        except (zipfile.BadZipFile, KeyError):
            continue
    detail = ' A different model was found: '+', '.join(mismatched) if mismatched else ''
    raise ValueError('Trained weights not found. Use Colab Files (folder icon) > Upload to add '
                     'ppe-video-gpu-bundle.zip or the project models/best.pt, then rerun this cell.'+detail)


def resolve_video(upload_root, selected=None):
    """Reuse one original video; require an explicit path when selection is ambiguous."""
    upload_root = Path(upload_root)
    selected = normalize_selected_path(selected)
    if selected:
        candidate = Path(selected)
        if not candidate.is_absolute():
            candidate = upload_root/candidate
        if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError('VIDEO_PATH must point to an uploaded MP4/MOV/AVI/MKV file')
        return candidate
    candidates = [p for p in upload_root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    cached = upload_root/'ppe-video-input'
    if cached.is_dir():
        candidates.extend(p for p in cached.glob('source.*') if p.suffix.lower() in VIDEO_EXTENSIONS)
    candidates = [p for p in candidates if p.stem.casefold() not in {'annotated', 'dashboard', 'preview'}]
    unique = {}
    for candidate in sorted(candidates):
        unique.setdefault(file_sha256(candidate), candidate)
    if not unique:
        raise ValueError('No original video found. Upload it through Colab Files (folder icon), then rerun this cell.')
    if len(unique) > 1:
        raise ValueError('Multiple videos found. Set VIDEO_PATH to the intended original file: '+
                         ', '.join(str(path) for path in unique.values()))
    return next(iter(unique.values()))


def inspect_video(path):
    """Decode one frame and validate metadata before the expensive model run."""
    import cv2
    import math

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('The video could not be opened. Try a readable MP4 export.')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError('The file contains no decodable video frame.')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('The video has no valid FPS metadata. Re-export it with a known constant frame rate.')
        return {'fps': fps, 'resolution': [frame.shape[1], frame.shape[0]]}
    finally:
        capture.release()
