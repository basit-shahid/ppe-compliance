"""Local UI services sharing the same detector, rules and report formats as CLI."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4


PROFILES = {"Balanced": 416, "Detail": 640, "Fast": 320}
MAX_IMAGE_PIXELS = 24_000_000
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def decode_image_bytes(data):
    """Validate uploads before preview libraries can try optional format plugins."""
    import cv2
    import numpy as np

    message = 'This file could not be decoded as an image. Try JPEG or PNG.'
    if not data:
        raise ValueError(message)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Image exceeds the 200 MB upload limit.')
    # Read the header before allocating a decoded OpenCV image. Compressed file
    # size alone does not bound memory use (e.g. a large, single-colour PNG).
    from PIL import BmpImagePlugin, JpegImagePlugin, PngImagePlugin, WebPImagePlugin
    # Use explicit readers: Ultralytics patches PIL.Image.open to try installing
    # a HEIF plugin when ANY corrupt file is unidentified. Upload validation must
    # remain offline and must never install packages in response to input data.
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        reader = PngImagePlugin.PngImageFile
    elif data.startswith(b'\xff\xd8\xff'):
        reader = JpegImagePlugin.JpegImageFile
    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        reader = WebPImagePlugin.WebPImageFile
    elif data.startswith(b'BM'):
        reader = BmpImagePlugin.BmpImageFile
    else:
        raise ValueError(message)
    try:
        with reader(BytesIO(data)) as image:
            width, height = image.size
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(message) from error
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError('Image exceeds 24 megapixels. Resize it before uploading.')
    try:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(message) from error
    if frame is None:
        raise ValueError(message)
    return frame


def assess_image_bytes(data, weights, image_size=640, confidence=.25, device="cpu",
                       output_dir=Path("outputs"), detail=False, show_scores=True):
    import cv2
    from ppe_compliance.detector import FrameDetector
    from ppe_compliance.detector import validate_inference_options
    from ppe_compliance.compliance import assess_detections, visibility_template
    from ppe_compliance.rendering import draw_assessment

    validate_inference_options(confidence, image_size)
    frame = decode_image_bytes(data)
    detector = FrameDetector(weights, confidence, image_size, device)
    _, raw = detector.detect(frame, "uploaded-image")
    assessment = assess_detections(raw, hashlib.sha256(data).hexdigest())
    annotated = draw_assessment(frame, assessment, raw['detections'], detail=detail, show_scores=show_scores)
    run = Path(output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    run.parent.mkdir(parents=True, exist_ok=True)
    # Publish the run only after all four artifacts succeed. An interrupted or
    # full-disk export cannot appear as a completed result in the workspace.
    with tempfile.TemporaryDirectory(prefix='.ppe-image-', dir=run.parent) as staging:
        stage = Path(staging)
        if not cv2.imwrite(str(stage/'annotated.png'), annotated):
            raise OSError("Could not save the annotated image; check disk space and output permissions")
        for filename, value in (("detections.json", raw), ("compliance.json", assessment),
                                ("visibility-template.json", visibility_template(assessment['workers'], assessment['source_sha256']))):
            (stage/filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        stage.replace(run)
    return {"kind": "image", "run_directory": str(run), "assessment": assessment}


def browser_video(source):
    """Make a high-quality H.264 derivative without changing source pacing/resolution.

    OpenCV's mp4v export is retained for download. CRF 18 avoids aggressive web
    compression; no frame interpolation or invented detection frames are added.
    """
    import imageio_ffmpeg

    source = Path(source)
    if not source.is_file():
        raise OSError('The exported video is missing. Analyze the source again.')
    destination = source.with_name('preview.mp4')
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-i', str(source),
                             '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                             '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)],
                            capture_output=True, text=True, timeout=600,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError('Browser video conversion failed; the original MP4 is still available for download.')
    return destination


def prepare_browser_video(summary):
    """Reuse a checked direct H.264 export; convert legacy MP4V only once."""
    run = Path(summary['run_directory'])
    if summary.get('browser_video_file'):
        path = run/summary['browser_video_file']
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError('The browser video is missing or empty. Analyze the source again; inspect the encoder log if it repeats.')
        return path
    return browser_video(run/('dashboard.mp4' if summary.get('presentation_layout') == 'dashboard'
                             else 'annotated.mp4'))


def camera_command(root, python, camera=0, backend='auto', profile='Balanced',
                   width=1280, height=720, fps=30, device='cpu', record=False, confidence=.25):
    """Argument list, safe for Popen without shell interpolation."""
    command = [str(python), '-m', 'ppe_compliance.cli', 'webcam', '--camera', str(camera),
               '--camera-backend', backend, '--low-latency', '--track',
               '--weights', str(Path(root)/'models/best.pt'), '--image-size', str(PROFILES[profile]),
               '--capture-width', str(width), '--capture-height', str(height),
               '--capture-fps', str(fps), '--device', device, '--output-dir', str(Path(root)/'outputs')]
    command += ['--confidence', str(confidence)]
    if record:
        command.append('--record')
    return command


def recorded_review_source(data, suffix, output_dir):
    """Keep uploaded video alive for the independent desktop review process."""
    if suffix.lower() not in {'.mp4','.mov','.avi','.mkv'} or not data:
        raise ValueError('Choose a supported nonempty video file')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Video exceeds the 200 MB upload limit.')
    folder = Path(output_dir)/'review-inputs'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/(hashlib.sha256(data).hexdigest()+suffix.lower())
    if not target.is_file():
        temporary = folder/(uuid4().hex+'.tmp')
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def recorded_review_command(root, python, source, profile='Balanced', confidence=.25, device='cpu',
                            processing_height=540, max_frames=0, detail=False, show_scores=True):
    command = [str(python), '-m', 'ppe_compliance.cli', 'video', '--source', str(source),
        '--live-preview', '--track', '--weights', str(Path(root)/'models/best.pt'),
        '--image-size', str(PROFILES[profile]), '--confidence', str(confidence), '--device', device,
        '--processing-height', str(processing_height or 540), '--output-dir', str(Path(root)/'outputs')]
    if max_frames: command += ['--max-frames', str(max_frames)]
    if detail: command.append('--detail')
    if not show_scores: command.append('--hide-scores')
    return command
