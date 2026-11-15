"""Local UI services sharing the same detector, rules and report formats as CLI."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import uuid4


PROFILES = {"Balanced": 416, "Detail": 640, "Fast": 320}


def decode_image_bytes(data):
    """Validate uploads before preview libraries can try optional format plugins."""
    import cv2
    import numpy as np

    message = 'This file could not be decoded as an image. Try JPEG or PNG.'
    if not data:
        raise ValueError(message)
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
    from ppe_compliance.compliance import assess_detections, visibility_template
    from ppe_compliance.rendering import draw_assessment

    frame = decode_image_bytes(data)
    detector = FrameDetector(weights, confidence, image_size, device)
    _, raw = detector.detect(frame, "uploaded-image")
    assessment = assess_detections(raw, hashlib.sha256(data).hexdigest())
    annotated = draw_assessment(frame, assessment, raw['detections'], detail=detail, show_scores=show_scores)
    run = Path(output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(run/'annotated.png'), annotated):
        raise OSError("Could not save the annotated image")
    for filename, value in (("detections.json", raw), ("compliance.json", assessment),
                            ("visibility-template.json", visibility_template(assessment['workers'], assessment['source_sha256']))):
        (run/filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    return {"kind": "image", "run_directory": str(run), "assessment": assessment}


def browser_video(source):
    """Make a high-quality H.264 derivative without changing source pacing/resolution.

    OpenCV's mp4v export is retained for download. CRF 18 avoids aggressive web
    compression; no frame interpolation or invented detection frames are added.
    """
    import imageio_ffmpeg

    source = Path(source)
    destination = source.with_name('preview.mp4')
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-i', str(source),
                             '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)],
                            capture_output=True, text=True, timeout=600,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('Browser video conversion failed; the original MP4 is still available for download.')
    return destination


def prepare_browser_video(summary):
    """Reuse a checked direct H.264 export; convert legacy MP4V only once."""
    run = Path(summary['run_directory'])
    if summary.get('browser_video_file'):
        return run/summary['browser_video_file']
    return browser_video(run/('dashboard.mp4' if summary.get('presentation_layout') == 'dashboard'
                             else 'annotated.mp4'))


def camera_command(root, python, camera=0, backend='auto', profile='Balanced',
                   width=1280, height=720, fps=30, device='cpu', record=False):
    """Argument list, safe for Popen without shell interpolation."""
    command = [str(python), '-m', 'ppe_compliance.cli', 'webcam', '--camera', str(camera),
               '--camera-backend', backend, '--low-latency', '--track',
               '--weights', str(Path(root)/'models/best.pt'), '--image-size', str(PROFILES[profile]),
               '--capture-width', str(width), '--capture-height', str(height),
               '--capture-fps', str(fps), '--device', device, '--output-dir', str(Path(root)/'outputs')]
    if record:
        command.append('--record')
    return command


def recorded_review_source(data, suffix, output_dir):
    """Keep uploaded video alive for the independent desktop review process."""
    if suffix.lower() not in {'.mp4','.mov','.avi','.mkv'} or not data:
        raise ValueError('Choose a supported nonempty video file')
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
