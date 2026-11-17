"""Rehearse real CPU inference and artifact integrity without network or a camera."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def verify(output, skip_tests=False):
    import cv2
    import torch
    from ppe_compliance.detector import file_sha256
    from ppe_compliance.interface import assess_image_bytes, decode_image_bytes, prepare_browser_video
    from ppe_compliance.video_pipeline import process_video

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'platform': platform.platform(), 'torch': torch.__version__, 'device': 'cpu',
                'scope': 'Integration and software correctness, not independent accuracy or sustained live performance.',
                'checks': {}, 'status': 'running'}
    started = perf_counter()
    try:
        dependencies = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, capture_output=True, text=True)
        evidence['checks']['dependencies'] = {'passed': dependencies.returncode == 0, 'output': dependencies.stdout.strip()}
        if dependencies.returncode:
            raise RuntimeError('Dependency check failed: '+dependencies.stdout+dependencies.stderr)
        if not skip_tests:
            tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                   cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (output/'automated-tests.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
            evidence['checks']['automated_tests'] = {'passed': tests.returncode == 0,
                'summary': tests.stderr.strip().splitlines()[-4:]}
            if tests.returncode:
                raise RuntimeError('Automated tests failed; inspect automated-tests.log')
        weights = ROOT/'models/best.pt'
        if not weights.is_file():
            raise ValueError('Add the trained model at models/best.pt; see docs/phase-3-colab.md')
        evidence['weights_sha256'] = file_sha256(weights)
        source = ROOT/'app/assets/presentation-sample.jpg'
        data = source.read_bytes()
        result = assess_image_bytes(data, str(weights), image_size=416, output_dir=output/'image')
        run = Path(result['run_directory'])
        decoded = decode_image_bytes(data)
        annotated = cv2.imread(str(run/'annotated.png'))
        if annotated is None or annotated.shape != decoded.shape:
            raise RuntimeError('Image export changed source dimensions or cannot decode')
        assessment = json.loads((run/'compliance.json').read_text(encoding='utf-8'))
        if not assessment['workers']:
            raise RuntimeError('The included worker sample produced no worker evidence')
        for filename in ['detections.json', 'compliance.json', 'visibility-template.json']:
            json.loads((run/filename).read_text(encoding='utf-8'))
        evidence['checks']['image'] = {'passed': True, 'source_sha256': file_sha256(source),
            'resolution': [decoded.shape[1], decoded.shape[0]], 'summary': assessment['summary'],
            'artifacts': sorted(p.name for p in run.iterdir())}
        # A deliberately synthetic 8-frame still sequence tests plumbing, timing
        # and encoding. It does not measure natural-motion detection accuracy.
        video = output/'rehearsal-input.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 8, (640, 360))
        if not writer.isOpened():
            raise RuntimeError('Cannot create the rehearsal video')
        try:
            for _ in range(8):
                writer.write(cv2.resize(decoded, (640, 360)))
        finally:
            writer.release()
        summary = process_video(video, str(weights), image_size=416, output_dir=output/'video',
                                tracking=True, view='dashboard', export_profile='efficient')
        browser = prepare_browser_video(summary)
        capture = cv2.VideoCapture(str(browser))
        frames = 0
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (720, 1280):
                    raise RuntimeError('Unexpected dashboard dimensions')
                frames += 1
        finally:
            capture.release()
        rows = [json.loads(line) for line in (Path(summary['run_directory'])/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        if frames != 8 or len(rows) != 8 or abs(fps-8) > .001:
            raise RuntimeError('Video frames, report rows or playback rate changed')
        if [r['source_time_seconds'] for r in rows] != [i/8 for i in range(8)]:
            raise RuntimeError('Source frame timestamps changed')
        evidence['checks']['video'] = {'passed': True, 'fixture': '8-frame repeated validation still; synthetic motion',
            'decoded_frames': frames, 'jsonl_rows': len(rows), 'playback_fps': fps,
            'resolution': [1280, 720], 'status': summary['status'], 'stop_reason': summary['stop_reason'],
            'processing_fps': summary['end_to_end_processing_fps'], 'total_wall_seconds': summary['total_wall_seconds'],
            'video_sha256': file_sha256(browser)}
        for bad in [b'', b'corrupt', b'\xff\xd8\xffbroken']:
            try:
                decode_image_bytes(bad)
            except ValueError:
                continue
            raise RuntimeError('Invalid image unexpectedly accepted')
        evidence['checks']['invalid_images'] = {'passed': True, 'rejected_cases': 3}
        evidence['status'] = 'passed'
    except Exception as error:
        evidence.update(status='failed', error=str(error))
        raise
    finally:
        evidence['wall_seconds'] = perf_counter()-started
        (output/'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(f"{evidence['status'].upper()}: {output/'verification.json'}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New, nonexisting evidence directory')
    parser.add_argument('--skip-tests', action='store_true', help='Only rerun dependency and real-model smoke checks')
    args = parser.parse_args()
    output = args.output or ROOT/'outputs'/('presentation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6])
    try:
        verify(output, args.skip_tests)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Rehearsal failed: {error}\n')


if __name__ == '__main__':
    main()
