"""Compare CPU/GPU video inference and complete exports using identical inputs."""

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter

from ppe_compliance.detector import file_sha256
from ppe_compliance.video_pipeline import process_video


def summarize_timings(rows, warmup_frames=5):
    """Separate first-frame startup from steady samples; never invent zero latency."""
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise ValueError('warmup_frames must be a nonnegative integer')
    selected = rows[warmup_frames:]
    stages = ('decode', 'frame_setup', 'inference', 'assessment_and_hash', 'render', 'encode_submission')
    values = {stage: [row['timing_seconds'][stage] for row in selected] for stage in stages}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value < 0 for group in values.values() for value in group):
        raise ValueError('Frame timings must be finite and nonnegative')
    inference = values['inference']
    milliseconds = {key: mean(value)*1000 if value else None for key, value in values.items()}
    return {'warmup_frames_excluded': min(warmup_frames, len(rows)), 'steady_frames': len(selected),
            'mean_stage_ms': milliseconds,
            'median_inference_ms': median(inference)*1000 if inference else None,
            'inference_only_fps': 1/mean(inference) if inference and mean(inference) > 0 else None,
            'first_inference_ms': rows[0]['timing_seconds']['inference']*1000 if rows else None}


def verify_export(path):
    """Decode every output frame outside the timed processing interval."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    count, dimensions = 0, None
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open exported video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, image = capture.read()
            if not ok:
                break
            current = [image.shape[1], image.shape[0]]
            if dimensions is not None and current != dimensions:
                raise RuntimeError('Exported video dimensions changed')
            dimensions = current
            count += 1
    finally:
        capture.release()
    return {'decoded_frames': count, 'resolution': dimensions, 'playback_fps': fps}


def benchmark_video(source, weights, output, *, devices=('cpu', '0'), views=('frame', 'dashboard'),
                    image_size=416, confidence=.25, max_frames=300, repeats=2,
                    warmup_frames=5, browser_preview=True, on_progress=None, export_profiles=('standard',),
                    processing_height=None):
    """Run the same frame range/settings on each device; save partial evidence too.

    Each job loads a model as the application does. Steady inference statistics
    exclude warmup rows, while whole-job throughput retains startup and exports.
    Frame detection extraction transfers boxes to CPU, completing GPU work before
    the pipeline's inference timer stops. This is wall-clock detector-call time,
    not CUDA-kernel-only latency.
    """
    import torch
    from ppe_compliance.interface import prepare_browser_video

    source, weights, output = Path(source), Path(weights), Path(output)
    if not source.is_file() or not weights.is_file():
        raise ValueError('Provide an existing local video and trained weights file')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError('max_frames must be a positive integer')
    if type(image_size) is not int or image_size < 1 or not 0 < confidence <= 1:
        raise ValueError('Invalid model input size or confidence')
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise ValueError('warmup_frames must be a nonnegative integer')
    if not devices or len(set(devices)) != len(devices) or any(d not in {'cpu', '0'} for d in devices):
        raise ValueError('Select cpu and/or 0 once each')
    if '0' in devices and not torch.cuda.is_available():
        raise ValueError('CUDA GPU is unavailable; select a Colab GPU runtime or use devices=(cpu,)')
    if not views or len(set(views)) != len(views) or any(v not in {'frame', 'dashboard'} for v in views):
        raise ValueError('Select frame and/or dashboard once each')
    if not export_profiles or len(set(export_profiles)) != len(export_profiles) or any(
            p not in {'standard', 'efficient'} for p in export_profiles):
        raise ValueError('Select standard and/or efficient export profiles once each')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'source_sha256': file_sha256(source),
              'weights_sha256': file_sha256(weights), 'image_size': image_size, 'confidence': confidence,
              'max_frames': max_frames, 'repeats': repeats, 'warmup_frames': warmup_frames,
              'tracking': True, 'show_scores': True, 'equipment_detail': False,
              'browser_preview_enabled': browser_preview,
              'export_profiles': list(export_profiles),
              'processing_height_limit': processing_height,
              'pipeline_source_sha256': {name: file_sha256(Path(__file__).with_name(name))
                  for name in ('detector.py', 'association.py', 'compliance.py', 'video_pipeline.py',
                               'rendering.py', 'preview_ui.py', 'tracking.py', 'benchmark.py',
                               'video_export.py', 'interface.py')},
              'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              'torch_cpu_threads': torch.get_num_threads()},
              'runs': [],
              'scope': 'Uploaded-file performance check, not accuracy evaluation. No frame skipping. '
                       'Upload/download and readback verification excluded from job time. '
                       'Job time includes model startup, all pipeline exports and optional H264 conversion. '
                       'Inference-only FPS excludes decode, association, rendering and encoding. '
                       'A Colab CPU/GPU comparison does not predict laptop speed.'}

    def save():
        (output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        for repeat in range(1, repeats+1):
            # Reverse order on alternate repetitions to reduce systematic warm-cache bias.
            ordered_devices = devices if repeat % 2 else tuple(reversed(devices))
            for device in ordered_devices:
                profiles = export_profiles if repeat % 2 else tuple(reversed(export_profiles))
                for view, export_profile in product(views, profiles):
                    label = f'{"gpu" if device == "0" else "cpu"}-{view}-r{repeat}'
                    if export_profile != 'standard':
                        label += '-'+export_profile
                    if on_progress:
                        on_progress(label)
                    if device == '0':
                        torch.cuda.synchronize()
                    started = perf_counter()
                    summary = process_video(source, str(weights), confidence, image_size, device,
                                            output/label, max_frames=max_frames, tracking=True, view=view,
                                            export_profile=export_profile,
                                            processing_height=processing_height,
                                            on_progress=(lambda count, seconds: on_progress(
                                                f'{label}: {count} frames ({count/max(seconds, .001):.1f} FPS)'))
                                                if on_progress else None)
                    if summary['stop_reason'] == 'keyboard_interrupt':
                        raise KeyboardInterrupt
                    if (summary['source_sha256'] != report['source_sha256'] or
                            summary['weights_sha256'] != report['weights_sha256']):
                        raise RuntimeError('Video or weights changed between benchmark jobs')
                    run = Path(summary['run_directory'])
                    selected = run/('dashboard.mp4' if view == 'dashboard' else 'annotated.mp4')
                    conversion_seconds = 0.
                    if browser_preview:
                        before = perf_counter()
                        preview = prepare_browser_video(summary)
                        conversion_seconds = 0. if summary.get('browser_video_file') else perf_counter()-before
                    else:
                        preview = None
                    if device == '0':
                        torch.cuda.synchronize()
                    job_seconds = perf_counter()-started
                    rows = [json.loads(line) for line in (run/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
                    verification = verify_export(preview or selected)
                    if verification['decoded_frames'] != summary['processed_frames']:
                        raise RuntimeError('Export readback frame count differs from assessed frames')
                    source_fps = summary['source_fps_metadata']
                    entry = {'label': label, 'device': device, 'view': view, 'repeat': repeat,
                             'run_directory': str(run), 'frames': summary['processed_frames'],
                             'export_profile': export_profile,
                             'source_fps': source_fps, 'output_fps': summary['output_fps'],
                             'source_resolution': summary['source_resolution'],
                             'processing_resolution': summary.get('processing_resolution'),
                             'frame_hash_scope': summary.get('frame_hash_scope'),
                             'model_load_seconds': summary['model_load_seconds'],
                             'pipeline_fps': summary['end_to_end_processing_fps'],
                             'job_seconds': job_seconds, 'job_fps': len(rows)/job_seconds,
                             'source_seconds_processed': len(rows)/source_fps if source_fps else None,
                             'h264_conversion_seconds': conversion_seconds,
                             'preview_file': str(preview) if preview else None,
                             'decoded_frame_range_sha256': hashlib.sha256(''.join(
                                 row['assessment']['source_sha256'] for row in rows).encode('ascii')).hexdigest(),
                             'verification': verification, **summarize_timings(rows, warmup_frames)}
                    entry['detection_validation'] = summary.get('detection_validation')
                    entry['assessment_evidence_sha256'] = hashlib.sha256(json.dumps([
                        {'detections': row['detections'], 'workers': row['assessment']['workers']}
                        for row in rows], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if rows and 'assessment_timing_seconds' in rows[0]:
                        entry['mean_assessment_detail_ms'] = {
                            key: mean(row['assessment_timing_seconds'][key] for row in rows[warmup_frames:])*1000
                            if len(rows) > warmup_frames else None
                            for key in rows[0]['assessment_timing_seconds']}
                    report['runs'].append(entry)
                    save()
        if len({r['frames'] for r in report['runs']}) != 1:
            raise RuntimeError('Benchmark jobs processed different frame counts; comparison is invalid')
        if len({r['decoded_frame_range_sha256'] for r in report['runs']}) != 1:
            raise RuntimeError('Benchmark jobs decoded different source frames; comparison is invalid')
        # Export changes must not change detection or rule evidence on one device.
        report['evidence_matches_between_exports'] = all(len({
            r['assessment_evidence_sha256'] for r in report['runs'] if r['device'] == device}) == 1
            for device in devices)
        if not report['evidence_matches_between_exports']:
            raise RuntimeError('Detection/assessment evidence differed between export jobs on the same device')
        report['status'] = 'completed'
    except KeyboardInterrupt:
        report.update(status='interrupted', error='Stopped by the user; partial outputs are retained.')
        raise
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('models/best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', choices=('cpu', '0'), default=['cpu', '0'])
    parser.add_argument('--views', nargs='+', choices=('frame', 'dashboard'), default=['frame', 'dashboard'])
    parser.add_argument('--export-profiles', nargs='+', choices=('standard', 'efficient'), default=['standard'])
    parser.add_argument('--processing-height', type=int)
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup-frames', type=int, default=5)
    parser.add_argument('--no-browser-preview', action='store_true')
    args = parser.parse_args()
    try:
        report = benchmark_video(args.source, args.weights, args.output, devices=args.devices, views=args.views,
                                 image_size=args.image_size, max_frames=args.max_frames, repeats=args.repeats,
                                 warmup_frames=args.warmup_frames, browser_preview=not args.no_browser_preview,
                                 export_profiles=args.export_profiles,
                                 processing_height=args.processing_height,
                                 on_progress=lambda label: print('Running '+label, flush=True))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, str(error)+'\n')
    for run in report['runs']:
        print(f"{run['label']}: pipeline {run['pipeline_fps']:.1f} FPS; complete job {run['job_fps']:.1f} FPS")


if __name__ == '__main__':
    main()
