"""Build a Colab notebook with verified source embedded and no upload widgets."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from textwrap import dedent
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_bytes(files):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            member = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, data)
    return buffer.getvalue()


def build(root=ROOT, destination=None):
    root = Path(root)
    destination = Path(destination) if destination is not None else root/'outputs/colab-video'
    members = sorted(p.relative_to(root).as_posix() for p in (root/'src/ppe_compliance').glob('*.py'))
    members += ['pyproject.toml', 'README.md', 'models/best.pt']
    missing = [name for name in members if not (root/name).is_file()]
    if missing or not (root/'src/ppe_compliance/benchmark.py').is_file():
        raise ValueError('Missing video bundle inputs: '+', '.join(missing or ['benchmark.py']))
    destination.mkdir(parents=True, exist_ok=True)
    content = {name: (root/name).read_bytes() for name in members}
    manifest = {'schema_version': 2, 'files': {name: digest(root/name) for name in members},
                'note': 'Current source and selected weights only; no video, credentials or datasets.'}
    bundle = destination/'ppe-video-gpu-bundle.zip'
    bundle.write_bytes(archive_bytes({**content, 'bundle-manifest.json': json.dumps(manifest, indent=2).encode()}))
    source_content = {name: data for name, data in content.items() if name != 'models/best.pt'}
    source_manifest = {name: hashlib.sha256(data).hexdigest() for name, data in source_content.items()}
    source_archive = archive_bytes({**source_content, 'source-manifest.json': json.dumps(source_manifest).encode()})
    source_hash = hashlib.sha256(source_archive).hexdigest()
    cells = []

    def add(kind, source, form=False):
        cell = {'cell_type': kind, 'metadata': {}, 'source': dedent(source).strip()+'\n'}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
            if form:
                cell['metadata']['cellView'] = 'form'
        cells.append(cell)

    add('markdown', '''
    # PPE video GPU test - rebuilt edition

    The invalid-box fix and current dashboard source are built into this notebook.
    **No repair script and no notebook file-upload widget are needed.**

    1. Select Runtime > Change runtime type > GPU (T4 if available).
    2. Open the **Files** panel (folder icon on the left) and upload
       `ppe-video-gpu-bundle.zip` **and your original video**. Wait for uploads to finish.
       You can use `models/best.pt` instead of the ZIP. Existing verified uploads
       in the same runtime are reused.
    3. Run cells from the top. A short GPU smoke test checks the model, malformed-box
       handling, dashboard export and browser encoding before the main run.
    4. Inspect the scored video and timing table, then download the results ZIP.

    The default runs **GPU first, dashboard only, single H.264 export**. Detection
    and evidence hashes use the selected processing resolution (1080p by default).
    Original-resolution analysis is available with PROCESSING_HEIGHT=0. CPU,
    native export and frame-only comparisons are optional.
    No retraining or frame skipping is used.
    This tests performance/integration, not independent accuracy. Colab GPU access
    varies; save/download outputs before the temporary runtime disconnects.
    ''')
    add('code', '''
    #@title 1. Select runtime and prepare dependencies
    import os, sys, subprocess, json, hashlib, platform
    from pathlib import Path
    import torch
    PREFERRED_DEVICE = 'auto' #@param ['auto', '0', 'cpu']
    INSTALL_DEPENDENCIES = True #@param {type:'boolean'}
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    RUNTIME_ROOT = Path('/content') if IN_COLAB else Path(os.environ.get('PPE_NOTEBOOK_ROOT', Path.cwd()/'outputs/notebook-runtime')).resolve()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if PREFERRED_DEVICE == '0' and not torch.cuda.is_available():
        raise RuntimeError('GPU explicitly selected but CUDA is unavailable. Select a GPU runtime or use auto/cpu diagnostic mode.')
    SELECTED_DEVICE = ('0' if torch.cuda.is_available() else 'cpu') if PREFERRED_DEVICE == 'auto' else PREFERRED_DEVICE
    DEVICE_LABEL = 'GPU' if SELECTED_DEVICE == '0' else 'CPU'
    os.environ['YOLO_CONFIG_DIR'] = str(RUNTIME_ROOT/'ppe-video-yolo-config')
    os.environ['MPLCONFIGDIR'] = str(RUNTIME_ROOT/'ppe-video-matplotlib')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    if INSTALL_DEPENDENCIES:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
            'ultralytics==8.4.212', 'opencv-python>=4.8,<5', 'Pillow>=10.1,<14',
            'imageio-ffmpeg>=0.5,<1', 'pandas>=2,<3'])
        # These wheels share the cv2 import/files. Keep one supported provider.
        from importlib.metadata import distributions
        conflicting = sorted({d.metadata['Name'] for d in distributions()
            if d.metadata['Name'] and d.metadata['Name'].lower() in {
                'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'}})
        if conflicting:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', *conflicting])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
            '--no-deps', 'opencv-python>=4.8,<5'])
    print('Device:', torch.cuda.get_device_name(0) if SELECTED_DEVICE == '0' else 'CPU', '| Torch:', torch.__version__, '| CUDA:', torch.version.cuda)
    if SELECTED_DEVICE == 'cpu':
        print('CPU diagnostic mode: this is not a GPU performance result. Processing may take longer.')
    WORK = RUNTIME_ROOT/'ppe-video-ready-project'
    UPLOADS = RUNTIME_ROOT
    def sha256(path):
        with Path(path).open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    ''', form=True)
    add('code', f'''
    #@title 2. Restore the fixed source (embedded; nothing to upload here)
    import base64, importlib, io, zipfile
    PREFLIGHT_PASSED = False
    SOURCE_PAYLOAD = {base64.b64encode(source_archive).decode('ascii')!r}
    SOURCE_SHA256 = {source_hash!r}
    code_archive = base64.b64decode(SOURCE_PAYLOAD)
    assert hashlib.sha256(code_archive).hexdigest() == SOURCE_SHA256, 'Embedded source checksum mismatch.'
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(code_archive)) as archive:
        for member in archive.infolist():
            assert (WORK/member.filename).resolve().is_relative_to(WORK.resolve()), 'Unsafe source path'
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Unsupported symlink'
        archive.extractall(WORK)
    for name, expected in json.loads((WORK/'source-manifest.json').read_text()).items():
        assert sha256(WORK/name) == expected, name
    sys.path.insert(0, str(WORK/'src'))
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == 'ppe_compliance' or name.startswith('ppe_compliance.'):
            del sys.modules[name]
    import ppe_compliance
    assert Path(ppe_compliance.__file__).resolve().is_relative_to(WORK.resolve()), 'Wrong source loaded.'
    from ppe_compliance.detector import validated_model_detections
    from ppe_compliance.colab_setup import normalize_selected_path
    assert normalize_selected_path('""') is None, 'Quoted empty-path self-check failed.'
    accepted, diagnostics = validated_model_detections([
        {{'class_id': 0, 'class_name': 'person', 'confidence': .9, 'bbox_xyxy': [1, 1, 10, 20]}},
        {{'class_id': 1, 'class_name': 'helmet', 'confidence': .9, 'bbox_xyxy': [10, 2, 10, 8]}}
    ], 100, 100)
    assert len(accepted) == 1 and diagnostics['rejected_count'] == 1, 'Invalid-box self-check failed.'
    print('Fixed source loaded. Invalid-box self-check passed.')
    ''', form=True)
    add('markdown', '''
    ## 3. Select the uploaded inputs

    Use the Files panel to upload the model ZIP (or best.pt) and original video.
    This cell reuses them and does not open an upload dialog. If several different
    videos are present, use Files > right-click the intended file > Copy path and
    paste that into VIDEO_PATH. Leave both form fields blank for automatic selection;
    do not type quotation marks to represent an empty field. Quoted pasted paths
    are normalized automatically.
    An older model bundle is usable when its best.pt matches: its old code is ignored.
    ''')
    add('code', f'''
    #@title 3. Inputs and profile
    WEIGHTS_PATH = '' #@param {{type:'string'}}
    VIDEO_PATH = '' #@param {{type:'string'}}
    IMAGE_SIZE = 416 #@param [320, 416, 640]
    PROCESSING_HEIGHT = 1080 #@param [0, 720, 1080]
    PREFLIGHT_PASSED = False
    from datetime import datetime, timezone
    from uuid import uuid4
    from ppe_compliance.colab_setup import resolve_weights, resolve_video, inspect_video, normalize_selected_path
    print('Model selection:', normalize_selected_path(WEIGHTS_PATH) or 'automatic')
    print('Video selection:', normalize_selected_path(VIDEO_PATH) or 'automatic')
    WEIGHTS = resolve_weights(WORK, UPLOADS, {digest(root/'models/best.pt')!r}, WEIGHTS_PATH or None)
    CLIP = resolve_video(UPLOADS, VIDEO_PATH or None)
    video_info = inspect_video(CLIP)
    SESSION_RESULTS = RUNTIME_ROOT/'ppe-video-ready-results'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    SESSION_RESULTS.mkdir(parents=True, exist_ok=False)
    PREFLIGHT_PASSED = False
    print('Video:', CLIP.name, '| Source:', video_info['resolution'], '| FPS:', round(video_info['fps'], 3))
    print('Processing height cap:', PROCESSING_HEIGHT or 'original', '| Lower resolution may miss small PPE.')
    print('Weights verified:', sha256(WEIGHTS))
    print('Results folder:', SESSION_RESULTS)
    ''', form=True)
    add('code', '''
    #@title 4. Short smoke test before the long run
    from ppe_compliance.video_pipeline import process_video
    from ppe_compliance.interface import prepare_browser_video
    from ppe_compliance.benchmark import verify_export
    from IPython.display import display, Video
    PREFLIGHT_PASSED = False
    smoke = process_video(CLIP, str(WEIGHTS), image_size=IMAGE_SIZE, device=SELECTED_DEVICE,
        output_dir=SESSION_RESULTS/'smoke', max_frames=8, tracking=True, view='dashboard', export_profile='efficient',
        processing_height=PROCESSING_HEIGHT or None)
    smoke_preview = prepare_browser_video(smoke)
    readback = verify_export(smoke_preview)
    assert readback['decoded_frames'] == smoke['processed_frames'] > 0, 'Smoke export readback failed.'
    PREFLIGHT_SIGNATURE = (sha256(CLIP), sha256(WEIGHTS), IMAGE_SIZE, SOURCE_SHA256, SELECTED_DEVICE, PROCESSING_HEIGHT)
    PREFLIGHT_PASSED = True
    print(DEVICE_LABEL+' smoke test PASSED:', readback['decoded_frames'], 'frames; valid dashboard and H.264 preview.')
    print('Invalid detections ignored and audited:', smoke['detection_validation']['rejected_count'])
    display(Video(filename=str(smoke_preview), embed=True, width=1000))
    ''', form=True)
    add('markdown', '''
    ## 5. Main GPU run or optional CPU comparison

    Default: first 300 frames on GPU with an efficient dashboard export, one repetition. Set
    COMPARE_CPU=True for the same clip on Colab CPU too; this can take several
    minutes. Set COMPARE_VIEWS=True to include the native frame-only export.
    For a more stable comparison choose REPEATS=2. The first five frames are
    excluded only from steady timing averages, never from video or whole-job time.
    The smoke test has already initialized CUDA: these are warm-runtime tests.
    Set COMPARE_EXPORTS=True to compare native drawing + MP4V + H.264 conversion
    against display-sized drawing + one direct H.264 file on the same device.
    Both profiles analyze the same source frame range at the selected processing resolution;
    the report checks their detection and assessment evidence for equality.
    ''')
    add('code', '''
    #@title 5. Run the selected tests
    COMPARE_CPU = False #@param {type:'boolean'}
    COMPARE_VIEWS = False #@param {type:'boolean'}
    COMPARE_EXPORTS = False #@param {type:'boolean'}
    MAX_FRAMES = 300 #@param {type:'integer'}
    REPEATS = 1 #@param [1, 2, 3]
    MAKE_BROWSER_PREVIEW = True #@param {type:'boolean'}
    assert globals().get('PREFLIGHT_PASSED', False), 'Run the smoke test first.'
    assert PREFLIGHT_SIGNATURE == (sha256(CLIP), sha256(WEIGHTS), IMAGE_SIZE, SOURCE_SHA256, SELECTED_DEVICE, PROCESSING_HEIGHT), 'Inputs/source/device changed: rerun the smoke test.'
    from ppe_compliance.benchmark import benchmark_video
    DEVICES = ('0', 'cpu') if COMPARE_CPU and SELECTED_DEVICE == '0' else (SELECTED_DEVICE,)
    if COMPARE_CPU and SELECTED_DEVICE != '0':
        print('CPU diagnostic run only: a GPU comparison requires a CUDA runtime.')
    VIEWS = ('frame', 'dashboard') if COMPARE_VIEWS else ('dashboard',)
    EXPORT_PROFILES = ('standard', 'efficient') if COMPARE_EXPORTS else ('efficient',)
    OUTPUT = SESSION_RESULTS/('benchmark-'+uuid4().hex[:8])
    try:
        report = benchmark_video(CLIP, WEIGHTS, OUTPUT, devices=DEVICES, views=VIEWS,
            image_size=IMAGE_SIZE, max_frames=MAX_FRAMES, repeats=REPEATS,
            warmup_frames=5, browser_preview=MAKE_BROWSER_PREVIEW,
            export_profiles=EXPORT_PROFILES,
            processing_height=PROCESSING_HEIGHT or None,
            on_progress=lambda message: print(message, flush=True))
    finally:
        (SESSION_RESULTS/'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        (SESSION_RESULTS/'source-sha256.txt').write_text(SOURCE_SHA256)
        (SESSION_RESULTS/'smoke-summary.json').write_text(json.dumps(smoke, indent=2))
    print('Main run complete. Continue to the results cell.')
    ''', form=True)
    add('markdown', '''
    ## 6. Results and scored playback

    Inference-only FPS excludes decoding/rules/rendering/encoding. Pipeline FPS
    includes the actual video loop and MP4 finalization. Complete job FPS includes
    model loading and H.264 conversion too. Upload/download and readback checking
    are excluded. Playback FPS remains the source rate, not AI speed.

    GPU inference can be fast while dashboard drawing and encoding remain CPU-bound.
    Colab's CPU differs from your laptop; do not report Colab CPU/GPU ratios as
    laptop speedups. Invalid model boxes are audited rather than aborting the video.
    ''')
    add('code', '''
    #@title 6. Show results
    import pandas as pd
    report = json.loads((OUTPUT/'benchmark.json').read_text())
    print('Benchmark status:', report['status'])
    if report.get('error'):
        print('Failure details:', report['error'], '| Partial evidence:', OUTPUT)
    table = pd.DataFrame([{
        'Run': r['label'], 'Frames': r['frames'], 'Model load s': r['model_load_seconds'],
        'Steady inference FPS': r['inference_only_fps'], 'Pipeline FPS': r['pipeline_fps'],
        'Complete job FPS': r['job_fps'], 'Playback FPS': r['output_fps'],
        'H264 seconds': r['h264_conversion_seconds'],
        'Rejected detections': (r.get('detection_validation') or {}).get('rejected_count', 0)
    } for r in report['runs']])
    display(table.round(2))
    table.to_csv(SESSION_RESULTS/'comparison.csv', index=False)
    stages = pd.DataFrame([{'Run': r['label'], **r['mean_stage_ms']} for r in report['runs']])
    print('Warm stage times: milliseconds/frame')
    display(stages.round(2))
    print('Same-device detection/assessment evidence matches:', report.get('evidence_matches_between_exports'))
    print('H264 seconds is the extra conversion only; direct encoding is included in pipeline time.')
    stages.to_csv(SESSION_RESULTS/'stage-times-ms.csv', index=False)
    for run in report['runs']:
        if run['device'] == SELECTED_DEVICE and run['view'] == 'dashboard' and run['preview_file']:
            display(Video(filename=run['preview_file'], embed=True, width=1000))
            break
    ''', form=True)
    add('code', '''
    #@title 7. Optional full video (no frame cap)
    RUN_FULL_VIDEO = False #@param {type:'boolean'}
    if RUN_FULL_VIDEO:
        assert globals().get('PREFLIGHT_PASSED', False) and PREFLIGHT_SIGNATURE == (sha256(CLIP), sha256(WEIGHTS), IMAGE_SIZE, SOURCE_SHA256, SELECTED_DEVICE, PROCESSING_HEIGHT), 'Rerun smoke test first.'
        full = process_video(CLIP, str(WEIGHTS), image_size=IMAGE_SIZE, device=SELECTED_DEVICE,
            output_dir=SESSION_RESULTS/'full-video', tracking=True, view='dashboard', export_profile='efficient',
            processing_height=PROCESSING_HEIGHT or None,
            on_progress=lambda count, seconds: print(f'{count} frames | {count/max(seconds,.001):.1f} FPS', flush=True))
        full_preview = prepare_browser_video(full)
        display(Video(filename=str(full_preview), embed=True, width=1000))
    else:
        print('Full video skipped. Set RUN_FULL_VIDEO=True if wanted.')
    ''', form=True)
    add('code', '''
    #@title 8. Download results (also works for partial benchmark evidence)
    import shutil
    assert 'SESSION_RESULTS' in globals() and SESSION_RESULTS.is_dir(), 'Run the input/smoke cells first.'
    result_zip = shutil.make_archive(str(RUNTIME_ROOT/('ppe-video-gpu-results-'+SESSION_RESULTS.name)), 'zip', root_dir=SESSION_RESULTS)
    print('Download:', result_zip)
    if IN_COLAB:
        from google.colab import files
        files.download(result_zip)
    else:
        from IPython.display import FileLink
        display(FileLink(result_zip))
    ''', form=True)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'accelerator': 'GPU', 'colab': {'name': 'PPE_Video_GPU_Ready.ipynb'},
                     'ppe_source_sha256': source_hash, 'ppe_model_bundle_sha256': digest(bundle),
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': cells}
    for index, cell in enumerate(cells):
        cell['id'] = f'ppe-ready-{index:02d}'
    notebook_dir = root/'notebooks';notebook_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(notebook, indent=2, ensure_ascii=False)+'\n'
    notebook_path = notebook_dir/'PPE_Video_GPU_Ready.ipynb'
    notebook_path.write_text(serialized, encoding='utf-8')
    # Old links also receive the fixed workflow; new runs should use the Ready filename.
    (notebook_dir/'PPE_Video_GPU_Benchmark.ipynb').write_text(serialized, encoding='utf-8')
    return notebook_path, bundle


def build_repair(root=ROOT, destination=None):
    """Make a small source-only repair for an already running Colab session."""
    root = Path(root)
    destination = Path(destination) if destination is not None else root/'outputs/colab-video'
    members = sorted(path.name for path in (root/'src/ppe_compliance').glob('*.py'))
    payloads = {'src/ppe_compliance/'+name: (root/'src/ppe_compliance'/name).read_bytes().decode('utf-8')
                for name in members}
    script = '# Generated source-only repair. No downloads, weights replacement or training.\n'
    script += '_FILES = '+json.dumps(payloads, ensure_ascii=True, indent=2)+'\n'
    script += dedent('''

    def apply_fix(project='/content/ppe-video-project'):
        import hashlib
        import importlib
        import json
        from pathlib import Path
        import sys

        project = Path(project).resolve()
        if not (project/'pyproject.toml').is_file() or not (project/'src/ppe_compliance/detector.py').is_file():
            raise RuntimeError('Existing PPE project not found. Run the notebook setup first.')
        for relative, source in _FILES.items():
            target = (project/relative).resolve()
            if not target.is_relative_to(project/'src/ppe_compliance'):
                raise RuntimeError('Repair path is outside the PPE package')
        checksums = {}
        for relative, source in _FILES.items():
            data = source.encode('utf-8')
            (project/relative).write_bytes(data)
            checksums[relative] = hashlib.sha256(data).hexdigest()
        (project/'colab-box-fix.json').write_text(json.dumps({
            'fix': 'Reject and audit invalid model predictions before strict association',
            'source_sha256': checksums}, indent=2))
        importlib.invalidate_caches()
        for name, module in list(sys.modules.items()):
            location = getattr(module, '__file__', None)
            if ((name == 'ppe_compliance' or name.startswith('ppe_compliance.')) and location
                    and Path(location).resolve().is_relative_to(project)):
                del sys.modules[name]
        if str(project/'src') not in sys.path:
            sys.path.insert(0, str(project/'src'))
        print('Box validation fix applied. Your video, weights and GPU session were preserved.')
        print('Rerun the benchmark settings/run cell; it imports the fixed code and creates a new output directory.')

    if __name__ == '__main__':
        apply_fix()
    ''')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/'ppe_colab_box_fix.py'
    path.write_text(script, encoding='utf-8')
    return path


if __name__ == '__main__':
    for path in build():
        print(path)
    print(build_repair())
