"""Run locally: python -m streamlit run app/streamlit_app.py."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import streamlit as st

from ppe_compliance.interface import PROFILES, assess_image_bytes, prepare_browser_video, camera_command, decode_image_bytes
from ppe_compliance.video_pipeline import process_video
from ppe_compliance.playback import playback_monitor
from ppe_compliance.interface import recorded_review_source, recorded_review_command
from ppe_compliance.workspace import empty_stage, project_guide, validation_view, worker_evidence
from ppe_compliance.browser_preview import start_browser_preview, render_browser_preview

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/best.pt'
OUTPUTS = ROOT / 'outputs'

st.set_page_config(page_title='PPE Vision · Inspection workspace', page_icon='🦺', layout='wide', initial_sidebar_state='expanded')
st.markdown('<style>' + (ROOT/'app/workspace.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark">V</div><div>
        <div class="brand-name">PPE VISION</div><div class="brand-sub">WORKSITE INTELLIGENCE</div></div></div>''', unsafe_allow_html=True)
    st.caption('01 / INSPECTION SOURCE')
    mode = st.radio('Source', ['Image', 'Video', 'Live camera'], label_visibility='collapsed')
    st.divider()
    profile = st.selectbox('Processing profile', list(PROFILES), index=0)
    st.caption({'Balanced': 'Everyday use · moderate detail and speed.',
                'Detail': 'Larger model input · better small-item detail, slower on CPU.',
                'Fast': 'Responsive on CPU · small equipment may be missed.'}[profile])
    with st.expander('Advanced settings'):
        confidence = st.slider('Detection confidence', .05, .95, .25, .05)
        device = st.selectbox('Compute device', ['cpu', '0', 'mps'], help='CPU works on any supported machine. CUDA 0 requires an NVIDIA GPU; mps requires Apple silicon.')
        detail = st.checkbox('Show equipment boxes', value=False, disabled=mode == 'Live camera')
        show_scores = True
        if mode != 'Live camera':
            show_scores = st.checkbox('Show confidence scores', value=True,
                                      help='P = person, H = matched helmet, V = matched vest. These are model confidence scores.')
        else:
            st.caption('The live monitor displays confidence scores.')
    st.divider()
    page = st.radio('Workspace', ['Inspection', 'Project guide', 'Validation'], key='workspace_page')
    st.divider()
    st.caption('Green: PPE detected · Amber: review · Red: check')
    st.caption('Visible PPE evidence only. Missing detections require review.')

st.markdown(f'''<div class="topline"><div class="eyebrow">COMPUTER VISION / {page.upper()}</div>
    <div class="tag">LOCAL PROCESSING · {'MODEL READY' if WEIGHTS.is_file() else 'MODEL REQUIRED'}</div></div>''', unsafe_allow_html=True)
st.title({'Inspection': 'Worksite inspection', 'Project guide': 'Inside the pipeline', 'Validation': 'Evidence & validation'}[page])
st.caption({'Inspection': 'A focused workspace for people, helmets and safety vests.',
            'Project guide': 'Understand every step, from the first pixel to the final report.',
            'Validation': 'Measured results, clear definitions and an honest view of what remains.'}[page])
browser_job = st.session_state.get('browser_preview')
if browser_job is not None and browser_job.running and (
        page != 'Inspection' or (mode == 'Live camera') != (browser_job.kind == 'camera') or mode == 'Image'):
    browser_job.stop()
if page == 'Project guide':
    project_guide(ROOT)
    st.stop()
if page == 'Validation':
    validation_view(ROOT)
    st.stop()
st.markdown('''<div class="steps"><span><b>01</b> Select a source</span><span><b>02</b> Analyze visible PPE</span>
    <span><b>03</b> Review & export</span></div>''', unsafe_allow_html=True)
if not WEIGHTS.is_file():
    st.error('PPE model is missing. Add trained weights at models/best.pt. See docs/phase-3-colab.md for setup.')
    st.stop()

if mode == 'Live camera':
    st.subheader('Live camera preview')
    st.write('Choose a camera and press Start. The live feed and separate AI evidence will appear below.')
    location = st.radio('Preview location', ['In this browser', 'Desktop window'], horizontal=True)
    st.caption('Uses a camera connected to the computer running this app. No browser camera-permission popup is needed.')
    left, right = st.columns([2, 1], gap='large')
    with left:
        camera = st.number_input('Camera index', 0, 20, 0, step=1)
        backend = st.selectbox('Camera backend', ['auto', 'msmf', 'dshow'])
        resolution = st.selectbox('Capture quality', ['HD · 1280 × 720', 'Standard · 640 × 480', 'Full HD · 1920 × 1080'])
        fps = st.select_slider('Requested camera FPS', [15, 24, 30, 60], value=30)
        record = st.checkbox('Save analyzed frames', help='Records AI frames at 10 FPS playback; this is separate from smooth live display.')
        dimensions = {'HD · 1280 × 720': (1280, 720), 'Standard · 640 × 480': (640, 480), 'Full HD · 1920 × 1080': (1920, 1080)}[resolution]
        command = camera_command(ROOT, sys.executable, camera, backend, profile, *dimensions, fps, device, record, confidence)
        browser_running = browser_job is not None and browser_job.running
        process = st.session_state.get('camera_process')
        running = process is not None and process.poll() is None
        if st.button('Start live camera' if location == 'In this browser' else 'Open camera monitor',
                     type='primary', disabled=running or browser_running):
            try:
                if location == 'In this browser':
                    st.session_state.browser_preview = start_browser_preview(ROOT,sys.executable,
                        kind='camera',source=int(camera),profile=profile,confidence=confidence,device=device,
                        backend=backend,width=dimensions[0],height=dimensions[1],fps=fps,record=record)
                    st.rerun()
                OUTPUTS.mkdir(exist_ok=True)
                with (OUTPUTS/'camera-launch.log').open('ab') as log:
                    st.session_state.camera_process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                st.success('Monitor starting on this computer. Press Q or Esc in its window to stop.')
                st.rerun()
            except OSError as error:
                st.error(f'The monitor could not start: {error}')
        if running:
            st.info('Monitor is running. Press Q or Esc in its window to release the camera.')
            st.button('Refresh monitor status')
        if process is not None and process.poll() not in (None, 0):
            st.error('The monitor could not finish. Check outputs/camera-launch.log and try another index/backend.')
        with st.expander('Equivalent terminal command'):
            st.code(subprocess.list2cmdline(command), language='powershell')
    with right:
        st.markdown('**How to use the preview**')
        st.write('1. Select your camera index (usually 0 or 1).\n2. Click Start live camera.\n3. Wait for the feed below; AI warms up separately.\n4. Click Stop live preview to release the camera.')
        st.caption('Close other camera applications first. If index 0 fails, try 1, then MSMF on Windows. Changed settings apply after Stop and Start.')
        if location == 'Desktop window':
            st.caption('The optional desktop window uses Tab to switch focus, F for fullscreen and Q/Esc to stop.')
    if browser_job is not None and browser_job.kind == 'camera':
        render_browser_preview(browser_job)
    elif location == 'In this browser':
        st.info('Your camera preview will appear here after you click Start live camera.')
    st.stop()

is_video = mode == 'Video'
st.subheader('01 / Source footage' if is_video else '01 / Source image')
upload = st.file_uploader('Choose a video' if is_video else 'Choose an image',
    type=['mp4', 'mov', 'avi', 'mkv'] if is_video else ['jpg', 'jpeg', 'png', 'webp'],
    help='Files are processed on this computer. Maximum upload: 200 MB. Images: up to 24 megapixels.')
if not is_video:
    sample = ROOT/'app/assets/presentation-sample.jpg'
    if upload is not None:
        st.session_state.use_sample = False
    use_sample = sample.is_file() and st.checkbox('Use the included presentation sample', disabled=upload is not None,
            key='use_sample', help='A Construction-PPE validation example for demonstrating the interface. It is not independent test data.')
    if use_sample and upload is None:
        sample_data = sample.read_bytes()
        upload = SimpleNamespace(name=sample.name, size=len(sample_data), file_id='included-presentation-sample',
                                 getvalue=lambda: sample_data)
        st.caption('Development validation sample · Ultralytics Construction-PPE · CC BY 4.0. Attribution: app/assets/README.md.')
limit = 0
video_view = 'dashboard'
export_profile = 'efficient'
processing_height = None
video_behaviour = 'Analyze every frame'
if is_video:
    video_behaviour = st.radio('Video review', ['Smooth live review', 'Analyze every frame'],horizontal=True,
        help='Smooth live review displays a live dashboard in this browser while AI scores selected frames. Full analysis scores every frame before playback.')
    if video_behaviour=='Smooth live review':
        review_location = st.radio('Preview location', ['In this browser', 'Desktop window'], horizontal=True)
        st.caption('Start live review to watch the source and timestamped AI evidence here. The browser shows sampled preview updates; every-frame export is a separate option.')
    with st.expander('Video options'):
        if video_behaviour!='Smooth live review':
            video_view = st.selectbox('Presentation', ['dashboard', 'frame'],
                format_func=lambda value: 'Monitor dashboard' if value == 'dashboard' else 'Video only')
        quality_options = (['Preview · 540p', 'HD · 720p', 'Detail · 1080p'] if video_behaviour=='Smooth live review'
                           else ['Balanced · 1080p', 'Lower resolution · 720p', 'Original resolution'])
        video_quality = st.selectbox('Video processing', quality_options,
            help='Smaller frames leave more CPU capacity for playback; larger frames can retain more small PPE detail.')
        processing_height = {'Preview · 540p':540,'HD · 720p':720,'Detail · 1080p':1080,
            'Balanced · 1080p':1080,'Lower resolution · 720p':720,'Original resolution':None}[video_quality]
        limit = st.number_input('Frame limit (0 = full video)', 0, 100000, 300, step=30)
        if video_behaviour!='Smooth live review':
            native_export = st.checkbox('Keep a native resolution export', value=False,
                help='Also creates the full-size annotated video. Uses more time and disk space.')
            export_profile = 'standard' if native_export else 'efficient'
        if video_behaviour=='Smooth live review':
            st.caption('AI scores selected frames in a separate pane with their source times. Choose 0 to review the full video. Browser preview refresh is independent of source FPS.')
        else:
            st.caption('Every frame is analyzed at the selected resolution. Playback keeps the source FPS. '
                       'The original upload stays unchanged. Choose 0 for the full video.')

# Cache the content hash by upload identity; Streamlit reruns on each widget edit.
# The content digest still binds results to exact uploaded bytes.
upload_identity = (getattr(upload, 'file_id', None), upload.name, upload.size) if upload else None
if upload_identity is not None and (upload_identity[0] is None or st.session_state.get('upload_identity') != upload_identity):
    st.session_state.upload_digest = hashlib.sha256(upload.getvalue()).hexdigest()
    st.session_state.upload_identity = upload_identity
fingerprint = (mode, st.session_state.get('upload_digest') if upload else None,
               profile, confidence, device, detail, show_scores, limit, video_view, export_profile, processing_height,video_behaviour)
review_process = st.session_state.get('recorded_review_process')
review_running = review_process is not None and review_process.poll() is None
smooth_review = is_video and video_behaviour=='Smooth live review'
browser_review = smooth_review and review_location == 'In this browser'
if browser_job is not None and browser_job.running and is_video and not browser_review:
    browser_job.stop()
if st.button(('Start live review' if browser_review else 'Open desktop review') if smooth_review else ('Analyze video' if is_video else 'Analyze image'),
             type='primary', disabled=upload is None or (smooth_review and (review_running or (browser_job is not None and browser_job.running)))):
    st.session_state.pop('result', None)
    try:
        if smooth_review:
            source = recorded_review_source(upload.getvalue(),Path(upload.name).suffix,OUTPUTS)
            if browser_review:
                st.session_state.browser_preview = start_browser_preview(ROOT,sys.executable,
                    kind='video',source=source,profile=profile,confidence=confidence,device=device,
                    processing_height=processing_height,max_frames=int(limit),detail=detail,show_scores=show_scores)
                st.rerun()
            command = recorded_review_command(ROOT,sys.executable,source,profile,confidence,device,
                                             processing_height, int(limit),detail,show_scores)
            with (OUTPUTS/'recorded-review-launch.log').open('ab') as log:
                st.session_state.recorded_review_process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            st.rerun()
        with st.spinner('Analyzing visible PPE…'):
            if is_video:
                progress = st.empty()
                with tempfile.TemporaryDirectory(prefix='ppe-upload-') as folder:
                    source = Path(folder)/('upload'+Path(upload.name).suffix.lower())
                    source.write_bytes(upload.getvalue())
                    summary = process_video(source, str(WEIGHTS), confidence, PROFILES[profile], device,
                        OUTPUTS, max_frames=int(limit) or None, tracking=True,
                        view=video_view, detail=detail, show_scores=show_scores, export_profile=export_profile,
                        processing_height=processing_height,
                        on_progress=lambda count, elapsed: progress.info(f'{count} frames analyzed · {count/max(elapsed,.001):.1f} frames/s'))
                progress.empty()
                run = Path(summary['run_directory'])
                preview_error = None
                preview_file = None
                try:
                    preview_file = prepare_browser_video(summary)
                except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    preview_error = str(error)
                result = {'kind': 'video', 'run_directory': str(run), 'summary': summary,
                          'preview_error': preview_error, 'preview_file': str(preview_file) if preview_file else None}
            else:
                result = assess_image_bytes(upload.getvalue(), str(WEIGHTS), PROFILES[profile], confidence,
                                            device, OUTPUTS, detail, show_scores)
            st.session_state.result = (fingerprint, result)
    except (ValueError, OSError, RuntimeError, ImportError, AssertionError) as error:
        st.error(f'Analysis could not finish: {error}')
        st.caption('Try CPU in Advanced settings, a smaller source, or another readable file. For video failures, partial evidence and encoder logs remain in the output run.')

if browser_review and browser_job is not None and browser_job.kind == 'video':
    render_browser_preview(browser_job)
elif browser_review:
    st.info('Upload a video, then click Start live review. The preview will appear here while the model works.')
if smooth_review and not browser_review and review_process is not None:
    if review_running:
        st.info('Recorded review is running in its desktop window. AI warms up first, then source playback starts. '
                'Tab changes the large pane; Q/Esc closes the window.')
        if st.button('Refresh review status'): st.rerun()
    elif review_process.returncode:
        st.error('Recorded review could not finish. Details are saved in outputs/recorded-review-launch.log.')
    else:
        st.success('Recorded review finished. Selected-frame evidence is saved in the latest output run.')

saved = st.session_state.get('result')
if saved and saved[0] == fingerprint:
    result = saved[1]
    run = Path(result['run_directory'])
    validation = (result['assessment'] if result['kind'] == 'image' else result['summary']).get('detection_validation') or {}
    if validation.get('rejected_count', 0):
        st.warning(f"Ignored {validation['rejected_count']} malformed model detections. "
                   'Valid detections and source frames were retained; details are in the JSON reports.')
    if result['kind'] == 'image':
        st.subheader('02 / Inspection results')
        assessment = result['assessment']
        counts = assessment['summary']
        metrics = st.columns(4)
        for col, label, key in zip(metrics, ['Workers', 'PPE detected', 'Needs review', 'Potential violations'],
                                    ['workers', 'PPE detected', 'review required', 'potential violation']):
            col.metric(label, counts[key])
        st.markdown('''<div class="legend"><span>PPE detected</span><span class="review">Review required</span>
            <span class="violation">Potential violation</span></div>''', unsafe_allow_html=True)
        annotated_tab, compare_tab, evidence_tab = st.tabs(['Annotated view', 'Source comparison', 'Worker evidence'])
        with annotated_tab:
            image_path = run/'annotated.png'
            if image_path.is_file():
                st.image(str(image_path), width='stretch')
            else:
                st.warning('The saved image was moved or removed. Analyze again to restore the export.')
        with compare_tab:
            before, after = st.columns(2)
            with before:
                st.caption('ORIGINAL SOURCE')
                st.image(decode_image_bytes(upload.getvalue()), channels='BGR', width='stretch')
            with after:
                st.caption('ANNOTATED EVIDENCE')
                if image_path.is_file():
                    st.image(str(image_path), width='stretch')
        with evidence_tab:
            worker_evidence(assessment)
        st.caption('Image IDs are local to this frame. Confidence scores describe model detections; they are not safety percentages.')
    else:
        summary = result['summary']
        cols = st.columns(3)
        cols[0].metric('Frames analyzed', summary['processed_frames'])
        cols[1].metric('Processing FPS', f"{summary['end_to_end_processing_fps']:.1f}")
        cols[2].metric('Playback FPS', f"{summary['output_fps']:.1f}")
        if summary.get('processing_resolution'):
            w, h = summary['processing_resolution']
            st.caption(f'Analysis: {w} × {h} · Saved playback retains every processed source frame.')
        if result.get('preview_error'):
            st.warning(result['preview_error'])
        else:
            video_path = Path(result.get('preview_file') or run/'preview.mp4')
            if video_path.is_file():
                st.video(str(video_path))
            else:
                st.warning('The saved video was moved or removed. Analyze again to restore playback.')
        st.caption('Scores: P = person, H = matched helmet, V = matched vest. -- means no matched score. These are model confidence scores, not a safety percentage.')
        with st.expander('Check playback smoothness'):
            playback_monitor(summary['output_fps'])
        if summary['stop_reason'] == 'max_frames':
            st.info('Preview frame limit reached. Set the frame limit to 0 and analyze again for the full video.')
        st.caption('Worker observations repeat across frames; they are not unique-person totals. Export contains no audio.')
    with st.expander('03 / Download evidence', expanded=True):
        files = ['annotated.png', 'compliance.json', 'detections.json', 'visibility-template.json'] if result['kind'] == 'image' else ['preview.mp4', 'dashboard.mp4', 'annotated.mp4', 'summary.json', 'frames.jsonl', 'events.jsonl']
        available = [name for name in files if (run/name).is_file()]
        if available:
            selected = st.selectbox('Export file', available)
            path = run/selected
            try:
                if path.stat().st_size <= 200*1024*1024:
                    st.download_button('Download '+selected, path.read_bytes(), file_name=selected, key='download-'+selected)
                else:
                    st.caption(f'{selected} exceeds 200 MB. Open the saved run directory to copy it locally.')
            except OSError as error:
                st.warning(f'This export could not be read: {error}. Analyze again to regenerate it.')
        else:
            st.warning('Saved exports are unavailable. Analyze again to regenerate them.')
        st.caption(f'Saved locally: {run}')
elif saved:
    st.info('Settings or source changed. Analyze again to generate matching results.')
elif upload is not None and not is_video:
    try:
        st.image(decode_image_bytes(upload.getvalue()), channels='BGR', caption='Source image · ready to analyze', width='stretch')
    except (ValueError, OSError):
        st.error('This image could not be decoded. Try another JPEG or PNG.')
elif not browser_review:
    empty_stage(is_video)
st.markdown('<div class="footer-note">PPE VISION / Local review workspace · Visible equipment evidence with human review.</div>', unsafe_allow_html=True)
