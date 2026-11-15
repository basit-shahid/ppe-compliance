"""Run locally: python -m streamlit run app/streamlit_app.py."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import streamlit as st

from ppe_compliance.interface import PROFILES, assess_image_bytes, prepare_browser_video, camera_command, decode_image_bytes
from ppe_compliance.video_pipeline import process_video
from ppe_compliance.rendering import worker_scores
from ppe_compliance.playback import playback_monitor
from ppe_compliance.interface import recorded_review_source, recorded_review_command

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/best.pt'
OUTPUTS = ROOT / 'outputs'

st.set_page_config(page_title='PPE Monitor', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 2rem; }
h1 { font-weight: 650; letter-spacing: -.04em; }
[data-testid="stMetricValue"] { font-size: 1.65rem; }
[data-testid="stMetric"] { padding: .3rem 0; }
[data-testid="stSidebar"] { border-right: 1px solid #2B3340; }
[data-testid="stImage"] img { max-height: 65vh; width: auto !important; max-width: 100%; margin: 0 auto; object-fit: contain; }
.stButton > button { border-radius: 8px; }
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### ◉ PPE Monitor')
    st.caption('LOCAL WORKSPACE')
    mode = st.radio('Source', ['Image', 'Video', 'Live camera'], label_visibility='collapsed')
    st.divider()
    profile = st.selectbox('Processing profile', list(PROFILES), index=0)
    st.caption({'Balanced': 'Everyday use · moderate detail and speed.',
                'Detail': 'Larger model input · better small-item detail, slower on CPU.',
                'Fast': 'Responsive on CPU · small equipment may be missed.'}[profile])
    with st.expander('Advanced settings'):
        confidence = st.slider('Detection confidence', .05, .95, .25, .05)
        device = st.selectbox('Compute device', ['cpu', '0', 'mps'], help='Select 0 only with configured CUDA; mps requires Apple silicon.')
        detail = st.checkbox('Show equipment boxes', value=False, disabled=mode == 'Live camera')
        show_scores = True
        if mode != 'Live camera':
            show_scores = st.checkbox('Show confidence scores', value=True,
                                      help='P = person, H = matched helmet, V = matched vest. These are model confidence scores.')
        else:
            st.caption('The live monitor displays confidence scores.')
    st.divider()
    st.caption('Green: PPE detected · Amber: review · Red: check')
    st.caption('Visible PPE evidence only. Missing detections require review.')

st.title('A clearer view of workplace PPE')
st.caption('Review workers, helmets and safety vests in one focused workspace.')
if not WEIGHTS.is_file():
    st.error('PPE model is missing. Add trained weights at models/best.pt. See docs/phase-3-colab.md for setup.')
    st.stop()

if mode == 'Live camera':
    st.subheader('Live monitor')
    st.write('Open the desktop monitor for smooth camera display with a separate, timestamped analysis view.')
    left, right = st.columns([2, 1], gap='large')
    with left:
        camera = st.number_input('Camera index', 0, 20, 0, step=1)
        backend = st.selectbox('Camera backend', ['auto', 'msmf', 'dshow'])
        resolution = st.selectbox('Capture quality', ['HD · 1280 × 720', 'Standard · 640 × 480', 'Full HD · 1920 × 1080'])
        fps = st.select_slider('Requested camera FPS', [15, 24, 30, 60], value=30)
        record = st.checkbox('Save analyzed frames', help='Records AI frames at 10 FPS playback; this is separate from smooth live display.')
        dimensions = {'HD · 1280 × 720': (1280, 720), 'Standard · 640 × 480': (640, 480), 'Full HD · 1920 × 1080': (1920, 1080)}[resolution]
        command = camera_command(ROOT, sys.executable, camera, backend, profile, *dimensions, fps, device, record)
        process = st.session_state.get('camera_process')
        running = process is not None and process.poll() is None
        if st.button('Open camera monitor', type='primary', disabled=running):
            OUTPUTS.mkdir(exist_ok=True)
            try:
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
        st.markdown('**While monitoring**')
        st.write('Tab switches the main view. F toggles full screen. Q or Esc stops capture.')
        st.caption('The live view keeps moving while AI processes selected frames. Old results disappear automatically. Camera settings depend on the driver.')
    st.stop()

is_video = mode == 'Video'
upload = st.file_uploader('Choose a video' if is_video else 'Choose an image',
    type=['mp4', 'mov', 'avi', 'mkv'] if is_video else ['jpg', 'jpeg', 'png', 'webp'],
    help='Files are processed on this computer. Maximum upload: 200 MB.')
limit = 0
video_view = 'dashboard'
export_profile = 'efficient'
processing_height = None
video_behaviour = 'Analyze every frame'
if is_video:
    video_behaviour = st.radio('Video review', ['Smooth live review', 'Analyze every frame'],horizontal=True,
        help='Smooth review plays the video in a desktop window while AI scores selected frames. Full analysis scores every frame before playback.')
    if video_behaviour=='Smooth live review':
        st.caption('Opens a desktop window on this computer with moving source video and a separate, timestamped AI view.')
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
            st.caption('Opens a desktop review window on this computer. The main video retains original motion; '
                       'AI scores selected frames in a separate pane with their source times. Choose 0 to play the full video.')
        else:
            st.caption('Every frame is analyzed at the selected resolution. Playback keeps the source FPS. '
                       'The original upload stays unchanged. Choose 0 for the full video.')

fingerprint = (mode, hashlib.sha256(upload.getvalue()).hexdigest() if upload else None,
               profile, confidence, device, detail, show_scores, limit, video_view, export_profile, processing_height,video_behaviour)
review_process = st.session_state.get('recorded_review_process')
review_running = review_process is not None and review_process.poll() is None
smooth_review = is_video and video_behaviour=='Smooth live review'
if st.button('Open desktop review' if smooth_review else ('Analyze video' if is_video else 'Analyze image'),
             type='primary', disabled=upload is None or (smooth_review and review_running)):
    st.session_state.pop('result', None)
    try:
        if smooth_review:
            source = recorded_review_source(upload.getvalue(),Path(upload.name).suffix,OUTPUTS)
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
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        st.error(f'Analysis could not finish: {error}')

if smooth_review and review_process is not None:
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
        assessment = result['assessment']
        counts = assessment['summary']
        metrics = st.columns(4)
        for col, label, key in zip(metrics, ['Workers', 'PPE detected', 'Needs review', 'Potential violations'],
                                    ['workers', 'PPE detected', 'review required', 'potential violation']):
            col.metric(label, counts[key])
        st.image(str(run/'annotated.png'), width='stretch')
        with st.expander('Worker evidence', expanded=False):
            if assessment['workers']:
                st.dataframe([{'Worker': w['worker_id'], 'Person confidence': worker_scores(w)[0],
                               'Status': w['status'], 'Helmet confidence': worker_scores(w)[1],
                               'Vest confidence': worker_scores(w)[2], 'Helmet': w['helmet']['state'],
                               'Vest': w['vest']['state'], 'Reason': '; '.join(w['reasons'])}
                              for w in assessment['workers']], hide_index=True, width='stretch')
            else:
                st.info('No workers detected. This does not establish an empty or safe scene.')
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
            st.video(result.get('preview_file') or str(run/'preview.mp4'))
        st.caption('Scores: P = person, H = matched helmet, V = matched vest. -- means no matched score. These are model confidence scores, not a safety percentage.')
        with st.expander('Check playback smoothness'):
            playback_monitor(summary['output_fps'])
        if summary['stop_reason'] == 'max_frames':
            st.info('Preview frame limit reached. Set the frame limit to 0 and analyze again for the full video.')
        st.caption('Worker observations repeat across frames; they are not unique-person totals. Export contains no audio.')
    with st.expander('Download results'):
        files = ['annotated.png', 'compliance.json', 'detections.json', 'visibility-template.json'] if result['kind'] == 'image' else ['preview.mp4', 'dashboard.mp4', 'annotated.mp4', 'summary.json', 'frames.jsonl', 'events.jsonl']
        for name in files:
            path = run/name
            if path.is_file():
                if path.stat().st_size <= 200*1024*1024:
                    st.download_button(name, path.read_bytes(), file_name=name, key='download-'+name)
                else:
                    st.caption(f'{name} exceeds 200 MB. Open the saved run directory to copy it locally.')
        st.caption(f'Saved locally: {run}')
elif saved:
    st.info('Settings or source changed. Analyze again to generate matching results.')
elif upload is not None and not is_video:
    try:
        st.image(decode_image_bytes(upload.getvalue()), channels='BGR', caption='Source image · ready to analyze', width='stretch')
    except (ValueError, OSError):
        st.error('This image could not be decoded. Try another JPEG or PNG.')
else:
    st.markdown('#### Your review starts here')
    st.write('Choose a file, select a processing profile, then analyze. Results include a clean annotated view and downloadable evidence.')
