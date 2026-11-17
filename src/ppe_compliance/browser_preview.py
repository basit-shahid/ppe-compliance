"""Isolated live-preview workers and a bounded latest-JPEG browser handoff.

The detector stays in a child process. Streamlit only polls atomic files, so
Start/Stop remains responsive and inference never calls Streamlit from a thread.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic, time, sleep
from uuid import uuid4

LEASE_SECONDS = 20.


def atomic_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        temporary.write_bytes(data)
        # Windows readers can briefly hold a file without delete-sharing. Keep
        # the old complete snapshot until the reader closes; never truncate it.
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                sleep(.003 * (attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def write_status(folder, **state):
    atomic_write(Path(folder)/'status.json', json.dumps(state, allow_nan=False).encode('utf-8'))


class BrowserPreview:
    """One session's process, stop request and browser heartbeat."""

    def __init__(self, folder, process, kind):
        self.folder, self.process, self.kind = Path(folder), process, kind
        self.stop_at = None

    @property
    def running(self):
        return self.process.poll() is None

    def heartbeat(self):
        if self.running:
            (self.folder/'heartbeat').touch()
        # Give the pipeline time to finalize reports; bounded native hangs must
        # not leave a camera occupied indefinitely after Stop is requested.
        if self.stop_at is not None and self.running and monotonic()-self.stop_at > 6:
            self.process.terminate()

    def stop(self):
        if self.stop_at is None:
            (self.folder/'stop').touch()
            self.stop_at = monotonic()

    def snapshot(self):
        try:
            state = json.loads((self.folder/'status.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {'status':'starting', 'message':'Opening the source and loading the model…'}
        if not self.running and state.get('status') in {'starting','running'}:
            if self.stop_at is not None:
                state.update(status='stopped', message='Preview stopped.')
            else:
                state.update(status='failed', message='Preview process exited before it was ready. Open the diagnostic log below.')
        try:
            state['image'] = (self.folder/'frame.jpg').read_bytes()
            state['image_age'] = max(0.,time()-(self.folder/'frame.jpg').stat().st_mtime)
        except OSError:
            state['image'] = None
        if self.stop_at is not None and self.running:
            state.update(status='stopping', message='Stopping the preview and releasing the source…')
        return state

    def log_tail(self):
        try:
            with (self.folder/'worker.log').open('rb') as log:
                log.seek(0,2)
                log.seek(max(0,log.tell()-6000))
                return log.read().decode('utf-8',errors='replace')
        except OSError:
            return 'No diagnostic log was created.'


def start_browser_preview(root, python, *, kind, source, profile='Balanced', confidence=.25,
                          device='cpu', backend='auto', width=1280, height=720, fps=30,
                          record=False, processing_height=540, max_frames=0,
                          detail=False, show_scores=True):
    from ppe_compliance.interface import PROFILES
    if kind not in {'camera','video'}:
        raise ValueError('Preview source must be camera or video')
    root = Path(root)
    folder = root/'outputs'/'.browser-previews'/uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    config = dict(kind=kind,source=str(source) if kind=='video' else int(source),
        weights=str(root/'models/best.pt'),output_dir=str(root/'outputs'),
        confidence=confidence,image_size=PROFILES[profile],device=device,
        backend=backend,width=width,height=height,fps=fps,record=record,
        processing_height=processing_height or 540,max_frames=max_frames,
        detail=detail,show_scores=show_scores)
    (folder/'config.json').write_text(json.dumps(config),encoding='utf-8')
    (folder/'heartbeat').touch()
    write_status(folder,status='starting',message='Opening the source and loading the model…')
    with (folder/'worker.log').open('ab') as log:
        process = subprocess.Popen([str(python),'-m','ppe_compliance.browser_preview','--job',str(folder)],
            cwd=root,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return BrowserPreview(folder,process,kind)


def run_worker(folder):
    import cv2
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    last_publish = -float('inf')

    def stopped():
        if (folder/'stop').exists():
            return True
        try:
            return time()-(folder/'heartbeat').stat().st_mtime > LEASE_SECONDS
        except OSError:
            return True

    def publish(frame, info):
        nonlocal last_publish
        now = monotonic()
        if now-last_publish < .1:
            return
        ok, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:
            raise RuntimeError('Could not encode the browser preview image')
        atomic_write(folder/'frame.jpg',encoded.tobytes())
        write_status(folder,status='running',message=('Loading AI; the source preview is available.'
            if info.get('loading') else 'Live preview is running in this browser.'),
            updated_at=time(),**info)
        last_publish = now

    common = dict(weights=config['weights'],confidence=config['confidence'],image_size=config['image_size'],
        device=config['device'],output_dir=Path(config['output_dir']),preview=False,
        tracking=True,max_frames=config['max_frames'] or None,on_preview=publish,
        stop_requested=stopped,shutdown_timeout=3.)
    try:
        if stopped():
            write_status(folder,status='stopped',message='Preview cancelled before startup.')
            return
        if config['kind']=='camera':
            from ppe_compliance.live_pipeline import process_live
            from ppe_compliance.video_pipeline import local_camera_context
            summary = process_live(camera=config['source'],record=config['record'],preview_fps=10.,
                capture_context_factory=lambda:local_camera_context(config['source'],config['backend'],
                    width=config['width'],height=config['height'],fps=config['fps']),**common)
        else:
            from ppe_compliance.recorded_preview import review_recorded_video
            summary = review_recorded_video(config['source'],processing_height=config['processing_height'],
                detail=config['detail'],show_scores=config['show_scores'],**common)
        write_status(folder,status=summary['status'],message='Preview finished. Evidence is saved locally.',
            summary=summary,finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        write_status(folder,status='failed',message=str(error),finished_at=datetime.now(timezone.utc).isoformat())
        raise


def render_browser_preview(job):
    """Render and refresh only the live panel, not the whole upload form."""
    import streamlit as st

    @st.fragment(run_every=.15 if job.running else None)
    def panel():
        try:
            job.heartbeat()
            state = job.snapshot()
            active = job.running
            if not active and st.session_state.get('finished_browser_preview') != job.folder.name:
                st.session_state.finished_browser_preview = job.folder.name
                st.rerun()
            if st.button('Stop live preview',key='stop-'+job.folder.name,
                         disabled=not active or job.stop_at is not None):
                job.stop()
                st.rerun()
            status = state.get('status')
            if status=='failed':
                st.error('Live preview could not run: '+state.get('message','Unknown error'))
                st.caption('For cameras: close any desktop monitor or other camera app, then try another index/backend. Use CPU unless CUDA is configured.')
            elif status in {'completed','stopped'}:
                st.success(state.get('message','Preview finished.'))
            else:
                st.info(state.get('message','Starting live preview…'))
            if state.get('image') and (not active or state.get('image_age',0)<3):
                st.image(state['image'],caption='Live browser preview' if active else 'Last preview frame · session ended',width='stretch')
                if active:
                    sequence=state.get('source_sequence',-1)
                    source_label=str(sequence) if sequence>=0 else 'preparing'
                    st.caption(f"Source frame: {source_label} · AI samples: {state.get('processed_frames',0)}. Browser updates are sampled; source and AI panes refer to their own frames.")
            elif active:
                st.caption('The preview will appear here as soon as readable source frames arrive. Model startup can take a few seconds.')
            summary = state.get('summary')
            if summary:
                st.caption(f"Source frames: {summary.get('source_frames_played',summary.get('captured_frames',0))} · AI samples: {summary.get('processed_frames',0)} · Stop reason: {summary.get('stop_reason','—')}")
                st.caption(f"Saved evidence: {summary.get('run_directory','outputs/')}")
            if status=='failed':
                with st.expander('Preview diagnostic log',expanded=False):
                    st.code(job.log_tail(),language='text')
        except OSError as error:
            st.error(f'Preview files could not be updated: {error}')
    panel()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    run_worker(parser.parse_args().job)


if __name__=='__main__':
    main()
