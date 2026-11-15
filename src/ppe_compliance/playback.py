"""Browser counters and source-paced playback of completed recorded video."""

import math
from pathlib import Path
from time import perf_counter


def play_recorded_output(summary):
    """Review the finished file at its encoded rate, with bounded GUI waits.

    All frames remain in order. A slow decoder reanchors the clock rather than
    skipping evidence or presenting a burst of overdue frames. Q/Esc closes it.
    """
    import cv2

    name = summary.get('browser_video_file') or summary.get('dashboard_file') or summary.get('annotated_file') or 'annotated.mp4'
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Invalid saved playback filename')
    path = Path(summary['run_directory'])/name
    if not path.is_file():
        raise ValueError(f'Saved video not found: {path}')
    capture = cv2.VideoCapture(str(path))
    window = 'Recorded PPE review - Q or Esc to close'
    opened = False
    frames, resets = 0, 0
    fps = None
    started = deadline = None
    status = 'completed'

    def wait_until(target):
        while True:
            remaining = target-perf_counter()
            delay = max(1, min(20, math.ceil(remaining*1000))) if remaining > 0 else 1
            key = cv2.waitKey(delay) & 0xFF
            try:
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = 0
            if key in (ord('q'), ord('Q'), 27) or visible < 1:
                return False
            if perf_counter() >= target:
                return True

    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot decode saved playback: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Saved playback has no valid frame rate')
        period = 1/fps
        advertised = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected_frames = int(advertised) if math.isfinite(advertised) and advertised > 0 else None
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            opened = True
            cv2.resizeWindow(window, 1280, 720)
        except cv2.error as error:
            raise RuntimeError(f'Playback window unavailable; open the saved video in a media player: {path}') from error
        while True:
            ok, frame = capture.read()
            if not ok:
                if frames == 0:
                    raise RuntimeError('Saved video returned no readable frames')
                if expected_frames is not None and frames < expected_frames:
                    raise RuntimeError(f'Playback decoding ended early ({frames}/{expected_frames} frames)')
                if not wait_until(deadline): status = 'stopped'
                break
            now = perf_counter()
            if started is None:
                started = deadline = now
            elif now > deadline+period:
                deadline = now
                resets += 1
            elif not wait_until(deadline):
                status = 'stopped'
                break
            cv2.imshow(window, frame)
            frames += 1
            deadline += period
            if not wait_until(perf_counter()):
                status = 'stopped'
                break
    except KeyboardInterrupt:
        status = 'stopped'
    finally:
        capture.release()
        if opened:
            try: cv2.destroyWindow(window)
            except cv2.error: pass
    elapsed = perf_counter()-started if started is not None else 0.
    return {'status': status, 'file': str(path), 'displayed_frames': frames,
            'playback_fps': fps, 'wall_seconds': elapsed,
            'measured_submission_fps': frames/elapsed if elapsed else None,
            'pacing_resets': resets, 'intentional_frame_skipping': False}


def playback_monitor(expected_fps):
    import streamlit as st

    rate = float(expected_fps or 0)
    if not math.isfinite(rate) or rate < 0:
        rate = 0.
    html = '''<!doctype html><html><head><style>
      body{margin:0;color:#EBEFF5;background:transparent;font:13px system-ui,sans-serif}
      #status{color:#9DAABA;margin:6px 0}button{border:1px solid #475367;border-radius:5px;
      color:#EBEFF5;background:#191F29;padding:5px 9px;cursor:pointer}
      pre{font-size:11px;white-space:pre-wrap;user-select:text}
      </style></head><body><div id="values">Waiting for the saved video…</div>
      <div id="status">Keep the player onscreen while testing.</div>
      <button id="copy">Copy playback test</button><pre id="details" hidden></pre>
      <script>
      let previous=null; const expected=__FPS__;
      function sample(){
        try{
          const v=parent.document.querySelector('video');
          if(!v || !v.getVideoPlaybackQuality)return;
          const q=v.getVideoPlaybackQuality(),r=v.getBoundingClientRect(),now=performance.now();
          const visible=Math.max(0,Math.min(r.right,parent.innerWidth)-Math.max(r.left,0))*
            Math.max(0,Math.min(r.bottom,parent.innerHeight)-Math.max(r.top,0))/Math.max(1,r.width*r.height);
          const active=!v.paused&&!v.ended&&!v.seeking;
          const fps=previous&&previous.active&&active?Math.max(0,(q.totalVideoFrames-q.droppedVideoFrames-previous.frames)/(now-previous.at)*1000):null;
          let ahead=0;for(let i=0;i<v.buffered.length;i++)if(v.buffered.start(i)<=v.currentTime&&v.buffered.end(i)>=v.currentTime)ahead=v.buffered.end(i)-v.currentTime;
          window.PPEPlaybackReport={expectedPlaybackFPS:expected,displayedFPS:fps,
            droppedFrames:q.droppedVideoFrames,totalFrames:q.totalVideoFrames,
            bufferedAheadSeconds:ahead,playerVisiblePercent:Math.round(visible*100),
            tabVisible:parent.document.visibilityState==='visible',playing:active,
            mediaTimeSeconds:v.currentTime,readyState:v.readyState};
          document.getElementById('values').textContent=`Expected ${expected} FPS · Displayed ${fps===null?'--':fps.toFixed(1)} FPS · Dropped ${q.droppedVideoFrames} / ${q.totalVideoFrames} frames`;
          document.getElementById('status').textContent=visible<.75||parent.document.visibilityState!=='visible'
            ?`Player ${Math.round(visible*100)}% in view. Bring it onscreen for a valid motion test.`
            :active?`Buffered ahead: ${ahead.toFixed(1)} s · Measuring playback on this device.`
            :'Play the saved video to measure motion on this device.';
          previous={frames:q.totalVideoFrames-q.droppedVideoFrames,at:now,active};
        }catch(error){document.getElementById('status').textContent='Playback counters are unavailable in this browser.';}
      }
      document.getElementById('copy').onclick=async()=>{
        const result=JSON.stringify(window.PPEPlaybackReport||{status:'Play the video first'},null,2);
        try{await navigator.clipboard.writeText(result);document.getElementById('copy').textContent='Copied';}
        catch(error){const details=document.getElementById('details');details.hidden=false;details.textContent=result;}
      };
      sample();const timer=setInterval(sample,1000);addEventListener('pagehide',()=>clearInterval(timer));
      </script></body></html>'''.replace('__FPS__', str(rate))
    if hasattr(st, 'iframe'):
        st.iframe(html, height=110, alt='Playback frame counters and buffering test')
    else:  # Compatibility with the supported earlier Streamlit versions.
        import streamlit.components.v1 as components
        components.html(html, height=110, scrolling=True)
