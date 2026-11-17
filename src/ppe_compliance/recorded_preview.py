"""Source-paced recorded playback with a bounded latest-frame AI worker."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from queue import Queue, Empty, Full
import subprocess
from threading import Event, Thread
from time import perf_counter, sleep
from uuid import uuid4

from ppe_compliance.association import AssociationRules
from ppe_compliance.compliance import assess_detections
from ppe_compliance.detector import FrameDetector, file_sha256
from ppe_compliance.live_pipeline import FramePacket, ResultPacket, LatestSlot
from ppe_compliance.preview_ui import DashboardRenderer
from ppe_compliance.rendering import draw_assessment, resize_frame
from ppe_compliance.tracking import WorkerTracker


def prepare_playback_source(source, source_hash, resolution, fps, expected, output_dir, run, stop):
    """Cache a lightweight, source-paced copy for expensive large-video decode."""
    import cv2
    import imageio_ffmpeg

    cache = Path(output_dir)/'.playback-cache'
    cache.mkdir(parents=True,exist_ok=True)
    target = cache/f'{source_hash}-{resolution[0]}x{resolution[1]}-v1.mp4'
    manifest = target.with_suffix('.json')

    def inspect(path):
        capture = cv2.VideoCapture(str(path))
        count = 0
        try:
            if not capture.isOpened(): raise RuntimeError('Prepared video cannot be decoded')
            rate = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(rate) or abs(rate-fps) > .001: raise RuntimeError('Prepared video frame rate changed')
            while not stop.is_set():
                ok, image = capture.read()
                if not ok: break
                if image.shape[:2] != (resolution[1],resolution[0]): raise RuntimeError('Prepared video dimensions changed')
                count += 1
        finally: capture.release()
        if not stop.is_set() and (not count or (expected is not None and count != expected)):
            raise RuntimeError('Prepared video frame count differs from the source')
        return count

    if target.is_file() and manifest.is_file():
        try:
            info = json.loads(manifest.read_text())
            if (info['source_sha256']==source_hash and info['sha256']==file_sha256(target)
                    and info['resolution']==list(resolution) and abs(info['fps']-fps)<.001
                    and (expected is None or info['frames']==expected)):
                return target, {**info,'cached':True}
        except (OSError,ValueError,KeyError): pass
    temporary = cache/(uuid4().hex+'.mp4')
    temporary_manifest = cache/(uuid4().hex+'.json')
    process = None
    try:
        with (Path(run)/'prepare.log').open('wb') as log:
            process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-hide_banner','-loglevel','error',
                '-threads','2','-i',str(source),'-map','0:v:0','-an','-sn','-dn','-filter_threads','1',
                '-vf',f'scale={resolution[0]}:{resolution[1]}:flags=area',
                '-fps_mode','passthrough','-c:v','libx264','-preset','veryfast','-crf','18',
                '-threads','1','-tune','fastdecode','-bf','0','-g',str(max(1,round(fps*2))),
                '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],
                stdout=subprocess.DEVNULL,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while process.poll() is None:
                if stop.wait(.05):
                    process.terminate()
                    break
            try: code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
                raise RuntimeError('Playback preparation did not finalize')
        if stop.is_set(): return source, {'cancelled':True}
        if code: raise RuntimeError(f'Playback preparation failed; see {Path(run)/"prepare.log"}')
        count = inspect(temporary)
        if stop.is_set(): return source, {'cancelled':True}
        info = {'source_sha256':source_hash,'sha256':file_sha256(temporary),
                'resolution':list(resolution),'fps':fps,'frames':count}
        temporary_manifest.write_text(json.dumps(info,indent=2))
        temporary.replace(target)
        temporary_manifest.replace(manifest)
        return target, {**info,'cached':False}
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait()
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


class ScaledVideoReader:
    """Stream decoded BGR frames through a bounded pipe; never queue the video."""

    def __init__(self, source, resolution, run):
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError('Live recorded review needs imageio-ffmpeg; install the project [ui] extra.') from error
        self.width, self.height = resolution
        self.log_path = Path(run)/'decode.log'
        self.log = self.log_path.open('wb')
        self.process = None
        try:
            self.process = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-threads', '2', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
                '-filter_threads', '1', '-vf', f'scale={self.width}:{self.height}:flags=bilinear',
                '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.log.close()
            raise

    def read(self):
        import numpy as np

        pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        buffer = memoryview(pixels).cast('B')
        received = 0
        while received < len(buffer):
            count = self.process.stdout.readinto(buffer[received:])
            if not count:
                if received:
                    raise RuntimeError(f'Truncated decoded frame; see {self.log_path}')
                code = self.process.wait(timeout=10)
                if code:
                    raise RuntimeError(f'Video decoder failed (exit {code}); see {self.log_path}')
                return False, None
            received += count
        return True, pixels

    def release(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdout.close()
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self.log.close()

    def cancel(self):
        process = self.process
        if process is not None and process.poll() is None:
            try: process.terminate()
            except OSError:
                if process.poll() is None: raise


class PrefetchedReader:
    """Three ordered source frames overlap decoding and GUI work without drops."""

    def __init__(self, source, resolution, run):
        self.reader = ScaledVideoReader(source,resolution,run)
        self.queue = Queue(maxsize=3)
        self.stop = Event()
        self.close_error = None
        self.worker = Thread(target=self.decode,name='ppe-recorded-decode',daemon=True)
        try:
            self.worker.start()
        except BaseException:
            self.reader.release()
            raise

    def publish(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item,timeout=.05)
                return
            except Full: pass

    def decode(self):
        try:
            while not self.stop.is_set():
                item = self.reader.read()
                self.publish(item)
                if not item[0]: break
        except Exception as error:
            if not self.stop.is_set(): self.publish(error)
        finally:
            try: self.reader.release()
            except Exception as error: self.close_error = error

    def read(self):
        while True:
            try: item = self.queue.get(timeout=.1)
            except Empty:
                if not self.worker.is_alive():
                    raise RuntimeError('Decoder stopped without a final frame status')
                continue
            if isinstance(item,Exception): raise RuntimeError(str(item)) from item
            return item

    def release(self):
        self.stop.set()
        # Signal the subprocess, but let its owning thread close the pipe.
        cancel = getattr(self.reader,'cancel',None)
        if cancel is not None: cancel()
        self.worker.join(timeout=6)
        if self.worker.is_alive(): raise RuntimeError('Decode worker did not finalize within timeout')
        if self.close_error: raise RuntimeError(str(self.close_error)) from self.close_error


def review_recorded_video(source, weights='models/best.pt', confidence=.25, image_size=416,
                          device='cpu', output_dir=Path('outputs'), processing_height=540,
                          max_frames=None, rules=None, tracking=True, preview=True,
                          detail=False, show_scores=True, on_progress=None, shutdown_timeout=30., analysis_fps=2.5,
                          on_preview=None, stop_requested=None):
    """Display every source frame in order; score only fresh selected frames.

    Results always retain their own source image/time in the separate analysis
    pane. JSONL contains sampled evidence, never an all-frame compliance report.
    No frame interpolation or old boxes on the moving video are used.
    """
    import cv2

    source = Path(source).resolve()
    if any(value is not None and not callable(value) for value in (on_preview, stop_requested)):
        raise ValueError('Preview and stop hooks must be callable')
    if not source.is_file():
        raise ValueError(f'Source video not found: {source}')
    if type(processing_height) is not int or processing_height <= 0:
        raise ValueError('Processing height must be a positive integer')
    if type(image_size) is not int or image_size <= 0 or isinstance(confidence, bool) or not 0 < confidence <= 1:
        raise ValueError('Invalid model input size or confidence')
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError('max_frames must be a positive source-frame limit')
    if any(type(value) is not bool for value in (tracking, preview, detail, show_scores)):
        raise ValueError('Preview, tracking and score options must be boolean flags')
    if isinstance(shutdown_timeout, bool) or not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
        raise ValueError('Shutdown timeout must be positive and finite')
    if isinstance(analysis_fps,bool) or not math.isfinite(analysis_fps) or analysis_fps <= 0:
        raise ValueError('Analysis FPS limit must be positive and finite')
    metadata = cv2.VideoCapture(str(source))
    try:
        if not metadata.isOpened():
            raise ValueError('Cannot open the source video')
        fps = metadata.get(cv2.CAP_PROP_FPS)
        advertised = metadata.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, first = metadata.read()
        if not ok or first is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError('The video needs readable frames and valid FPS metadata')
    finally:
        metadata.release()
    expected = int(round(advertised)) if math.isfinite(advertised) and advertised > 0 else None
    original_resolution = (first.shape[1], first.shape[0])
    ratio = min(1., processing_height/first.shape[0], (processing_height*16/9)/first.shape[1])
    resolution = (max(1, round(first.shape[1]*ratio)), max(1, round(first.shape[0]*ratio)))
    if original_resolution[0]*original_resolution[1] > 1920*1080:
        # H.264/yuv420p playback copies require even dimensions. Keep the
        # processing/report geometry consistent rather than padding secretly.
        resolution = tuple(max(2,value-value%2) for value in resolution)
    warmup_frame = resize_frame(first, resolution) if resolution != original_resolution else first
    source_hash = file_sha256(source)
    run = Path(output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    stop, ready, source_ready = Event(), Event(), Event()
    frames, results = LatestSlot(), LatestSlot()
    state = {'processed': 0, 'error': None, 'weights_sha256': None, 'warmup_seconds': None,
             'worker_observations': Counter(), 'latency_sum': 0., 'max_latency': 0.,
             'rejected_count': 0, 'inference_threads': None,
             'playback_source':str(source),'source_preparation':None,'preparation_seconds':0.}
    rules = rules or AssociationRules()
    started = perf_counter()

    def prepare():
        try:
            # A full AI scan is never done here. Only large inputs need the
            # cheap playback representation; normal HD files stream directly.
            if original_resolution[0]*original_resolution[1] > 1920*1080:
                prepared, info = prepare_playback_source(source,source_hash,resolution,fps,expected,output_dir,run,stop)
                state['playback_source'] = str(prepared)
                state['source_preparation'] = info
            state['preparation_seconds'] = perf_counter()-started
            source_ready.set()
        except Exception as failure:
            state['error'] = str(failure) or type(failure).__name__
            stop.set()

    def analyze():
        old_threads = None
        try:
            detector = FrameDetector(weights, confidence, image_size, device)
            state['weights_sha256'] = detector.weights_sha256
            if not {'person','helmet','vest'} <= {v.casefold() for v in detector.model_classes.values()}:
                raise ValueError('Recorded review needs PPE-trained person/helmet/vest weights')
            detector.detect(warmup_frame.copy(), f'{source}#warmup=0')
            # Reserve CPU capacity for source decoding and the GUI. This mode
            # trades analysis frequency for responsive playback intentionally.
            import torch
            if device == 'cpu':
                old_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                state['inference_threads'] = 1
            tracker = WorkerTracker() if tracking else None
            state['warmup_seconds'] = perf_counter()-started
            ready.set()
            previous = -1
            next_analysis = 0.
            with (run/'frames.jsonl').open('w',encoding='utf-8') as log, (run/'events.jsonl').open('w',encoding='utf-8') as events:
                while not stop.is_set():
                    if stop.wait(max(0,next_analysis-perf_counter())): break
                    packet = frames.next_frame(previous, stop)
                    if packet is None:
                        continue
                    analysis_started = perf_counter()
                    _, raw = detector.detect(packet.frame.copy(), f'{source}#frame={packet.sequence}')
                    if stop.is_set(): break
                    assessment = assess_detections(raw, hashlib.sha256(memoryview(packet.frame)).hexdigest(), rules)
                    if tracker:
                        assessment = tracker.update(assessment, packet.sequence/fps)
                    assessment['detection_id_scope'] = 'zero-based index into this sampled row detections'
                    annotated = draw_assessment(packet.frame,assessment,raw['detections'],
                                                fit=(944,532),detail=detail,show_scores=show_scores)
                    completed = perf_counter()
                    if stop.is_set(): break
                    result = ResultPacket(packet,completed,annotated,assessment,
                                          tuple(tracker.history) if tracker else ())
                    log.write(json.dumps({'frame_index':packet.sequence,'source_time_seconds':packet.sequence/fps,
                        'analysis_sample_index':state['processed'],'source_sha256':source_hash,
                        'source_coordinate_scale':[original_resolution[i]/resolution[i] for i in (0,1)],
                        'frame_hash_format':'SHA256 of streamed processing BGR pixels; dimensions in assessment',
                        'detections':raw['detections'],'assessment':assessment},separators=(',',':'))+'\n')
                    log.flush()
                    for event in assessment.get('status_events',[]): events.write(json.dumps(event)+'\n')
                    events.flush()
                    state['processed'] += 1
                    state['worker_observations'].update({k:v for k,v in assessment['summary'].items() if k!='workers'})
                    latency = completed-packet.received_at
                    state['latency_sum'] += latency
                    state['max_latency'] = max(state['max_latency'],latency)
                    state['rejected_count'] += (raw.get('detection_validation') or {}).get('rejected_count',0)
                    results.publish(result)
                    previous = packet.sequence
                    next_analysis = analysis_started+1/analysis_fps
        except Exception as error:
            state['error'] = str(error) or type(error).__name__
            stop.set()
        finally:
            if old_threads is not None:
                import torch
                torch.set_num_threads(old_threads)

    worker = Thread(target=analyze,name='ppe-recorded-ai',daemon=True)
    prepare_worker = Thread(target=prepare,name='ppe-recorded-prepare',daemon=True)
    reader = None
    renderer = DashboardRenderer()
    window = 'Recorded video | original motion + sampled AI | Q/Esc stop'
    opened = False
    focus = 'live'
    played = resets = 0
    stage_totals = Counter()
    playback_started = playback_ended = None
    last_progress = -math.inf
    error = None
    reason = 'end_of_video'
    canvas = None
    canvas_key = None
    ui_updated_at = -math.inf

    def events_until(deadline, pump=False):
        nonlocal focus
        while True:
            if stop_requested is not None and stop_requested():
                stop.set()
                return
            remaining = deadline-perf_counter()
            if remaining <= 0 and not pump: return
            if preview:
                key = cv2.waitKey(max(1,min(10,math.ceil(remaining*1000))) if remaining > 0 else 1)&0xFF
                if key == 9: focus = 'analyzed' if focus == 'live' else 'live'
                try: visible = cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)
                except cv2.error: visible = 0
                if key in (ord('q'),ord('Q'),27) or visible < 1: stop.set()
            elif remaining > 0:
                sleep(min(.01,remaining))
            if stop.is_set() or perf_counter() >= deadline: return

    def show(packet, loading=False):
        nonlocal canvas, canvas_key, ui_updated_at
        if not preview and on_preview is None: return
        now = perf_counter()
        result = results.snapshot()
        elapsed = now-playback_started if playback_started is not None else 0.
        key = (result.source.sequence if result else -1,focus,loading)
        if canvas is None or key!=canvas_key or now-ui_updated_at>=.25:
            canvas = renderer.render(packet,result,result is not None,now,
                {'mode':'recorded_live','loading':loading,'show_scores':show_scores,
                 'live_rate':max(0,played-1)/elapsed if elapsed else 0., 'source_fps':fps,
                 'source_time_seconds':max(0,packet.sequence)/fps,
                 'analysis_time_seconds':result.source.sequence/fps if result else None,
                 'preparation_status':f"Model {'ready' if ready.is_set() else 'warming'} | Video {'ready' if source_ready.is_set() else 'preparing'} | {now-started:.0f} s",
                 'ai_rate':state['processed']/elapsed if elapsed else 0.},focus=focus)
            canvas_key,ui_updated_at = key,now
        else:
            renderer.update_source(canvas,packet,focus)
        if on_preview is not None:
            on_preview(canvas, {'loading':loading, 'source_sequence':packet.sequence,
                'source_time_seconds':max(0,packet.sequence)/fps,
                'processed_frames':state['processed'],
                'analysis_sequence':result.source.sequence if result is not None else None,
                'source_kind':'video'})
        if preview:
            cv2.imshow(window,canvas)

    try:
        if preview:
            cv2.namedWindow(window,cv2.WINDOW_NORMAL)
            opened = True
            cv2.resizeWindow(window,1280,720)
        worker.start()
        prepare_worker.start()
        placeholder = FramePacket(-1,perf_counter(),warmup_frame)
        show(placeholder,loading=True)
        loading_updated = perf_counter()
        while not (ready.is_set() and source_ready.is_set()) and not stop.is_set():
            events_until(perf_counter()+.01)
            if perf_counter()-loading_updated >= .5:
                show(placeholder,loading=True)
                loading_updated = perf_counter()
        if state['error']: raise RuntimeError(state['error'])
        if stop.is_set(): reason = 'preview_stop'
        else:
            reader = PrefetchedReader(state['playback_source'],resolution,run)
            deadline = None
            period = 1/fps
            while not stop.is_set():
                if state['error']: raise RuntimeError(state['error'])
                stage_started = perf_counter()
                ok, image = reader.read()
                read_done = perf_counter()
                if not ok:
                    if expected is not None and played < expected:
                        raise RuntimeError(f'Source decoding ended early ({played}/{expected} frames)')
                    if deadline is not None: events_until(deadline)
                    break
                if image.shape != (resolution[1],resolution[0],3) or image.dtype.name!='uint8':
                    raise RuntimeError('Decoded processing frame dimensions or pixel type changed')
                now = perf_counter()
                if deadline is None: playback_started = deadline = now
                if now > deadline+period:
                    deadline = now
                    resets += 1
                events_until(deadline)
                wait_done = perf_counter()
                if stop.is_set(): break
                image.flags.writeable = False
                packet = FramePacket(played,perf_counter(),image)
                frames.publish(packet)
                played += 1
                show(packet)
                render_done = perf_counter()
                deadline += period
                events_until(perf_counter(),pump=True)
                poll_done = perf_counter()
                stage_totals.update(read=read_done-stage_started,wait=wait_done-read_done,
                                    render=render_done-wait_done,gui_poll=poll_done-render_done)
                if on_progress and perf_counter()-last_progress >= 1.:
                    on_progress(played,state['processed'])
                    last_progress = perf_counter()
                if max_frames is not None and played >= max_frames:
                    reason = 'max_source_frames'
                    events_until(deadline)
                    break
            if stop.is_set(): reason = 'preview_stop'
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except Exception as failure:
        error, reason = str(failure) or type(failure).__name__, 'error'
    finally:
        playback_ended = perf_counter()
        stop.set()
        if reader is not None:
            try: reader.release()
            except Exception as failure: error = error or str(failure)
        if opened:
            try: cv2.destroyWindow(window)
            except cv2.error: pass
        shutdown_deadline = perf_counter()+shutdown_timeout
        for thread in (worker,prepare_worker):
            if thread.ident is not None: thread.join(timeout=max(0,shutdown_deadline-perf_counter()))
    unfinished = worker.is_alive() or prepare_worker.is_alive()
    if unfinished: error = error or 'AI or playback preparation did not finalize before the shutdown timeout'
    error = error or state['error']
    try:
        if file_sha256(source)!=source_hash: error=error or 'Source video changed during review'
    except OSError:
        error=error or 'Source video became unavailable during review'
    if error: reason = 'shutdown_timeout' if unfinished else 'error'
    seconds = playback_ended-playback_started if playback_started is not None else 0.
    summary = {'schema_version':1,'mode':'recorded-live-preview','status':'failed' if error else ('completed' if reason=='end_of_video' else 'stopped'),
        'stop_reason':reason,'error':error,'run_directory':str(run),'source':str(source),'source_sha256':source_hash,
        'source_fps_metadata':fps,'source_frame_count_metadata':expected,'source_resolution':list(original_resolution),
        'processing_resolution':list(resolution),'processing_height_limit':processing_height,'image_size':image_size,
        'source_coordinate_scale':[original_resolution[i]/resolution[i] for i in (0,1)],
        'frame_hash_scope':'streamed_processing_bgr','analysis_fps_limit':analysis_fps,
        'playback_source':state['playback_source'],'playback_source_preparation':state['source_preparation'],
        'preparation_seconds':state['preparation_seconds'],
        'browser_preview_enabled':on_preview is not None,
        'weights':str(weights),'weights_sha256':state['weights_sha256'],'device':device,
        'source_frames_played':played,'processed_frames':state['processed'],
        'source_frames_not_analyzed':played-state['processed'],'intentional_analysis_sampling':True,
        'intentional_source_frame_skipping':False,'interpolation':False,'playback_seconds':seconds,
        'source_submission_fps':played/seconds if seconds else None,'playback_pacing_resets':resets,
        'mean_playback_stage_ms':{k:v/played*1000 for k,v in stage_totals.items()} if played else {},
        'warmup_seconds':state['warmup_seconds'],'inference_cpu_threads':state['inference_threads'],
        'total_wall_seconds':perf_counter()-started,'worker_observations':dict(state['worker_observations']),
        'detection_validation':{'rejected_count':state['rejected_count']},
        'analysis_latency_seconds':{'mean':state['latency_sum']/state['processed'] if state['processed'] else None,
                                    'max':state['max_latency'] if state['processed'] else None},
        'note':'Every source frame is submitted in order at nominal source FPS when decoding/rendering keep up. '
               'Large inputs use a cached resized playback copy prepared alongside AI warm-up; original files remain unchanged. '
               'AI consumes selected latest frames only. The separate scored pane retains its exact frame/time; '
               'no AI boxes are drawn on moving source video. JSONL is sampled evidence, not an all-frame report. '
               'No scored movie export or audio playback. Source timestamps use nominal FPS, not variable-rate PTS.'}
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if error: raise RuntimeError(f'{error}; partial results: {run}')
    return summary
