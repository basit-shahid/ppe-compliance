"""Video-first native monitor with cached typography and resized image panes."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ppe_compliance.rendering import worker_scores


def _color(value):
    return tuple(int(value[i:i+2], 16) for i in (4, 2, 0))


BG = _color('10151D')
PANEL = _color('191F29')
LINE = _color('2B3340')
TEXT = _color('EBEFF5')
MUTED = _color('9DAABA')
BLUE = _color('76B8EE')
GREEN = _color('62CE8C')
AMBER = _color('EFC16A')
RED = _color('F08484')
LARGE_PANE, SMALL_PANE = (24,126,944,532), (992,126,264,150)


@lru_cache(maxsize=32)
def _font(size, bold=False):
    paths = [Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
             Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    # Colab does not always include either platform font. Pillow's scalable
    # fallback must receive the requested size, otherwise every label is tiny.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 9/10.0 compatibility.
        return ImageFont.load_default()


@lru_cache(maxsize=512)
def _label(value, size, bold, color):
    """Cache small text tiles instead of converting the entire dashboard in PIL."""
    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(value, anchor='ls')
    surface = Image.new('RGBA', (max(1, right-left), max(1, bottom-top)))
    ImageDraw.Draw(surface).text((-left, -top), value, font=font, fill=(*color, 255), anchor='ls')
    pixels = np.asarray(surface)
    return pixels[:, :, :3].astype(np.uint16), pixels[:, :, 3:4].astype(np.uint16), left, top


def visible_workers(result, fresh):
    return result.assessment['workers'] if fresh and result is not None else []


class DashboardRenderer:
    """Rebuild text only on evidence changes; resize each source sequence once."""

    def __init__(self):
        self.key = None
        self.chrome = None
        self.build_count = 0
        self.panes = {}

    def _chrome(self, frame, result, fresh, stats, focus):
        canvas = np.full((720, 1280, 3), BG, np.uint8)
        labels = []

        def text(value, x, y, size=16, color=TEXT, bold=False, limit=None):
            value = str(value)
            font = _font(size, bold)
            if limit:
                while value and font.getlength(value) > limit:
                    value = value[:-4] + '...' if not value.endswith('...') else value[:-4] + '...'
                    if value == '...':
                        break
            labels.append((value, x, y, size, bold, color))

        workers = visible_workers(result, fresh)
        sampled = stats.get('mode') == 'recorded_live'
        recorded = sampled or stats.get('mode') == 'recorded'
        scores = stats.get('show_scores', True)
        text('PPE Monitor', 24, 37, 25, TEXT, True)
        text('Original video + selected AI frames' if sampled else ('Recorded video review' if recorded else 'Visible equipment assessment'), 24, 62, 14, MUTED)
        text('PLAYBACK FPS' if recorded else 'LIVE FPS', 650, 27, 11, MUTED)
        text('AI / S' if sampled else ('PROCESS / S' if recorded else 'AI / S'), 800, 27, 11, MUTED)
        text('AI FRAME TIME' if sampled else ('VIDEO TIME' if recorded else 'RESULT AGE'), 930, 27, 11, MUTED)
        text('SOURCE', 1100, 27, 11, MUTED)
        text(f'{frame.frame.shape[1]} x {frame.frame.shape[0]}', 1100, 54, 17)
        cv2.line(canvas, (24, 82), (1256, 82), LINE, 1)
        large_ai = focus == 'analyzed'
        source_label = 'SOURCE VIDEO' if sampled else ('SOURCE FRAME' if recorded else 'LIVE CAMERA')
        text('ANALYZED FRAME' if large_ai else source_label, 24, 111, 13, BLUE, True)
        text(source_label if large_ai else 'LATEST ANALYSIS', 992, 111, 12, MUTED, True)
        cv2.rectangle(canvas, (24, 126), (968, 658), (8, 11, 16), -1)
        cv2.rectangle(canvas, (992, 126), (1256, 276), (8, 11, 16), -1)
        if not fresh:
            x, y = (270, 380) if large_ai else (1006, 196)
            text(('Preparing review...' if sampled else 'Preparing AI...') if stats.get('loading') else 'Awaiting fresh result', x, y,
                 20 if large_ai else 15, MUTED)
        text(f'Workers  {len(workers) if fresh else "--"}', 992, 335, 20, TEXT, True)
        if not fresh or not workers:
            text('No current decisions' if not fresh else 'No workers detected', 992, 372, 14, MUTED)
        for index, worker in enumerate(workers[:3]):
            y = 375 + index*76
            color = {'PPE detected': GREEN, 'review required': AMBER, 'potential violation': RED}[worker['status']]
            person, helmet, vest = worker_scores(worker)
            text(worker['worker_id'] + (f'  {person}' if scores else ''), 992, y, 14, TEXT, True, limit=103)
            status = {'PPE detected': 'PPE detected', 'review required': 'Review', 'potential violation': 'Check'}[worker['status']]
            text(status, 1100, y, 14, color, True, 150)
            text(f"Helmet: {worker['helmet']['state']}" + (f'  {helmet}' if scores else ''), 992, y+23, 13, MUTED, limit=262)
            text(f"Vest: {worker['vest']['state']}" + (f'  {vest}' if scores else ''), 992, y+43, 13, MUTED, limit=262)
            cv2.line(canvas, (992, y+57), (1256, y+57), LINE, 1)
        if len(workers) > 3:
            text(f'+ {len(workers)-3} in the report', 992, 608, 13, MUTED)
        events = list(getattr(result, 'recent_events', ())) if result else []
        if events:
            event = events[-1]
            text('LAST EVENT (HISTORY)', 992, 641, 10, MUTED)
            text(f"{event['track_id']}  {event.get('status', event['type'].replace('_', ' '))}",
                 992, 662, 12, MUTED, limit=260)
        text('TAB  Switch view   |   P: person   H: helmet   V: vest   |   Selected-frame scores' if sampled else
             'P: person   H: helmet   V: vest   |   Scores: model confidence' if recorded else
             'TAB  Switch view     F  Full screen     Q / ESC  Stop', 24, 699, 13, MUTED)
        text('Sampled AI review' if sampled else ('Recorded analysis' if recorded else ('Recording analysis' if stats.get('recording') else 'Preview only')), 992, 699, 13, BLUE)
        for value, x, y, size, bold, color in labels:
            pixels, alpha, left, top = _label(value, size, bold, color)
            x, y = x+left, y+top
            x0, y0, x1, y1 = max(0, x), max(0, y), min(1280, x+pixels.shape[1]), min(720, y+pixels.shape[0])
            if x1 > x0 and y1 > y0:
                tile = pixels[y0-y:y1-y, x0-x:x1-x]
                mask = alpha[y0-y:y1-y, x0-x:x1-x]
                region = canvas[y0:y1, x0:x1]
                region[:] = (tile*mask + region.astype(np.uint16)*(255-mask) + 127)//255
        self.chrome = canvas
        self.build_count += 1

    def _put(self, view, image, sequence, name, rect, fast=False):
        x, y, w, h = rect
        key = (sequence, image.shape, w, h,fast)
        cached = self.panes.get(name)
        if cached is None or cached[0] != key:
            ih, iw = image.shape[:2]
            scale = min(w/iw, h/ih)
            nw, nh = max(1, round(iw*scale)), max(1, round(ih*scale))
            # Linear resampling is much cheaper for every live frame. Analysis
            # uses area reduction and is cached until the next AI result.
            interpolation = cv2.INTER_LINEAR if fast or name == 'live' or scale >= 1 else cv2.INTER_AREA
            pixels = cv2.resize(image, (nw, nh), interpolation=interpolation)
            cached = (key, pixels)
            self.panes[name] = cached
        pixels = cached[1]
        ph, pw = pixels.shape[:2]
        left, top = x+(w-pw)//2, y+(h-ph)//2
        view[top:top+ph, left:left+pw] = pixels

    def update_source(self, canvas, frame, focus='live'):
        """Update only the moving video between slower dashboard refreshes."""
        self._put(canvas,frame.frame,frame.sequence,'live',SMALL_PANE if focus=='analyzed' else LARGE_PANE)

    def render(self, frame, result, fresh, now, stats, focus='live', width=1280):
        workers = visible_workers(result, fresh)
        events = list(getattr(result, 'recent_events', ())) if result else []
        key = (fresh, focus, stats.get('mode'), stats.get('show_scores', True), stats.get('loading'), stats.get('recording'), frame.frame.shape,
               len(workers), tuple((w['worker_id'], w['status'], w['helmet']['state'], w['vest']['state'],
                                   worker_scores(w) if stats.get('show_scores', True) else ()) for w in workers[:3]),
               events[-1].get('event_id') if events else None)
        if key != self.key:
            self._chrome(frame, result, fresh, stats, focus)
            self.key = key
        view = self.chrome.copy()
        large, small = LARGE_PANE, SMALL_PANE
        self.update_source(view,frame,focus)
        if fresh:
            self._put(view, result.annotated, result.source.sequence, 'analysis', large if focus == 'analyzed' else small,
                      fast=stats.get('mode')=='recorded_live')
        age = now-result.source.received_at if result is not None else None
        sampled = stats.get('mode') == 'recorded_live'
        recorded = sampled or stats.get('mode') == 'recorded'
        timestamp = stats.get('source_time_seconds')
        time_label = f'{timestamp:.2f} s' if timestamp is not None else '--'
        analysis_time = stats.get('analysis_time_seconds')
        analysis_label = f'{analysis_time:.2f} s' if analysis_time is not None else '--'
        for value, x, color in ((f"{stats.get('playback_fps' if recorded and not sampled else 'live_rate', 0):.1f}", 650, TEXT),
                                (f"{stats.get('ai_rate', 0):.1f}", 800, TEXT),
                                (analysis_label if sampled else (time_label if recorded else (f'{age*1000:.0f} ms' if age is not None else '--')), 930, BLUE if fresh else AMBER)):
            cv2.putText(view, value, (x, 54), cv2.FONT_HERSHEY_SIMPLEX, .56, color, 1, cv2.LINE_AA)
        note = f'Analyzed #{result.source.sequence}' if focus == 'analyzed' and result else f'Live #{frame.sequence}'
        if recorded:
            note = f'Frame {frame.sequence+1} | Source time {time_label} | Visible PPE evidence only'
        if sampled:
            note = f'Source {time_label} | Analyzed frame at {analysis_label} | AI scores refer to the analyzed pane only'
            if stats.get('loading'):
                note = 'Preparing review | '+stats.get('preparation_status','Model warming up')
        if focus == 'analyzed' and not fresh:
            note = 'No recent analysis; old decisions hidden'
        cv2.putText(view, note, (24, 679), cv2.FONT_HERSHEY_SIMPLEX, .37, MUTED, 1, cv2.LINE_AA)
        if width != 1280:
            view = cv2.resize(view, (width, round(720*width/1280)), interpolation=cv2.INTER_AREA if width < 1280 else cv2.INTER_LINEAR)
        return view


def dashboard(frame, result, fresh, now, stats, focus='live', width=1280):
    return DashboardRenderer().render(frame, result, fresh, now, stats, focus, width)
