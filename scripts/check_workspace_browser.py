"""Check the running local app through an existing Chromium DevTools session.

Optional QA utility: requires websockets and a headless Chromium browser launched
with a dedicated profile and --remote-debugging-port. Never attaches to a personal
browser profile. See docs/presentation-guide.md.
"""

import argparse
import base64
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.request import urlopen


class Browser:
    def __init__(self, port):
        from websockets.sync.client import connect
        with urlopen(f'http://127.0.0.1:{port}/json', timeout=5) as response:
            pages = json.load(response)
        page = next(p for p in pages if p['type'] == 'page')
        self.connection = connect(page['webSocketDebuggerUrl'], max_size=30_000_000)
        self.socket = self.connection.__enter__()
        self.identifier = 0
        self.errors = []
        self.call('Runtime.enable')
        self.call('Page.enable')

    def call(self, method, **params):
        self.identifier += 1
        identifier = self.identifier
        self.socket.send(json.dumps({'id': identifier, 'method': method, 'params': params}))
        while True:
            reply = json.loads(self.socket.recv(timeout=30))
            if reply.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(reply['params'])
            if reply.get('id') == identifier:
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                return reply.get('result', {})

    def evaluate(self, expression):
        reply = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
        if 'exceptionDetails' in reply:
            raise RuntimeError(reply['exceptionDetails'])
        return reply.get('result', {}).get('value')

    def wait(self, expression, timeout=45):
        until = monotonic()+timeout
        while monotonic() < until:
            if self.evaluate("document.readyState !== 'loading' && document.body !== null && ("+expression+")"):
                return
            sleep(.15)
        raise RuntimeError('Browser condition timed out: '+expression)

    def click(self, selector, text):
        found = self.evaluate(f"""(() => {{ const el = [...document.querySelectorAll({json.dumps(selector)})]
            .find(x=>x.textContent.trim()==={json.dumps(text)}); if(!el) return null;
            el.scrollIntoView({{block:'center'}}); const r=el.getBoundingClientRect();
            return {{x:r.x+r.width/2,y:r.y+r.height/2}}; }})()""")
        if not found:
            raise RuntimeError(f'No browser control: {text}')
        self.call('Input.dispatchMouseEvent', type='mousePressed', button='left', clickCount=1, **found)
        self.call('Input.dispatchMouseEvent', type='mouseReleased', button='left', clickCount=1, **found)

    def screenshot(self, path):
        data = self.call('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(data['data']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug-port', type=int, default=9234)
    parser.add_argument('--url', default='http://127.0.0.1:8502')
    parser.add_argument('--output', type=Path, default=Path('outputs/presentation-browser-check'))
    parser.add_argument('--video', type=Path, help='Optional short local clip for real UI export/playback QA')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    browser = Browser(args.debug_port)
    checks = []
    report = {'checks': checks, 'status': 'running'}
    try:
        browser.call('Emulation.setDeviceMetricsOverride', width=1440, height=1080, deviceScaleFactor=1, mobile=False)
        browser.call('Page.navigate', url=args.url)
        browser.wait("document.body.innerText.includes('Worksite inspection') && !!document.querySelector('input[type=file]')")
        browser.screenshot(args.output/'01-workspace.png')
        checks.append('Desktop workspace rendered')
        browser.click('[data-testid="stCheckbox"] label', 'Use the included presentation sample')
        browser.wait("[...document.querySelectorAll('button')].some(x=>x.textContent.trim()==='Analyze image'&&!x.disabled)")
        browser.click('button', 'Analyze image')
        browser.wait("document.body.innerText.includes('02 / Inspection results') && !!document.querySelector('[data-testid=stImage] img')", timeout=60)
        browser.wait("[...document.querySelectorAll('[data-testid=stImage] img')].every(x=>x.complete&&x.naturalWidth>0)")
        browser.evaluate("document.querySelector('[data-testid=stMetric]').scrollIntoView({block:'start'})")
        browser.screenshot(args.output/'02-image-analysis.png')
        checks.append('Included sample analyzed with real model; image decoded in browser')
        browser.click('[role=tab]', 'Source comparison')
        browser.wait("[...document.querySelectorAll('[data-testid=stImage] img')].filter(x=>x.getBoundingClientRect().width>0&&x.complete&&x.naturalWidth>0).length>=2")
        checks.append('Original and annotated source comparison images decoded')
        browser.click('[role=tab]', 'Worker evidence')
        browser.wait("document.body.innerText.includes('Filter worker evidence')")
        browser.screenshot(args.output/'03-worker-evidence.png')
        checks.append('Worker evidence tab opened')
        browser.click('[data-testid="stRadio"] label', 'Project guide')
        browser.wait("document.body.innerText.includes('Inside the pipeline')")
        browser.screenshot(args.output/'04-pipeline-guide.png')
        checks.append('Project guide rendered')
        browser.click('[data-testid="stRadio"] label', 'Validation')
        browser.wait("document.body.innerText.includes('90.08%')")
        browser.screenshot(args.output/'05-validation.png')
        checks.append('Saved validation metrics rendered')
        browser.click('[data-testid="stRadio"] label', 'Inspection')
        browser.wait("document.body.innerText.includes('Worksite inspection')")
        browser.click('[data-testid="stRadio"] label', 'Video')
        browser.wait("document.body.innerText.includes('Choose a video')")
        browser.screenshot(args.output/'06-video.png')
        checks.append('Video review controls rendered')
        if args.video:
            browser.click('[data-testid="stRadio"] label', 'Analyze every frame')
            browser.wait("document.body.innerText.includes('Analyze video')")
            document = browser.call('DOM.getDocument')['root']['nodeId']
            node = browser.call('DOM.querySelector', nodeId=document, selector='input[type=file]')['nodeId']
            browser.call('DOM.setFileInputFiles', nodeId=node, files=[str(args.video.resolve())])
            browser.wait("[...document.querySelectorAll('button')].some(x=>x.textContent.trim()==='Analyze video'&&!x.disabled)")
            browser.click('button', 'Analyze video')
            browser.wait("document.querySelector('video')?.readyState>=2", timeout=60)
            browser.evaluate("(async()=>{const v=document.querySelector('video');v.muted=true;await v.play();return true})()")
            browser.wait("document.querySelector('video')?.currentTime>.1")
            playback = browser.evaluate("(()=>{const v=document.querySelector('video');v.pause();return {error:v.error?.message||null,duration:v.duration,currentTime:v.currentTime,decodedFrames:v.getVideoPlaybackQuality().totalVideoFrames}})()")
            if playback['error'] or playback['decodedFrames'] < 1:
                raise RuntimeError('Browser failed to decode the completed video')
            report['video_playback'] = playback
            browser.evaluate("document.querySelector('video').scrollIntoView({block:'center'})")
            browser.screenshot(args.output/'06b-video-playback.png')
            checks.append('Short clip uploaded, analyzed, exported and played in the actual browser')
        browser.click('[data-testid="stRadio"] label', 'Live camera')
        browser.wait("document.body.innerText.includes('Start live camera')")
        browser.screenshot(args.output/'07-camera.png')
        checks.append('Camera controls rendered; hardware not opened')
        browser.call('Emulation.setDeviceMetricsOverride', width=390, height=844, deviceScaleFactor=1, mobile=True)
        browser.wait("document.querySelector('h1') !== null")
        browser.screenshot(args.output/'08-mobile.png')
        overflow = browser.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2')
        if overflow:
            raise RuntimeError('Mobile viewport has horizontal overflow')
        checks.append('390px viewport has no page overflow')
        if browser.evaluate("document.querySelectorAll('[data-testid=stException]').length") or browser.errors:
            raise RuntimeError('Browser or application exception detected')
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['browser_exceptions'] = browser.errors
        (args.output/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.connection.__exit__(None, None, None)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
