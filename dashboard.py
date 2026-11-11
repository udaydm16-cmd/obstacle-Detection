"""Local browser dashboard for Road Obstacle Detection."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock, Thread
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2

from board import Board
from history import History, site_key
from monitor import run

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
REPLAYS = {
    'traffic': ('Highway traffic', 'highway-traffic.mp4', [151,29,208,29,319,184,89,184], None),
    'pothole': ('Kumasi pothole', 'kumasi-road-pothole.mp4', [216,232,8,232,151,110,225,111], (.38,.33,.28)),
    'blockage': ('Highway blockage', 'highway-obstruction.mp4', [198,224,103,224,162,91,207,91], (.33,.21,.09)),
    'bump': ('Speed breaker', 'speed-breaker.mp4', [20,218,20,120,304,120,304,218], (.80,.59,.38)),
}
MAX_VIDEO = 250 * 1024 * 1024


class Dashboard:
    def __init__(self, settings):
        self.settings = settings
        self.history = History()
        self.settings.road = settings.road or self.history.calibration(settings.camera)
        self.lock = Lock()
        self.stop = Event()
        self.frame = None
        self.revision = 0
        self.run_id = 0
        self.overlay = False
        self.state = dict(running=False, source='No source selected', message='Choose a video or start the camera.',
                          mode=None, board='USB idle' if settings.port else 'Laptop mode', passages=0, frames=None,
                          site=site_key(str(settings.camera), settings.road) if settings.road else None)

    def snapshot(self):
        with self.lock:
            state = dict(self.state, revision=self.revision, run_id=self.run_id, corners=self.settings.road)
        state['counts'] = self.history.counts(state['site'])
        return state

    def save_calibration(self, corners):
        if (not isinstance(corners, list) or len(corners) != 8 or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= (319 if index%2 == 0 else 239)
                    for index,value in enumerate(corners))):
            raise ValueError('Mark four corners inside the camera image.')
        polygon = [(corners[index],corners[index+1]) for index in range(0,8,2)]
        turns = []
        for i in range(4):
            a,b,c = polygon[i], polygon[(i+1)%4], polygon[(i+2)%4]
            turns.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        if not (min(turns)>50 or max(turns)<-50):
            raise ValueError('The corners must form a clear, non-crossing road outline.')
        with self.lock:
            if not self.state['running'] or self.state['mode'] != 'calibrate' or not self.frame:
                raise ValueError('Open the camera preview before marking lanes.')
            self.settings.road = self.history.calibration(self.settings.camera, corners)
            self.state.update(site=site_key(str(self.settings.camera), corners))
            self.stop.set()

    def start(self, key, upload=None):
        if key not in REPLAYS and key not in ('camera','auto','calibrate','upload'):
            raise ValueError('Choose a video or camera.')
        with self.lock:
            if self.state['running']:
                raise ValueError('Stop the current run before starting another.')
            if key == 'camera' and not self.settings.road:
                key = 'auto'
            self.stop.clear()
            self.frame = None
            self.overlay = False
            self.run_id += 1
            self.revision += 1
            name, filename, road, gates = REPLAYS.get(
                key, ('Lane setup' if key=='calibrate' else 'Live camera' if key in ('camera','auto') else 'Uploaded video', None,
                      self.settings.road if key=='camera' else None, None))
            source = str(DATA / filename) if filename else (str(self.settings.camera) if key in ('camera','auto','calibrate') else str(upload))
            self.state.update(running=True, mode=key, source=name, message='Opening source…', passages=0, frames=None,
                              board='Connecting over USB' if self.settings.port else 'Laptop mode')
        args = SimpleNamespace(source=source, source_label=name, fps=5, threshold=100,
                               speed_breaker=False, headless=True, seconds=None,
                               playback_rate=int(key not in ('camera','auto','calibrate')), road=road, gates=gates, history=self.history,
                               vehicle_model=None if key in ('traffic','calibrate') else DATA/'vehicle-detector.onnx',
                               pavement_model=DATA/'pothole.onnx' if key=='pothole' else None,
                               speed_bump_model=DATA/'speed-bump.onnx' if key=='bump' else None,
                               tracking_end=15 if key=='blockage' else None,
                               obstruction_region=[(61,133),(119,128),(124,151),(57,157)] if key=='blockage' else None,
                               mode=key, on_source=self.camera_preview if key=='calibrate' else None,
                               on_progress=self.progress if key!='calibrate' else None,
                               preview_enabled=lambda: self.overlay,
                               output=ROOT/'evidence')
        self.worker = Thread(target=self.work, args=(args, upload), daemon=True)
        self.worker.start()

    def progress(self, passages, frames):
        with self.lock:
            self.state.update(passages=passages, frames=f'{frames:,}')

    def camera_preview(self, frame):
        h,w = frame.shape[:2]
        scale = min(1, 640/max(h,w))
        self.preview(cv2.resize(frame,(round(w*scale),round(h*scale))),0)

    def preview(self, frame, passages):
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self.lock:
            self.state['passages'] = passages
            if ok:
                self.frame = jpg.tobytes()
                self.revision += 1

    def work(self, args, upload):
        board = None
        try:
            if args.mode == 'auto':
                from auto_lanes import calibrate_camera
                with self.lock:
                    self.state['message'] = 'Looking for three lane markings'
                corners = calibrate_camera(self.settings.camera, self.stop)
                if self.stop.is_set():
                    with self.lock:
                        self.state['message'] = 'Stopped'
                    return
                if corners is None:
                    with self.lock:
                        self.state['message'] = 'Automatic marking needs a clearer view of three lane markings.'
                    return
                with self.lock:
                    self.settings.road = self.history.calibration(self.settings.camera, corners)
                    self.state.update(mode='camera', site=site_key(str(self.settings.camera), corners))
                args.road, args.mode = corners, 'camera'
            if self.settings.port and args.mode != 'calibrate':
                board = Board(self.settings.port)
                board.evaluate(0,0,0,0,0,100,False)
            args.board = board
            with self.lock:
                self.state.update(board='STM32 connected' if board else ('USB idle' if self.settings.port else 'Laptop mode'),
                                  message='Mark the four road corners' if args.mode=='calibrate' else 'Monitoring in the background' if args.mode=='camera' else 'Analysing footage')
            result = run(args, self.preview if args.mode not in ('camera','calibrate') else None,
                         should_stop=self.stop.is_set, wait=self.stop.wait)
            expected, read = result['expected_frames'], result['frames_read']
            with self.lock:
                self.state.update(frames=f'{read:,}/{expected:,}' if expected else f'{read:,}',
                                  message=('Lanes saved. Start monitoring when ready.' if args.mode=='calibrate' and self.settings.road else 'Stopped') if self.stop.is_set() else
                                  ('Video ended early' if expected and read < expected else 'Run complete'))
        except (Exception, SystemExit) as error:
            with self.lock:
                self.state.update(message=str(error), board='USB check needed' if not board and self.settings.port else self.state['board'])
        finally:
            if board:
                board.close()
            if upload:
                upload.unlink(missing_ok=True)
            with self.lock:
                self.state['running'] = False


def handler_for(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_data(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def json(self, value, status=200):
            self.send_data(json.dumps(value).encode(), 'application/json', status)

        def valid_host(self):
            host = self.headers.get('Host', '').split(':',1)[0]
            if host in ('127.0.0.1', 'localhost'):
                return True
            self.send_error(403)
            return False

        def do_GET(self):
            if not self.valid_host():
                return
            path = urlsplit(self.path).path
            assets = {'/': ('index.html','text/html; charset=utf-8'),
                      '/style.css': ('style.css','text/css'),
                      '/dashboard.js': ('dashboard.js','text/javascript')}
            if path in assets:
                filename, content = assets[path]
                self.send_data((ROOT/'web'/filename).read_bytes(),content)
            elif path.startswith('/media/'):
                key = path.removeprefix('/media/')
                if key not in REPLAYS:
                    self.send_error(404)
                    return
                name = REPLAYS[key][1]
                compact = DATA/'preview'/name
                self.send_video(compact if compact.is_file() else DATA/name)
            elif path == '/api/status':
                self.json(dashboard.snapshot())
            elif path == '/api/events':
                self.json([{**event, 'evidence': '/evidence/'+Path(event['evidence']).name if event['evidence'] else None}
                           for event in dashboard.history.events()])
            elif path == '/frame.jpg' and dashboard.frame:
                self.send_data(dashboard.frame or b'','image/jpeg')
            elif path == '/urgent.wav':
                self.send_data((DATA/'urgent.wav').read_bytes(),'audio/wav')
            elif path.startswith('/evidence/'):
                file = ROOT/'evidence'/Path(path).name
                if file.suffix == '.jpg' and file.is_file():
                    self.send_data(file.read_bytes(),'image/jpeg')
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def send_video(self, path):
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            first, last = 0, size-1
            header = self.headers.get('Range')
            if header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
                if match and any(match.groups()):
                    start, end = match.groups()
                    if start:
                        first = int(start)
                        last = min(last,int(end)) if end else last
                    else:
                        first = max(0,size-int(end))
                if not match or not any(match.groups()) or first > last:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            self.send_response(206 if header else 200)
            self.send_header('Content-Type','video/mp4')
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','private, max-age=300')
            if header:
                self.send_header('Content-Range',f'bytes {first}-{last}/{size}')
            self.send_header('Content-Length',str(last-first+1))
            self.end_headers()
            try:
                with path.open('rb') as file:
                    file.seek(first)
                    remaining = last-first+1
                    while remaining:
                        chunk = file.read(min(remaining,128*1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self.valid_host():
                return
            if self.headers.get('X-Road-Dashboard') != '1':
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            upload = None
            try:
                if path == '/api/stop':
                    dashboard.stop.set()
                elif path in ('/api/overlay/on','/api/overlay/off'):
                    with dashboard.lock:
                        dashboard.overlay = path.endswith('/on')
                elif path == '/api/calibration':
                    length = int(self.headers.get('Content-Length','0'))
                    if not 0 < length <= 2048:
                        raise ValueError('Invalid lane calibration.')
                    dashboard.save_calibration(json.loads(self.rfile.read(length)))
                elif path == '/api/upload':
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= MAX_VIDEO:
                        raise ValueError('Choose a video up to 250 MB.')
                    with NamedTemporaryFile(suffix='.video', delete=False) as file:
                        upload = Path(file.name)
                        remaining = length
                        while remaining:
                            block = self.rfile.read(min(remaining, 1024*1024))
                            if not block:
                                raise ValueError('Upload ended early.')
                            file.write(block)
                            remaining -= len(block)
                    dashboard.start('upload', upload)
                    upload = None
                elif path.startswith('/api/start/'):
                    dashboard.start(path.rsplit('/',1)[1])
                else:
                    self.send_error(404)
                    return
                self.json({'ok':True})
            except (ValueError,OSError) as error:
                self.json({'error':str(error)},400)
            finally:
                if upload:
                    upload.unlink(missing_ok=True)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='USB device, e.g. /dev/ttyACM0')
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--road', type=float, nargs=8, help='Live lane corners in the 320 × 240 reference view')
    settings = parser.parse_args()
    if settings.port:
        settings.port = settings.port.strip().rstrip(',')
    dashboard = Dashboard(settings)
    with ThreadingHTTPServer(('127.0.0.1',settings.http_port),handler_for(dashboard)) as server:
        print(f'Road Obstacle Detection · http://127.0.0.1:{settings.http_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            dashboard.stop.set()
            if hasattr(dashboard, 'worker'):
                dashboard.worker.join(timeout=5)
