"""Desktop camera and results screen."""
import argparse
from collections import deque
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog
from types import SimpleNamespace

import cv2

from board import Board
from history import History
from monitor import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='STM32 USB serial device, e.g. /dev/ttyACM0')
    parser.add_argument('--road', nargs=8, type=float,
                        metavar='PX', help='Road corners in the 320 × 240 preview')
    cli = parser.parse_args()
    if cli.port:
        cli.port = cli.port.strip().rstrip(",")
    root = tk.Tk()
    root.title('Road Obstacle Detection')
    root.configure(bg='#fff7ef')
    events = Queue()
    latest = deque(maxlen=1)
    stop = threading.Event()
    worker = None
    camera = tk.StringVar(value='0')
    status = tk.StringVar(value='Ready')
    result = tk.StringVar(value='Alerts will appear here.')
    counts = tk.StringVar(value='0 passages')
    last_frame = None

    def label(parent, **kwargs):
        return tk.Label(parent, bg='#fff7ef', fg='#382016', **kwargs)

    label(root, text='Road Obstacle Detection', font=('Sans', 24, 'bold')).pack(pady=(14, 5))
    label(root, text='STM32 over USB' if cli.port else 'Laptop test mode').pack()
    controls = tk.Frame(root, bg='#fff7ef')
    controls.pack(pady=10)
    label(root, text='Source view                                      Tracking overlay').pack()
    preview = label(root, text='Choose Real traffic replay, Start camera, or Open video.')
    preview.pack(padx=16, pady=8)
    label(root, textvariable=counts).pack()
    label(root, textvariable=result, font=('Sans', 13, 'bold'), wraplength=620).pack(pady=10)
    suggestion = tk.StringVar()
    label(root, textvariable=suggestion, font=('Sans', 12, 'bold')).pack()
    evidence = label(root)
    evidence.pack()
    label(root, textvariable=status, wraplength=620).pack(pady=(0, 12))

    def start(source, replay=None):
        nonlocal worker
        if not source or (worker and worker.is_alive()):
            return
        stop.clear()
        latest.clear()
        result.set('Watching for repeated slowdown or merging.')
        evidence.configure(image='')
        suggestion.set('')
        preview.configure(image='', text='Opening video…')
        gates = vehicle_model = pavement_model = None
        if replay:
            title, road, gates, vehicle_model, pavement_model = replay
            status.set(f'{title} · 1× playback')
        else:
            status.set('Starting…')
            road = cli.road
            vehicle_model = data / 'vehicle-detector.onnx'
        counts.set('0 passages')
        args = SimpleNamespace(source=source, fps=5, threshold=100,
                               speed_breaker=False, headless=True, seconds=None,
                               playback_rate=int(not source.isdecimal()),
                               speed_bump_model=data / "speed-bump.onnx" if replay is speed_breaker else None,
                               tracking_end=15 if replay is blockage else None,
                               obstruction_region=[(61,133),(119,128),(124,151),(57,157)]
                               if replay is blockage else None,
                               road=road, gates=gates, vehicle_model=vehicle_model, pavement_model=pavement_model,
                               output=Path('evidence'), history=History(), source_label=title if replay else source)

        def work():
            board = None
            try:
                if cli.port:
                    board = Board(cli.port)
                    board.evaluate(0, 0, 0, 0, 0, 100, args.speed_breaker)
                args.board = board
                summary = run(args, lambda frame, n: latest.append((frame, n)),
                              events.put, stop.is_set, stop.wait)
                expected = summary['expected_frames']
                read = summary['frames_read']
                ended_early = expected > 0 and read < expected
                outcome = 'Stopped' if stop.is_set() else ('Video ended early' if ended_early else 'Run finished')
                frame_status = f'{read}/{expected} frames read' if expected else f'{read} frames read'
                events.put({'status': f"{outcome} · {summary['completed_passages']} passages · "
                                      f"{summary['slowed_passages']} slowed · {summary['merged_passages']} lane changes · {summary['alerts']} alerts\n"
                                      f"{summary['source_seconds']:g}s read · {frame_status}",
                            'uncalibrated': not summary['lane_calibrated'],
                            'blockage_replay': replay is blockage, 'speed_breaker': summary['speed_bump_detected']})
            except (Exception, SystemExit) as error:
                events.put({'status': str(error), 'failed': True})
            finally:
                if board:
                    board.close()

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

    data = Path(__file__).resolve().parent / 'data'
    replays = tk.Frame(root, bg='#fff7ef')
    replays.pack(before=controls)
    highway = ('Recorded highway traffic', [151, 29, 208, 29, 319, 184, 89, 184], None, None, None)
    pothole = ('Recorded Kumasi pothole footage', [216, 232, 8, 232, 151, 110, 225, 111],
               (.38, .33, .28), data / 'vehicle-detector.onnx', data / 'pothole.onnx')
    tk.Button(replays, text='Real traffic replay',
              command=lambda: start(str(data / 'highway-traffic.mp4'), replay=highway)).pack(side='left', padx=4)
    tk.Button(replays, text='Real pothole replay',
              command=lambda: start(str(data / 'kumasi-road-pothole.mp4'), replay=pothole)).pack(side='left', padx=4)
    blockage = ('Recorded highway obstruction', [198, 224, 103, 224, 162, 91, 207, 91],
                (.33, .21, .09), data / 'vehicle-detector.onnx', None)
    tk.Button(replays, text='Real blockage replay',
              command=lambda: start(str(data / 'highway-obstruction.mp4'), replay=blockage)).pack(side='left', padx=4)
    speed_breaker = ('Recorded speed breaker', [20, 218, 20, 120, 304, 120, 304, 218],
                     (.80, .59, .38), data / 'vehicle-detector.onnx', None)
    tk.Button(replays, text='Real speed breaker replay',
              command=lambda: start(str(data / 'speed-breaker.mp4'), replay=speed_breaker)).pack(side='left', padx=4)
    tk.Entry(controls, width=3, textvariable=camera).pack(side='left')
    tk.Button(controls, text='Start camera', command=lambda: start(camera.get())).pack(side='left', padx=4)
    tk.Button(controls, text='Open video', command=lambda: start(filedialog.askopenfilename())).pack(side='left', padx=4)
    tk.Button(controls, text='Stop', command=stop.set).pack(side='left', padx=4)

    def show(widget, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        widget.photo = tk.PhotoImage(data=f'P6\n{width} {height}\n255\n'.encode() + rgb.tobytes(), format='PPM')
        widget.configure(image=widget.photo, text='')

    def play_urgent():
        player = shutil.which('pw-play') or shutil.which('paplay')
        if player:
            try:
                audio = subprocess.Popen([player, str(data / 'urgent.wav')],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                root.bell()
                return

            def check_audio():
                code = audio.poll()
                if code is None:
                    root.after(100, check_audio)
                elif code != 0:
                    root.bell()

            root.after(100, check_audio)
        else:
            root.bell()

    def poll():
        nonlocal last_frame
        if latest and latest[-1] is not last_frame:
            last_frame = latest[-1]
            frame, number = last_frame
            show(preview, frame)
            counts.set(f'{number} passages')
        while not events.empty():
            event = events.get_nowait()
            if 'status' in event:
                status.set(event['status'])
                if event.get('failed'):
                    preview.configure(text='Unable to start. See the message below.')
                    result.set('Run stopped.')
                elif result.get() == 'Watching for repeated slowdown or merging.':
                    message = 'STM32: No alert raised.' if cli.port else 'No alert raised.'
                    if event.get('uncalibrated'):
                        message = 'Vehicle detection complete. Set lane boundaries to enable lane alerts.'
                    if event.get('speed_breaker'):
                        message += ' Pothole alerts suppressed after visual speed-breaker detection.'
                    if event.get('blockage_replay'):
                        message += ' Urgent review requires 3 merges from the same lane within 20 seconds.'
                    result.set(message)
            else:
                if event['event'] == 'image_suggestion':
                    suggestion.set(event['reason'])
                else:
                    result.set(event['reason'])
                if event['evidence']:
                    image = cv2.imread(event['evidence'])
                    if image is not None:
                        show(evidence, image)
                if event['event'] == 'blockage':
                    play_urgent()
        root.after(200, poll)

    def close():
        stop.set()
        if worker and worker.is_alive():
            root.after(100, close)
        else:
            root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    poll()
    root.mainloop()


if __name__ == '__main__':
    main()
