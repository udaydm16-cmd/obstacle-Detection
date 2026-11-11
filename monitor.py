"""Low-resolution, two-lane road behavior prototype."""
import argparse
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class Passage:
    gates: list = field(default_factory=list)


class RoadMonitor:
    # Compare equal lengths immediately before and within the monitored segment.
    gates = (.80, .58, .36)

    def __init__(self, threshold=100, speed_breaker=False, board=None, gates=None, history=None, site=None):
        self.board = board
        if gates is not None:
            self.gates = gates
        self.threshold = threshold
        self.speed_breaker = speed_breaker
        self.passages = {}
        self.recent = deque()
        self.alerts = set()
        self.completed = 0
        self.slowed = self.merged = 0
        self.history, self.site = history, site
        self.saved_alerts = {}
        self.clock_start = time.time()
        if history and site:
            rows, self.saved_alerts = history.restore(site, self.clock_start)
            self.recent.extend(rows)

    def observe(self, track_id, previous, current, timestamp):
        if previous is None or current[1] >= previous[1]:
            return []
        passage = self.passages.setdefault(track_id, Passage())
        for index, gate in enumerate(self.gates):
            if previous[1] >= gate > current[1] and len(passage.gates) == index:
                passage.gates.append((timestamp, int(current[0] >= .5)))
        if len(passage.gates) != 3:
            return []
        times, lanes = zip(*passage.gates)
        # Mark this passage complete even if the tracker reports it again.
        passage.gates.append(None)
        self.completed += 1
        before, inside = times[1]-times[0], times[2]-times[1]
        slow = before > 0 and inside >= 1.6 * before
        merged = lanes[1] != lanes[2]
        self.slowed += int(slow and not merged)
        self.merged += int(merged)
        if self.history and self.site:
            timestamp = time.time()
        record = (timestamp, lanes[1], slow, merged)
        if self.history and self.site:
            self.history.passage(self.site, record)
        self.recent.append(record)
        while self.recent and timestamp - self.recent[0][0] > 72 * 3600:
            self.recent.popleft()
        return [(kind, message, event_lane) for lane in (0, 1)
                for kind, message, event_lane in self.evaluate(timestamp, lane)]

    def evaluate(self, now, lane):
        merges = sum(merged for ts, origin, _, merged in self.recent
                     if now-ts <= 20 and origin == lane)
        target = [slow for _, origin, slow, merged in self.recent if origin == lane and not merged]
        other = [slow for _, origin, slow, merged in self.recent if origin != lane and not merged]
        slowed = sum(target)
        other_slowed = sum(other)
        if self.board:
            flags = self.board.evaluate(len(target), slowed, len(other), other_slowed,
                                        merges, self.threshold, self.speed_breaker)
        else:
            flags = int(merges >= 3)
            if (not self.speed_breaker and slowed >= self.threshold and len(other) >= 3
                    and 5*slowed >= 3*len(target) and 4*other_slowed <= len(other)):
                flags |= 2
        alerts = []
        if flags & 1 and ('blockage', lane) not in self.alerts:
            alerts.append(('blockage', f'Urgent lane blockage review: {merges} vehicles merged out of lane {lane+1}', lane))
        last_pothole = self.saved_alerts.get(lane, float('-inf'))
        if flags & 2 and now-last_pothole > 72*3600:
            self.saved_alerts[lane] = now
            alerts.append(('pothole', f'Possible pothole: {slowed} of {len(target)} passages slowed in lane {lane+1}', lane))
        self.alerts.update((kind, lane) for kind, _, _ in alerts)
        return alerts


def road_positions(frame, subtractor):
    mask = subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    height, width = frame.shape[:2]
    return [(x/width, y/height) for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if 140 <= cv2.contourArea(contour) <= width*height*.12
            for bx, by, bw, bh in [cv2.boundingRect(contour)]
            for x, y in [(bx+bw/2, by+bh)]]


def evidence_frame(image):
    height, width = image.shape[:2]
    scale = min(320/width, 240/height)
    return cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))),
                      interpolation=cv2.INTER_AREA)


def run(args, on_frame=None, on_result=None, should_stop=lambda: False, wait=time.sleep):
    cv2.setNumThreads(2)
    camera = args.source.isdecimal()
    capture = cv2.VideoCapture(int(args.source) if camera else args.source)
    if not capture.isOpened():
        raise SystemExit(f'Cannot open source {args.source}. Check the camera index or video path.')
    width, height = 320, 240
    if camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps/args.fps))
    road_configured = args.road is not None
    road_corners = args.road if road_configured else [0, 0, 319, 0, 319, 239, 0, 239]
    points = np.float32(road_corners).reshape(4, 2)
    transform = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]))
    inverse = np.linalg.inv(transform)

    def camera_points(coords):
        projected = cv2.perspectiveTransform(np.float32(coords).reshape(-1, 1, 2), inverse)
        return np.rint(projected.reshape(-1, 2)).astype(np.int32)

    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
    detector = None
    if getattr(args, 'vehicle_model', None):
        from vehicle_detector import VehicleDetector
        detector = VehicleDetector(args.vehicle_model)
    history = getattr(args, 'history', None)
    site = None
    if history and camera and road_configured:
        from history import site_key
        site = site_key(args.source, args.road, getattr(args, 'gates', None))
    monitor = RoadMonitor(args.threshold, args.speed_breaker, getattr(args, 'board', None),
                          getattr(args, 'gates', None), history, site)

    def publish(event):
        if history:
            history.event(getattr(args, 'source_label', args.source), site, event)
        print(json.dumps(event), flush=True)
        if on_result:
            on_result(event)
    tracks = {}
    next_id = frames = scans = evidence_bytes = 0
    start = time.monotonic()
    latest = 'Watching two lanes' if road_configured else 'Full-frame vehicle detection'
    tracking_end = getattr(args, 'tracking_end', None)
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not camera else 0
    last_vehicle_count = 0
    playback_rate = getattr(args, 'playback_rate', 0)
    alert_view = None
    pavement = None
    pavement_scans = 0
    image_suggestions = 0
    next_image_check = None
    checks_left = 0
    obstruction = None
    if detector and getattr(args, 'obstruction_region', None):
        from obstruction import ObstructionCheck
        obstruction = ObstructionCheck(detector, args.obstruction_region)
    visual_alerted = False
    speed_bump_weights = getattr(args, 'speed_bump_model', None)
    speed_bump_check = None
    speed_bump_detected = False
    speed_bump_scans = 0
    next_speed_scan = 0
    pavement_weights = getattr(args, 'pavement_model', None)
    on_source = getattr(args, 'on_source', None)
    on_progress = getattr(args, 'on_progress', None)
    preview_enabled = getattr(args, 'preview_enabled', lambda: True)
    try:
        while not should_stop() and capture.grab():
            frames += 1
            if frames % stride:
                continue
            ok, image = capture.retrieve()
            if not ok:
                break
            now = time.monotonic()-start if camera else (frames-1)/fps
            if playback_rate and not camera:
                wait(max(0, start + now/playback_rate - time.monotonic()))
                if should_stop():
                    break
            if on_source:
                on_source(image)
            alert_this_frame = False
            preview = cv2.resize(image, (width, height))
            tracking = road_configured and (tracking_end is None or now < tracking_end)
            scans += 1
            boxes = detector.boxes(image if not road_configured else preview) if detector else []
            last_vehicle_count = len(boxes)
            if (tracking and speed_bump_weights and not monitor.speed_breaker
                    and speed_bump_scans < 14 and now + 1e-6 >= next_speed_scan):
                if speed_bump_check is None:
                    from speed_bump import SpeedBumpCheck
                    speed_bump_check = SpeedBumpCheck(speed_bump_weights)
                speed_bump_scans += 1
                next_speed_scan = now + 1
                box = speed_bump_check.scan(image)
                if box:
                    if monitor.board:
                        monitor.board.evaluate(0, 0, 0, 0, 0, monitor.threshold, True)
                    monitor.speed_breaker = speed_bump_detected = True
                    speed_bump_check = None
                    evidence = evidence_frame(image)
                    eh, ew = evidence.shape[:2]
                    x1,y1,x2,y2 = (round(v*k) for v,k in zip(box, (ew/320, eh/180, ew/320, eh/180)))
                    cv2.rectangle(evidence, (x1,y1), (x2,y2), (0,100,255), 2)
                    args.output.mkdir(parents=True, exist_ok=True)
                    path = args.output / f'speed-bump-{time.time_ns()}.jpg'
                    saved = cv2.imwrite(str(path), evidence, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if saved:
                        evidence_bytes += path.stat().st_size
                    event = dict(event='image_suggestion', source_seconds=round(now, 2),
                                 reason='AI suggestion: Possible speed breaker visible',
                                 evidence=str(path) if saved else None)
                    publish(event)
            observations = []
            if tracking:
                if detector:
                    observations = detector.positions(boxes, transform, width, height)
                else:
                    road = cv2.warpPerspective(preview, transform, (width, height))
                    observations = road_positions(road, subtractor)
            if scans < 8:
                observations = []
            tracks = {tid: (position, timestamp) for tid, (position, timestamp) in tracks.items() if now-timestamp <= .8}
            monitor.passages = {tid: state for tid, state in monitor.passages.items() if tid in tracks}
            unused = set(tracks)
            visible_positions = []
            for position in observations:
                x, y = position
                if not .03 < x < .97:
                    continue
                distances = [(np.hypot(x-tracks[tid][0][0], y-tracks[tid][0][1]), tid) for tid in unused]
                distance, tid = min(distances, default=(float('inf'), None))
                if distance <= (.17 if detector else .13):
                    previous = tracks[tid][0]
                    unused.remove(tid)
                else:
                    next_id += 1
                    tid, previous = next_id, None
                tracks[tid] = (position, now)
                completed_before = monitor.completed
                decisions = monitor.observe(tid, previous, position, now)
                if (pavement_weights and not monitor.speed_breaker and not image_suggestions
                        and not checks_left and monitor.completed != completed_before):
                    _, _, slowed, merged = monitor.recent[-1]
                    if slowed and not merged:
                        checks_left = 12
                        next_image_check = now + .2
                        if pavement is not None:
                            pavement.previous.clear()
                for kind, latest, lane in decisions:
                    alert_this_frame = True
                    args.output.mkdir(parents=True, exist_ok=True)
                    path = args.output / f'{kind}-{time.time_ns()}.jpg'
                    if cv2.imwrite(str(path), evidence_frame(image), [cv2.IMWRITE_JPEG_QUALITY, 60]):
                        evidence_bytes += path.stat().st_size
                    event = dict(event=kind, lane=lane, source_seconds=round(now, 2), reason=latest,
                                 evidence=str(path) if path.is_file() else None)
                    publish(event)
                visible_positions.append((x*width, y*height))
            if tracking and obstruction and not visual_alerted:
                box = obstruction.scan(image, now)
                if box:
                    flags = (monitor.board.evaluate(0, 0, 0, 0, 0, 100, False, visual_blockage=True)
                             if monitor.board else 1)
                    if flags & 1:
                        visual_alerted = True
                        alert_this_frame = True
                        latest = 'Urgent lane blockage review: vehicle moving sideways in an active lane'
                        evidence = evidence_frame(image)
                        eh, ew = evidence.shape[:2]
                        x1,y1,x2,y2 = (round(v*k) for v,k in zip(box, (ew/width, eh/height, ew/width, eh/height)))
                        cv2.rectangle(evidence, (x1,y1), (x2,y2), (0,100,255), 2)
                        args.output.mkdir(parents=True, exist_ok=True)
                        path = args.output / f'visual-blockage-{time.time_ns()}.jpg'
                        saved = cv2.imwrite(str(path), evidence, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if saved:
                            evidence_bytes += path.stat().st_size
                        event = dict(event='blockage', source_seconds=round(now, 2), reason=latest,
                                     evidence=str(path) if saved else None, basis='visual')
                        publish(event)
            if tracking and checks_left and now + 1e-6 >= next_image_check:
                if pavement is None:
                    from pavement import PavementCheck
                    pavement = PavementCheck(pavement_weights, args.road)
                pavement_scans += 1
                checks_left -= 1
                next_image_check = now + .4
                candidate = pavement.scan(preview)
                if candidate:
                    _, box = candidate
                    evidence = evidence_frame(image)
                    eh, ew = evidence.shape[:2]
                    x1, y1, x2, y2 = (round(value*scale) for value, scale in
                                      zip(box, (ew/width, eh/height, ew/width, eh/height)))
                    cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 100, 255), 2)
                    args.output.mkdir(parents=True, exist_ok=True)
                    path = args.output / f'pavement-{time.time_ns()}.jpg'
                    saved = cv2.imwrite(str(path), evidence, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if saved:
                        evidence_bytes += path.stat().st_size
                    event = dict(event='image_suggestion', source_seconds=round(now, 2),
                                 reason='AI suggestion: Possible pothole visible',
                                 evidence=str(path) if saved else None)
                    publish(event)
                    image_suggestions += 1
                    checks_left = 0
            if on_progress:
                on_progress(monitor.completed, frames)
            if not args.headless or (on_frame and preview_enabled()):
                source_h, source_w = image.shape[:2]
                panel_w, panel_h = 480, 300
                scale = min(panel_w/source_w, panel_h/source_h)
                display_w, display_h = max(1, round(source_w*scale)), max(1, round(source_h*scale))
                source_view = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
                overlay = source_view.copy()
                display_scale = np.float32([display_w/width, display_h/height])

                def display_points(coords):
                    return np.rint(camera_points(coords)*display_scale).astype(np.int32)

                for x1,y1,x2,y2 in boxes:
                    cv2.rectangle(overlay, (round(x1*display_w), round(y1*display_h)),
                                  (round(x2*display_w), round(y2*display_h)), (255, 190, 40), 1)
                if tracking:
                    boundary = np.rint(points*display_scale).astype(np.int32)
                    cv2.polylines(overlay, [boundary], True, (220, 220, 220), 1)
                    divider = display_points([(width/2, 0), (width/2, height-1)])
                    cv2.line(overlay, tuple(divider[0]), tuple(divider[1]), (255, 255, 255), 1)
                    for gate in monitor.gates:
                        line = display_points([(0, gate*height), (width-1, gate*height)])
                        cv2.line(overlay, tuple(line[0]), tuple(line[1]), (70, 220, 100), 1)
                if visible_positions:
                    for point in display_points(visible_positions):
                        cv2.circle(overlay, tuple(point), 3, (50, 200, 255), -1)
                caption = 'Camera view changed; lane counts paused' if road_configured and not tracking else latest
                cv2.putText(overlay, caption[:47], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 170, 255), 1)
                view = np.zeros((panel_h, panel_w*2, 3), dtype=np.uint8)
                y0, x0 = (panel_h-display_h)//2, (panel_w-display_w)//2
                for offset, panel in ((0, source_view), (panel_w, overlay)):
                    view[y0:y0+display_h, offset+x0:offset+x0+display_w] = panel
                if on_frame:
                    if alert_this_frame:
                        alert_view = view
                    on_frame(view, monitor.completed)
                if not args.headless:
                    cv2.imshow('Road Obstacle Detection | Q to quit', view)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
            if args.seconds and now >= args.seconds:
                break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if camera and not should_stop() and not args.seconds:
        raise RuntimeError('Camera stopped delivering frames.')
    if playback_rate and alert_view is not None and preview_enabled() and not should_stop():
        on_frame(alert_view, monitor.completed)
    summary = dict(frames_analyzed=scans, frames_read=frames, expected_frames=expected_frames,
                   source_seconds=round(frames/fps, 1) if not camera else round(time.monotonic()-start, 1),
                   lane_calibrated=road_configured, vehicles_visible=last_vehicle_count,
                   completed_passages=monitor.completed,
                   alerts=len(monitor.alerts)+int(visual_alerted), evidence_bytes=evidence_bytes,
                   speed_bump_scans=speed_bump_scans, speed_bump_detected=speed_bump_detected,
                   pavement_scans=pavement_scans, image_suggestions=image_suggestions,
                   merged_passages=monitor.merged,
                   slowed_passages=monitor.slowed,
                   processing_seconds=round(time.monotonic()-start, 2))
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='USB camera index or video file')
    parser.add_argument('--road', nargs=8, type=float, default=[0, 0, 319, 0, 319, 239, 0, 239],
                        metavar='PX', help='Road corners in a 320x240 preview, far-left, far-right, near-right, near-left')
    parser.add_argument('--speed-bump-model', type=Path)
    parser.add_argument('--pavement-model', type=Path)
    parser.add_argument('--vehicle-model', type=Path)
    parser.add_argument('--gates', nargs=3, type=float)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--threshold', type=int, default=100)
    parser.add_argument('--speed-breaker', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', type=Path, default=Path('evidence'))
    args = parser.parse_args()
    if args.fps <= 0 or args.threshold < 1:
        parser.error('fps and threshold must be positive')
    run(args)
