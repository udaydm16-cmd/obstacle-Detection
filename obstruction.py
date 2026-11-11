"""Inspect a calibrated active-lane crop using the existing vehicle detector."""
from math import hypot

import cv2
import numpy as np


class ObstructionCheck:
    def __init__(self, detector, polygon):
        self.detector = detector
        self.polygon = np.float32(polygon)
        self.tracks = []
        self.next_sample = 0

    def scan(self, image, now):
        if now + 1e-6 < self.next_sample:
            return None
        self.next_sample = now + .5
        height, width = image.shape[:2]
        # Give the model some context around the monitored lane.
        x, y, w, h = cv2.boundingRect(self.polygon)
        left, top = max(0, x-15), max(0, y-18)
        right, bottom = min(320, x+w+15), min(240, y+h+12)
        crop = image[round(top*height/240):round(bottom*height/240),
                     round(left*width/320):round(right*width/320)]
        current = []
        available = self.tracks.copy()
        for a,b,c,d in self.detector.boxes(crop):
            box = (left+a*(right-left), top+b*(bottom-top),
                   left+c*(right-left), top+d*(bottom-top))
            x1,y1,x2,y2 = box
            center = ((x1+x2)/2, y2)
            if cv2.pointPolygonTest(self.polygon, center, False) < 0:
                continue
            previous = min(available, key=lambda track: hypot(center[0]-track[1][0],
                           center[1]-track[1][1]), default=None)
            if previous and now-previous[2] <= 1 and hypot(
                    center[0]-previous[1][0], center[1]-previous[1][1]) <= 10:
                available.remove(previous)
                start, _, _, origin = previous
            else:
                start, origin = now, center
            current.append((start, center, now, origin))
            dx, dy = abs(center[0]-origin[0]), abs(center[1]-origin[1])
            # A vehicle creeping sideways without advancing along the road can block a lane.
            if now-start >= 3 and 10 <= dx <= 1.6*(x2-x1) and dy <= 5:
                return box
        self.tracks = current
        return None
