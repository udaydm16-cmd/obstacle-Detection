"""Bounded, local pothole inference on small overlapping pavement crops."""
import cv2
import numpy as np


class PavementCheck:
    def __init__(self, weights, road):
        self.net = cv2.dnn.readNetFromONNX(str(weights))
        self.road = np.float32(road).reshape(4, 2)
        self.previous = []

    def scan(self, frame):
        # Restore the recording's wide aspect ratio for the model.
        image = cv2.resize(frame, (320, 180))
        candidates = []
        for top in (72, 90):
            for left in (0, 80, 160):
                crop = image[top:top+90, left:left+160]
                padded = cv2.copyMakeBorder(cv2.resize(crop, (320, 180)), 70, 70, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
                self.net.setInput(cv2.dnn.blobFromImage(padded, 1/255, swapRB=True))
                for x, y, w, h, score, manhole in self.net.forward()[0].T:
                    if score < .22 or score <= manhole:
                        continue
                    x1 = max(left, left+(x-w/2)/2)
                    x2 = min(left+160, left+(x+w/2)/2)
                    y1 = max(top, top+(y-h/2-70)/2)*4/3
                    y2 = min(top+90, top+(y+h/2-70)/2)*4/3
                    if x2-x1 < 4 or y2-y1 < 4:
                        continue
                    box = (x1, y1, x2, y2)
                    if all(cv2.pointPolygonTest(self.road, (float(a),float(b)), False) >= 0
                           for a in (x1,x2) for b in (y1,y2)):
                        candidates.append((float(score), box))
        matches = [(score, box) for score, box in candidates
                   if any(self.overlap(box, old) >= .3 for _, old in self.previous)]
        self.previous = candidates
        return max(matches, default=None)

    @staticmethod
    def overlap(a, b):
        x = max(0, min(a[2], b[2])-max(a[0], b[0]))
        y = max(0, min(a[3], b[3])-max(a[1], b[1]))
        intersection = x*y
        return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
