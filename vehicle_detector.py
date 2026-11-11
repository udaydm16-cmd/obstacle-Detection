"""Vehicle observations from a small local ONNX model."""
import cv2
import numpy as np


class VehicleDetector:
    def __init__(self, weights):
        self.net = cv2.dnn.readNetFromONNX(str(weights))

    def boxes(self, frame):
        height, width = frame.shape[:2]
        scale = 320 / max(height, width)
        scaled_width, scaled_height = max(1, round(width*scale)), max(1, round(height*scale))
        resized = cv2.resize(frame, (scaled_width, scaled_height))
        square = cv2.copyMakeBorder(resized, 0, 320-scaled_height, 0, 320-scaled_width,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.net.setInput(cv2.dnn.blobFromImage(square, 1/255, (320, 320), swapRB=True))
        rows = self.net.forward()[0].T
        scores = rows[:, [6, 7, 9, 11]].max(axis=1)
        selected = scores >= .27
        boxes = rows[selected, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores[selected].tolist(), .27, .45)
        if not len(indices):
            return []
        detected = []
        for x, y, w, h in boxes[np.asarray(indices).ravel()]:
            x1, y1 = max(0, x/scaled_width), max(0, y/scaled_height)
            x2, y2 = min(1, (x+w)/scaled_width), min(1, (y+h)/scaled_height)
            if x2 > x1 and y2 > y1:
                detected.append((x1, y1, x2, y2))
        return detected

    @staticmethod
    def positions(boxes, transform, width, height):
        if not boxes:
            return []
        points = np.float32([((x1+x2)*width/2, min(y2*height, height-1))
                             for x1,y1,x2,y2 in boxes])
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform)
        return [(float(x/width), float(y/height)) for ((x,y),) in mapped if 0 < y < height]
