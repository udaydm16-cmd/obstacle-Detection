"""Confirm a modular road hump with texture and a YOLO speed-bump prediction."""
import cv2
import numpy as np

from pavement import PavementCheck


class SpeedBumpCheck:
    def __init__(self, weights):
        self.net = cv2.dnn.readNetFromONNX(str(weights))
        self.previous = None

    def scan(self, frame):
        image = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        smooth = cv2.GaussianBlur(gray, (0, 0), 2)
        roughness = cv2.blur(np.abs(gray-smooth), (9, 9))
        mask = np.uint8((roughness > 7) & (smooth < 173))*255
        mask[:100] = mask[174:] = 0
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        candidates = []
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour)/(w*h)
            if 95 <= w <= 190 and 48 <= h <= 75 and .32 <= fill <= .65:
                candidates.append((x,y,x+w,y+h))
        if not candidates:
            self.previous = None
            return None

        # Present the strip in the orientation used by the model's road-facing training views.
        left, top, right = 47, 90, 273
        height, width = frame.shape[:2]
        crop = frame[round(top*height/180):,round(left*width/320):round(right*width/320)]
        self.net.setInput(cv2.dnn.blobFromImage(crop, 1/255, (320,320), swapRB=True))
        predictions = self.net.forward()[0]
        accepted = None
        for x,y,w,h,obj,pothole,bump in predictions:
            if obj*bump < .035 or bump <= pothole:
                continue
            cx,cy = left+x*(right-left)/320, top+y*(180-top)/320
            bw,bh = w*(right-left)/320, h*(180-top)/320
            prediction = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
            accepted = next((box for box in candidates
                             if PavementCheck.overlap(box, prediction) >= .28), None)
            if accepted:
                break
        confirmed = (accepted and self.previous and
                     PavementCheck.overlap(accepted, self.previous) >= .65)
        self.previous = accepted
        return accepted if confirmed else None
