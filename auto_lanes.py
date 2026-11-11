"""One-shot lane calibration using the YOLOP lane mask."""
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent / 'data' / 'yolop.onnx'


def lane_mask(image, net):
    h,w = image.shape[:2]
    scale = 640/max(h,w)
    width,height = round(w*scale),round(h*scale)
    x,y = (640-width)//2,(640-height)//2
    rgb = cv2.cvtColor(cv2.resize(image,(width,height)),cv2.COLOR_BGR2RGB)
    canvas = np.full((640,640,3),114,np.uint8)
    canvas[y:y+height,x:x+width] = rgb
    normalized = (canvas.astype(np.float32)/255-np.float32([.485,.456,.406]))/np.float32([.229,.224,.225])
    net.setInput(normalized.transpose(2,0,1)[None])
    background, lane = net.forward('lane_line_seg')[0]
    mask = np.uint8(lane[y:y+height,x:x+width] > background[y:y+height,x:x+width])*255
    return cv2.resize(mask,(320,240),interpolation=cv2.INTER_NEAREST)


def calibrate(image, net):
    mask = lane_mask(image, net)
    mask[:85] = 0
    options = [fit(mask,far,near) for far,near in ((125,170),(135,180),(145,190),(155,205),(165,220))]
    return max((option for option in options if option), default=(0,None))[1]


def fit(mask,far,near):
    lines = cv2.HoughLinesP(mask,1,np.pi/180,14,minLineLength=18,maxLineGap=34)
    if lines is None:
        return None
    candidates = []
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(y2-y1)<17 or max(y1,y2)<far+15:
            continue
        slope = (x2-x1)/(y2-y1)
        xf,xn = (x1+(row-y1)*slope for row in (far,near))
        if not 0 <= min(xf,xn) <= max(xf,xn) <= 319:
            continue
        strength = 0
        for row in range(far,near):
            col = round(x1+(row-y1)*slope)
            strength += bool(mask[row,max(0,col-3):col+4].any())
        if strength>=10:
            candidates.append((strength,xf,xn))
    unique=[]
    for candidate in sorted(candidates,reverse=True):
        if all(abs(candidate[2]-other[2])>12 for other in unique):
            unique.append(candidate)
    best=None
    for left,mid,right in combinations(sorted(unique,key=lambda item:item[2]),3):
        a,b = mid[2]-left[2],right[2]-mid[2]
        c,d = mid[1]-left[1],right[1]-mid[1]
        if min(a,b)<24 or max(a,b)>2*min(a,b) or min(c,d)<12 or max(c,d)>2*min(c,d):
            continue
        if a+b < c+d:
            continue
        score=sum(item[0] for item in (left,mid,right))
        if best is None or score>best[0]:
            best=(score,[round(left[1]),far,round(right[1]),far,
                         round(right[2]),near,round(left[2]),near])
    return best


def calibrate_camera(index, stop):
    net = cv2.dnn.readNetFromONNX(str(WEIGHTS))
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError('Cannot open the camera for automatic lane marking.')
    matches = []
    try:
        for index in range(9):
            if stop.is_set():
                return None
            ok, frame = camera.read()
            if not ok:
                break
            if index < 3:
                continue
            corners = calibrate(frame, net)
            if corners:
                for earlier in matches:
                    if max(abs(a-b) for a,b in zip(corners,earlier)) <= 12:
                        return [round((a+b)/2) for a,b in zip(corners,earlier)]
                matches.append(corners)
            stop.wait(.15)
    finally:
        camera.release()
    return None
