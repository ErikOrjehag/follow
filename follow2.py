import cv2
import numpy as np
cap = cv2.VideoCapture(0)

def getGrayFrame():
  size = 0.5
  ret, frame = cap.read()
  frame = cv2.resize(frame, (0, 0), fx=size, fy=size)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return gray, frame

prev, frame = getGrayFrame()
hsv = np.zeros_like(frame)
hsv[...,1] = 255

while(1):
    next, frame = getGrayFrame()

    flow = cv2.calcOpticalFlowFarneback(prev, next, 0.5, 3, 15, 3, 5, 1.2, 0)

    cv2.imshow('flow0', flow[..., 0])
    cv2.imshow('flow1', flow[..., 1])

    prev = next

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()