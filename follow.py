import numpy as np
import cv2
from math import sqrt

import perception
import control
import motion
import sys
from time import sleep

per = perception.Perception(sys.argv[1])
ctrl = control.Control(sys.argv[1])

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(
  maxCorners = 100,
  qualityLevel = 0.3,
  minDistance = 4,
  blockSize = 7
)

height = 480
width = 640

# Parameters for lucas kanade optical flow
lk_params = dict(
  winSize  = (15, 15),
  maxLevel = 8,
  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create some random colors
colors = np.random.randint(0, 255, (100, 3))
line_mask = None

old_frame = None
old_gray = None
p0 = None

new_track_point = None
track_point_radius = None
dragging = False

def dist(x1, y1, x2, y2):
  return int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def on_click(event, x, y, flags, param):
  global new_track_point
  global track_point_radius
  global dragging

  if event == cv2.EVENT_LBUTTONDOWN:
    new_track_point = (x, y)
    track_point_radius = 0
    dragging = True

  elif event == cv2.EVENT_LBUTTONUP:
    dragging = False

  elif event == cv2.EVENT_MOUSEMOVE and dragging:
    track_point_radius = dist(x, y, new_track_point[0], new_track_point[1])

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", on_click)

per.use_upper_camera()
ctrl.walk_init()

#ctrl.turn_head(0, 0.2, 0.1)

while True:

  #sleep(3)
  #print "Bild!"

  #ret, frame = cap.read()
  frame = per.get_image()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #cv2.rectangle(frame, (0,0), (width/2, height/2), (255, 255, 255), -1)

  if dragging:
    cv2.circle(frame, new_track_point, track_point_radius, (0, 0, 255), 1)

  elif track_point_radius is not None:
    feature_mask = np.zeros_like(old_gray)
    cv2.circle(feature_mask, new_track_point, track_point_radius, (255, 255, 255), -1)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=feature_mask, **feature_params)
    line_mask = np.zeros_like(frame)
    new_track_point = None
    track_point_radius = None


  if p0 is not None:

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      if len(good_new):

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(line_mask, (a, b), (c, d), colors[i].tolist(), 1)
            cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
        frame = cv2.add(frame, line_mask)

        average = [0, 0]
        for pt in good_new:
          average[0] += pt[0]
          average[1] += pt[1]
        average[0] /= len(good_new)
        average[1] /= len(good_new)
        average = (int(average[0]), int(average[1]))

        cv2.circle(frame, average, 5, (0, 0, 255), 2)

        # Update previous points
        p0 = good_new.reshape(-1, 1, 2)

        turn = (average[0] - width / 2) * -0.001

        ctrl.move(0.4, 0, turn)
  else:
    ctrl.stop()

  # Update previous frame
  old_gray = frame_gray.copy()

  cv2.imshow("frame", frame)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
      break

ctrl.stop()

cv2.destroyAllWindows()
cap.release()