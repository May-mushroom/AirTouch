import time

import cv2 as cv
from hand_tracker import HandLandmarker
from ui import draw_fps, draw_hands

from gestures import detect_gesture

# Open the default camera
cam = cv.VideoCapture(0)
hand_landmarker = HandLandmarker()

prev_time = time.time()
while True:
    ret, frame = cam.read()
    if not ret:
        print("frame is None")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)
    hand_landmarker.detect_landmark(frame=rgb_frame, timestamp_ms=timestamp_ms)

    result = hand_landmarker.get_result()
    # Detection runs on the raw (unflipped) frame above; flip only for display
    # so drawn text/labels come out right-side-up instead of mirrored.
    display_frame = cv.flip(frame, 1)
    if result: print(detect_gesture(result.hand_landmarks))
    gesture = detect_gesture(result.hand_landmarks) if result else None
    
    # TODO: swap in a real per-hand gesture list, aligned with
    # result.hand_landmarks, once gestures.py is ready, e.g. [("Pinch", 0.9), None]
    draw_hands(display_frame, result, gestures=gesture)

    now = time.time()
    fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
    prev_time = now
    draw_fps(display_frame, fps)

    cv.imshow("Air Touch", display_frame)

    if cv.waitKey(1) == ord('q'):
        hand_landmarker.close()
        break

# Release the capture and writer objects
cam.release()
cv.destroyAllWindows()
