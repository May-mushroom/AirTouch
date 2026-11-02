import time

import cv2 as cv
from hand_tracker import HandLandmarker

# Open the default camera
cam = cv.VideoCapture(0)
hand_landmarker = HandLandmarker()
while True:
    ret, frame = cam.read()
    data = cv.flip(frame, 1)
    timestamp = time.time() * 1000
    hand_landmarker.detect_landmark(frame=data, timestamp_ms=int(timestamp))
    
    res_frame = hand_landmarker.get_output_image()
    if not res_frame: print("frame is None")
    # else: cv.imshow("frame", res_frame)
    
    if cv.waitKey(1) == ord('q'):
        hand_landmarker.close()
        break
    

# Release the capture and writer objects
cam.release()
cv.destroyAllWindows()
