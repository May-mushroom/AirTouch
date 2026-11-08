import queue

import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision

from model import HAND_LANDMARKER_PATH

HandLandmarkerResult = vision.HandLandmarkerResult

class HandLandmarker():
    def __init__(self, model_path = HAND_LANDMARKER_PATH):
        self.options = vision.HandLandmarkerOptions(
            base_options = tasks.BaseOptions(model_asset_path = str(model_path)),
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback = self.__set_detection_results,
            num_hands = 2
        )
        self.land_marker = (vision.HandLandmarker).create_from_options(self.options)
        self.results = queue.Queue(maxsize=1)
    
    # callback function for land marker
    def __set_detection_results(self, result, output_image: mp.Image, timestamp_ms: int):
        landmarks = result.hand_landmarks
        if (self.results.full()):
            try:
                self.results.get_nowait()
            except queue.Empty: pass
        self.results.put_nowait(result)
    
    # get latest HandLandmarkerResult (landmarks + handedness), or None if nothing new yet
    def get_result(self):
        res = None
        try:
            res = self.results.get_nowait()
        except queue.Empty: pass
        return res
        
    # detect hand landmarks
    def detect_landmark(self, frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.land_marker.detect_async(mp_image, timestamp_ms)
        return self.results
    
    # close landmark 
    def close(self):
        self.land_marker.close()
            