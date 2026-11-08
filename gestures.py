# HARDCODE TO TEST CAMERA MODULE FOR NOW. 
# WORKING GESTURE DECTECTING MODEL 

import math 
from mediapipe.tasks.python.vision import HandLandmarkerResult
import numpy as np

FINGER_TIP = {
  "index": 8,
  "middle": 12,
  "thumb": 4
}


class GestureDectector():
  def __init__(self):
    self.detected_gesture = None


# angle >= 160: extended, otherwise curled
def is_curled(landmarks) -> bool: 
  res = False
  angle = calculate_angle(landmarks)
  if angle < 160: res = True
  return res
  
# group landmarks based on finger
def detect_hand_config(landmarks): 
  # in case no hand detected
  if not landmarks: return
  fingers = {
    "thumb": get_landmark_position(landmarks, [1,2,3,4]),
    "index": get_landmark_position(landmarks, [5,6,7,8]),
    "middle": get_landmark_position(landmarks, [9,10,11,12]),
    "ring": get_landmark_position(landmarks, [13,14,15,16]),
    "pinky": get_landmark_position(landmarks, [17,18,19,20]),     
  }
  hand_config = {
    "thumb": is_curled(fingers["thumb"]),
    "index": is_curled(fingers["index"]),
    "middle": is_curled(fingers["middle"]),
    "ring": is_curled(fingers["ring"]),
    "pinky": is_curled(fingers["pinky"]),
  }
  return hand_config
    
# switch case at the end, return none = no gesture detected.
def detect_gesture(landmarks): 
  if not landmarks: return
  hand_config = detect_hand_config(landmarks)
  thumb_coordinate = get_coordinate_finger_tip(landmarks, FINGER_TIP["thumb"])
  index_coordinate = get_coordinate_finger_tip(landmarks, FINGER_TIP["index"])
  middle_coordinate = get_coordinate_finger_tip(landmarks, FINGER_TIP["middle"])
  thumb_index_dist = math.dist(thumb_coordinate, index_coordinate) if thumb_coordinate and index_coordinate else None
  thumb_middle_dist = math.dist(thumb_coordinate, middle_coordinate) if thumb_coordinate and middle_coordinate else None
  gestures = {
    "cursor": not hand_config["index"] and hand_config["middle"] and hand_config["ring"] and hand_config["pinky"],
    "left_click": thumb_index_dist < 0.1 if thumb_index_dist else False,
    "righ_click": thumb_middle_dist < 0.1 if thumb_middle_dist else False
  }
  for attr, value in gestures.items():
    if value: 
      return attr
  return None
    
# UTILS
# get all marks of 1 finger (4 marks) and wrist(1 mark)
def get_landmark_position(landmarks, positions):
  if landmarks == []: return []
  res = []
  for p in positions: res.append(landmarks[0][p])
  return res

def get_coordinate_finger_tip(landmarks, pos):
  if landmarks == []: return None
  return (landmarks[0][pos].x, landmarks[0][pos].y)

def extract_features(landmarks):
  if landmarks == []: return None
  index_tip = get_landmark_position(landmarks, "index_tip"),
  thumb_tip = get_landmark_position(landmarks, "thumb_tip"),
  middle_tip = get_landmark_position(landmarks, "middle_tip"),
  return {
    "index_tip": index_tip,
    "thumb_tip": thumb_tip,
    "middle_tip": middle_tip
  }
# accept 4 landmarks of 1 finger, starting from knuckle(0) to tip(3)
def calculate_angle(landmarks):
    A = np.array([landmarks[0].x, landmarks[0].y])
    B = np.array([landmarks[1].x, landmarks[1].y])
    C = np.array([landmarks[2].x, landmarks[2].y])

    BA = A - B
    BC = C - B

    cosine = np.dot(BA, BC) / (
        np.linalg.norm(BA) * np.linalg.norm(BC)
    )

    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return angle

  
# flow: capture video -> landmark hands -> [lock interest target(index, thumb, middle) 
# -> constantly check gesture] -> map gesutre to action -> OS actio

# 2. lock interest: extract_landmark -> classify into group.
# 3. detect hand config: group of hand config for each gesture
  # cursor: index extend and thumb are the only extended, 3 fingers curl
  # left click: index touches thumb, 3 curls
  # right click: mid touch thumb, ring pink curl, index might extend
  # scroll: thumb extend, index+mid extend + next to each other, ring + pinky curl, vertical direction.
# 3. detect gesture (calculate distance bwt finger) -> constantly pool to check gestures on live -> raise flag for gesture
# 4. if flag -> map action -> execute action

# index_tip and thumb only, the rest of fingers near palm -> track cursor
# index thumb touch -> left click, remain touch and move -> dragging
# thumb & middle touch -> right click
# index + thumb + middle + direction -> scroll and switch tab



