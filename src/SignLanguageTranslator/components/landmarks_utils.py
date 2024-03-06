import os
import mediapipe as mp
from SignLanguageTranslator import logger



def initialize_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.1)

    return mp_drawing, mp_drawing_styles, mp_holistic, mp_hands, holistic, hands



class landmarks_data:
    def __init__(self, landmark_type, i, x, y, z, video_id) -> None:
        self.landmark_type = landmark_type
        self.landmark_index = i
        self.x = x
        self.y = y
        self.z = z
        self.video_id = video_id

    @staticmethod
    def get_landmark_data(landmark_type, i, x, y, z, video_id):
        return {
            'type': landmark_type,
            'landmark_index': i,
            'x': x,
            'y': y,
            'z': z,
            'video_id': video_id,
        }
