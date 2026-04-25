import cv2
import numpy as np
import mediapipe as mp
import os

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Actions (classes)
actions = ['A', 'B', 'C']

# Dataset settings
no_sequences = 30        # number of videos per action
sequence_length = 30     # frames per video
DATA_PATH = os.path.join('MP_Data')


# 🔹 1. Mediapipe Detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# 🔹 2. Draw Landmarks (IMPORTANT FIXED VERSION)
def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )


# 🔹 3. Extract Keypoints (IMPORTANT)
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    else:
        return np.zeros(21 * 3)