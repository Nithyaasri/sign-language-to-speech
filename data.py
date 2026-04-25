from function import *
import cv2

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    for action in actions:
        for sequence in range(no_sequences):

            img_path = f'Image/{action}/{sequence}.jpg'
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            image, results = mediapipe_detection(frame, hands)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            np.save(os.path.join(DATA_PATH, action, str(sequence), '0'), keypoints)

            print(f"{action} {sequence} done")