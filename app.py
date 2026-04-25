import cv2
import numpy as np
import tensorflow as tf
import json
import time

# Load model
model = tf.keras.models.load_model("sign_model.h5")

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Convert index → letter
classes = list(labels.keys())

cap = cv2.VideoCapture(0)

sentence = ""
last_pred = ""
last_time = time.time()

while True:
    ret, frame = cap.read()

    # ROI box
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
    roi = frame[100:300, 100:300]

    # Preprocess
    img = cv2.resize(roi, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)
    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    # Add letter only if stable
    if confidence > 0.8:
        if label != last_pred:
            last_time = time.time()
            last_pred = label

        # add after delay (avoid repeats)
        if time.time() - last_time > 1:
            sentence += label
            last_time = time.time()

    # Display
    cv2.putText(frame, f'Letter: {label}', (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f'Sentence: {sentence}', (50,450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Sign Language", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):  # clear sentence
        sentence = ""

cap.release()
cv2.destroyAllWindows()