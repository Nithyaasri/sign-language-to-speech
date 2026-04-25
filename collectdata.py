import cv2
import os

cap = cv2.VideoCapture(0)

directory = 'Image/'

for folder in ['A', 'B', 'C']:
    os.makedirs(directory + folder, exist_ok=True)

while True:
    ret, frame = cap.read()

    cv2.rectangle(frame, (0,40), (300,400), (255,255,255), 2)
    roi = frame[40:400, 0:300]

    cv2.imshow('Frame', frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(10)

    if key == ord('a'):
        count = len(os.listdir(directory+'A'))
        cv2.imwrite(f"{directory}A/{count}.jpg", roi)

    if key == ord('b'):
        count = len(os.listdir(directory+'B'))
        cv2.imwrite(f"{directory}B/{count}.jpg", roi)

    if key == ord('c'):
        count = len(os.listdir(directory+'C'))
        cv2.imwrite(f"{directory}C/{count}.jpg", roi)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()