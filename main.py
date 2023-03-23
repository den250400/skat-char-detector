import cv2

from detector import Detector
import imutils


detector = Detector(model_path="./models/0.pth", alphabet_path="./data/armenian_alphabet")
capture = cv2.VideoCapture(0)


while True:
    ret, frame = capture.read()
    detections = detector.detect(frame)
    detector.draw_detections(frame, detections)

    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
