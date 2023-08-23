import cv2
from pathlib import Path

from detector import Detector
from coordinate_writer import CoordinateWriter
import imutils


detector = Detector(model_path="./models/0.pth", alphabet_path="./data/armenian_alphabet")
coordinate_writer = CoordinateWriter(path=Path.home() / "object-coordinates.csv", lat=5614479, lon=3498443)
capture = cv2.VideoCapture(0)


while True:
    ret, frame = capture.read()
    detections = detector.detect(frame)
    print(detections)
    if len(detections) > 0:
        coordinate_writer.append(detections[0]['id'])
    
    #detector.draw_detections(frame, detections)

    #cv2.imshow('Frame', frame)

    #keyboard = cv2.waitKey(30)
    #if keyboard == 'q' or keyboard == 27:
    #    break
