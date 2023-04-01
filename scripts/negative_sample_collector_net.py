import cv2
import os
import numpy as np

from detector import Detector
import imutils


PATH = "../data/negative_samples"
MODEL_PATH = "../models/0.pth"
ALPHABET_PATH = "../data/armenian_alphabet"
CONFIDENCE_THRESH = 0.5
OUTPUT_SIZE = (64, 64)


def init_counter(path):
    filenames = os.listdir(path)
    numbers = [int(f.split('.')[0]) for f in filenames]
    if len(numbers) > 0:
        return np.array(numbers).max()
    else:
        return 0


detector = Detector(model_path=MODEL_PATH, alphabet_path=ALPHABET_PATH)
capture = cv2.VideoCapture("/home/denis/Downloads/5 Best Simple & Easy Cinematic Drone Shots - DJI Drones.mp4")

counter = init_counter(PATH)
while True:
    ret, frame = capture.read()
    detections = detector.detect(frame, confidence_thresh=CONFIDENCE_THRESH)

    if len(detections) > 0:
        for detection in detections:
            candidate_img = imutils.warp_perspective(frame, detection['coords'])
            candidate_binarized = 255 - imutils.binarize_image(candidate_img)
            cv2.imshow('Candidate', candidate_binarized)
            cv2.imwrite(os.path.join(PATH, "%i.png" % counter), cv2.resize(candidate_binarized, OUTPUT_SIZE))
            counter += 1

        detector.draw_detections(frame, detections)
    frame = cv2.putText(frame, "Images: %i" % counter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break