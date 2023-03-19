import cv2
import os
import numpy as np

from candidate_detector import CandidateDetector
import imutils


PATH = "../data/negative_samples"
OUTPUT_SIZE = (64, 64)


def init_counter(path):
    filenames = os.listdir(path)
    numbers = [int(f.split('.')[0]) for f in filenames]
    if len(numbers) > 0:
        return np.array(numbers).max()
    else:
        return 0


detector = CandidateDetector(corner_saturation_thresh=0.3, min_cnt_angle=30, max_cnt_aspect_ratio=10, gamma=1)
capture = cv2.VideoCapture(0)

counter = init_counter(PATH)
while True:
    ret, frame = capture.read()
    contours = detector.detect_candidates(frame)

    if len(contours) > 0:
        for cnt in contours:
            candidate_img = imutils.warp_perspective(frame, contours[0])
            candidate_binarized = 255 - imutils.binarize_image(candidate_img)
            cv2.imshow('Candidate', candidate_binarized)
            cv2.imwrite(os.path.join(PATH, "%i.png" % counter), cv2.resize(candidate_binarized, OUTPUT_SIZE))
            counter += 1

        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)
    frame = cv2.putText(frame, "Images: %i" % counter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
