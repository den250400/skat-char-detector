import cv2

from candidate_detector import CandidateDetector
import imutils


detector = CandidateDetector()
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    contours = detector.detect_candidates(frame)

    if len(contours) > 0:
        candidate_img = imutils.warp_perspective(frame, contours[0])
        candidate_binarized = 255 - imutils.binarize_image(candidate_img)
        cv2.imshow('Candidate', candidate_binarized)

        cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
