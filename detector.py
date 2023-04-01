import numpy as np
import cv2
import os

from candidate_detector import CandidateDetector
from classifier import Classifier
import imutils


class Detector:
    def __init__(self, model_path: str, alphabet_path: str = None):
        """

        :param model_path: path to model state dict (.pth)
        """
        self.classifier = Classifier(model_path)
        self.candidate_detector = CandidateDetector()
        self.alphabet = None
        if alphabet_path is not None:
            self.load_alphabet(alphabet_path)

    def load_alphabet(self, alphabet_path):
        alphabet_files = os.listdir(alphabet_path)
        classes = sorted([int(filename.split('.')[0]) for filename in alphabet_files])
        self.alphabet = []
        for c in classes:
            self.alphabet.append(cv2.imread(os.path.join(alphabet_path, '%i.png' % c)))

    def draw_detections(self, img: np.array, detections: list, cnt_color: tuple=(0, 255, 0), letter_size: tuple=(48, 48)):
        for detection in detections:
            cv2.drawContours(img, detection['coords'].reshape(1, -1, 2), -1, cnt_color, 3)
            if self.alphabet is not None:
                col1, row2 = detection['coords'][0]
                col2 = col1 + letter_size[0]
                row1 = row2 - letter_size[1]

                col_diff = letter_size[0]
                row_diff = 0
                if col2 > img.shape[1]:
                    col_diff = img.shape[1] - col2
                    col2 = img.shape[1]
                if row1 < 0:
                    row_diff = row1
                    row1 = 0

                letter = cv2.resize(self.alphabet[detection['id']], letter_size)[-row_diff:, 0:col_diff]
                img[row1:row2, col1:col2] = letter

    def detect(self, img: np.array, confidence_thresh: float = 0.5):
        detections = []
        contours = self.candidate_detector.detect_candidates(img)
        for cnt in contours:
            detection = {}
            candidate_img = imutils.warp_perspective(img, cnt)
            candidate_binarized = 255 - imutils.binarize_image(candidate_img)
            confidence, prediction = self.classifier.predict(cv2.resize(candidate_binarized, (48, 48)))

            if confidence > confidence_thresh:
                detection['coords'] = cnt
                detection['confidence'] = confidence
                detection['id'] = prediction
                detections.append(detection)

        return detections
