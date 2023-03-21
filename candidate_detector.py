import cv2
import numpy as np
import math

import imutils


class CandidateDetector:
    def __init__(self, adaptive_thresh_min_scale=3, adaptive_thresh_scale_step=10, adaptive_thresh_n_scales=3,
                 min_perimeter_rate=0.05, max_cnt_aspect_ratio=1.5, min_cnt_angle=70, marker_corner_size=0.1,
                 corner_saturation_thresh=0.3, gamma=0.3):
        self.adaptive_thresh_min_scale = adaptive_thresh_min_scale
        self.adaptive_thresh_scale_step = adaptive_thresh_scale_step
        self.adaptive_thresh_n_scales = adaptive_thresh_n_scales
        self.min_perimeter_rate = min_perimeter_rate
        self.max_cnt_aspect_ratio = max_cnt_aspect_ratio
        self.min_cnt_angle = min_cnt_angle
        self.marker_corner_size = marker_corner_size
        self.corner_saturation_thresh = corner_saturation_thresh
        self.gamma = gamma

    @staticmethod
    def compute_aspect_ratio(cnt: np.array) -> float:
        """
        Compute contour's aspect ratio (max_length / min_length)

        :param cnt: np.array(shape=(4, 2)) candidate contour polygonal approximation
        :return:
        """
        #cnt = cnt.reshape(-1, 2)
        point1 = cnt[:-1]
        point2 = cnt[1:]
        diff = point2 - point1
        lengths = np.hypot(diff[:, 0], diff[:, 1])

        return np.max(lengths) / np.min(lengths)

    @staticmethod
    def compute_min_angle(cnt: np.array) -> float:
        """
        Compute contour's min angle

        :param cnt: np.array(shape=(4, 2)) candidate contour polygonal approximation
        :return: Contour's min angle in degrees
        """
        #cnt = cnt.reshape(-1, 2)
        cnt_shifted = cnt[1:]
        cnt_shifted = np.append(cnt_shifted, cnt[0].reshape(1, 2), axis=0)

        vectors = cnt_shifted - cnt
        vectors_shifted = vectors[1:]
        vectors_shifted = np.append(vectors_shifted, vectors[0].reshape(1, 2), axis=0)

        angles = np.arccos(np.matmul(vectors.reshape(-1, 1, 2), vectors_shifted.reshape(-1, 2, 1)).reshape(-1) /
                           (np.linalg.norm(vectors, axis=1) * np.linalg.norm(vectors_shifted, axis=1))) * 180 / np.pi

        return angles.min()

    @staticmethod
    def preprocess_image(img: np.array, thresh_ksize: int) -> np.array:
        """
        Preprocess the image for candidate detection

        :param img: np.array(shape=(H, W, 3)) - Input color image
        :param thresh_ksize: kernel size for adaptive thresholding
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh_ksize, 3)
        cv2.imshow('Gray', np.append(gray, thresh, axis=1))

        return thresh

    def detect_polygons(self, img: np.array) -> list:
        """
        Detect 4-corner polygons in image

        :param img: Input color image
        :return:
        """
        polygons = []
        polygons_valid = []

        for i in range(self.adaptive_thresh_n_scales):
            scale = self.adaptive_thresh_min_scale + self.adaptive_thresh_scale_step * i
            thresh_img = self.preprocess_image(img, scale)
            img_diag = math.sqrt(thresh_img.shape[0] ** 2 + thresh_img.shape[1] ** 2)
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, len(cnt) * 0.03, True).reshape(-1, 2)
                arc_len = cv2.arcLength(approx, True)

                if len(approx) != 4 or not cv2.isContourConvex(approx) or self.compute_min_angle(approx) < self.min_cnt_angle:
                    continue

                aspect_ratio = self.compute_aspect_ratio(approx)
                if arc_len < img_diag * self.min_perimeter_rate or aspect_ratio > self.max_cnt_aspect_ratio:
                    continue

                polygons.append(approx)

        # Filter out parent contours
        for i in range(len(polygons)):
            is_parent = False
            for j in range(i+1, len(polygons)):
                if i == j:
                    continue
                for k in range(len(polygons[j])):
                    pt = (int(polygons[j][k][0]), int(polygons[j][k][1]))
                    cnt = polygons[i].reshape(-1, 1, 2).astype(int)
                    if cv2.pointPolygonTest(cnt, pt, measureDist=False) != -1:
                        is_parent = True
                        break
                if is_parent:
                    break

            if not is_parent:
                polygons_valid.append(polygons[i])

        return polygons_valid

    @staticmethod
    def reorder_corners(cnt: np.array):
        """
        Enforce clock-wise direction of contour's corners

        :param cnt: np.array(shape=(4, 2)) candidate contour polygonal approximation
        :return:
        """
        cnt = cnt.reshape(-1, 2)
        vec1 = cnt[1] - cnt[0]
        vec2 = cnt[-1] - cnt[0]

        if np.cross(vec1, vec2) > 0:
            # Enforce clock-wise direction
            temp = cnt[1].copy()
            cnt[1] = cnt[3]
            cnt[3] = temp

    def is_marker(self, candidate_img: np.array) -> bool:
        """
        Check if candidate's corners are white enough

        :param candidate_img:
        :return:
        """
        corner_h = int(candidate_img.shape[0] * self.marker_corner_size)
        corner_w = int(candidate_img.shape[1] * self.marker_corner_size)
        h = candidate_img.shape[0]
        w = candidate_img.shape[1]
        hsv_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2HSV)

        corners = np.empty(shape=(4, corner_h, corner_w))
        corners[0] = hsv_img[0:corner_h, 0:corner_w, 1]  # Top left corner
        corners[1] = hsv_img[0:corner_h, w - corner_w:, 1]  # Top right corner
        corners[2] = hsv_img[h - corner_h:, w - corner_w:, 1]  # Bottom right corner
        corners[3] = hsv_img[h - corner_h:, 0:corner_w, 1]  # Bottom left corner

        mean_corner_value = np.mean(corners, axis=(1, 2)) / 255

        if (mean_corner_value < self.corner_saturation_thresh).all():
            return True
        else:
            return False

    def detect_candidates(self, img: np.array) -> list:
        candidates = []
        contours = self.detect_polygons(imutils.adjust_gamma(img, gamma=self.gamma))

        if len(contours) > 0:
            for cnt in contours:
                self.reorder_corners(cnt)
                candidate_img = imutils.warp_perspective(img, cnt.reshape(-1, 2))

                if self.is_marker(candidate_img):
                    candidates.append(cnt)

        return candidates

