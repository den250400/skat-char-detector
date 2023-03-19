import cv2
import numpy as np


def pad_to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def warp_perspective(img: np.array, corners: np.array, result_size: tuple = (480, 480)) -> np.array:
    """
    Perform a perspective transform

    :param img: Input color image
    :param corners: np.array(shape=(4, 2))
    :param result_size: size of transformed image
    :return:
    """
    output_pts = np.float32([[0, 0],
                             [0, result_size[1] - 1],
                             [result_size[0] - 1, result_size[1] - 1],
                             [result_size[0] - 1, 0]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), output_pts)

    warped_img = cv2.warpPerspective(img, M, result_size, flags=cv2.INTER_LINEAR)

    return warped_img


def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh
