import cv2
import os

from imutils import pad_to_shape


def align_letter(letter_img, output_size=(640, 640)):
    img_h, img_w = letter_img.shape[0], letter_img.shape[1]
    contours, hierarchy = cv2.findContours(letter_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    letter_aligned = letter_img[y:y+h, x:x+w]
    letter_aligned = pad_to_shape(letter_aligned, (img_h, img_w))
    letter_aligned = cv2.resize(letter_aligned, output_size)

    return letter_aligned


IMG_PATH = "../data/armenian_alphabet_merged.png"
OUTPUT_PATH = "../data/armenian_alphabet"
CROP_POINT1 = (0, 165)
CROP_POINT2 = (552, 1180)
ROWS = 4
COLS = 9

img = cv2.imread(IMG_PATH)[CROP_POINT1[0]:CROP_POINT2[0], CROP_POINT1[1]:CROP_POINT2[1]]
img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Cropped", img)
cv2.waitKey()
cv2.destroyAllWindows()

row_step = img.shape[0] // ROWS
col_step = img.shape[1] // COLS

for i in range(ROWS):
    for j in range(COLS):
        letter = align_letter(img[row_step*i:row_step*(i+1), col_step*j:col_step*(j+1)])
        max_dim = max(letter.shape[0], letter.shape[1])
        letter = pad_to_shape(letter, (max_dim, max_dim))
        cv2.imwrite(os.path.join(OUTPUT_PATH, "%s.png" % str(i*COLS+j)), letter)



