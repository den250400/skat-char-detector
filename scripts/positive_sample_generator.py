import cv2
from tqdm import tqdm
import numpy as np
import random
import os


ALPHABET_PATH = "../data/armenian_alphabet"
OUTPUT_PATH = "../data/positive_samples"
SAMPLES = 2000  # Number of samples per 1 class


def generate_sample(img, angle_range=(-7, 7), scale_range=(0.9, 1.5), rotate=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if rotate:
        gray = random_rot(gray)
    gray = random_line(gray)
    gray = random_scale(gray, scale_range)
    gray = random_rotate(gray, angle_range)

    return gray


def random_rot(gray):
    n_rot = random.randint(0, 3)
    new_gray = gray.copy()
    for _ in range(n_rot):
        new_gray = np.rot90(new_gray)

    return new_gray.copy()


def random_line(img, color=0, thickness_range=(1,2)):
    coord1 = (int(random.uniform(0, img.shape[1])), 0)
    coord2 = (int(random.uniform(0, img.shape[1])), img.shape[0])
    thickness = random.randint(*thickness_range)

    return cv2.line(img, coord1, coord2, color, thickness)


def random_scale(img, scale_range):
    scale = random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 0, scale)

    return cv2.warpAffine(img, M, img.shape, borderValue=(255, 255, 255))


def random_rotate(img, angle_range):
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)

    return cv2.warpAffine(img, M, img.shape, borderValue=(255, 255, 255))


alphabet_files = os.listdir(ALPHABET_PATH)
classes = [filename.split('.')[0] for filename in alphabet_files]

alphabet_files = [os.path.join(ALPHABET_PATH, f) for f in alphabet_files]
alphabet = [cv2.imread(f) for f in alphabet_files]

# Create class directories
class_dirs = [os.path.join(OUTPUT_PATH, c) for c in classes]
for class_dir in class_dirs:
    try:
        os.makedirs(class_dir)
    except FileExistsError:
        continue


print("Generating samples...")
for i in tqdm(range(len(class_dirs))):
    for j in range(SAMPLES):
        transformed_img = generate_sample(alphabet[i])
        cv2.imwrite(os.path.join(class_dirs[i], "%i.png" % j), transformed_img)
        """
        cv2.imshow("Sample", cv2.resize(transformed_img, (512, 512), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey()
        cv2.destroyAllWindows()
        """


