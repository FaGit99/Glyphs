"""Skin backgrounds: lesion-free native-resolution crops of real HAM10000 images.

Files are split by index parity so backgrounds (parity 0) and real comparison
images (parity 1) never share a source image.
"""
import glob
import os
import cv2
import numpy as np

HAM = "/mnt/samsung_data/mole_data/data/raw/"
HAM_SEG = "/mnt/samsung_data/mole_data/data/Segmentation/HAM10000_segmentations_lesion_tschandl/"


def ham_files(parity):
    files = sorted(glob.glob(HAM + "HAM10000_images_part_1/*.jpg") + glob.glob(HAM + "HAM10000_images_part_2/*.jpg"))
    return files[parity::2]


def procedural(rng, size):
    base = np.array([rng.uniform(190, 225), rng.uniform(140, 175), rng.uniform(120, 155)])
    img = np.ones((size, size, 3)) * base
    for s, a in ((size / 3, 10), (size / 12, 5), (1.5, 3)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), s)
        img += (n / (n.std() + 1e-6) * a)[..., None] * np.array([1, .9, .85])
    return np.clip(img, 0, 255).astype(np.uint8)


def real_patch(rng, size, tries=150, files=None):
    """Native-resolution (no resize) lesion-free window of `size` px."""
    files = files or ham_files(0)
    for _ in range(tries):
        f = files[rng.integers(len(files))]
        img = cv2.imread(f)
        m = cv2.imread(HAM_SEG + os.path.basename(f)[:-4] + "_segmentation.png", 0)
        if img is None or m is None:
            continue
        m = cv2.dilate((m > 127).astype(np.uint8), np.ones((9, 9), np.uint8))  # allow skin right next to the lesion
        h, w = m.shape
        for _ in range(25):
            y, x = rng.integers(0, h - size), rng.integers(0, w - size)
            if m[y:y + size, x:x + size].any():
                continue
            crop = img[y:y + size, x:x + size]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if g.mean() < 55 or g.std() > 75:  # only reject near-black corners / extreme clutter; hair and rulers stay
                continue
            return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), os.path.basename(f)
    return None


def make_skin(rng, size=224, prefer_real=True, files=None):
    if prefer_real:
        r = real_patch(rng, size, files=files)
        if r is not None:
            return r
    return procedural(rng, size), "procedural"
