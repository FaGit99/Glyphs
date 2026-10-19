"""Per-view 28x28 coverage maps for ISIC 2018 Task 2 attributes + Task 1 lesion segmentation (training images only).

Same geometry as prep_real.py (short side 450 -> 5 tiles + whole view), so index [i, v] matches cache/isic18_img.npy.
"""
import json, time
import cv2, numpy as np
from multiprocessing import Pool
from prep_real import RAW, ATTRS, resize_min450, views

B = RAW + "isic2018/"
G = 28


def cov(m):  # bool/uint8 224x224 -> 28x28 coverage 0-255
    return np.clip(cv2.resize((m > 127).astype(np.float32), (G, G), interpolation=cv2.INTER_AREA) * 255, 0, 255).astype(np.uint8)


def job(name):
    base = name[:-4]
    out = np.zeros((6, len(ATTRS), G, G), np.uint8)
    for k, a in enumerate(ATTRS):
        m = cv2.imread(f"{B}ISIC2018_Task2_Training_GroundTruth_v3/{base}_attribute_{a}.png", 0)
        for v, x in enumerate(views(resize_min450(m, cv2.INTER_NEAREST), cv2.INTER_NEAREST)):
            out[v, k] = cov(x)
    seg = cv2.imread(f"{B}ISIC2018_Task1_Training_GroundTruth/{base}_segmentation.png", 0)
    les = np.stack([cov(x) for x in views(resize_min450(seg, cv2.INTER_NEAREST), cv2.INTER_NEAREST)])
    return out, les


if __name__ == "__main__":
    t = time.time()
    names = json.load(open("cache/isic18_names.json"))[:2594]
    with Pool(16) as p:
        res = p.map(job, names, chunksize=16)
    np.save("cache/isic18_maskmaps.npy", np.stack([r[0] for r in res]))
    np.save("cache/isic18_lesion.npy", np.stack([r[1] for r in res]))
    print(f"{len(res)} images, {time.time() - t:.0f}s")
