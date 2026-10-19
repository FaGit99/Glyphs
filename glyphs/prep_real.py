"""Cache real evaluation data as 224 px views at the native HAM pitch (short side -> 450 px).

Per image: 5 tiles (4 corners + centre of the centre square) and 1 'whole' view (square resized to 224).
ISIC 2018 Task 2: per-view fraction of mask pixels for the 5 attributes. Derm7pt: image-level labels.
"""
import csv, glob, json, os, sys, time
import cv2, numpy as np
from multiprocessing import Pool

RAW = "/mnt/samsung_data/mole_data/data/raw/"
ATTRS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
POS = [(0, 0), (226, 0), (0, 226), (226, 226), (113, 113)]
cv2.setNumThreads(1)


def views(arr, interp):
    """arr: HxW(xC) with min side resized to 450 -> square 450 -> 5 tiles + whole, all 224."""
    h, w = arr.shape[:2]
    y0, x0 = (h - 450) // 2, (w - 450) // 2
    sq = arr[y0:y0 + 450, x0:x0 + 450]
    out = [sq[y:y + 224, x:x + 224] for x, y in POS]
    out.append(cv2.resize(sq, (224, 224), interpolation=interp))
    return out


def resize_min450(img, interp):
    h, w = img.shape[:2]
    s = 450 / min(h, w)
    return cv2.resize(img, (max(450, int(round(w * s))), max(450, int(round(h * s)))), interpolation=interp)


def isic_job(a):
    img_path, mask_prefix = a
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.cvtColor(resize_min450(img, cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    v = np.stack(views(img, cv2.INTER_AREA))
    frac = np.zeros((6, len(ATTRS)), np.float32)
    for k, at in enumerate(ATTRS):
        m = cv2.imread(f"{mask_prefix}_attribute_{at}.png", 0)
        if m is None:
            return None
        m = resize_min450(m, cv2.INTER_NEAREST)
        frac[:, k] = [(x > 127).mean() for x in views(m, cv2.INTER_NEAREST)]
    return v, frac


def d7_job(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(resize_min450(img, cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.stack(views(img, cv2.INTER_AREA))


def find_ci(base, rel):
    """Resolve each path component case-insensitively (metadata and folders differ in case)."""
    p = base
    for part in rel.split("/"):
        if os.path.exists(os.path.join(p, part)):
            p = os.path.join(p, part)
            continue
        hit = [x for x in os.listdir(p) if x.lower() == part.lower()]
        if not hit:
            return None
        p = os.path.join(p, hit[0])
    return p


if __name__ == "__main__":
    t = time.time()
    which = sys.argv[1:] or ["derm7pt", "isic2018"]
    if "derm7pt" in which:
        base = RAW + "Derm7_pt/images/"
        rows = list(csv.DictReader(open(RAW + "Derm7_pt/meta/meta.csv")))
        paths = [find_ci(base, r["derm"]) for r in rows]
        keep = [i for i, p in enumerate(paths) if p]
        with Pool(16) as pool:
            res = pool.map(d7_job, [paths[i] for i in keep], chunksize=8)
        ok = [i for i, r in zip(keep, res) if r is not None]
        np.save("cache/d7_img.npy", np.stack([r for r in res if r is not None]))
        keys = ["pigment_network", "streaks", "dots_and_globules", "blue_whitish_veil", "regression_structures", "vascular_structures", "diagnosis"]
        json.dump({k: [rows[i][k] for i in ok] for k in keys}, open("cache/d7_labels.json", "w"))
        print(f"derm7pt: {len(ok)} images, {time.time() - t:.0f}s", flush=True)
    if "isic2018" in which:
        b = RAW + "isic2018/"
        jobs = []
        for inp, gt in (("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task2_Training_GroundTruth_v3"),
                        ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task2_Validation_GroundTruth"),
                        ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task2_Test_GroundTruth")):
            for f in sorted(glob.glob(b + inp + "/*.jpg")):
                jobs.append((f, b + gt + "/" + os.path.basename(f)[:-4]))
        with Pool(16) as pool:
            res = pool.map(isic_job, jobs, chunksize=8)
        good = [(j, r) for j, r in zip(jobs, res) if r is not None]
        np.save("cache/isic18_img.npy", np.stack([r[0] for _, r in good]))
        np.save("cache/isic18_frac.npy", np.stack([r[1] for _, r in good]))
        json.dump([os.path.basename(j[0]) for j, _ in good], open("cache/isic18_names.json", "w"))
        print(f"isic2018: {len(good)}/{len(jobs)} images, {time.time() - t:.0f}s", flush=True)
