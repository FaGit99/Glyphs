"""Derm7pt data for MIL training/eval: lesion-overlapping native-pitch tiles per image + 7-point-checklist labels + official split."""
import csv, json, os
import cv2, numpy as np

RAWD = "/mnt/samsung_data/mole_data/data/raw/Derm7_pt/"
TMAX, TILE, STRIDE = 16, 224, 112
# image-level targets: (name, field, predicate on the label string)
TARGETS = [
    ("network_any", "pigment_network", lambda v: v != "absent"), ("network_atypical", "pigment_network", lambda v: v == "atypical"),
    ("streaks_any", "streaks", lambda v: v != "absent"), ("streaks_irregular", "streaks", lambda v: v == "irregular"),
    ("dots_any", "dots_and_globules", lambda v: v != "absent"), ("dots_irregular", "dots_and_globules", lambda v: v == "irregular"),
    ("veil", "blue_whitish_veil", lambda v: v == "present"), ("regression_any", "regression_structures", lambda v: v != "absent"),
    ("pigmentation_irregular", "pigmentation", lambda v: "irregular" in v),
    ("vessels_dotted", "vascular_structures", lambda v: v == "dotted"), ("vessels_hairpin", "vascular_structures", lambda v: v == "hairpin"),
    ("vessels_linear_irregular", "vascular_structures", lambda v: v == "linear irregular"), ("vessels_comma", "vascular_structures", lambda v: v == "comma"),
    ("vessels_arborizing", "vascular_structures", lambda v: v == "arborizing"),
]
TNAMES = [t[0] for t in TARGETS]


def meta():
    return list(csv.DictReader(open(RAWD + "meta/meta.csv")))


def labels():
    rows = meta()
    return np.array([[float(f(r[fld])) for _, fld, f in TARGETS] for r in rows], np.float32)


def split():
    return {k: np.array([int(r[0]) for r in csv.reader(open(RAWD + f"meta/{k}_indexes.csv")) if r and r[0].isdigit()]) for k in ("train", "valid", "test")}


def melanoma():
    return np.array([r["diagnosis"].startswith("melanoma") for r in meta()], np.int64)


def build_cache():
    """tiles uint8 [N,TMAX,224,224,3], count [N]. Tiles ranked by lesion coverage (>=15 %), at most TMAX per image."""
    import analyze as A, lesion_seg as LS, prep_real as PR
    M = A.models(); rows = meta(); N = len(rows)
    tiles = np.zeros((N, TMAX, TILE, TILE, 3), np.uint8); cnt = np.zeros(N, np.int64); fr = np.zeros((N, TMAX), np.float32)
    for n, r in enumerate(rows):
        p = PR.find_ci(RAWD + "images/", r["derm"])
        raw = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); s = 450 / min(raw.shape[:2])
        img = cv2.resize(raw, (int(round(raw.shape[1] * s)), int(round(raw.shape[0] * s))), interpolation=cv2.INTER_AREA)
        les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
        cand = []
        for y in A.positions(h):
            for x in A.positions(w):
                f = les[y:y + TILE, x:x + TILE].mean()
                if f >= .15:
                    cand.append((f, x, y))
        if not cand:                                    # segmentation failed: fall back to the centre tile
            cand = [(0., (w - TILE) // 2, (h - TILE) // 2)]
        cand = sorted(cand, reverse=True)[:TMAX]
        for k, (f, x, y) in enumerate(cand):
            tiles[n, k] = img[y:y + TILE, x:x + TILE]; fr[n, k] = f
        cnt[n] = len(cand)
        if n % 200 == 199:
            print(n + 1, flush=True)
    np.save("cache/d7_tiles.npy", tiles); np.save("cache/d7_tile_count.npy", cnt); np.save("cache/d7_tile_frac.npy", fr)
    print("tiles per image: mean", cnt.mean(), "min", cnt.min(), "max", cnt.max(), "| images with fallback tile:", int((fr[:, 0] == 0).sum()))


if __name__ == "__main__":
    build_cache()
