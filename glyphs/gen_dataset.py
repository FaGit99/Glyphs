"""Generate a synthetic multi-label dataset with region targets.

images  [N,224,224,3] uint8
labels  [N,n_ids]     uint8   IDs recorded in the scene (visible in the frame)
regions [N,n_ids,28,28] uint8 region mask (instances unioned + closed, so scattered clods form one region), coverage 0-255 per cell
lesion  [N,56,56]     uint8   lesion mask (0/255 area-averaged), for zone statistics

Usage: python gen_dataset.py <n_train> <n_val>
"""
import sys, time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

REGION_CLOSE = 21   # px at 224: structures closer than this belong to one region
GRID = 28


def region_targets(masks, ids):
    out = np.zeros((len(ids), GRID, GRID), np.uint8)
    union = {}
    for key, m in masks.items():
        sid = key.split("_")[0]
        union[sid] = np.maximum(union.get(sid, 0), m.astype(np.uint8))
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REGION_CLOSE, REGION_CLOSE))
    for sid, m in union.items():
        m = cv2.morphologyEx(cv2.dilate(m, se1), cv2.MORPH_CLOSE, se2)
        out[ids.index(sid)] = np.clip(cv2.resize(m.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA) * 255, 0, 255)
    return out


def work(args):
    seed0, n = args
    import compose, structures as S, skin
    files = skin.ham_files(0)
    ids = sorted(S.SPEC)
    imgs, labs = np.zeros((n, 224, 224, 3), np.uint8), np.zeros((n, len(ids)), np.uint8)
    regs, les = np.zeros((n, len(ids), GRID, GRID), np.uint8), np.zeros((n, 56, 56), np.uint8)
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        k = (0, 7) if rng.random() < .9 else (1, 2)       # ~1/7 structure-free scenes, otherwise 1-6 IDs
        req = compose.random_scene_ids(rng, k=k)
        img, labels, masks, _, lesion = compose.compose(rng, req, 448, 224, skin_files=files)
        imgs[i] = img
        for lab in labels:
            labs[i, ids.index(lab["id"])] = 1
        regs[i] = region_targets(masks, ids)
        les[i] = np.clip(cv2.resize((lesion > .5).astype(np.float32), (56, 56), interpolation=cv2.INTER_AREA) * 255, 0, 255)
    return imgs, labs, regs, les


def build(name, seed0, n, workers=20, chunk=250):
    t = time.time()
    jobs = [(seed0 + j, min(chunk, n - j)) for j in range(0, n, chunk)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(work, jobs))
    import structures as S
    d = {k: np.concatenate([r[i] for r in res]) for i, k in enumerate(["images", "labels", "regions", "lesion"])}
    np.savez(f"data/synth_{name}.npz", ids=np.array(sorted(S.SPEC)), **d)
    print(f"{name}: {n} images in {time.time() - t:.0f}s; labels/img {d['labels'].sum(1).mean():.2f}; "
          f"mean region coverage of lesion-having images {(d['regions'] > 127).mean():.4f}", flush=True)


if __name__ == "__main__":
    build("val", 5_000_000, int(sys.argv[2]))
    build("train", 0, int(sys.argv[1]))
