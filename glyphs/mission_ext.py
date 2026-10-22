"""Mission C6: frozen regime models on the 4 external sources of external_check.py (melanoma and nevus images only).

  python mission_ext.py      -> cache/mission_ext.npz  (per image: group, source, top-3-mean image-head probabilities [76] per regime)
Tiles: as d7_data.py (450 px short side, 224 px tiles at stride 112, lesion coverage >= 15 %, at most 16 ranked by coverage), so
the Derm7pt features (d7_mil in mission_pred.py) and the external features come from the same tiling.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
import analyze as A, dense as D, external_check as X, lesion_seg as LS, mission_pred as MP

FINAL = {"R0": MP.M + "dense_v1.pt", "R0pipe": MP.M + "mission_r0pipe_f0.pt", "R1": MP.M + "real_only.pt", "R2": "frozen_V2/models_full/hybrid.pt",
         "R2ctrl": MP.M + "hybrid_v3_noeasy.pt", "R3": MP.M + "hybrid_v3_f0.pt", "R5": MP.M + "mission_r5_f0.pt"}


def tiles_of(img, seg):
    les, _ = A.lesion_mask(img, seg); h, w = les.shape; cand = []
    for y in A.positions(h):
        for x in A.positions(w):
            f = les[y:y + 224, x:x + 224].mean()
            if f >= .15:
                cand.append((f, x, y))
    if not cand:
        cand = [(0., (w - 224) // 2, (h - 224) // 2)]
    return np.stack([img[y:y + 224, x:x + 224] for _, x, y in sorted(cand, reverse=True)[:16]])


def main():
    seg = LS.SegViT(pretrained=False).to(D.dev); seg.load_state_dict(torch.load(MP.M + "lesion_seg.pt", weights_only=False)["state"]); seg.eval()
    models = {r: MP.load(p) for r, p in FINAL.items()}
    out = {r: [] for r in FINAL}; grp, src = [], []
    for name, items in X.datasets().items():
        items = [it for it in items if it[1] in ("MEL", "NV")]; t0 = time.time(); print(name, len(items), flush=True)
        with ThreadPoolExecutor(8) as ex:
            for n, ((path, g), img) in enumerate(zip(items, ex.map(lambda it: X._safe(X.load450, it[0]), items))):
                if img is None:
                    continue
                t = tiles_of(img, seg)
                for r, (m, k) in models.items():
                    pi, _ = MP.run(m, k, t); out[r].append(np.sort(pi, 0)[::-1][:3].mean(0))
                grp.append(g); src.append(name.split()[0])
                if n % 500 == 499:
                    print(f"  {n + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    np.savez("cache/mission_ext.npz", group=np.array(grp), source=np.array(src), **{r: np.stack(v) for r, v in out.items()}); print("wrote cache/mission_ext.npz", len(grp))


if __name__ == "__main__":
    main()
