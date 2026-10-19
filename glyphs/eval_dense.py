"""Localisation evaluation of a DenseViT checkpoint.

1. Synthetic val: patch-level AUC and best-threshold IoU per ID (upper bound).
2. Real ISIC 2018 Task 2 (5 attributes, native-pitch tiles): patch AUC / IoU against the attribute masks, pointing game.
3. Zone plausibility: share of predicted mass inside / border / outside the lesion (ISIC Task 1 masks) - real vs synthetic.
4. Overlay examples for visual review (out/loc_examples.png).
Usage: python eval_dense.py <tag>
"""
import csv, json, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import roc_auc_score, precision_recall_curve
import dense as D

tag = sys.argv[1]
ck = torch.load(f"models/{tag}.pt", map_location="cpu")
ids = [str(i) for i in ck["ids"]]
model = D.DenseViT(len(ids), pretrained=False).to(D.dev); model.load_state_dict(ck["state"]); model.eval()
tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
ATTRS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
amap = {a: [ids.index(i) for i in ids if i in tax and a in tax[i]["ISIC 2018 Task 2 label"]] for a in ATTRS}


def pool14(x):  # coverage maps [...,28,28] 0-255 -> [...,14,14] 0-1
    t = torch.as_tensor(x).float() / 255.
    return F.avg_pool2d(t.reshape(-1, 1, 28, 28), 2).reshape(*t.shape[:-2], 14, 14).numpy()


def best_iou(gt, p):
    """best IoU over thresholds for flat binary gt and scores p"""
    pr, rc, th = precision_recall_curve(gt, p)
    iou = pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1] - pr[:-1] * rc[:-1], 1e-9)
    k = int(np.argmax(iou)); return float(iou[k]), float(th[k])


def zones(les14, k=1):
    """les14 bool [14,14] -> (core, border, outside) bool; border = +-k cells around the contour."""
    d, e = binary_dilation(les14, iterations=k), binary_erosion(les14, iterations=k)
    return e, d & ~e, ~d


# ---------------- 1. synthetic validation ----------------
d = np.load("data/synth_val.npz"); Xv, Rv, Ev = d["images"], d["regions"], d["lesion"]
_, Pv = D.predict(model, Xv)
gtv = pool14(Rv)
print("=== 1. synthetic val (upper bound): patch-level localisation per ID ===")
res = []
for j, sid in enumerate(ids):
    g = (gtv[:, j] >= .25).ravel()
    if g.sum() < 20:
        continue
    iou, th = best_iou(g, Pv[:, j].ravel()); res.append((sid, roc_auc_score(g, Pv[:, j].ravel()), iou))
print(f"IDs evaluated {len(res)}; patch AUC mean {np.mean([r[1] for r in res]):.3f}, IoU mean {np.mean([r[2] for r in res]):.3f}")
res.sort(key=lambda r: r[2])
print("lowest IoU :", ", ".join(f"{s}:{i:.2f}" for s, a, i in res[:8]))
print("highest IoU:", ", ".join(f"{s}:{i:.2f}" for s, a, i in res[-6:]), flush=True)

# synthetic zone shares (reference for plausibility)
def zone_share(P, les14, thr=.5):
    """per image: share of predicted probability mass in outside zone (over cells; only cells above thr count)."""
    out = []
    for p, l in zip(P, les14):
        if l.sum() < 5:
            continue
        c, b, o = zones(l)
        m = p * (p > thr)
        tot = m.sum()
        if tot > 0:
            out.append(m[o].sum() / tot)
    return np.array(out)

les_v = F.avg_pool2d(torch.as_tensor(Ev).float()[:, None] / 255., 4)[:, 0].numpy() > .5   # 56 -> 14
syn_out = {sid: zone_share(Pv[:, j], les_v) for j, sid in enumerate(ids)}
allv = np.concatenate([v for v in syn_out.values() if len(v)])
print(f"synthetic: share of detected mass outside lesion+border: mean {allv.mean():.3f} (over {len(allv)} detections)\n", flush=True)

# ---------------- 2. real ISIC ----------------
img = np.load("cache/isic18_img.npy", mmap_mode="r")[:2594]
mm = np.load("cache/isic18_maskmaps.npy"); ls = np.load("cache/isic18_lesion.npy")      # [N,6,5,28,28], [N,6,28,28]
N = len(img); tiles = np.ascontiguousarray(img[:, :5].reshape(N * 5, 224, 224, 3))
_, P = D.predict(model, tiles)                                                              # [N*5, K, 14, 14]
gt = pool14(mm[:, :5].reshape(N * 5, 5, 28, 28)); les14 = (pool14(ls[:, :5].reshape(N * 5, 28, 28)) > .5)
inl = np.stack([binary_dilation(l, iterations=1) for l in les14])                           # lesion + border cells (diagnostic zone)
print("=== 2. real ISIC 2018 Task 2: patch-level localisation (tiles at native pitch) ===")
print(f"{'attribute':17s} {'AUC all':>8s} {'AUC lesion+rim':>15s} {'IoU@best':>9s} {'IoU@0.5':>8s} {'pointing':>9s}")
real = {}
for k, a in enumerate(ATTRS):
    if not amap[a]:
        continue
    s = P[:, amap[a]].max(1)                                  # [T,14,14]
    g = gt[:, k] >= .25
    ok = g.ravel().sum() > 20
    if not ok:
        continue
    auc_all = roc_auc_score(g.ravel(), s.ravel())
    zi = inl.ravel(); auc_in = roc_auc_score(g.ravel()[zi], s.ravel()[zi]) if 0 < g.ravel()[zi].sum() < zi.sum() else float("nan")
    iou, th = best_iou(g.ravel(), s.ravel())
    p5 = s.ravel() > .5; iou5 = (p5 & g.ravel()).sum() / max((p5 | g.ravel()).sum(), 1)
    has = np.nonzero(gt[:, k].reshape(len(gt), -1).max(1) >= .25)[0]
    hit = np.mean([g[t].ravel()[s[t].ravel().argmax()] for t in has])
    # baselines: "whole lesion+rim is positive" (no structure knowledge) and a random point inside lesion+rim
    bl = inl.ravel().astype(float)
    b_auc = roc_auc_score(g.ravel(), bl); b_iou = (inl.ravel() & g.ravel()).sum() / max((inl.ravel() | g.ravel()).sum(), 1)
    chance = np.mean([(g[t] & inl[t]).sum() / max(inl[t].sum(), 1) for t in has])
    print(f"{a:17s} {auc_all:8.3f} {auc_in:15.3f} {iou:9.3f} {iou5:8.3f} {hit:9.3f}  | baseline lesion=pos: AUC {b_auc:.3f} IoU {b_iou:.3f}; random-point-in-lesion hit {chance:.3f}", flush=True)
    real[a] = dict(baseline_auc=float(b_auc), baseline_iou=float(b_iou), pointing_chance=float(chance), auc_all=auc_all, auc_lesion_rim=auc_in, iou_best=iou, iou_at_05=float(iou5), pointing=float(hit), thr=th)

# ---------------- 3. zone plausibility on real tiles ----------------
print("\n=== 3. plausibility: where is the detected mass? (real ISIC tiles with a visible lesion) ===")
zon = {}
for k, a in enumerate(ATTRS):
    if not amap[a]:
        continue
    s = P[:, amap[a]].max(1)
    sh = zone_share(s, les14)
    pos_gt = [t for t in range(len(s)) if gt[t, k].max() >= .25 and les14[t].sum() >= 5]
    # background silence: mean prob outside lesion+border on tiles with a lesion
    bg = np.mean([s[t][~binary_dilation(les14[t], iterations=1)].mean() for t in range(len(s)) if les14[t].sum() >= 5 and (~binary_dilation(les14[t], iterations=1)).sum() > 10])
    print(f"{a:17s} detected tiles {len(sh):5d}: outside-share mean {sh.mean() if len(sh) else float('nan'):.3f}   mean prob in outside zone {bg:.3f}")
    zon[a] = dict(outside_share=float(sh.mean()) if len(sh) else None, outside_prob=float(bg), n=len(sh))
print(f"(synthetic reference outside-share: {allv.mean():.3f})")
json.dump(dict(synthetic=dict(patch_auc=float(np.mean([r[1] for r in res])), iou=float(np.mean([r[2] for r in res])), outside_share=float(allv.mean())),
               real=real, zones=zon), open(f"models/loc_{tag}.json", "w"), indent=1)

# ---------------- 4. overlays ----------------
rng = np.random.default_rng(0); cells = []; cmap = cv2.COLORMAP_TURBO
picks = []
for k, a in enumerate(ATTRS):
    has = [t for t in range(len(gt)) if gt[t, k].max() >= .3 and les14[t].sum() >= 20]
    picks += [(a, k, t) for t in rng.choice(has, 2, replace=False)] if len(has) >= 2 else []
for a, k, t in picks:
    im = cv2.cvtColor(cv2.resize(tiles[t], (320, 320), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    s = P[t, amap[a]].max(0)
    heat = cv2.applyColorMap((cv2.resize(s, (320, 320), interpolation=cv2.INTER_CUBIC).clip(0, 1) * 255).astype(np.uint8), cmap)
    ov = cv2.addWeighted(im, .55, heat, .45, 0)
    cl, _ = cv2.findContours((cv2.resize(les14[t].astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cl, -1, (255, 255, 255), 1)
    g = cv2.resize((gt[t, k] >= .25).astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    cg, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); im2 = im.copy(); cv2.drawContours(im2, cg, -1, (0, 255, 255), 2)
    cap = np.full((22, 640, 3), 255, np.uint8); cv2.putText(cap, f"{a}   left: ISIC mask (yellow)   right: predicted region (white = lesion)", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    cells.append(np.vstack([cap, np.hstack([im2, ov])]))
cv2.imwrite("out/loc_examples.png", np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells) - len(cells) % 2, 2)]))
print("wrote out/loc_examples.png")
