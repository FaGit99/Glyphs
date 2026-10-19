"""Hybrid training (synthetic + real ISIC masks) and calibrated localisation evaluation.

Channels: 71 taxonomy IDs (synthetic supervision only) + 5 ISIC attribute channels (shared: synthetic = union of the
mapped IDs, real = ISIC Task 2 masks). Real images are masked out of the loss for the 71 ID channels.

  python hybrid.py train hybrid   [epochs]   # synthetic + real train split
  python hybrid.py train real_only [epochs]  # real train split only, 5 outputs
  python hybrid.py eval                      # thresholds on the calibration split, metrics on the test split
"""
import csv, json, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import binary_dilation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
import dense as D

ATTRS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
MAP_W, POS_W = 5.0, 3.0
tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}


def id_list():
    return [str(i) for i in np.load("data/synth_val.npz")["ids"]]


def attr_ids(ids):
    return {a: [ids.index(i) for i in ids if i in tax and a in tax[i]["ISIC 2018 Task 2 label"]] for a in ATTRS}


def pool14(x):  # coverage 0-255 [...,28,28] -> 0-1 [...,14,14]
    t = torch.as_tensor(np.asarray(x)).float() / 255.
    return F.avg_pool2d(t.reshape(-1, 1, 28, 28), 2).reshape(*t.shape[:-2], 14, 14).numpy()


def real_split():
    """block split of the 2594 ISIC training images: blocks of 25 consecutive images -> 60/20/20 train/cal/test"""
    n = 2594
    b = (np.arange(n) // 25) % 5
    return {"train": np.nonzero(b <= 2)[0], "cal": np.nonzero(b == 3)[0], "test": np.nonzero(b == 4)[0]}


def real_data(idx):
    img = np.load("cache/isic18_img.npy", mmap_mode="r")
    mm = np.load("cache/isic18_maskmaps.npy", mmap_mode="r"); ls = np.load("cache/isic18_lesion.npy", mmap_mode="r")
    X = np.ascontiguousarray(img[idx][:, :5].reshape(-1, 224, 224, 3))
    A = np.ascontiguousarray(mm[idx][:, :5].reshape(-1, 5, 28, 28)); L = np.ascontiguousarray(ls[idx][:, :5].reshape(-1, 28, 28))
    return X, A, L


def synth_attr(regions, labels, amap):
    """attribute maps/labels from the ID region maps: union (max) over the mapped IDs"""
    A = np.stack([regions[:, amap[a]].max(1) for a in ATTRS], 1)
    y = np.stack([labels[:, amap[a]].max(1) for a in ATTRS], 1)
    return A, y


def masked_bce(logits, target, mask, pos_weight=None):
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (l * mask).sum() / mask.sum().clamp(min=1)


def train(mode, epochs):
    ids = id_list(); K = len(ids); amap = attr_ids(ids)
    sp = real_split(); Xr, Ar, _ = real_data(sp["train"])
    Xr = torch.from_numpy(Xr); Ar = torch.from_numpy(Ar)
    yr = (Ar.float().mean((2, 3)) / 255. >= .01).float()                       # tile-level attribute presence
    if mode == "hybrid":
        d = np.load("data/synth_train.npz")
        Xs, Rs, Ys = torch.from_numpy(d["images"]), torch.from_numpy(d["regions"]), torch.from_numpy(d["labels"]).float()
        As, ys = synth_attr(d["regions"], d["labels"], amap); As, ys = torch.from_numpy(As), torch.from_numpy(ys).float()
        n_out, nS, nR, per_epoch = K + 5, 96, 32, len(Xs) // 96
    else:
        n_out, nS, nR, per_epoch = 5, 0, 128, len(Xr) // 128
    model = D.DenseViT(n_out).to(D.dev)
    steps = epochs * per_epoch
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4},
                             {"params": list(model.img.parameters()) + list(model.pat.parameters()), "lr": 1e-3}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=.1)
    pw = torch.tensor(POS_W, device=D.dev); t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xs) if nS else len(Xr)); tl = 0
        for it in range(per_epoch):
            ri = torch.randint(0, len(Xr), (nR,))
            if nS:
                si = perm[it * nS:(it + 1) * nS]
                x = torch.cat([Xs[si], Xr[ri]])
                maps = torch.zeros(nS + nR, K + 5, 28, 28); maps[:nS, :K] = Rs[si].float(); maps[:nS, K:] = As[si].float(); maps[nS:, K:] = Ar[ri].float()
                yimg = torch.zeros(nS + nR, K + 5); yimg[:nS, :K] = Ys[si]; yimg[:nS, K:] = ys[si]; yimg[nS:, K:] = yr[ri]
                mask = torch.ones(nS + nR, K + 5); mask[nS:, :K] = 0
            else:
                ri = perm[it * nR:(it + 1) * nR]
                x, maps, yimg, mask = Xr[ri], Ar[ri].float(), yr[ri], torch.ones(nR, 5)
            x, maps = D.prep(x, maps, True)
            yimg, mask = yimg.to(D.dev), mask.to(D.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                li, lm = model(x)
            tgt = F.avg_pool2d(maps / 255., 2)
            loss = masked_bce(li.float(), yimg, mask) + MAP_W * masked_bce(lm.float(), tgt, mask[:, :, None, None].expand_as(tgt), pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        print(f"epoch {ep + 1}/{epochs} loss {tl / per_epoch:.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "ids": ids, "n_out": n_out, "mode": mode}, f"models/{mode}.pt")


# ------------------------------------------------------------------------------------------------
def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = D.DenseViT(ck.get("n_out", len(ck["ids"])), pretrained=False).to(D.dev); m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def zones_inl(les14):
    return np.stack([binary_dilation(l, iterations=1) for l in les14])


def best_thr(g, p):
    pr, rc, th = precision_recall_curve(g, p)
    iou = pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1] - pr[:-1] * rc[:-1], 1e-9)
    k = int(np.argmax(iou)); return float(th[k])


def ece(p, g, bins=10):
    e, n = 0.0, len(p)
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - g[m].mean())
    return e


def evaluate():
    ids = id_list(); amap = attr_ids(ids); sp = real_split()
    splits = {k: real_data(sp[k]) for k in ("cal", "test")}
    arms = {}
    m, _ = load_model("models/dense_v1.pt"); arms["synth_only"] = ("ids", m)
    m, _ = load_model("models/real_only.pt"); arms["real_only"] = ("attr5", m)
    m, _ = load_model("models/hybrid.pt"); arms["hybrid"] = ("attr_ext", m)

    def attr_maps(arm, tiles):
        kind, model = arms[arm]
        pi, pm = D.predict(model, tiles)
        if kind == "ids":
            return np.stack([pm[:, amap[a]].max(1) for a in ATTRS], 1), np.stack([pi[:, amap[a]].max(1) for a in ATTRS], 1)
        if kind == "attr5":
            return pm, pi
        return pm[:, -5:], pi[:, -5:]

    pred = {arm: {s: attr_maps(arm, splits[s][0]) for s in splits} for arm in arms}
    gt = {s: pool14(splits[s][1]) for s in splits}                # [T,5,14,14] coverage
    les = {s: pool14(splits[s][2]) > .5 for s in splits}
    inl = {s: zones_inl(les[s]) for s in splits}
    out = {}
    print(f"cal tiles {len(splits['cal'][0])}, test tiles {len(splits['test'][0])} (blocks of 25 consecutive images, no overlap)\n")
    print("per attribute on the TEST split; thresholds chosen on the CAL split (best patch IoU).")
    print(f"{'attribute':17s} {'arm':11s} {'thr':>5s} {'IoU':>6s} {'Dice':>6s} {'prec':>5s} {'rec':>5s} {'AUC(l+r)':>9s} {'point':>6s} {'tileAUC':>8s} {'ECE raw>platt':>14s}")
    for k, a in enumerate(ATTRS):
        gc, gtst = (gt["cal"][:, k] >= .25), (gt["test"][:, k] >= .25)
        has = np.nonzero(gtst.reshape(len(gtst), -1).any(1))[0]
        b_iou = (inl["test"].ravel() & gtst.ravel()).sum() / max((inl["test"].ravel() | gtst.ravel()).sum(), 1)
        chance = np.mean([(gtst[t] & inl["test"][t]).sum() / max(inl["test"][t].sum(), 1) for t in has])
        tile_y = (gt["test"][:, k].mean((1, 2)) >= .01).astype(int)
        counts = {}
        for arm in arms:
            pc, pt = pred[arm]["cal"][0][:, k], pred[arm]["test"][0][:, k]
            thr = best_thr(gc.ravel(), pc.ravel())
            pb = pt >= thr
            tp = (pb & gtst).sum(); fp = (pb & ~gtst).sum(); fn = (~pb & gtst).sum()
            iou = tp / max(tp + fp + fn, 1); dice = 2 * tp / max(2 * tp + fp + fn, 1); prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            zi = inl["test"].ravel(); auc = roc_auc_score(gtst.ravel()[zi], pt.ravel()[zi])
            point = np.mean([gtst[t].ravel()[pt[t].ravel().argmax()] for t in has])
            tauc = roc_auc_score(tile_y, pred[arm]["test"][0][:, k].reshape(len(pt), -1).max(1))
            # Platt calibration on the calibration split (lesion+rim patches), ECE before/after on the test split
            zc = inl["cal"].ravel(); lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            pl = LogisticRegression(C=100).fit(lg(pc.ravel()[zc])[:, None], gc.ravel()[zc])
            cal_p = pl.predict_proba(lg(pt.ravel()[zi])[:, None])[:, 1]
            e0, e1 = ece(pt.ravel()[zi], gtst.ravel()[zi]), ece(cal_p, gtst.ravel()[zi])
            print(f"{a:17s} {arm:11s} {thr:5.2f} {iou:6.3f} {dice:6.3f} {prec:5.2f} {rec:5.2f} {auc:9.3f} {point:6.3f} {tauc:8.3f} {e0:6.3f}>{e1:6.3f}")
            per_img = {n: np.bincount(np.arange(len(pb)) // 5, weights=v.reshape(len(pb), -1).sum(1)) for n, v in (("tp", pb & gtst), ("fp", pb & ~gtst), ("fn", ~pb & gtst))}
            counts[arm] = per_img
            out.setdefault(a, {})[arm] = dict(thr=thr, iou=float(iou), dice=float(dice), precision=float(prec), recall=float(rec), auc_lesion_rim=float(auc),
                                              pointing=float(point), tile_auc=float(tauc), ece_raw=float(e0), ece_platt=float(e1),
                                              platt=[float(pl.coef_[0, 0]), float(pl.intercept_[0])])
        # bootstrap over images (test split) for IoU and for the paired hybrid - real_only difference
        rng_b = np.random.default_rng(0); nimg = len(counts["hybrid"]["tp"]); bs = {arm: [] for arm in arms}; dd = []
        for _ in range(500):
            ix = rng_b.integers(0, nimg, nimg)
            io = {arm: counts[arm]["tp"][ix].sum() / max((counts[arm]["tp"][ix] + counts[arm]["fp"][ix] + counts[arm]["fn"][ix]).sum(), 1) for arm in arms}
            for arm in arms:
                bs[arm].append(io[arm])
            dd.append(io["hybrid"] - io["real_only"])
        ci = lambda v: f"[{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}]"
        print(f"{'':17s} IoU 95% CI  " + "  ".join(f"{arm} {ci(bs[arm])}" for arm in arms) + f"   hybrid-real_only {np.mean(dd):+.3f} {ci(dd)}")
        for arm in arms:
            out[a][arm]["iou_ci"] = [float(np.percentile(bs[arm], 2.5)), float(np.percentile(bs[arm], 97.5))]
        out[a]["hybrid_minus_real_only"] = [float(np.mean(dd)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        print(f"{'':17s} {'(baseline)':11s} {'':>5s} {b_iou:6.3f}  lesion=positive; random point in lesion+rim hits {chance:.3f}\n")
        out[a]["baseline"] = dict(iou=float(b_iou), pointing_chance=float(chance))

    # zone plausibility of the hybrid on the test tiles
    print("zone plausibility (hybrid, test tiles with a visible lesion): share of predicted mass outside lesion+border")
    for k, a in enumerate(ATTRS):
        thr = out[a]["hybrid"]["thr"]; sh = []
        for p, z in zip(pred["hybrid"]["test"][0][:, k], inl["test"]):
            if z.sum() < 5:
                continue
            m = p * (p >= thr)
            if m.sum() > 0:
                sh.append(m[~z].sum() / m.sum())
        print(f"  {a:17s} tiles with detections {len(sh):4d}  outside-share {np.mean(sh) if sh else float('nan'):.3f}")
        out[a]["hybrid"]["outside_share"] = float(np.mean(sh)) if sh else None

    # synthetic ID channels: did the hybrid lose anything vs dense_v1?
    d = np.load("data/synth_val.npz"); Rv = pool14(d["regions"]); res = {}
    for arm in ("synth_only", "hybrid"):
        pm = D.predict(arms[arm][1], d["images"])[1][:, :len(ids)]
        ious = []
        for j in range(len(ids)):
            g = (Rv[:, j] >= .25).ravel()
            if g.sum() >= 20:
                t = best_thr(g, pm[:, j].ravel()); p = pm[:, j].ravel() >= t; ious.append((p & g).sum() / max((p | g).sum(), 1))
        res[arm] = float(np.mean(ious)); print(f"synthetic val, 71-ID channels, {arm}: mean patch IoU {res[arm]:.3f} over {len(ious)} IDs")
    out["synthetic_val_iou"] = res
    json.dump(out, open("models/hybrid_eval.json", "w"), indent=1)
    cal = {a: dict(thr=out[a]["hybrid"]["thr"], platt=out[a]["hybrid"]["platt"]) for a in ATTRS}
    json.dump(cal, open("models/hybrid_calibration.json", "w"), indent=1)

    # overlays on the test split: GT | synthetic-only | hybrid (thresholded at the calibrated value)
    rng = np.random.default_rng(0); rows = []
    tiles = splits["test"][0]
    for k, a in enumerate(ATTRS):
        has = [t for t in range(len(tiles)) if (gt["test"][t, k] >= .25).sum() >= 12 and les["test"][t].sum() >= 30]
        for t in (rng.choice(has, 2, replace=False) if len(has) >= 2 else []):
            im = cv2.cvtColor(cv2.resize(tiles[t], (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
            cells = []
            for title, mk in (("ISIC mask", None), ("synthetic-only", "synth_only"), ("hybrid", "hybrid")):
                c = im.copy()
                if mk is None:
                    mm_ = cv2.resize((gt["test"][t, k] >= .25).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                else:
                    p = pred[mk]["test"][0][t, k]
                    mm_ = cv2.resize((p >= out[a][mk]["thr"]).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                    heat = cv2.applyColorMap((cv2.resize(p, (256, 256), interpolation=cv2.INTER_CUBIC).clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    c = cv2.addWeighted(c, .6, heat, .4, 0)
                cs, _ = cv2.findContours(mm_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(c, cs, -1, (0, 255, 255), 2)
                cl, _ = cv2.findContours(cv2.resize(les["test"][t].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(c, cl, -1, (255, 255, 255), 1)
                cv2.putText(c, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1); cells.append(c)
            cap = np.full((20, 768, 3), 255, np.uint8); cv2.putText(cap, f"{a}  (yellow = region above the calibrated threshold, white = lesion)", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
            rows.append(np.vstack([cap, np.hstack(cells)]))
    cv2.imwrite("out/hybrid_examples.png", np.vstack(rows)); print("wrote out/hybrid_examples.png, models/hybrid_eval.json, models/hybrid_calibration.json")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else (16 if sys.argv[2] == "hybrid" else 30))
    else:
        evaluate()
