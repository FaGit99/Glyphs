"""Transfer test: synthetic-only model on real labels (ISIC 2018 Task 2 masks, Derm7pt).

Arm A  zero-shot : synthetic-trained model, score = max prob over the IDs mapped to the real label
Arm B  probe-IN  : frozen ImageNet ViT-S features + logistic regression trained on real labels (CV)
Arm C  probe-SYN : frozen features of the synthetic-trained backbone + the same probe (CV)
B/C show how much the real labels are learnable at all and whether synthetic training helps the features.
"""
import csv, json, re, sys, time
import numpy as np, torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import train as T

tag = sys.argv[1]
dev = "cuda"
ckpt = torch.load(f"models/{tag}.pt", map_location="cpu", weights_only=False)  # own checkpoint; ids may be numpy str in v1
ids = [str(i) for i in ckpt["ids"]]
model = T.build(len(ids)); model.load_state_dict(ckpt["state"]); model.eval()
imnet = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0).to(dev).eval()

# ---- label mapping from the taxonomy CSV --------------------------------------------------
tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
drawn = [i for i in ids if i in tax]
isic_map = {a: [i for i in drawn if a in tax[i]["ISIC 2018 Task 2 label"]] for a in T_ISIC} if (T_ISIC := ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]) else {}
d7col = {i: tax[i]["Derm7pt label"] for i in drawn}
VASC = {"DOT": "dotted", "HP": "hairpin", "LIR": "linear irregular", "COM": "comma", "ARB": "arborizing"}
d7_map = {
    "network (any)": [i for i in drawn if "pigment_network" in d7col[i]],
    "network (atypical)": [i for i in drawn if "pigment_network" in d7col[i] and "ATP" in d7col[i] and "TYP" not in d7col[i]],
    "streaks": [i for i in drawn if d7col[i].startswith("streaks")],
    "dots/globules": [i for i in drawn if "dots_and_globules" in d7col[i]],
    "blue-whitish veil": [i for i in drawn if "blue_whitish_veil" in d7col[i]],
    "regression (white/blue)": [i for i in drawn if d7col[i].startswith("regression_structures")],
}
for code, name in VASC.items():
    d7_map[f"vessels: {name}"] = [i for i in drawn if re.search(rf"vascular_structures = {code}\b", d7col[i])]
print("ISIC mapping:", isic_map); print("Derm7pt mapping:", d7_map, flush=True)


@torch.no_grad()
def run(net, imgs, head=True, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        x = T.prep(torch.from_numpy(imgs[i:i + bs]), False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(x) if head else net.forward_head(net.forward_features(x), pre_logits=True)
        out.append((torch.sigmoid(o.float()) if head else o.float()).cpu())
    return torch.cat(out).numpy()


def auc_ci(y, s, groups, n=200, seed=0):
    if y.sum() < 3 or (1 - y).sum() < 3:
        return None
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed); ug = np.unique(groups); by = {g: np.nonzero(groups == g)[0] for g in ug}
    bs = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[idx].sum() < len(idx):
            bs.append(roc_auc_score(y[idx], s[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cv_probe(X, y, groups, folds=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=.1, max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        p[te] = clf.decision_function((X[te] - mu) / sd)
    return p


def fmt(r):
    return "  n/a  " if r is None else f"{r[0]:.3f} [{r[1]:.2f}-{r[2]:.2f}]"


results = {"tag": tag, "isic": {}, "derm7pt": {}}
t0 = time.time()
# ---- ISIC 2018 Task 2 : tile-level labels from the masks -----------------------------------
img = np.load("cache/isic18_img.npy", mmap_mode="r"); frac = np.load("cache/isic18_frac.npy")
N = len(img); flat = np.ascontiguousarray(img.reshape(N * 6, 224, 224, 3))
view = np.tile(np.arange(6), N); gid = np.repeat(np.arange(N), 6)
P = run(model, flat); FA = run(model, flat, head=False); FI = run(imnet, flat, head=False)
fr = frac.reshape(N * 6, -1)
print(f"ISIC features done {time.time() - t0:.0f}s: {N} images", flush=True)
print("\n=== ISIC 2018 Task 2 (tiles at native pitch = views 0-4; 'whole' = view 5) ===")
print(f"{'attribute':18s} {'#pos/#neg tiles':>16s}  {'A zero-shot':>20s}  {'B probe-ImageNet':>20s}  {'C probe-synth':>20s}   whole-view A")
for k, a in enumerate(T_ISIC):
    col = [ids.index(i) for i in isic_map[a]]
    if not col:
        continue
    score = P[:, col].max(1)
    y = np.where(fr[:, k] >= .01, 1, np.where(fr[:, k] == 0, 0, -1))
    m = (view < 5) & (y >= 0)
    ra = auc_ci(y[m], score[m], gid[m])
    rb = auc_ci(y[m], cv_probe(FI[m], y[m], gid[m]), gid[m]); rc = auc_ci(y[m], cv_probe(FA[m], y[m], gid[m]), gid[m])
    mw = (view == 5) & (y >= 0); rw = auc_ci(y[mw], score[mw], gid[mw])
    print(f"{a:18s} {int(y[m].sum()):7d}/{int((1 - y[m]).sum()):<8d} {fmt(ra):>20s}  {fmt(rb):>20s}  {fmt(rc):>20s}   {fmt(rw)}", flush=True)
    results["isic"][a] = dict(ids=isic_map[a], pos=int(y[m].sum()), neg=int((1 - y[m]).sum()), zero_shot=ra, probe_imagenet=rb, probe_synth=rc, whole_zero_shot=rw)

# ---- Derm7pt : image-level labels, score = max over the 5 tiles -------------------------------
d7 = np.load("cache/d7_img.npy"); lab = json.load(open("cache/d7_labels.json")); M = len(d7)
flat = d7.reshape(M * 6, 224, 224, 3); P = run(model, flat).reshape(M, 6, -1)
FA = run(model, flat, head=False).reshape(M, 6, -1)[:, :5].mean(1); FI = run(imnet, flat, head=False).reshape(M, 6, -1)[:, :5].mean(1)
vas = np.array(lab["vascular_structures"]); gid = np.arange(M)
tg = {"network (any)": np.array(lab["pigment_network"]) != "absent", "network (atypical)": np.array(lab["pigment_network"]) == "atypical",
      "streaks": np.array(lab["streaks"]) != "absent", "dots/globules": np.array(lab["dots_and_globules"]) != "absent",
      "blue-whitish veil": np.array(lab["blue_whitish_veil"]) == "present", "regression (white/blue)": np.array(lab["regression_structures"]) != "absent"}
for code, name in VASC.items():
    tg[f"vessels: {name}"] = vas == name
print("\n=== Derm7pt (image-level; A = max over 5 tiles) ===")
print(f"{'target':24s} {'#pos/#neg':>10s}  {'A zero-shot':>20s}  {'B probe-ImageNet':>20s}  {'C probe-synth':>20s}   whole-view A")
for name, y in tg.items():
    col = [ids.index(i) for i in d7_map[name]]
    if not col:
        continue
    y = y.astype(int); s = P[:, :5, col].max((1, 2)); sw = P[:, 5, col].max(1)
    ra, rw = auc_ci(y, s, gid), auc_ci(y, sw, gid)
    rb = auc_ci(y, cv_probe(FI, y, gid), gid); rc = auc_ci(y, cv_probe(FA, y, gid), gid)
    print(f"{name:24s} {y.sum():4d}/{(1 - y).sum():<5d} {fmt(ra):>20s}  {fmt(rb):>20s}  {fmt(rc):>20s}   {fmt(rw)}", flush=True)
    results["derm7pt"][name] = dict(ids=d7_map[name], pos=int(y.sum()), neg=int((1 - y).sum()), zero_shot=ra, probe_imagenet=rb, probe_synth=rc, whole_zero_shot=rw)
json.dump(results, open(f"models/transfer_{tag}.json", "w"), indent=1)
print(f"\ndone in {time.time() - t0:.0f}s")
