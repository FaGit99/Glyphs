"""Independent check on HAM10000 (never used for structure labels): melanoma vs nevus, 400 + 400 random images.

R   classic 7-point rule on the predicted checklist probabilities (no diagnosis training at all): new model / old model
B1  probe on the 14 checklist channels, fitted on Derm7pt train+valid out-of-fold outputs, applied to HAM
D   ImageNet-feature probe fitted on Derm7pt, applied to HAM
"""
import csv, json, sys, time
import cv2, numpy as np, torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import analyze as A, dense as D, hybrid as H, d7_data as DD, mil, skin

meta = {r["image_id"]: r["dx"] for r in csv.DictReader(open("/mnt/samsung_data/mole_data/data/metadata/HAM10000/HAM10000_metadata.csv"))}
files = {f.split("/")[-1][:-4]: f for f in skin.ham_files(1)}
rng = np.random.default_rng(0)
sel = []
for dx in ("mel", "nv"):
    c = sorted(i for i in files if meta.get(i) == dx); sel += [(i, int(dx == "mel")) for i in rng.choice(c, 400, replace=False)]
M = A.models(); new, ck = H.load_model("models/mel_final.pt"); old, _ = H.load_model("models/hybrid.pt")
imnet = timm.create_model(D.BACKBONE, pretrained=True, num_classes=0).to(D.dev).eval()
K = 76; P_new, P_old, F_in, y = [], [], [], []; t0 = time.time()
for n, (i, lab) in enumerate(sel):
    raw = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB); s = 450 / min(raw.shape[:2])
    img = cv2.resize(raw, (int(round(raw.shape[1] * s)), int(round(raw.shape[0] * s))), interpolation=cv2.INTER_AREA)
    les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
    cand = sorted([(les[yy:yy + 224, xx:xx + 224].mean(), xx, yy) for yy in A.positions(h) for xx in A.positions(w) if les[yy:yy + 224, xx:xx + 224].mean() >= .15], reverse=True)[:16]
    if not cand:
        cand = [(0., (w - 224) // 2, (h - 224) // 2)]
    x = np.stack([img[yy:yy + 224, xx:xx + 224] for _, xx, yy in cand])
    pn, _ = D.predict(new, x); po, _ = D.predict(old, x)
    P_new.append(np.sort(pn, 0)[::-1][:3].mean(0)); P_old.append(np.sort(po, 0)[::-1][:3].mean(0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        F_in.append(imnet(D.prep(torch.from_numpy(x), None, False)).float().mean(0).cpu().numpy())
    y.append(lab)
    if n % 100 == 99:
        print(f"{n + 1}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
P_new, P_old, F_in, y = np.stack(P_new), np.stack(P_old), np.stack(F_in), np.array(y)
np.savez("models/ham_check_feats.npz", P_new=P_new, P_old=P_old, F_in=F_in, y=y)

# --- evaluation ---
sp = DD.split(); pool = np.concatenate([sp["train"], sp["valid"]]); mel = DD.melanoma(); N = 1011
oof = np.zeros((N, K + 14), np.float32)
for k in range(5):
    z = np.load(f"models/mel_fold{k}_pred.npz"); oof[z["idx"]] = z["pred"]
FI7 = np.load("cache/d7_imnet_feat.npy")
def probe(Xtr, ytr, Xte):
    return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr).decision_function(Xte)
def ci(yy, s, n=500):
    r = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = r.integers(0, len(yy), len(yy))
        if 0 < yy[ix].sum() < len(ix):
            b.append(roc_auc_score(yy[ix], s[ix]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
nm = {n: DD.TNAMES.index(n) for n in DD.TNAMES}
def rule(P, get):
    return sum(w * get(P, n) for n, w in {"network_atypical": 2, "veil": 2, "streaks_irregular": 1, "pigmentation_irregular": 1, "dots_irregular": 1, "regression_any": 1}.items()) + 2 * np.maximum(get(P, "vessels_dotted"), get(P, "vessels_linear_irregular"))
get_new = lambda P, n: P[:, K + nm[n]]
ids = [str(i) for i in ck["ids"]]; idx = lambda l: [ids.index(i) for i in l]
BM = {"network_atypical": idx(["DS003", "DS004"]), "veil": idx(["DS049"]), "streaks_irregular": idx(["DS008", "DS009", "DS024"]) + [71 + 2], "pigmentation_irregular": [], "dots_irregular": idx(["DS025", "DS037"]) + [71 + 4],
      "regression_any": idx(["DS051"]), "vessels_dotted": idx(["DS060"]), "vessels_linear_irregular": idx(["DS065"])}
get_old = lambda P, n: P[:, BM[n]].max(1) if BM[n] else np.zeros(len(P))
print(f"\nHAM10000 melanoma vs nevus ({int(y.sum())}/{len(y)}), AUC [95% CI]")
rows = {"R 7-point rule, new glyph probabilities (no diagnosis training)": rule(P_new, get_new), "R 7-point rule, old glyph probabilities": rule(P_old, get_old),
        "B1 probe on 14 checklist channels (fitted on Derm7pt)": probe(oof[pool][:, K:], mel[pool], P_new[:, K:]), "D ImageNet-feature probe (fitted on Derm7pt)": probe(FI7[pool], mel[pool], F_in)}
res = {}
for k, s in rows.items():
    a = roc_auc_score(y, s); lo, hi = ci(y, s); res[k] = [float(a), float(lo), float(hi)]; print(f"{k:66s} {a:.3f} [{lo:.2f}-{hi:.2f}]")
json.dump(res, open("models/ham_check.json", "w"), indent=1)
