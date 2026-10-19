"""Evaluation of the melanoma-structure improvement on the official Derm7pt split (test = 395 images, never used for training).

Structure level : AUC of the 14 checklist channels, old (hybrid, zero-shot) vs new (fine-tuned with Derm7pt MIL labels).
Melanoma level  : one-vs-rest AUC on the test split for
   A  old glyph outputs (76 hybrid channels) -> logistic probe fitted on train+valid
   B  new glyph outputs (stacked: out-of-fold predictions on train+valid -> probe; final model on test)  [B1: only the 14 checklist channels]
   C  oracle: the TRUE checklist labels -> probe        D  frozen ImageNet ViT-S features (mean over tiles) -> probe
   R  classic 7-point rule (weights 2/2/2/1/1/1/1, no training) on predicted probabilities (new / old) and on the true labels
"""
import json, os
import numpy as np, torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dense as D, hybrid as H, d7_data as DD, mil, diagnosis_map as DM

sp = DD.split(); pool = np.concatenate([sp["train"], sp["valid"]]); test = sp["test"]
Y7 = DD.labels(); mel = DD.melanoma(); N = len(Y7); tiles, cnt = mil.load_tiles()
base, ck = H.load_model("models/hybrid.pt"); ids = [str(i) for i in ck["ids"]]; K = ck["n_out"]; nid = len(ids)

# ---- predictions -------------------------------------------------------------------------
if not os.path.exists("cache/d7_base_pred.npy"):
    np.save("cache/d7_base_pred.npy", mil.agg_predict(base, tiles, cnt, np.arange(N)))
P0 = np.load("cache/d7_base_pred.npy")                                   # [N,76] old model, all images
oof = np.zeros((N, K + len(DD.TARGETS)), np.float32)
for k in range(5):
    z = np.load(f"models/mel_fold{k}_pred.npz"); oof[z["idx"]] = z["pred"]
zf = np.load("models/mel_final_pred.npz"); oof[zf["idx"]] = zf["pred"]     # test rows come from the final model
if not os.path.exists("cache/d7_imnet_feat.npy"):
    net = timm.create_model(D.BACKBONE, pretrained=True, num_classes=0).to(D.dev).eval(); fe = []
    with torch.no_grad():
        for j in range(N):
            x = D.prep(torch.from_numpy(np.asarray(tiles[j, :cnt[j]])), None, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fe.append(net(x).float().mean(0).cpu().numpy())
    np.save("cache/d7_imnet_feat.npy", np.stack(fe))
FI = np.load("cache/d7_imnet_feat.npy")


def ci(y, s, n=500):
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            b.append(roc_auc_score(y[ix], s[ix]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auc(y, s):
    return (roc_auc_score(y, s), *ci(y, s))


fmt = lambda r: f"{r[0]:.3f} [{r[1]:.2f}-{r[2]:.2f}]"
# ---- structure level ---------------------------------------------------------------------------
attr_i = {a: nid + k for k, a in enumerate(H.ATTRS)}; idx = lambda l: [ids.index(i) for i in l]
BM = {"network_any": idx(["DS001", "DS002", "DS003", "DS004", "DS022"]) + [attr_i["pigment_network"]], "network_atypical": idx(["DS003", "DS004"]),
      "streaks_any": idx(["DS008", "DS009", "DS024"]) + [attr_i["streaks"]], "streaks_irregular": idx(["DS008", "DS009", "DS024"]) + [attr_i["streaks"]],
      "dots_any": idx(["DS025", "DS037"]) + [attr_i["globules"]], "dots_irregular": idx(["DS025", "DS037"]) + [attr_i["globules"]], "veil": idx(["DS049"]),
      "regression_any": idx(["DS051"]), "pigmentation_irregular": [], "vessels_dotted": idx(["DS060"]), "vessels_hairpin": idx(["DS064"]),
      "vessels_linear_irregular": idx(["DS065"]), "vessels_comma": idx(["DS067"]), "vessels_arborizing": idx(["DS072"])}
print(f"=== structure level, Derm7pt TEST ({len(test)} images): old zero-shot vs new fine-tuned (AUC [95% CI]) ===")
print(f"{'target':26s} {'#pos':>5s} {'old (hybrid)':>18s} {'new (fine-tuned)':>18s}   OOF train+valid (new)")
res = {"structures": {}}
for c, nm in enumerate(DD.TNAMES):
    y = Y7[test, c].astype(int); yo = Y7[pool, c].astype(int)
    if y.sum() < 5:
        print(f"{nm:26s} {y.sum():5d}  (too few positives)"); continue
    new = auc(y, oof[test, K + c]); o_oof = roc_auc_score(yo, oof[pool, K + c])
    old = auc(y, P0[test][:, BM[nm]].max(1)) if BM[nm] else None
    print(f"{nm:26s} {y.sum():5d} {fmt(old) if old else '       n/a':>18s} {fmt(new):>18s}   {o_oof:.3f}")
    res["structures"][nm] = dict(n_pos=int(y.sum()), old=old, new=new, oof=float(o_oof))

# ---- melanoma level -------------------------------------------------------------------------
def probe(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr)
    return clf.decision_function(Xte)

ym = mel[test]
feats = {"A old glyphs (76 ch)": (P0, P0), "B new glyphs (90 ch, stacked)": (oof, oof), "B1 new, 14 checklist ch only": (oof[:, K:], oof[:, K:]),
         "C oracle: true checklist labels": (Y7, Y7), "D ImageNet features (mean over tiles)": (FI, FI)}
print(f"\n=== melanoma vs rest, Derm7pt TEST ({int(ym.sum())}/{len(ym)} melanomas); probes fitted on train+valid ({len(pool)}) ===")
res["melanoma"] = {}
for nm, (Ftr, Fte) in feats.items():
    s = probe(Ftr[pool], mel[pool], Fte[test]); res["melanoma"][nm] = auc(ym, s); print(f"{nm:40s} {fmt(res['melanoma'][nm])}")
# classic 7-point rule (no training)
W = {"network_atypical": 2, "veil": 2, "streaks_irregular": 1, "pigmentation_irregular": 1, "dots_irregular": 1, "regression_any": 1}
def seven(get):
    s = sum(w * get(n) for n, w in W.items()) + 2 * np.maximum(get("vessels_dotted"), get("vessels_linear_irregular"))
    return s
ci_ = {n: DD.TNAMES.index(n) for n in DD.TNAMES}
true_s = seven(lambda n: Y7[test, ci_[n]])
new_s = seven(lambda n: oof[test, K + ci_[n]])
def old_get(n):
    return P0[test][:, BM[n]].max(1) if BM[n] else np.zeros(len(test))
old_s = seven(old_get)
print("\nclassic 7-point rule on test (score = 2*atypical network + 2*veil + 2*atypical vessels + 1*irregular streaks/pigmentation/dots + 1*regression):")
for nm, s in (("R true checklist labels", true_s), ("R new glyph probabilities", new_s), ("R old glyph probabilities", old_s)):
    res["melanoma"][nm] = auc(ym, s); print(f"{nm:40s} {fmt(res['melanoma'][nm])}")
# other diagnosis groups (context)
GRP = {"clark nevus": "NV", "reed or spitz nevus": "NV", "dermal nevus": "NV", "blue nevus": "NV", "congenital nevus": "NV", "combined nevus": "NV", "recurrent nevus": "NV", "seborrheic keratosis": "BKL",
       "lentigo": "BKL", "melanosis": "BKL", "basal cell carcinoma": "BCC", "dermatofibroma": "DF", "vascular lesion": "VASC", "miscellaneous": "OTHER"}
dg = np.array(["MEL" if r["diagnosis"].startswith("melanoma") else GRP[r["diagnosis"]] for r in DD.meta()])
print("\nother groups, one-vs-rest AUC on TEST (probe on new 90 channels | old 76 channels | ImageNet | oracle):")
for g in ("NV", "BCC", "BKL", "DF", "VASC"):
    y = (dg == g).astype(int)
    if y[test].sum() < 8:
        print(f"{g}: only {y[test].sum()} test cases"); continue
    r = [roc_auc_score(y[test], probe(F[pool], y[pool], F[test])) for F in (oof, P0, FI, Y7)]
    print(f"{g:5s} n={int(y[test].sum()):3d}  " + "  ".join(f"{x:.3f}" for x in r))
json.dump(res, open("models/melanoma_eval.json", "w"), indent=1)
