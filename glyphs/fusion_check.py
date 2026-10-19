"""Do glyph features and generic (ImageNet) features complement each other?  Everything is fitted on Derm7pt train+valid only.

G    probe on the 14 checklist channels (raw, out-of-fold on train+valid)
I    probe on frozen ImageNet ViT-S features (mean over lesion tiles)
R    7-point rule on calibrated probabilities (no training)
Variants declared before looking at external results:
 C1  one probe on [14 glyph channels | 384 ImageNet dims] (joint standardisation, same regularisation)
 F1  late fusion: z(G) + z(I)   (z from the out-of-fold Derm7pt decision values; equal weights)
 F2  stacked logistic regression on [dG, dI] fitted on the out-of-fold Derm7pt decision values
 F3  z(R) + z(I)   (unsupervised rule + generic probe)
"""
import glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import d7_data as DD

K = 76
sp = DD.split(); pool, test = np.concatenate([sp["train"], sp["valid"]]), sp["test"]; y7 = DD.melanoma()
oof = np.zeros((1011, K + 14), np.float32)
for k in range(5):
    z = np.load(f"models/mel_fold{k}_pred.npz"); oof[z["idx"]] = z["pred"]
zf = np.load("models/mel_final_pred.npz"); oof[zf["idx"]] = zf["pred"]
G7, I7 = oof[:, K:], np.load("cache/d7_imnet_feat.npy")
cal = json.load(open("models/mel_calibration.json")); T = DD.TNAMES
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def rule(raw14):
    c = {n: 1 / (1 + np.exp(-(cal[n]["platt"][0] * lg(raw14[:, i]) + cal[n]["platt"][1]))) for i, n in enumerate(T)}
    return 2 * c["network_atypical"] + 2 * c["veil"] + 2 * np.maximum(c["vessels_dotted"], c["vessels_linear_irregular"]) + c["streaks_irregular"] + c["pigmentation_irregular"] + c["dots_irregular"] + c["regression_any"]


mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=4000, class_weight="balanced"))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xc = np.hstack([G7, I7])
pG, pI, pC = mk().fit(G7[pool], y7[pool]), mk().fit(I7[pool], y7[pool]), mk().fit(Xc[pool], y7[pool])
oG = cross_val_predict(mk(), G7[pool], y7[pool], cv=cv, method="decision_function"); oI = cross_val_predict(mk(), I7[pool], y7[pool], cv=cv, method="decision_function")
muG, sdG, muI, sdI = oG.mean(), oG.std(), oI.mean(), oI.std(); Rp = rule(G7[pool]); muR, sdR = Rp.mean(), Rp.std()
stack = LogisticRegression(C=10, class_weight="balanced").fit(np.c_[oG, oI], y7[pool])
print("stacking weights on [z-free decision values G, I]:", stack.coef_.round(2), "| decision-value std G/I (Derm7pt OOF):", round(sdG, 2), round(sdI, 2))


def scores(G, I):
    dG, dI = pG.decision_function(G), pI.decision_function(I); zG, zI = (dG - muG) / sdG, (dI - muI) / sdI; R = rule(G)
    return {"R rule (no training)": R, "G probe glyphs": dG, "I probe ImageNet": dI, "C1 concat probe": pC.decision_function(np.hstack([G, I])),
            "F1 z(G)+z(I)": zG + zI, "F2 stacked LR": stack.decision_function(np.c_[dG, dI]), "F3 z(R)+z(I)": (R - muR) / sdR + zI}


def paired(y, a, b, n=400, seed=0):
    r = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)


def ci(y, s, n=300, seed=0):
    r = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


names = ["R rule (no training)", "G probe glyphs", "I probe ImageNet", "C1 concat probe", "F1 z(G)+z(I)", "F2 stacked LR", "F3 z(R)+z(I)"]
sets = {"Derm7pt TEST (in-domain)": (G7[test], I7[test], y7[test], np.ones(len(test), bool))}
for fn in sorted(glob.glob("models/ext_*.npz")):
    z = np.load(fn, allow_pickle=True); sets[fn[len("models/ext_"):-4]] = (z["agg_raw"][:, K:], z["imnet"], z["group"], None)
res = {}
for task, pos, neg in (("T1 melanoma vs nevus", ["MEL"], ["NV"]), ("T2 melanoma vs all other", ["MEL"], ["NV", "BCC", "BKL", "AKIEC", "DF", "VASC", "OTHER"])):
    print(f"\n=== {task}: AUC (95% CI) ===\n{'dataset (n pos/neg)':30s}" + "".join(f"{n.split()[0]:>16s}" for n in names))
    pooled = {n: [] for n in names}; yp = []; res[task] = {}
    for name, (G, I, g, _) in sets.items():
        if name.startswith("Derm7pt"):
            if "T2" in task:
                y = g.astype(int); m = np.ones(len(y), bool)
            else:
                continue                                  # in-domain only has mel vs rest
        else:
            m = np.isin(g, pos + neg); y = np.isin(g[m], pos).astype(int)
        sc = scores(G[m], I[m])
        print(f"{name + f' ({y.sum()}/{(1 - y).sum()})':30s}" + "".join(f"{roc_auc_score(y, sc[n]):16.3f}" for n in names))
        res[task][name] = {n: float(roc_auc_score(y, sc[n])) for n in names}
        if not name.startswith("Derm7pt"):
            yp.append(y); [pooled[n].append(sc[n]) for n in names]
    if yp:
        y = np.concatenate(yp); S = {n: np.concatenate(pooled[n]) for n in names}
        print(f"{'POOLED external (' + str(int(y.sum())) + '/' + str(int((1 - y).sum())) + ')':30s}" + "".join(f"{roc_auc_score(y, S[n]):16.3f}" for n in names))
        print("paired AUC differences, pooled external [95% CI]:")
        for a, b in (("F1 z(G)+z(I)", "G probe glyphs"), ("F1 z(G)+z(I)", "I probe ImageNet"), ("F2 stacked LR", "I probe ImageNet"), ("C1 concat probe", "I probe ImageNet"), ("F3 z(R)+z(I)", "R rule (no training)"), ("F3 z(R)+z(I)", "I probe ImageNet")):
            d = paired(y, S[a], S[b]); print(f"   {a:16s} - {b:20s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        res[task]["pooled"] = {n: float(roc_auc_score(y, S[n])) for n in names}
        # per dataset: the worst-case AUC (robustness)
        print("worst dataset AUC:  " + "  ".join(f"{n.split()[0]} {min(res[task][k][n] for k in res[task] if k != 'pooled' and not k.startswith('Derm7pt')):.3f}" for n in names))
json.dump(res, open("models/fusion_check.json", "w"), indent=1)
