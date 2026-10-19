"""Does the taxonomy-based glyph -> diagnosis heuristic carry information? Derm7pt (1011 cases with diagnosis), one-vs-rest AUC.

 H1  heuristic, raw evidence score       (taxonomy only, nothing trained on diagnoses)
 H2  heuristic, normalised by #IDs/group (pre-declared variant; avoids favouring groups with many listed IDs)
 P   supervised probe: logistic regression on the 71 ID probabilities (5-fold CV)  -> information content of the glyph outputs
 O   oracle: logistic regression on the REAL Derm7pt structure labels (pigment network, streaks, dots/globules, veil, regression, vessels, pigmentation) (CV)
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import dense as D, hybrid as H, diagnosis_map as DM

GRP = {"clark nevus": "NV", "reed or spitz nevus": "NV", "dermal nevus": "NV", "blue nevus": "NV", "congenital nevus": "NV", "combined nevus": "NV", "recurrent nevus": "NV",
       "seborrheic keratosis": "BKL", "lentigo": "BKL", "melanosis": "BKL", "basal cell carcinoma": "BCC", "dermatofibroma": "DF", "vascular lesion": "VASC", "miscellaneous": "OTHER"}
ids = H.id_list(); lab = json.load(open("cache/d7_labels.json")); M = len(lab["diagnosis"])
grp = np.array(["MEL" if d.startswith("melanoma") else GRP[d] for d in lab["diagnosis"]])
print("Derm7pt groups:", {g: int((grp == g).sum()) for g in DM.GROUPS}, flush=True)

if not os.path.exists("cache/d7_hybrid_probs.npy"):
    m, _ = H.load_model("models/hybrid.pt")
    d7 = np.load("cache/d7_img.npy"); tiles = np.ascontiguousarray(d7[:, :5].reshape(-1, 224, 224, 3))
    pi, _ = D.predict(m, tiles); np.save("cache/d7_hybrid_probs.npy", pi.reshape(M, 5, -1).max(1))
P = np.load("cache/d7_hybrid_probs.npy")[:, :len(ids)]          # image-level prob per ID (max over 5 tiles)

dm = DM.load()
def heur(norm):
    out = np.zeros((M, len(DM.GROUPS)))
    for n in range(M):
        conf = {i: float(P[n, k]) for k, i in enumerate(ids) if P[n, k] >= .5}
        s = DM.scores(conf, dm, normalise=norm); out[n] = [s[g] for g in DM.GROUPS]
    return out
H1, H2 = heur(False), heur(True)

# oracle features from the real labels
def onehot(vals, cats):
    return np.stack([np.array(vals) == c for c in cats], 1).astype(float)
O = np.hstack([onehot(lab["pigment_network"], ["typical", "atypical"]), onehot(lab["streaks"], ["regular", "irregular"]), onehot(lab["dots_and_globules"], ["regular", "irregular"]),
               onehot(lab["blue_whitish_veil"], ["present"]), onehot(lab["regression_structures"], ["white areas", "blue areas", "combinations"]),
               onehot(lab["vascular_structures"], ["arborizing", "comma", "hairpin", "within regression", "dotted", "linear irregular", "wreath"])])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
def probe(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000, class_weight="balanced"))
    return cross_val_predict(clf, X, y, cv=cv, method="decision_function")

def ci(y, s, n=300):
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            b.append(roc_auc_score(y[ix], s[ix]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

print(f"\n{'group':8s} {'n':>4s} {'H1 raw':>17s} {'H2 norm':>17s} {'P probe(71 IDs)':>17s} {'O oracle labels':>17s}")
res = {}
for gi, g in enumerate(DM.GROUPS):
    y = (grp == g).astype(int)
    if y.sum() < 15:
        continue
    row = {}
    for nm, s in (("H1", H1[:, gi]), ("H2", H2[:, gi]), ("P", probe(P, y)), ("O", probe(O, y))):
        a = roc_auc_score(y, s); lo, hi = ci(y, s); row[nm] = (float(a), float(lo), float(hi))
    print(f"{g:8s} {y.sum():4d} " + " ".join(f"{row[k][0]:6.3f} [{row[k][1]:.2f}-{row[k][2]:.2f}]" for k in ("H1", "H2", "P", "O")))
    res[g] = row
# top-1 agreement of the heuristic restricted to the 6 groups with enough cases (diagnostic argmax over groups) - for context only
gl = [g for g in DM.GROUPS if (grp == g).sum() >= 15]; ix = [list(DM.GROUPS).index(g) for g in gl]
for nm, Hs in (("H1", H1), ("H2", H2)):
    pred = np.array(gl)[Hs[:, ix].argmax(1)]; print(f"{nm} top-1 over {gl}: accuracy {np.mean(pred == grp):.3f}  (majority class {max((grp == g).mean() for g in gl):.3f}); n with evidence>0: {(Hs[:, ix].max(1) > 0).sum()}")
json.dump(res, open("models/diag_eval.json", "w"), indent=1)
