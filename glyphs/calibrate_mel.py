"""Platt calibration of the 14 Derm7pt checklist channels.

Fitted on out-of-fold predictions of train+valid (616 images; no model saw its own image), checked on the test split
(final model, 395 images, untouched). Writes models/mel_calibration.json.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import d7_data as DD

K = 76
sp = DD.split(); pool = np.concatenate([sp["train"], sp["valid"]]); test = sp["test"]; Y = DD.labels()
oof = np.zeros((1011, K + 14), np.float32)
for k in range(5):
    z = np.load(f"models/mel_fold{k}_pred.npz"); oof[z["idx"]] = z["pred"]
zf = np.load("models/mel_final_pred.npz"); oof[zf["idx"]] = zf["pred"]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def ece(p, y, bins=8):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e
out = {}
print(f"{'channel':26s} {'n_pos(tr+va/test)':>18s} {'AUC oof':>8s} {'AUC test':>9s} {'ECE raw>cal (test)':>20s} {'Brier raw>cal':>16s} reliable")
for c, nm in enumerate(DD.TNAMES):
    ytr, yte = Y[pool, c].astype(int), Y[test, c].astype(int)
    ptr, pte = oof[pool, K + c], oof[test, K + c]
    pl = LogisticRegression(C=10).fit(lg(ptr)[:, None], ytr)
    cte = pl.predict_proba(lg(pte)[:, None])[:, 1]
    a_oof = roc_auc_score(ytr, ptr); a_te = roc_auc_score(yte, pte) if 0 < yte.sum() < len(yte) else float("nan")
    rel = bool(ytr.sum() >= 40 and a_oof >= .75)                      # enough positives and out-of-fold AUC >= 0.75
    print(f"{nm:26s} {int(ytr.sum()):9d}/{int(yte.sum()):<8d} {a_oof:8.3f} {a_te:9.3f} {ece(pte, yte):9.3f}>{ece(cte, yte):<9.3f} {brier_score_loss(yte, pte):7.3f}>{brier_score_loss(yte, cte):<7.3f} {'yes' if rel else 'NO (few cases / weak)'}")
    out[nm] = dict(platt=[float(pl.coef_[0, 0]), float(pl.intercept_[0])], n_pos=int(ytr.sum()), auc_oof=float(a_oof), auc_test=None if np.isnan(a_te) else float(a_te), reliable=rel,
                   ece_raw=float(ece(pte, yte)), ece_cal=float(ece(cte, yte)))
json.dump(out, open("models/mel_calibration.json", "w"), indent=1)
