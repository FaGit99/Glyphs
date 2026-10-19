"""Consistency check: run the Derm7pt TEST images through analyze.py and compare the displayed checklist with melanoma labels."""
import json, time
import numpy as np
from sklearn.metrics import roc_auc_score
import analyze as A, d7_data as DD, prep_real as PR

rows = DD.meta(); test = DD.split()["test"]; mel = DD.melanoma(); Y = DD.labels(); t0 = time.time()
exp, pts, probs = [], [], []
for n, i in enumerate(test):
    R = A.analyze(PR.find_ci(DD.RAWD + "images/", rows[i]["derm"])); c = R["check"]
    exp.append(c["expected"]); pts.append(c["points"]); probs.append([r["prob"] for r in c["rows"]])
    if n % 100 == 99:
        print(f"{n + 1}/{len(test)} {time.time() - t0:.0f}s", flush=True)
exp, pts, probs = np.array(exp), np.array(pts), np.array(probs); y = mel[test]
def ci(s, n=500):
    r = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = r.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            b.append(roc_auc_score(y[ix], s[ix]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
print(f"\nanalyze.py on Derm7pt test ({int(y.sum())} melanomas / {len(y)}):")
for nm, s in (("expected points (displayed)", exp), ("points at >=50 % criteria (displayed)", pts)):
    lo, hi = ci(s); print(f"  melanoma AUC, {nm:40s} {roc_auc_score(y, s):.3f} [{lo:.2f}-{hi:.2f}]")
# calibration of the displayed score: mean expected vs true checklist score
true = (2 * Y[test, 1] + 2 * Y[test, 6] + 2 * np.maximum(Y[test, 9], Y[test, 11]) + Y[test, 3] + Y[test, 8] + Y[test, 5] + Y[test, 7])
print(f"  mean displayed expected score {exp.mean():.2f} vs mean true checklist score {true.mean():.2f}; correlation {np.corrcoef(exp, true)[0, 1]:.2f}")
for thr in (3,):
    pred = pts >= thr; print(f"  rule 'points >= {thr}': sensitivity {pred[y == 1].mean():.2f}, specificity {(~pred[y == 0]).mean():.2f}")
