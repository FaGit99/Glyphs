"""Glyph score from validated channels only vs. all 90 channels (same protocol as fusion_val_test.py: fit on v17 val, evaluate on v17 test).

Channel layout of agg_raw: 0-70 synthetic ID channels (not validated on real masks), 71-75 ISIC attributes (validated with real masks),
76-89 Derm7pt checklist channels (validated at image level).
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fusion_val_test as FV

V, T = FV.load_split("val"), FV.load_split("test")
VARIANTS = {"A all 90 channels (current)": np.arange(90), "B validated 19 (5 ISIC + 14 Derm7pt)": np.arange(71, 90), "C Derm7pt 14 only": np.arange(76, 90),
            "D ISIC 5 only": np.arange(71, 76), "E synthetic IDs only (71)": np.arange(71)}
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=5000))
ytest, ydir = T["y"], T["logit"]
thr_d = float(np.quantile(V["logit"][V["y"] == 1], .05)); neg = T["logit"] < thr_d
spec_at = lambda y, s, sens: (s[y == 0] < np.quantile(s[y == 1], 1 - sens)).mean()
conf = np.nonzero(T["status"] != "benign_weak")[0]
res, S = {}, {}
for name, cols in VARIANTS.items():
    gp = mk().fit(V["G"][:, cols], V["y"]); oof = np.zeros(len(V["y"]))
    for tr, te in GroupKFold(5).split(V["G"], V["y"], V["grp"]):
        oof[te] = mk().fit(V["G"][tr][:, cols], V["y"][tr]).decision_function(V["G"][te][:, cols])
    gT = gp.decision_function(T["G"][:, cols]); t90 = np.quantile(oof[V["y"] == 0], .90)
    s2 = LogisticRegression(C=10).fit(np.c_[V["logit"], oof], V["y"]); two = s2.decision_function(np.c_[ydir, gT])
    fl = neg & (gT >= t90)
    S[name] = dict(g=gT, two=two, flag=fl)
    res[name] = dict(auc=roc_auc_score(ytest, gT), spec95=spec_at(ytest, gT, .95), auc_conf=roc_auc_score(ytest[conf], gT[conf]), two=roc_auc_score(ytest, two),
                     flagged=int(fl.sum()), flag_mal=float(ytest[fl].mean()), caught=int(ytest[fl].sum()))
print(f"v17 test, usable images {len(ytest)}; direct model AUROC {roc_auc_score(ytest, ydir):.4f}; the direct model calls {int(neg.sum())} images benign, of which {int(ytest[neg].sum())} are malignant ({ytest[neg].mean():.2%})\n")
print(f"{'variant':42s} {'AUROC':>7s} {'spec@95':>8s} {'AUROC conf.':>11s} | {'2-stage AUROC':>13s} | {'flagged':>7s} {'malignant':>9s} {'lift':>5s} {'caught/105':>10s}")
for k, r in res.items():
    print(f"{k:42s} {r['auc']:7.4f} {r['spec95']:8.3f} {r['auc_conf']:11.4f} | {r['two']:13.4f} | {r['flagged']:7d} {r['flag_mal']:9.1%} {r['flag_mal'] / ytest[neg].mean():5.1f} {r['caught']:7d}/{int(ytest[neg].sum())}")
# paired cluster bootstrap: validated-19 minus all-90 (glyph AUROC, caught among flagged)
ug = sorted(set(T["grp"])); gi = {g: np.nonzero(T["grp"] == g)[0] for g in ug}; rng = np.random.default_rng(0); a, b = "A all 90 channels (current)", "B validated 19 (5 ISIC + 14 Derm7pt)"
d = []
for _ in range(300):
    ii = np.concatenate([gi[g] for g in rng.integers(0, len(ug), len(ug)) for g in [ug[g]]])
    if 0 < ytest[ii].sum() < len(ii):
        d.append((roc_auc_score(ytest[ii], S[b]["g"][ii]) - roc_auc_score(ytest[ii], S[a]["g"][ii]), roc_auc_score(ytest[ii], S[b]["two"][ii]) - roc_auc_score(ytest[ii], ydir[ii])))
d = np.array(d); print(f"\npaired: glyph AUROC validated-19 minus all-90 = {d[:, 0].mean():+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}]")
print(f"paired: two-stage fusion (validated-19) minus direct = {d[:, 1].mean():+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, {np.percentile(d[:, 1], 97.5):+.4f}]")
fa, fb = S[a]["flag"], S[b]["flag"]
print(f"flag overlap: all-90 flags {int(fa.sum())}, validated-19 flags {int(fb.sum())}, both {int((fa & fb).sum())}; missed malignant caught by both {int((ytest[fa & fb]).sum())}, only by all-90 {int(ytest[fa & ~fb].sum())}, only by validated-19 {int(ytest[fb & ~fa].sum())}")
json.dump(res, open("models/validated_score.json", "w"), indent=1, default=float)
