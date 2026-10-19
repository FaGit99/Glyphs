"""Fix and store everything the two-path view needs (models/twopath.json). Fitted on the v17 VALIDATION split only; test numbers are for the evidence texts.
VARIANT B (since 2026-10-15): the glyph probe uses only the 19 VALIDATED channels (71-75 ISIC attributes with real masks, 76-89 Derm7pt checklist criteria);
the earlier 90-channel version is kept in models/twopath_all90.json.

 direct path : threshold on the base_448_v17 logit = 5 % quantile of the malignant validation logits (95 % sensitivity)
 glyph path  : logistic probe (standardised, C=0.1) on the 90 raw glyph channels, fitted on validation; the flag threshold is the 90th percentile of the
               OUT-OF-FOLD probe scores of benign validation images (patient-grouped 5-fold), the same rule that was evaluated on test.
 flag        : direct logit < threshold AND glyph-probe score >= flag threshold.
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
COLS = np.arange(71, 90)   # validated channels
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=5000))
gp = mk().fit(V["G"][:, COLS], V["y"]); oof = np.zeros(len(V["y"]))
for tr, te in GroupKFold(5).split(V["G"], V["y"], V["grp"]):
    oof[te] = mk().fit(V["G"][tr][:, COLS], V["y"][tr]).decision_function(V["G"][te][:, COLS])
sc, lr = gp[0], gp[1]
thrD = float(np.quantile(V["logit"][V["y"] == 1], .05))
t90 = float(np.quantile(oof[V["y"] == 0], .90))
benign_q = np.quantile(oof[V["y"] == 0], np.linspace(0, 1, 101)).tolist()          # reference distribution for the percentile display
gT = gp.decision_function(T["G"][:, COLS]); y = T["y"]
dpos = T["logit"] >= thrD; neg = ~dpos; fl = neg & (gT >= t90)
both = dpos & (gT >= t90)
out = dict(
    direct=dict(name="EVA-02 base 448 (v17)", thr_logit=thrD, thr_prob=float(1 / (1 + np.exp(-thrD))), test_sens=float(dpos[y == 1].mean()), test_spec=float((~dpos[y == 0]).mean()), val_sens=0.95),
    glyph=dict(variant='validated-19', cols=COLS.tolist(), mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), flag_thr=t90, benign_quantiles=benign_q),
    flag=dict(n_direct_negative=int(neg.sum()), missed=int(y[neg].sum()), base_rate=float(y[neg].mean()), flagged=int(fl.sum()), flagged_share=float(fl.sum() / neg.sum()), flagged_malignant=float(y[fl].mean()),
              lift=float(y[fl].mean() / y[neg].mean()), caught=int(y[fl].sum()), review_per_catch=float(fl.sum() / max(y[fl].sum(), 1))),
    agreement=dict(both_high=int(both.sum()), malignant_rate_both_high=float(y[both].mean()), malignant_rate_direct_high_glyph_low=float(y[dpos & ~(gT >= t90)].mean()),
                   malignant_rate_both_low=float(y[neg & ~(gT >= t90)].mean())),
    fusion=dict(direct_auc=float(roc_auc_score(y, T['logit'])), two_stage_auc=float(roc_auc_score(y, LogisticRegression(C=10).fit(np.c_[V['logit'], oof], V['y']).decision_function(np.c_[T['logit'], gT])))),
    n_val=int(len(V["y"])), n_test=int(len(y)))
json.dump(out, open("models/twopath.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("direct", "flag", "agreement")}, indent=1))
