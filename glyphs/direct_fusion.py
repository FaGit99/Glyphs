"""Glyph path + the project's direct classifier (EVA-02 base_448, v17) on the v17 TEST split.

  python direct_fusion.py extract   # glyph features for the usable test images
  python direct_fusion.py eval

Direct-classifier scores: audit_reports/x_swap/X5_results/X5_BEST_epoch_013_..._test__resize_..._predictions.csv
(reproduces AUROC 0.9675 on the 9,374 test images = base_448 in the v17 PDF). Binary target: malignant vs benign.
Excluded for the glyph path: clinical close-ups (not dermoscopy), images my glyph / lesion models were trained on
(ISIC 2018 Task 1-2, HAM10000 index-parity-0 skin source).
"""
import csv, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import external_check as EC

Dx = "/mnt/samsung_data/mole_data/"
PRED = Dx + "audit_reports/x_swap/X5_results/X5_BEST_epoch_013_auroc_0.9734__test__resize__20260924T082542Z_predictions.csv"
conv = lambda s: Dx + s.split("mole_data\\", 1)[1].replace("\\", "/")


def usable():
    import skin
    names18 = {os.path.basename(f)[:-4] for sub in ("Training", "Validation", "Test") for f in glob.glob(Dx + f"data/raw/isic2018/ISIC2018_Task1-2_{sub}_Input/*.jpg")}
    ham0 = {os.path.basename(f)[:-4] for f in skin.ham_files(0)}
    rows = list(csv.DictReader(open(PRED)))
    keep = [r for r in rows if r["image_type"] == "dermoscopic" and r["isic_id"] not in names18 and r["isic_id"] not in ham0]
    return rows, keep


def extract():
    import torch, analyze as A, dense as D, hybrid as H
    rows, keep = usable()
    print(f"test {len(rows)} -> usable for the glyph path {len(keep)}", flush=True)
    M = A.models(); t0 = time.time(); out = {k: [] for k in ("id", "exp", "pts", "cal14", "agg_raw", "prob", "y", "group", "status", "source")}
    with ThreadPoolExecutor(8) as ex:
        for n, (r, img) in enumerate(zip(keep, ex.map(lambda r: EC._safe(EC.load450, conv(r["image_path"])), keep))):
            if img is None:
                continue
            les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
            c = A.checklist(img, les, [(x, y) for y in A.positions(h) for x in A.positions(w)], M, render=False, keep=True)
            for k, v in (("id", r["isic_id"]), ("exp", c["expected"]), ("pts", c["points"]), ("cal14", c["cal14"]), ("agg_raw", c["agg_raw"]), ("prob", float(r["prob"])),
                         ("y", int(float(r["target"]))), ("group", r["group_id"]), ("status", r["label_status"]), ("source", r["source_archive"])):
                out[k].append(v)
            if n % 1000 == 999:
                print(f"  {n + 1}/{len(keep)} {time.time() - t0:.0f}s", flush=True)
    np.savez("models/direct_feats.npz", **{k: np.array(v) for k, v in out.items()}); print(f"done {time.time() - t0:.0f}s, {len(out['id'])} images")


if __name__ == "__main__" and sys.argv[1] == "extract":
    extract()


# --------------------------------------------------------------------------------------------------------------------------
def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load("models/direct_feats.npz", allow_pickle=True)
    y, p, grp, status, src, R, G = z["y"].astype(int), z["prob"], z["group"], z["status"], z["source"], z["exp"], z["agg_raw"]
    n = len(y); lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); Gs = np.c_[G]
    print(f"{n} usable test images, {int(y.sum())} malignant ({y.mean():.1%}); groups {len(set(grp))}; confirmed benign {int((status == 'benign_confirmed').sum())}, weak benign {int((status == 'benign_weak').sum())}")
    mk = lambda C=.1: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    feats = {"D direct (EVA-02 base_448)": lg[:, None], "R glyph rule (no training)": R[:, None], "Gp glyph probe (90 ch, CV)": G, "FR direct + rule (CV)": np.c_[lg, R], "FG direct + 90 glyph ch (CV)": np.c_[lg, G]}
    sc = {"D direct (EVA-02 base_448)": lg, "R glyph rule (no training)": R}
    folds = list(GroupKFold(5).split(np.zeros(n), y, grp))
    for name, X in feats.items():
        if name in sc:
            continue
        s = np.zeros(n)
        for tr, te in folds:
            s[te] = mk().fit(X[tr], y[tr]).decision_function(X[te])
        sc[name] = s
    def spec_at(yy, s, sens):
        thr = np.quantile(s[yy == 1], 1 - sens); return (s[yy == 0] < thr).mean()
    names = list(sc)
    def metrics(idx):
        yy = y[idx]; return {k: (roc_auc_score(yy, sc[k][idx]), spec_at(yy, sc[k][idx], .90), spec_at(yy, sc[k][idx], .95), spec_at(yy, sc[k][idx], .97)) for k in names}
    subsets = {"ALL usable": np.arange(n), "confirmed labels only (malignant_confirmed vs benign_confirmed)": np.nonzero(status != "benign_weak")[0]}
    for s_ in sorted(set(src)):
        subsets[f"source {s_}"] = np.nonzero(src == s_)[0]
    gi = {g: np.nonzero(grp == g)[0] for g in set(grp)}; ug = list(gi); rng = np.random.default_rng(0); res = {}
    for sname, idx in subsets.items():
        if y[idx].sum() < 20 or (1 - y[idx]).sum() < 20:
            print(f"\n(skipped {sname}: only one class)"); continue
        m = metrics(idx); print(f"\n=== {sname} (n={len(idx)}, malignant {int(y[idx].sum())}) ===\n{'model':32s} {'AUROC':>7s} {'spec@90%sens':>13s} {'spec@95%':>9s} {'spec@97%':>9s}")
        for k in names:
            print(f"{k:32s} {m[k][0]:7.4f} {m[k][1]:13.3f} {m[k][2]:9.3f} {m[k][3]:9.3f}")
        # cluster bootstrap over groups for the differences
        sub = set(idx); gsub = [[i for i in gi[g] if i in sub] for g in ug]; gsub = [a for a in gsub if a]
        diffs = {("FG direct + 90 glyph ch (CV)", "D direct (EVA-02 base_448)"): [], ("FR direct + rule (CV)", "D direct (EVA-02 base_448)"): [], ("Gp glyph probe (90 ch, CV)", "R glyph rule (no training)"): []}
        for _ in range(300):
            ii = np.concatenate([gsub[j] for j in rng.integers(0, len(gsub), len(gsub))])
            if 0 < y[ii].sum() < len(ii):
                mb = {k: (roc_auc_score(y[ii], sc[k][ii]), spec_at(y[ii], sc[k][ii], .95)) for k in {a for a, _ in diffs} | {b for _, b in diffs}}
                for (a, b), l in diffs.items():
                    l.append((mb[a][0] - mb[b][0], mb[a][1] - mb[b][1]))
        print("paired differences [95% CI, cluster bootstrap over patient groups]:")
        for (a, b), l in diffs.items():
            l = np.array(l); print(f"   {a[:30]:30s} - {b[:26]:26s}  dAUROC {l[:, 0].mean():+.4f} [{np.percentile(l[:, 0], 2.5):+.4f}, {np.percentile(l[:, 0], 97.5):+.4f}]   dSpec@95%sens {l[:, 1].mean():+.3f} [{np.percentile(l[:, 1], 2.5):+.3f}, {np.percentile(l[:, 1], 97.5):+.3f}]")
        res[sname] = {k: list(map(float, m[k])) for k in names}
    json.dump(res, open("models/direct_fusion.json", "w"), indent=1)
    # where do the two paths disagree? direct classifier's false negatives at 95 % sensitivity
    thr = np.quantile(lg[y == 1], .05); fn = (y == 1) & (lg < thr); tp = (y == 1) & (lg >= thr)
    print(f"\ndirect classifier at 95 % sensitivity: {int(fn.sum())} missed malignant lesions of {int(y.sum())}")
    print(f"  glyph rule score (expected points): missed {R[fn].mean():.2f} vs detected malignant {R[tp].mean():.2f} vs benign {R[y == 0].mean():.2f}")
    top = R >= np.quantile(R[y == 0], .8)
    print(f"  of the missed malignant lesions, {top[fn].mean():.1%} have a glyph score above the 80th percentile of benign lesions (benign: 20 % by construction)")


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
