"""Proper fusion test: fit on the v17 VALIDATION split, evaluate on the v17 TEST split (direct classifier = base_448_v17, run by direct_model.py).

  python fusion_val_test.py extract <split>   # glyph features for the usable images of a split (val | test)
  python fusion_val_test.py eval
Usable for the glyph path: dermoscopic only; not an ISIC 2018 Task 1-2 image (my glyph training), not a HAM index-parity-0 image (synthetic skin source).
"""
import csv, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import direct_model as DM, external_check as EC

Dx = "/mnt/samsung_data/mole_data/"


def usable_rows(split):
    import skin
    names18 = {os.path.basename(f)[:-4] for sub in ("Training", "Validation", "Test") for f in glob.glob(Dx + f"data/raw/isic2018/ISIC2018_Task1-2_{sub}_Input/*.jpg")}
    ham0 = {os.path.basename(f)[:-4] for f in skin.ham_files(0)}
    rows = DM.split_rows(split)
    return rows, [r for r in rows if r["image_type"] == "dermoscopic" and r["isic_id"] not in names18 and r["isic_id"] not in ham0]


def extract(split):
    import analyze as A
    rows, keep = usable_rows(split); print(f"{split}: {len(rows)} -> usable {len(keep)}", flush=True)
    M = A.models(); t0 = time.time(); out = {k: [] for k in ("id", "exp", "cal14", "agg_raw")}
    with ThreadPoolExecutor(8) as ex:
        for n, (r, img) in enumerate(zip(keep, ex.map(lambda r: EC._safe(EC.load450, DM.ARCHIVE + r["image_relpath"]), keep))):
            if img is None:
                continue
            les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
            c = A.checklist(img, les, [(x, y) for y in A.positions(h) for x in A.positions(w)], M, render=False, keep=True)
            for k, v in (("id", r["isic_id"]), ("exp", c["expected"]), ("cal14", c["cal14"]), ("agg_raw", c["agg_raw"])):
                out[k].append(v)
            if n % 2000 == 1999:
                print(f"  {n + 1}/{len(keep)} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"models/glyph_feats_{split}.npz", **{k: np.array(v) for k, v in out.items()}); print(f"done {time.time() - t0:.0f}s, {len(out['id'])} images")


if __name__ == "__main__" and sys.argv[1] == "extract":
    extract(sys.argv[2])


# ---------------------------------------------------------------------------------------------------------------------------
def load_split(split):
    d = np.load(f"models/direct_base448_{split}.npz", allow_pickle=True); g = np.load(f"models/glyph_feats_{split}.npz", allow_pickle=True)
    meta = {r["isic_id"]: r for r in DM.split_rows(split)}; pos = {i: k for k, i in enumerate(d["ids"])}
    ix = np.array([pos[i] for i in g["id"]]); ids = g["id"]
    return dict(id=ids, logit=d["logit"][ix], y=d["y"][ix].astype(int), G=g["agg_raw"], R=g["exp"], status=np.array([meta[i]["label_status"] for i in ids]), grp=np.array([meta[i]["group_id"] for i in ids]))


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    V, T = load_split("val"), load_split("test")
    print(f"val {len(V['y'])} images ({V['y'].mean():.1%} malignant), test {len(T['y'])} images ({T['y'].mean():.1%} malignant)")
    mk = lambda C=.1: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    # glyph probe, fitted on val; inner out-of-fold glyph scores on val for the stacking stage
    gp = mk().fit(V["G"], V["y"]); gV_oof = np.zeros(len(V["y"]))
    for tr, te in GroupKFold(5).split(V["G"], V["y"], V["grp"]):
        gV_oof[te] = mk().fit(V["G"][tr], V["y"][tr]).decision_function(V["G"][te])
    gT = gp.decision_function(T["G"])
    s1 = mk().fit(np.c_[V["logit"], V["G"]], V["y"]); s2 = LogisticRegression(C=10).fit(np.c_[V["logit"], gV_oof], V["y"]); s3 = LogisticRegression(C=10).fit(np.c_[V["logit"], V["R"]], V["y"])
    print("stacking weights [direct logit, glyph-probe score]:", s2.coef_.round(2), "| [logit, rule]:", s3.coef_.round(2))
    sc_T = {"D direct (base_448_v17)": T["logit"], "R glyph rule (no training)": T["R"], "Gp glyph probe (90 ch, fit on val)": gT,
            "S1 one-stage [logit + 90 ch]": s1.decision_function(np.c_[T["logit"], T["G"]]), "S2 two-stage [logit, glyph probe]": s2.decision_function(np.c_[T["logit"], gT]), "S3 [logit, rule]": s3.decision_function(np.c_[T["logit"], T["R"]])}
    sc_V = {"D direct (base_448_v17)": V["logit"], "S2 two-stage [logit, glyph probe]": s2.decision_function(np.c_[V["logit"], gV_oof])}
    names = list(sc_T)

    def spec_at(y, s, sens):
        return (s[y == 0] < np.quantile(s[y == 1], 1 - sens)).mean()
    ug = sorted(set(T["grp"])); gi = {g: np.nonzero(T["grp"] == g)[0] for g in ug}; rng = np.random.default_rng(0); res = {}
    for sname, idx in (("ALL usable test images", np.arange(len(T["y"]))), ("confirmed labels only", np.nonzero(T["status"] != "benign_weak")[0])):
        y = T["y"][idx]; print(f"\n=== {sname} (n={len(idx)}, malignant {int(y.sum())}) ===\n{'model':36s} {'AUROC':>7s} {'spec@90':>8s} {'spec@95':>8s} {'spec@97':>8s}")
        for k in names:
            s = sc_T[k][idx]; print(f"{k:36s} {roc_auc_score(y, s):7.4f} {spec_at(y, s, .90):8.3f} {spec_at(y, s, .95):8.3f} {spec_at(y, s, .97):8.3f}")
        sub = set(idx); gs = [[i for i in gi[g] if i in sub] for g in ug]; gs = [a for a in gs if a]
        for k in ("S1 one-stage [logit + 90 ch]", "S2 two-stage [logit, glyph probe]", "S3 [logit, rule]"):
            d = []
            for _ in range(300):
                ii = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))])
                if 0 < T["y"][ii].sum() < len(ii):
                    d.append((roc_auc_score(T["y"][ii], sc_T[k][ii]) - roc_auc_score(T["y"][ii], sc_T[names[0]][ii]), spec_at(T["y"][ii], sc_T[k][ii], .95) - spec_at(T["y"][ii], sc_T[names[0]][ii], .95)))
            d = np.array(d); print(f"   {k} - direct:  dAUROC {d[:, 0].mean():+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}]   dSpec@95%sens {d[:, 1].mean():+.3f} [{np.percentile(d[:, 1], 2.5):+.3f}, {np.percentile(d[:, 1], 97.5):+.3f}]")
        res[sname] = {k: float(roc_auc_score(y, sc_T[k][idx])) for k in names}
    # deployment-style operating point: threshold fixed on VAL for 95 % sensitivity, applied to TEST
    print("\n=== threshold fixed on val (95 % sensitivity), applied to test ===")
    for k in ("D direct (base_448_v17)", "S2 two-stage [logit, glyph probe]"):
        thr = np.quantile(sc_V[k][V["y"] == 1], .05); p = sc_T[k] >= thr; y = T["y"]
        print(f"{k:36s} sensitivity {p[y == 1].mean():.3f}  specificity {(~p[y == 0]).mean():.3f}")
    # disagreement flag: direct says benign, glyph path is suspicious
    thr = np.quantile(V["logit"][V["y"] == 1], .05); neg = T["logit"] < thr; y = T["y"]
    print(f"\n=== disagreement flag among the {int(neg.sum())} test images the direct model calls benign (95 % sens threshold from val) ===")
    print(f"missed malignant: {int(y[neg].sum())} ({y[neg].mean():.2%} of these images)")
    for q in (.90, .95, .98):
        t = np.quantile(gV_oof[V["y"] == 0], q); fl = neg & (gT >= t)
        print(f"flag = glyph-probe score above the {q:.0%} quantile of benign val images: flagged {int(fl.sum())} images, malignant among flagged {y[fl].mean():.2%} (lift x{y[fl].mean() / y[neg].mean():.1f}), catches {int(y[fl].sum())} of {int(y[neg].sum())} missed malignant")
    json.dump(res, open("models/fusion_val_test.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
