"""Externe Prüfung der 7-Punkte-Regel (Melanom gegen Naevus bzw. gegen alle übrigen) nur auf histologisch gesicherten Bildern.

  python histo_external.py [v17_dir]   -> models/histo_external.json

Verwendet die gespeicherten Merkmale models/ext_<Quelle>.npz (external_check.py) und verknüpft sie über die ISIC-ID mit den V17-Metadaten
(train + val + test; V17 enthält nur zugelassene Bilder). Gesichert = diagnosis_confirm_type 'histopathology'. Bilder ohne V17-Eintrag
(ausgeschlossen oder nicht im Archiv) fallen aus der Teilmenge. Keine Anpassung an den externen Daten. Cluster-Bootstrap über V17-group_id.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

V17 = sys.argv[1] if len(sys.argv) > 1 else "/mnt/samsung_data/mole_data/e_reports_v17/splits/v17"
meta = pd.concat([pd.read_csv(f"{V17}/{s}.csv", low_memory=False, usecols=["isic_id", "diagnosis_confirm_type", "label_status", "group_id", "target"]) for s in ("train", "val", "test")]).set_index("isic_id")
rng = np.random.default_rng(0); out = {}
tid = lambda p: os.path.basename(p).rsplit(".", 1)[0].replace("_downsampled", "")


def auc_ci(y, s, grp, nb=300):
    a = roc_auc_score(y, s); ug = np.unique(grp); gi = {g: np.nonzero(grp == g)[0] for g in ug}; b = []
    for _ in range(nb):
        ii = np.concatenate([gi[g] for g in ug[rng.integers(0, len(ug), len(ug))]])
        if 0 < y[ii].sum() < len(ii):
            b.append(roc_auc_score(y[ii], s[ii]))
    return float(a), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


pool = {"R": [], "y1": [], "y2": [], "hist": [], "grp": [], "src": [], "pts": [], "inv17": []}
for name in ("HAM10000", "BCN20000", "ISIC2019", "ISIC2020"):
    z = np.load(f"models/ext_{name}.npz", allow_pickle=True); ids = [tid(p) for p in z["path"]]; g = z["group"]
    inm = np.array([i in meta.index for i in ids]); m = meta.reindex(ids)
    hist = (m["diagnosis_confirm_type"] == "histopathology").values
    grp = np.where(inm, m["group_id"].fillna("NA").values, np.array(ids))
    pool["R"] += list(z["exp"]); pool["pts"] += list(z["pts"]); pool["y1"] += list(g == "MEL"); pool["hist"] += list(hist); pool["grp"] += list(grp); pool["src"] += [name] * len(ids); pool["inv17"] += list(inm)
    pool["y2"] += list(g == "MEL")
    pool.setdefault("dx", []).extend(list(g))
P = {k: np.array(v) for k, v in pool.items()}
for sname in list(dict.fromkeys(P["src"])) + ["gepoolt"]:
    sel = np.ones(len(P["R"]), bool) if sname == "gepoolt" else P["src"] == sname; r = {}
    for task in ("T1 Melanom gegen Naevus", "T2 Melanom gegen alle übrigen"):
        for lab, flt in (("alle gezogenen", sel), ("in V17 enthalten", sel & P["inv17"]), ("nur histologisch gesichert", sel & P["hist"])):
            k = flt & ((P["dx"] == "MEL") | (P["dx"] == "NV")) if task.startswith("T1") else flt
            y = (P["dx"][k] == "MEL").astype(int)
            if y.sum() < 15 or (1 - y).sum() < 15:
                r[f"{task} | {lab}"] = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), auc=None); continue
            a, lo, hi = auc_ci(y, P["R"][k], P["grp"][k]); p3 = P["pts"][k] >= 3
            r[f"{task} | {lab}"] = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), auc=a, ci=[lo, hi], sens3=float(p3[y == 1].mean()), spec3=float((~p3[y == 0]).mean()))
    out[sname] = r
    print(f"\n=== {sname} ===")
    for k, v in r.items():
        print(f"  {k:55s} pos {v['n_pos']:4d} neg {v['n_neg']:5d}  " + ("zu klein" if v["auc"] is None else f"AUC {v['auc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  >=3 Punkte: Sens {v['sens3']:.2f} Spez {v['spec3']:.2f}"))
json.dump(out, open("models/histo_external.json", "w"), indent=1, ensure_ascii=False)
