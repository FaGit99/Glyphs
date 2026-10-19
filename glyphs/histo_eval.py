"""Kennzahlen auf histologisch gesicherten Fällen (V17-Metadaten) im Vergleich zu allen nutzbaren Fällen.

  python histo_eval.py [v17_dir]        -> models/histo_eval.json + Tabelle auf der Konsole

Definition "histologisch gesichert": V17-Spalte diagnosis_confirm_type == 'histopathology'
(= label_status malignant_confirmed oder benign_confirmed). Alle übrigen benignen Fälle (label_status benign_weak) sind
Verlaufskontrolle ohne Änderung, Experten-Konsens oder Konfokalmikroskopie.

Protokoll wie fusion_val_test.py / validated_score.py / twopath_fit.py: Sonden und Schwellen nur auf der V17-Validierung,
Auswertung auf dem V17-Test; nutzbar sind nur dermoskopische Bilder ohne ISIC-2018-Task-1/2-Bilder und ohne HAM-Indexhälfte 0 (8 054 Test, 8 043 Val).
Varianten: A = 90 Kanäle (Stand Paper V1), B = 19 validierte Kanäle (ausgelieferter Stand, models/twopath.json).
Bootstrap: Cluster über group_id (V17-Gruppen), 300 Ziehungen, Seed 0.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V17 = sys.argv[1] if len(sys.argv) > 1 else "/mnt/samsung_data/mole_data/e_reports_v17/splits/v17"
NB = 300
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=5000))
spec_at = lambda y, s, sens: float((s[y == 0] < np.quantile(s[y == 1], 1 - sens)).mean())


def load(split):
    m = pd.read_csv(f"{V17}/{split}.csv", low_memory=False).set_index("isic_id")
    d = np.load(f"models/direct_base448_{split}.npz", allow_pickle=True); g = np.load(f"models/glyph_feats_{split}.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(d["ids"])}; ix = np.array([pos[i] for i in g["id"]]); mm = m.loc[g["id"]]
    return dict(id=g["id"], logit=d["logit"][ix], y=d["y"][ix].astype(int), G=g["agg_raw"], R=g["exp"], status=mm["label_status"].values,
                conf=mm["diagnosis_confirm_type"].fillna("NA").values, grp=mm["group_id"].values)


V, T = load("val"), load("test")
for D in (V, T):
    D["histo"] = D["conf"] == "histopathology"
    assert (D["histo"] == (D["status"] != "benign_weak")).all()          # histologisch == (malignant_confirmed | benign_confirmed)
yT = T["y"]
VARIANTS = {"A (90 Kanäle)": np.arange(90), "B (19 validierte Kanäle)": np.arange(71, 90)}


def fit_variant(cols, Vsel=None):
    """Sonde + OOF-Scores auf (einer Auswahl der) Validierung. Rückgabe: Sonde, 2-Stufen-Modell, 90.-Perzentil der benignen OOF-Scores, Direktschwelle."""
    s = np.arange(len(V["y"])) if Vsel is None else np.nonzero(Vsel)[0]
    y, G, lg, gr = V["y"][s], V["G"][s][:, cols], V["logit"][s], V["grp"][s]
    gp = mk().fit(G, y); oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(G, y, gr):
        oof[te] = mk().fit(G[tr], y[tr]).decision_function(G[te])
    s2 = LogisticRegression(C=10).fit(np.c_[lg, oof], y)
    return gp, s2, float(np.quantile(oof[y == 0], .90)), float(np.quantile(lg[y == 1], .05)), oof[y == 0]


def scores(cols, fit):
    gp, s2, t90, thrD, _ = fit
    g = gp.decision_function(T["G"][:, cols]); return g, s2.decision_function(np.c_[T["logit"], g]), t90, thrD


def hint(idx, g, t90, thrD):
    neg = (T["logit"] < thrD)[idx]; y = yT[idx]; fl = neg & (g[idx] >= t90)
    return dict(n=int(len(idx)), n_direct_neg=int(neg.sum()), missed=int(y[neg].sum()), base_rate=float(y[neg].mean()), flagged=int(fl.sum()),
                flagged_malignant=float(y[fl].mean()) if fl.sum() else None, lift=float(y[fl].mean() / y[neg].mean()) if fl.sum() else None, caught=int(y[fl].sum()))


subsets = {"alle nutzbaren Testbilder": np.arange(len(yT)), "nur histologisch gesichert": np.nonzero(T["histo"])[0],
           "maligne + benign_weak (Kontrast)": np.nonzero(~T["histo"] | (yT == 1))[0]}
fits = {k: fit_variant(c) for k, c in VARIANTS.items()}
fits_h = {"B (19 validierte Kanäle)": fit_variant(VARIANTS["B (19 validierte Kanäle)"], V["histo"])}    # Sensitivität: alles auf histologischer Validierung angepasst
S = {"D Direktmodell": T["logit"], "R 7-Punkte-Regel (ohne Training)": T["R"]}; TH = {}
for k, c in VARIANTS.items():
    g, two, t90, thrD = scores(c, fits[k]); S[f"Gp Sonde {k}"] = g; S[f"S2 zweistufig {k}"] = two; TH[k] = (g, t90, thrD)
gh, twoh, t90h, thrDh = scores(VARIANTS["B (19 validierte Kanäle)"], fits_h["B (19 validierte Kanäle)"])

ug = sorted(set(T["grp"])); gi = {g: np.nonzero(T["grp"] == g)[0] for g in ug}; rng = np.random.default_rng(0); res = {}
for sname, idx in subsets.items():
    y = yT[idx]; r = {"n": int(len(idx)), "malignant": int(y.sum()), "benign": int((1 - y).sum()), "auroc": {}, "spec95": {}}
    for k, s in S.items():
        r["auroc"][k] = float(roc_auc_score(y, s[idx])); r["spec95"][k] = spec_at(y, s[idx], .95)
    sub = set(idx.tolist()); gs = [[i for i in gi[g] if i in sub] for g in ug]; gs = [a for a in gs if a]; boot = {k: [] for k in S}; dif = {k: [] for k in S if k.startswith("S2")}
    for _ in range(NB):
        ii = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]); yy = yT[ii]
        if 0 < yy.sum() < len(yy):
            a = {k: roc_auc_score(yy, s[ii]) for k, s in S.items()}
            for k in S: boot[k].append(a[k])
            for k in dif: dif[k].append(a[k] - a["D Direktmodell"])
    r["auroc_ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    r["d_auroc_vs_direct"] = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in dif.items()}
    r["direct_at_val_thr"] = dict(sens=float((T["logit"][idx] >= TH["A (90 Kanäle)"][2])[y == 1].mean()), spec=float((T["logit"][idx] < TH["A (90 Kanäle)"][2])[y == 0].mean()))
    r["hint"] = {k: hint(idx, *TH[k]) for k in VARIANTS}
    r["hint"]["B, auf histologischer Validierung angepasst"] = hint(idx, gh, t90h, thrDh)
    # Cluster-Bootstrap für den Abweichungshinweis (Anteil maligner unter den Markierungen, Anreicherung, gefundene Fälle)
    for hk, (gg, t90_, thrD_) in {**TH, "B, auf histologischer Validierung angepasst": (gh, t90h, thrDh)}.items():
        neg_ = T["logit"] < thrD_; fl_ = neg_ & (gg >= t90_); pp, ll, cc = [], [], []
        for _ in range(NB):
            ii = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]); n_, f_ = neg_[ii], fl_[ii]; yy = yT[ii]
            if f_.sum() and yy[n_].sum():
                pp.append(yy[f_].mean()); ll.append(yy[f_].mean() / yy[n_].mean()); cc.append(yy[f_].sum() / yy[n_].sum())
        r["hint"][hk]["ci"] = dict(flagged_malignant=[float(np.percentile(pp, 2.5)), float(np.percentile(pp, 97.5))], lift=[float(np.percentile(ll, 2.5)), float(np.percentile(ll, 97.5))],
                                    caught_share=[float(np.percentile(cc, 2.5)), float(np.percentile(cc, 97.5))])
    res[sname] = r
# Hinweisgrenze: Quantile 90/95/98 % der benignen OOF-Scores der Validierung (wie Tabelle im Paper), je Variante und Teilmenge
res["_hint_quantiles"] = {}
for vk, cols in VARIANTS.items():
    g_, _, _, thrD_ = scores(cols, fits[vk]); ben = fits[vk][4]
    for sname in ("alle nutzbaren Testbilder", "nur histologisch gesichert"):
        idx = subsets[sname]; rows = []
        for q in (.90, .95, .98):
            h_ = hint(idx, g_, float(np.quantile(ben, q)), thrD_); h_["q"] = q; rows.append(h_)
        res["_hint_quantiles"][f"{vk} | {sname}"] = rows
# Vier Zustände (Auslieferung, Variante B): Direktmodell über/unter Schwelle x Glyphen-Score auffällig/unauffällig
gB, t90B, thrDB = TH["B (19 validierte Kanäle)"]; dposB = T["logit"] >= thrDB; gfB = gB >= t90B
res["_states_B"] = {}
for sname in ("alle nutzbaren Testbilder", "nur histologisch gesichert"):
    idx = subsets[sname]; y_ = yT[idx]; dp, gf = dposB[idx], gfB[idx]; st = {}
    for nm, m_ in (("beide_auffaellig", dp & gf), ("nur_direkt", dp & ~gf), ("abweichung", ~dp & gf), ("beide_unauffaellig", ~dp & ~gf)):
        st[nm] = dict(n=int(m_.sum()), malignant=int(y_[m_].sum()), rate=float(y_[m_].mean()) if m_.sum() else None)
    res["_states_B"][sname] = st; print("\nVier Zustände B |", sname, {k: (v["n"], round(v["rate"] * 100, 1)) for k, v in st.items()})
for k, rows in res["_hint_quantiles"].items():
    print("\nHinweisgrenzen", k)
    for h_ in rows:
        print(f"  q={h_['q']:.2f}: markiert {h_['flagged']}, maligne {h_['flagged_malignant']:.1%}, x{h_['lift']:.1f}, gefunden {h_['caught']}/{h_['missed']}")
json.dump(res, open("models/histo_eval.json", "w"), indent=1, ensure_ascii=False)

for sname, r in [(k, v) for k, v in res.items() if not k.startswith('_')]:
    print(f"\n=== {sname}: n={r['n']}, maligne {r['malignant']}, benigne {r['benign']} ===")
    print(f"direkt @ Val-Schwelle: Sens {r['direct_at_val_thr']['sens']:.3f}  Spez {r['direct_at_val_thr']['spec']:.3f}")
    print(f"{'Modell':36s} {'AUROC':>7s} {'95%-KI':>16s} {'Spez@95':>8s}")
    for k in S:
        lo, hi = r["auroc_ci"][k]; print(f"{k:36s} {r['auroc'][k]:7.4f} [{lo:.4f},{hi:.4f}] {r['spec95'][k]:8.3f}")
    for k, v in r["d_auroc_vs_direct"].items():
        print(f"   {k} - Direkt: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
    for k, h in r["hint"].items():
        fm = "-" if h["flagged_malignant"] is None else f"{h['flagged_malignant']:.1%}"; li = "-" if h["lift"] is None else f"{h['lift']:.1f}"
        print(f"   Hinweis {k}: Direkt-benigne {h['n_direct_neg']}, davon maligne {h['missed']} ({h['base_rate']:.1%}); markiert {h['flagged']}, maligne {fm}, Anreicherung x{li} (KI {h['ci']['lift'][0]:.1f}-{h['ci']['lift'][1]:.1f}), gefunden {h['caught']}/{h['missed']}")
