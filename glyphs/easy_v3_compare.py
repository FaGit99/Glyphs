"""V2 vs Kontrolle (ohne EASY) vs V3 Out-of-fold auf EASY: AUROC je Merkmal/Kanal mit gepaartem Bootstrap über Bilder.

  python easy_v3_compare.py     -> models/easy_v3_compare.json + Tabelle
Voraussetzung: cache/easy_pred_{v2check,v3noeasy,v3oof}.npz (easy_v3_pred.py)
"""
import json
import numpy as np
import easy_eval as EE
from hybrid3 import SKIP

arms = {"V2": "_v2check", "ctrl": "_v3noeasy", "V3oof": "_v3oof"}
g = np.load("cache/easy_gt.npz"); nrd, feats = g["nrd"], [str(f) for f in g["feats"]]
P = {}
for a, t in arms.items():
    z = np.load(f"cache/easy_pred{t}.npz"); P[a] = (z["pm"].astype(np.float32).max((1, 3, 4)), [str(i) for i in z["ids"]])
ids = P["V2"][1]; assert all(P[a][1] == ids for a in P)
grp = np.load("cache/easy_v3.npz")["group"]
rng = np.random.default_rng(0); res = {}; rows = []
for fi, f in enumerate(feats):
    mids = EE.MAP[f][0]
    if len(mids) != 1:
        continue
    c = ids.index(mids[0]); pos = nrd[:, fi] >= 2; neg = nrd[:, fi] == 0; use = pos | neg
    y = pos[use]
    if y.sum() < 5:
        continue
    S = {a: P[a][0][use, c] for a in P}
    A = {a: EE.auc(y, S[a]) for a in S}
    bs = {a: [] for a in S}; n = len(y)
    for _ in range(500):
        b = rng.integers(0, n, n)
        if y[b].sum() == 0 or y[b].sum() == n:
            continue
        for a in S:
            bs[a].append(EE.auc(y[b], S[a][b]))
    d = np.array(bs["V3oof"]) - np.array(bs["V2"]); d2 = np.array(bs["V3oof"]) - np.array(bs["ctrl"])
    res[f] = dict(channel=mids[0], n_pos=int(y.sum()), n_neg=int((~y).sum()), trained=f not in SKIP, auroc=A,
                  d_v3_v2=[float(A["V3oof"] - A["V2"]), *np.percentile(d, [2.5, 97.5]).tolist()],
                  d_v3_ctrl=[float(A["V3oof"] - A["ctrl"]), *np.percentile(d2, [2.5, 97.5]).tolist()])
    rows.append((f, mids[0], int(y.sum()), f not in SKIP, A, res[f]["d_v3_v2"], res[f]["d_v3_ctrl"]))
print(f"{'Merkmal':52s} {'ID':6s} {'n+':>3s} {'trn':>3s} {'V2':>5s} {'ctrl':>5s} {'V3oof':>5s}  {'V3-V2 [KI]':>20s}  {'V3-ctrl [KI]':>20s}")
for f, i, n, t, A, d, d2 in rows:
    print(f"{f:52s} {i:6s} {n:3d} {'ja' if t else '':>3s} {A['V2']:.3f} {A['ctrl']:.3f} {A['V3oof']:.3f}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}]")
for name, sel in (("trainiert (n+>=10)", lambda r: r[3] and r[2] >= 10), ("nicht trainiert (n+>=10)", lambda r: (not r[3]) and r[2] >= 10)):
    rr = [r for r in rows if sel(r)]
    if rr:
        print(f"Mittel {name}: n={len(rr)}  " + "  ".join(f"{a} {np.mean([r[4][a] for r in rr]):.3f}" for a in arms))
        res["_mean_" + name] = {a: float(np.mean([r[4][a] for r in rr])) for a in arms}
json.dump(res, open("models/easy_v3_compare.json", "w"), indent=1, ensure_ascii=False)
