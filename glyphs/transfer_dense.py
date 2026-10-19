"""Zero-shot Derm7pt check of the 71 synthetic ID channels: dense_v1 (synthetic only) vs hybrid (real fine-tuned on 5 attributes)."""
import csv, json, re
import numpy as np
from sklearn.metrics import roc_auc_score
import dense as D, hybrid as H

ids = H.id_list(); tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
col = {i: tax[i]["Derm7pt label"] for i in ids}
VASC = {"DOT": "dotted", "HP": "hairpin", "LIR": "linear irregular", "COM": "comma", "ARB": "arborizing"}
mp = {"network (any)": [i for i in ids if "pigment_network" in col[i]], "network (atypical)": [i for i in ids if "pigment_network" in col[i] and "ATP" in col[i] and "TYP" not in col[i]],
      "streaks": [i for i in ids if col[i].startswith("streaks")], "dots/globules": [i for i in ids if "dots_and_globules" in col[i]],
      "blue-whitish veil": [i for i in ids if "blue_whitish_veil" in col[i]], "regression": [i for i in ids if col[i].startswith("regression_structures")]}
for c, n in VASC.items():
    mp[f"vessels: {n}"] = [i for i in ids if re.search(rf"vascular_structures = {c}\b", col[i])]
lab = json.load(open("cache/d7_labels.json")); M = len(lab["diagnosis"])
tg = {"network (any)": np.array(lab["pigment_network"]) != "absent", "network (atypical)": np.array(lab["pigment_network"]) == "atypical",
      "streaks": np.array(lab["streaks"]) != "absent", "dots/globules": np.array(lab["dots_and_globules"]) != "absent",
      "blue-whitish veil": np.array(lab["blue_whitish_veil"]) == "present", "regression": np.array(lab["regression_structures"]) != "absent"}
for c, n in VASC.items():
    tg[f"vessels: {n}"] = np.array(lab["vascular_structures"]) == n
d7 = np.load("cache/d7_img.npy"); tiles = np.ascontiguousarray(d7[:, :5].reshape(-1, 224, 224, 3))
res = {}
for tag in ("dense_v1", "hybrid"):
    m, _ = H.load_model(f"models/{tag}.pt")
    pi, pm = D.predict(m, tiles); pi = pi[:, :len(ids)].reshape(M, 5, -1); pm = pm[:, :len(ids)].reshape(M, 5, len(ids), -1).max(-1)
    res[tag] = {n: (roc_auc_score(tg[n], pi[:, :, [ids.index(i) for i in mp[n]]].max((1, 2))), roc_auc_score(tg[n], pm[:, :, [ids.index(i) for i in mp[n]]].max((1, 2)))) for n in tg if mp[n]}
print(f"{'target (n pos)':32s} {'dense_v1 img/map':>18s} {'hybrid img/map':>18s}")
for n in res["dense_v1"]:
    a, b = res["dense_v1"][n], res["hybrid"][n]
    print(f"{n + f' ({tg[n].sum()})':32s} {a[0]:8.3f} /{a[1]:6.3f}   {b[0]:8.3f} /{b[1]:6.3f}")
json.dump(res, open("models/transfer_dense_d7.json", "w"), indent=1)
