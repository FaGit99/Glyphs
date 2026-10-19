"""One example report per state of the two-path view (random draw with a fixed seed), written to out/twopath/."""
import json, os, numpy as np
import fusion_val_test as FV, direct_model as DM, analyze as A
os.makedirs("out/twopath", exist_ok=True)
for f in os.listdir("out/twopath"):
    if f.endswith(".html"): os.remove("out/twopath/" + f)
T = FV.load_split("test"); P = json.load(open("models/twopath.json")); g = P["glyph"]; rows = {r["isic_id"]: r for r in DM.split_rows("test")}
cols = np.array(g.get("cols", list(range(90))))
gs = ((T["G"][:, cols] - np.array(g["mean"])) / np.array(g["scale"])) @ np.array(g["coef"]) + g["intercept"]
d_high = T["logit"] >= P["direct"]["thr_logit"]; g_high = gs >= g["flag_thr"]; rng = np.random.default_rng(1)
cases = [("flag_malignant", (~d_high) & g_high & (T["y"] == 1)), ("flag_benign", (~d_high) & g_high & (T["y"] == 0)), ("both_high_malignant", d_high & g_high & (T["y"] == 1)),
         ("direct_only_malignant", d_high & ~g_high & (T["y"] == 1)), ("direct_only_benign", d_high & ~g_high & (T["y"] == 0)), ("both_low_benign", (~d_high) & ~g_high & (T["y"] == 0))]
for name, m in cases:
    i = int(rng.choice(np.nonzero(m)[0])); iid = T["id"][i]; R = A.analyze(DM.ARCHIVE + rows[iid]["image_relpath"]); tp = R["twopath"]
    open(f"out/twopath/{name}_{iid}.html", "w", encoding="utf-8").write(A.html(R))
    print(f"{name:22s} {iid} truth={'malignant' if T['y'][i] else 'benign':9s} state={tp['state']:11s} direct {tp['direct_prob']:.2f}  glyph score {tp['g_score']:+.2f} (stored {gs[i]:+.2f}) pct {tp['g_pct']:.0f}")
