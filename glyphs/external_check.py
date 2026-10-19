"""Extended external test of the glyph path (calibrated checklist) on four independent sources.

  python external_check.py extract   # run the pipeline on the sampled images, cache features in models/ext_<name>.npz
  python external_check.py eval

Sources (no image used for any training): HAM10000 (index-parity-1 half), BCN20000 (Hospital Clinic Barcelona), ISIC 2019 images that are neither
HAM, BCN nor ISIC-2018 (MSK etc.), ISIC 2020 (SIIM). Nothing here is fitted on the external data; probes are fitted on Derm7pt only.
"""
import csv, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, torch, timm
from PIL import Image
import analyze as A, dense as D, hybrid as H, d7_data as DD, mil, skin

Dp = "/mnt/samsung_data/mole_data/data/"
CAP = {"MEL": 500, "NV": 800, "OTHER": 200}
GROUPS = ["MEL", "NV", "BCC", "BKL", "AKIEC", "DF", "VASC", "OTHER"]


def sample(items, seed=0):
    """items: [(path, group)] -> capped sample (MEL 500, NV 800, every other group 200)"""
    rng = np.random.default_rng(seed); out = []
    for g in GROUPS:
        sel = [x for x in items if x[1] == g]
        k = CAP.get(g, CAP["OTHER"]); out += [sel[i] for i in rng.permutation(len(sel))[:k]]
    return out


def datasets():
    names18 = {os.path.basename(f)[:-4] for sub in ("Training", "Validation", "Test") for f in glob.glob(Dp + f"raw/isic2018/ISIC2018_Task1-2_{sub}_Input/*.jpg")}
    ham = {r["image_id"]: r["dx"] for r in csv.DictReader(open(Dp + "metadata/HAM10000/HAM10000_metadata.csv"))}
    hmap = {"mel": "MEL", "nv": "NV", "bcc": "BCC", "bkl": "BKL", "akiec": "AKIEC", "df": "DF", "vasc": "VASC"}
    fh = {f.split("/")[-1][:-4]: f for f in skin.ham_files(1)}
    d = {"HAM10000": sample([(f, hmap[ham[i]]) for i, f in fh.items() if i in ham])}
    bcn = {r["isic_id"]: r["diagnosis"] for r in csv.DictReader(open(Dp + "metadata/bcn20000/metadata.csv"))}
    bmap = {"melanoma": "MEL", "nevus": "NV", "basal cell carcinoma": "BCC", "seborrheic keratosis": "BKL", "solar lentigo": "BKL", "lichenoid keratosis": "BKL", "actinic keratosis": "AKIEC",
            "squamous cell carcinoma": "AKIEC", "dermatofibroma": "DF", "vascular lesion": "VASC", "scar": "OTHER", "other": "OTHER", "atypical melanocytic proliferation": "OTHER", "angioma": "VASC"}
    d["BCN20000"] = sample([(Dp + f"raw/bcn20000/images/{i}.jpg", bmap[dx]) for i, dx in bcn.items() if dx in bmap and os.path.exists(Dp + f"raw/bcn20000/images/{i}.jpg")])
    g19 = []
    for r in csv.DictReader(open(Dp + "metadata/isic2019/ground_truth.csv")):
        i = r["image"]
        if i in names18 or i in ham or i in bcn:
            continue
        c = [k for k in ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC") if float(r[k]) == 1.0]
        if c:
            g19.append((Dp + f"raw/isic2019/ISIC_2019_Training_Input/{i}.jpg", {"AK": "AKIEC", "SCC": "AKIEC"}.get(c[0], c[0])))
    d["ISIC2019 (MSK etc.)"] = sample(g19)
    g20 = []
    for r in csv.DictReader(open(Dp + "metadata/isic2020/ground_truth.csv")):
        grp = "MEL" if r["diagnosis"] == "melanoma" else "NV" if r["diagnosis"] == "nevus" else "OTHER" if r["target"] == "0" else None
        if grp:
            g20.append((Dp + f"raw/isic2020/train/{r['image_name']}.jpg", grp))
    d["ISIC2020"] = sample(g20)
    return d


def load450(path):
    with Image.open(path) as im:
        w, h = im.size
    r = 1
    while min(w, h) / (r * 2) >= 450 and r < 8:
        r *= 2
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[r]
    raw = cv2.cvtColor(cv2.imread(path, flag), cv2.COLOR_BGR2RGB); s = 450 / min(raw.shape[:2])
    return cv2.resize(raw, (int(round(raw.shape[1] * s)), int(round(raw.shape[0] * s))), interpolation=cv2.INTER_AREA)


def extract():
    M = A.models(); old, _ = H.load_model("models/hybrid.pt")
    net = timm.create_model(D.BACKBONE, pretrained=True, num_classes=0).to(D.dev).eval()
    for name, items in datasets().items():
        fn = f"models/ext_{name.split()[0]}.npz"
        if os.path.exists(fn):
            continue
        print(f"{name}: {len(items)} images " + str({g: sum(1 for x in items if x[1] == g) for g in GROUPS}), flush=True); t0 = time.time()
        feats = {k: [] for k in ("exp", "pts", "cal14", "agg_raw", "old", "imnet", "group", "path", "ntiles")}; skipped = 0
        with ThreadPoolExecutor(8) as ex:
            for n, ((path, grp), img) in enumerate(zip(items, ex.map(lambda x: _safe(load450, x[0]), items))):
                if img is None:
                    skipped += 1; continue
                les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
                pos = [(x, y) for y in A.positions(h) for x in A.positions(w)]
                c = A.checklist(img, les, pos, M, render=False, keep=True)
                po, _ = D.predict(old, c["tiles"])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    fi = net(D.prep(torch.from_numpy(c["tiles"]), None, False)).float().mean(0).cpu().numpy()
                for k, v in (("exp", c["expected"]), ("pts", c["points"]), ("cal14", c["cal14"]), ("agg_raw", c["agg_raw"]), ("old", np.sort(po, 0)[::-1][:3].mean(0)), ("imnet", fi), ("group", grp), ("path", path), ("ntiles", c["n_tiles"])):
                    feats[k].append(v)
                if n % 500 == 499:
                    print(f"  {n + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        np.savez(fn, **{k: np.array(v) for k, v in feats.items()}); print(f"  done {time.time() - t0:.0f}s, skipped {skipped}", flush=True)


def _safe(f, p):
    try:
        return f(p)
    except Exception:
        return None


if __name__ == "__main__":
    extract() if sys.argv[1] == "extract" else None


# ------------------------------------------------------------------------------------------------------------------------------
def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sp = DD.split(); pool = np.concatenate([sp["train"], sp["valid"]]); mel7 = DD.melanoma(); K = 76
    oof = np.zeros((1011, K + 14), np.float32)
    for k in range(5):
        z = np.load(f"models/mel_fold{k}_pred.npz"); oof[z["idx"]] = z["pred"]
    FI7 = np.load("cache/d7_imnet_feat.npy")
    GRP = {"clark nevus": "NV", "reed or spitz nevus": "NV", "dermal nevus": "NV", "blue nevus": "NV", "congenital nevus": "NV", "combined nevus": "NV", "recurrent nevus": "NV", "seborrheic keratosis": "BKL",
           "lentigo": "BKL", "melanosis": "BKL", "basal cell carcinoma": "BCC", "dermatofibroma": "DF", "vascular lesion": "VASC", "miscellaneous": "OTHER"}
    g7 = np.array(["MEL" if r["diagnosis"].startswith("melanoma") else GRP[r["diagnosis"]] for r in DD.meta()])
    probe = lambda Xtr, y, Xte: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight="balanced")).fit(Xtr, y).decision_function(Xte)
    ids = [str(i) for i in H.load_model("models/hybrid.pt")[1]["ids"]]; ix = lambda l: [ids.index(i) for i in l]
    BM = {"network_atypical": ix(["DS003", "DS004"]), "veil": ix(["DS049"]), "streaks_irregular": ix(["DS008", "DS009", "DS024"]) + [73], "pigmentation_irregular": [], "dots_irregular": ix(["DS025", "DS037"]) + [75],
          "regression_any": ix(["DS051"]), "vessels_dotted": ix(["DS060"]), "vessels_linear_irregular": ix(["DS065"])}
    gnew = lambda P, n: P[:, K + DD.TNAMES.index(n)]
    def old_rule(P):
        g = lambda n: P[:, BM[n]].max(1) if BM[n] else np.zeros(len(P))
        return 2 * g("network_atypical") + 2 * g("veil") + g("streaks_irregular") + g("pigmentation_irregular") + g("dots_irregular") + g("regression_any") + 2 * np.maximum(g("vessels_dotted"), g("vessels_linear_irregular"))
    def ci(y, s, n=300):
        r = np.random.default_rng(0); b = []
        for _ in range(n):
            i = r.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                b.append(roc_auc_score(y[i], s[i]))
        return np.percentile(b, 2.5), np.percentile(b, 97.5)
    fmt = lambda y, s: (lambda a, c: f"{a:.3f} [{c[0]:.2f}-{c[1]:.2f}]")(roc_auc_score(y, s), ci(y, s))
    res, allf = {}, {}
    for fn in sorted(glob.glob("models/ext_*.npz")):
        name = fn[len("models/ext_"):-4]; z = np.load(fn, allow_pickle=True); allf[name] = z
    print("Probes fitted on Derm7pt train+valid only; the rule needs no diagnosis training at all.\n")
    for task, pos, neg in (("T1  melanoma vs nevus", ["MEL"], ["NV"]), ("T2  melanoma vs all other lesions", ["MEL"], [g for g in GROUPS if g != "MEL"])):
        print(f"=== {task}: AUC [95% CI] ===")
        print(f"{'dataset (n mel / n other)':30s} {'R calibrated rule':>19s} {'R points>=50%':>17s} {'B1 probe 14ch':>17s} {'D ImageNet probe':>17s} {'old glyph rule':>17s} | pts>=3: sens/spec")
        pooled = {k: [] for k in ("y", "R", "P", "B1", "D", "O", "pts")}
        for name, z in allf.items():
            g = z["group"]; m = np.isin(g, pos + neg); y = np.isin(g[m], pos).astype(int)
            if y.sum() < 20:
                continue
            sc = {"R": z["exp"][m], "P": z["pts"][m].astype(float), "B1": probe(oof[pool][:, K:], mel7[pool], z["agg_raw"][m][:, K:]), "D": probe(FI7[pool], mel7[pool], z["imnet"][m]), "O": old_rule(z["old"][m])}
            pr = z["pts"][m] >= 3
            print(f"{name + f' ({y.sum()}/{(1 - y).sum()})':30s} {fmt(y, sc['R']):>19s} {fmt(y, sc['P']):>17s} {fmt(y, sc['B1']):>17s} {fmt(y, sc['D']):>17s} {fmt(y, sc['O']):>17s} | {pr[y == 1].mean():.2f}/{(~pr[y == 0]).mean():.2f}")
            res.setdefault(task, {})[name] = {k: float(roc_auc_score(y, v)) for k, v in sc.items()} | dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), sens3=float(pr[y == 1].mean()), spec3=float((~pr[y == 0]).mean()))
            for k, v in (("y", y), ("R", sc["R"]), ("P", sc["P"]), ("B1", sc["B1"]), ("D", sc["D"]), ("O", sc["O"]), ("pts", pr)):
                pooled[k].append(v)
        y = np.concatenate(pooled["y"]); print(f"{'POOLED (' + str(int(y.sum())) + '/' + str(int((1 - y).sum())) + ')':30s} " + " ".join(f"{fmt(y, np.concatenate(pooled[k])):>{w}s}" for k, w in (("R", 19), ("P", 17), ("B1", 17), ("D", 17), ("O", 17))), end="")
        pr = np.concatenate(pooled["pts"]); print(f" | {pr[y == 1].mean():.2f}/{(~pr[y == 0]).mean():.2f}\n")
    # other groups with a 90-channel probe fitted on Derm7pt (one vs rest)
    print("=== other groups, one-vs-rest AUC: 90-channel glyph probe | ImageNet probe (both fitted on Derm7pt) ===")
    for grp in ("BCC", "BKL"):
        for name, z in allf.items():
            g = z["group"]; y = (g == grp).astype(int)
            if y.sum() < 20:
                continue
            a = probe(oof[pool], (g7[pool] == grp).astype(int), z["agg_raw"]); b = probe(FI7[pool], (g7[pool] == grp).astype(int), z["imnet"])
            print(f"{grp:4s} {name + f' ({y.sum()}/{(1 - y).sum()})':32s} {fmt(y, a):>19s} | {fmt(y, b):>19s}")
            res.setdefault(grp, {})[name] = dict(glyph=float(roc_auc_score(y, a)), imnet=float(roc_auc_score(y, b)))
    json.dump(res, open("models/external_check.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
