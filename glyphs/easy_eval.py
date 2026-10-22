"""Auswertung OHNE Neutraining: hybrid.pt auf den 248 EASY-Bildern (Liopyris et al. 2023, ISIC-Collection 166).

  python easy_eval.py masks   # Reader-Masken -> Abdeckung je Ansicht/Merkmal (cache/easy_gt.npz)
  python easy_eval.py pred    # Vorhersagen (GPU/CPU, EASY_DEV) -> cache/easy_pred.npz
  python easy_eval.py eval    # Kennzahlen -> models/easy_eval.json

Geometrie wie prep_real.py: kurze Seite -> 450 px, mittleres Quadrat, 4 Eck-Tiles + Mitte + Gesamtansicht (je 224 px).
GT je Merkmal/Ansicht/14x14-Zelle = Summe der Leser-Abdeckungen (0..n_Leser). Positive Bilder: >=2 Leser markieren das Merkmal;
negative Bilder: kein Leser. Bilder mit genau 1 Leser sind mehrdeutig und werden aus dem AUROC ausgenommen.
"""
import csv, glob, json, os, sys, time
import cv2, numpy as np
from multiprocessing import Pool

E = "external/expert-annotation-agreement-data/"
POS = [(0, 0), (226, 0), (0, 226), (226, 226), (113, 113)]
cv2.setNumThreads(1)

# Handgepflegte Zuordnung EASY-Merkmal -> Taxonomie-IDs (registry.csv); ISIC-Attribut nur wo eine Entsprechung besteht.
MAP = {
 "Dots : Irregular": (["DS037"], "globules"), "Dots : Regular": (["DS037"], "globules"),
 "Globules / Clods : Cobblestone pattern": (["DS029"], "globules"), "Globules / Clods : Irregular": (["DS025"], "globules"),
 "Globules / Clods : Regular": (["DS025"], "globules"), "Globules / Clods : Milky red": (["DS033"], "globules"),
 "Globules / Clods : Rim of brown globules": (["DS026"], "globules"),
 "Lines : Angulated lines / Polygons / Zig-zag pattern": (["DS019", "DS020"], None),
 "Lines : Branched streaks": (["DS007"], "streaks"), "Lines : Pseudopods": (["DS024"], "streaks"),
 "Lines : Radial streaming": (["DS009"], "streaks"),
 "Network : Atypical pigment network / Reticulation": (["DS004"], "pigment_network"),
 "Network : Broadened pigment network / Reticulation": (["DS003"], "pigment_network"),
 "Network : Delicate Pigment Network / Reticulation": (["DS002"], "pigment_network"),
 "Network : Negative pigment network": (["DS005"], "negative_network"),
 "Network : Typical pigment network / Reticulation": (["DS022"], "pigment_network"),
 "Pattern : Homogeneous : NOS": (["DS055"], None), "Pattern : Starburst": (["DS023"], None),
 "Regression structures : Peppering / Granularity": (["DS038"], None),
 "Regression structures : Scarlike depigmentation": (["DS051"], None),
 "Shiny white structures : Shiny white streaks": (["DS006"], None),
 "Structureless : Blotch irregular": (["DS048"], None), "Structureless : Blotch regular": (["DS048"], None),
 "Structureless : Blue-whitish veil": (["DS049"], None), "Structureless : Milky red areas": (["DS050"], None),
 "Structureless : Structureless brown (tan)": (["DS057"], None),
 "Vessels : Comma": (["DS067"], None), "Vessels : Corkscrew": (["DS066"], None), "Vessels : Dotted": (["DS060"], None),
 "Vessels : Linear irregular": (["DS065"], None), "Vessels : Polymorphous": (["DS069"], None),
}
ATTRS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
FEATS = sorted(MAP)


def fname(feat):
    return feat.replace("(", "").replace(")", "").replace(" : ", "_-_").replace(" / ", "_+_").replace(" ", "_")


def resize_min450(img, interp):
    h, w = img.shape[:2]; s = 450 / min(h, w)
    return cv2.resize(img, (max(450, int(round(w * s))), max(450, int(round(h * s)))), interpolation=interp)


def views(arr, interp):
    h, w = arr.shape[:2]; y0, x0 = (h - 450) // 2, (w - 450) // 2
    sq = arr[y0:y0 + 450, x0:x0 + 450]
    return [sq[y:y + 224, x:x + 224] for x, y in POS] + [cv2.resize(sq, (224, 224), interpolation=interp)]


def mask_job(a):
    path = a
    m = cv2.imread(path, 0)
    if m is None:
        return None
    m = resize_min450((m > 127).astype(np.float32), cv2.INTER_AREA)
    return np.stack([cv2.resize(v, (14, 14), interpolation=cv2.INTER_AREA) for v in views(m, cv2.INTER_AREA)])


def do_masks(limit=150):
    """Fortsetzbar: speichert Zwischenstand (cache/easy_gt_partial.npz), bricht nach `limit` Sekunden ab; erneut aufrufen bis fertig."""
    import pickle
    t = time.time(); j = json.load(open(E + "markups.json")); names = [l.strip() for l in open(E + "image_list.txt") if l.strip()]
    jobs, keys = [], []
    for ii, n in enumerate(names):
        for rd, fs in j[n].items():
            for f in fs:
                p = f"{E}masks/{n}_{rd}_{fname(f)}.png"
                if not os.path.exists(p):
                    print("fehlt:", p); continue
                jobs.append(p); keys.append((ii, FEATS.index(f)))
    part = "cache/easy_gt_partial.pkl"; os.makedirs("cache", exist_ok=True)
    done = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    pend = [k for k in range(len(jobs)) if k not in done]
    print(len(jobs), "Masken,", len(done), "fertig,", len(pend), "offen", flush=True)
    pool = Pool(2)
    try:
        for k, r in zip(pend, pool.imap(mask_job, [jobs[k] for k in pend], chunksize=2)):
            done[k] = r
            if time.time() - t > limit:
                break
    finally:
        pool.terminate()
    pickle.dump(done, open(part, "wb"))
    print(len(done), "/", len(jobs), f"nach {time.time() - t:.0f}s", flush=True)
    if len(done) < len(jobs):
        return
    cov = np.zeros((len(names), len(FEATS), 6, 14, 14), np.float32); nrd = np.zeros((len(names), len(FEATS)), np.int16)
    for k, (ii, fi) in enumerate(keys):
        if done[k] is not None:
            cov[ii, fi] += done[k]; nrd[ii, fi] += 1
    np.savez("cache/easy_gt.npz", cov=cov, nrd=nrd, names=np.array(names), feats=np.array(FEATS))
    print("easy_gt.npz geschrieben")


TAG = os.environ.get("EASY_TAG", "")   # z. B. "_gpu" -> getrennte Ausgabedateien
TLIM = float(os.environ.get("EASY_TLIM", "1e9"))


def do_pred():
    import torch
    DEV = os.environ.get("EASY_DEV", "cuda" if torch.cuda.is_available() else "cpu")
    src = open("dense.py").read().replace('dev = "cuda"', f'dev = "{DEV}"'); D = type(sys)("dense_dev"); exec(src, D.__dict__)
    if DEV == "cpu":
        torch.set_num_threads(2)
    ck = torch.load("models/hybrid.pt", map_location="cpu", weights_only=False)
    m = D.DenseViT(ck["n_out"], pretrained=False); m.load_state_dict(ck["state"]); m.to(DEV).eval()
    ids = [str(i) for i in ck["ids"]]; names = [l.strip() for l in open(E + "image_list.txt") if l.strip()]
    mean = np.array([0.485, 0.456, 0.406], np.float32) * 255; std = np.array([0.229, 0.224, 0.225], np.float32) * 255
    part = f"cache/easy_pred{TAG}_partial.npz"; t = time.time()
    if os.path.exists(part):
        z = np.load(part); out, ok = z["pm"], z["ok"]
    else:
        out = np.zeros((len(names), 6, ck["n_out"], 14, 14), np.float16); ok = np.zeros(len(names), bool)
    for i, n in enumerate(names):
        if ok[i]:
            continue
        if time.time() - t > TLIM:
            break
        img = cv2.imread(f"{E}images/{n}.jpg")
        img = cv2.cvtColor(resize_min450(img, cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (np.stack(views(img, cv2.INTER_AREA)).astype(np.float32) - mean) / std
        with torch.no_grad():
            _, pm = m(torch.from_numpy(x).permute(0, 3, 1, 2).to(DEV))
        out[i] = torch.sigmoid(pm).float().cpu().numpy().astype(np.float16); ok[i] = True
    np.savez(part, pm=out, ok=ok)
    print(int(ok.sum()), "/", len(names), f"nach {time.time() - t:.0f}s", flush=True)
    if ok.all():
        np.savez(f"cache/easy_pred{TAG}.npz", pm=out, ids=np.array(ids)); print(f"easy_pred{TAG}.npz geschrieben")


def auc(y, s):
    """AUROC ueber Raenge (Mann-Whitney), Bindungen mit mittlerem Rang."""
    y = np.asarray(y, bool); s = np.asarray(s, np.float64); o = np.argsort(s, kind="mergesort"); ss = s[o]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(y, s, n=300):
    rng = np.random.default_rng(0); v = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].min() == y[ix].max():
            continue
        v.append(auc(y[ix], s[ix]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def do_eval():
    g = np.load("cache/easy_gt.npz"); p = np.load(f"cache/easy_pred{TAG}.npz")
    cov, nrd, feats = g["cov"], g["nrd"], [str(f) for f in g["feats"]]; pm = p["pm"].astype(np.float32); ids = [str(i) for i in p["ids"]]
    chan = ids + ATTRS; K = len(ids)
    img_score = pm.max((1, 3, 4))                       # [N, C] max ueber Ansichten und Zellen
    generic = img_score.mean(1)                         # Kontrolle: Mittel ueber alle 76 Kanaele (unspezifisch)
    anyreg = cov.sum(1) >= .25                          # [N,6,14,14] von irgendeinem Leser/Merkmal markiert = Naeherung fuer die Laesion
    res = {}
    print(f"{'Merkmal':50s} {'Kanal':16s} {'n+':>3s} {'n-':>4s} {'AUROC':>6s} {'95%-KI':>11s} {'generisch':>9s} {'Rang/76':>7s} {'Zeigen':>6s} {'Zufall*':>7s} {'ZellAUC*':>8s}")
    for fi, f in enumerate(feats):
        pos = nrd[:, fi] >= 2; neg = nrd[:, fi] == 0; use = pos | neg
        ids_m, attr = MAP[f]
        cand = {",".join(ids_m): [chan.index(i) for i in ids_m]}
        if attr:
            cand[attr + "*"] = [K + ATTRS.index(attr)]
        allauc = np.array([auc(pos[use], img_score[use, c]) for c in range(len(chan))])
        gen = auc(pos[use], generic[use]); best = chan[int(allauc.argmax())]
        for cname, cs in cand.items():
            s_ = img_score[:, cs].max(1)
            a_ = auc(pos[use], s_[use]); ci = boot(pos[use], s_[use])
            rank = int(1 + (allauc > a_).sum()) if len(cs) == 1 else None
            hits, ch_in, ys, ss = [], [], [], []
            for n in np.nonzero(pos)[0]:
                P = pm[n][:, cs].max(1)
                v = int(P.max((1, 2)).argmax()); cell = np.unravel_index(P[v].argmax(), (14, 14))
                gt = cov[n, fi, v] >= .25; reg = anyreg[n, v]
                if gt.any() and reg.any():
                    hits.append(gt[cell]); ch_in.append((gt & reg).sum() / reg.sum())
                for vv in range(6):
                    gt2 = cov[n, fi, vv] >= .5; r2 = anyreg[n, vv]
                    if (gt2 & r2).any() and (~gt2 & r2).any():
                        ys.append(gt2[r2]); ss.append(pm[n, vv][cs].max(0)[r2])
            cell_auc = auc(np.concatenate(ys), np.concatenate(ss)) if ys else float("nan")
            print(f"{f:50s} {cname:16s} {pos.sum():3d} {neg.sum():4d} {a_:6.3f} [{ci[0]:.2f},{ci[1]:.2f}] {gen:9.3f} {str(rank) if rank else '-':>7s} {np.mean(hits):6.2f} {np.mean(ch_in):7.2f} {cell_auc:8.2f}")
            res.setdefault(f, {})[cname] = dict(n_pos=int(pos.sum()), n_neg=int(neg.sum()), image_auroc=a_, ci=ci, generic_auroc=gen, rank_of_76=rank,
                                                best_channel=best, best_auroc=float(allauc.max()), pointing=float(np.mean(hits)),
                                                pointing_chance_in_annotated_region=float(np.mean(ch_in)), n_pointing=len(hits), cell_auroc_in_annotated_region=cell_auc)
    json.dump(res, open(f"models/easy_eval{TAG}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"masks": do_masks, "pred": do_pred, "eval": do_eval}[sys.argv[1]]()
