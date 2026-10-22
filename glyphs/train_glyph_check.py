"""Glyph inventory of the V3 model (models/hybrid.pt) on its own real training sources: ISIC 2018 Task 2 (2594 images) and EASY (248).

  python train_glyph_check.py        # -> models/train_glyph_check.json, out/train_glyph_check.md

Per image and channel: detected = at least MIN_CELLS cells (14x14 patch grid, 5 native-pitch tiles) inside lesion + 1-cell border
above the threshold (ISIC attribute channels: calibrated threshold from hybrid_calibration.json; ID channels: raw 0.5, as in analyze.py).
Each detection is then sorted by what the training data said about that image/channel:
  ISIC 2018:  attr (5 channels, real masks; trained on the train split, cal/test = held out)
              id->attr  ID that rolls up to an ISIC attribute (taxonomy column): consistent / contradicts the image's attribute mask
              id_new    ID with no ISIC label at all -> information the ISIC labels do not contain
                        (split into: validated on EASY out-of-fold, or synthetic-only = unvalidated)
  EASY:       easy_trained  24 ID channels with EASY reader targets: >=2 readers / 0 readers / 1 reader (ambiguous)
              easy_skip     ID of an EASY feature that was labelled but not trained (mapping too coarse)
              id_new        ID outside the EASY vocabulary
Diagnoses (ISIC 2019 ground truth / EASY metadata) are used only as a plausibility check of the 'new' detections.
"""
import csv, json
import numpy as np
from scipy.ndimage import binary_dilation
import dense as D, hybrid as H, easy_eval as EE, hybrid3 as H3

MIN_CELLS = 3
tax = H.tax
ids = H.id_list(); K = len(ids); ATTRS = H.ATTRS; amap = H.attr_ids(ids)
id2attr = {ids[j]: a for a in ATTRS for j in amap[a]}
CAL = json.load(open("models/hybrid_calibration.json"))
THR = np.array([.5] * K + [CAL[a]["thr"] for a in ATTRS])
name = {i: (tax[i]["Descriptive term (IDS)"] if tax[i]["Metaphoric term"].startswith("(no") else tax[i]["Metaphoric term"]) for i in ids}
name.update({a: a for a in ATTRS})
chan = ids + ATTRS


def detect(pm, zone):
    """pm [N,5,C,14,14], zone [N,5,14,14] bool -> det [N,C] bool, score [N,C] (max inside zone)"""
    z = zone[:, :, None]
    above = (pm >= THR[None, None, :, None, None]) & z
    det = above.sum((1, 3, 4)) >= MIN_CELLS
    score = np.where(z, pm, 0).max((1, 3, 4))
    return det, score


def zone_of(les):  # les [N,5,14,14] bool
    return np.stack([[binary_dilation(l, iterations=1) for l in v] for v in les])


def easy_oof_auc():
    e = json.load(open("models/easy_v3_compare.json")); best = {}
    for f, r in e.items():
        if isinstance(r, dict) and r.get("trained") and r.get("channel") in ids:
            best[r["channel"]] = max(best.get(r["channel"], 0), r["auroc"]["V3oof"])
    return best


def dx_isic(names):
    g = {}
    for r in csv.DictReader(open("/mnt/samsung_data/mole_data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv")):
        c = [k for k in r if k not in ("image", "UNK") and r[k] in ("1", "1.0")]
        if c:
            g[r["image"]] = c[0]
    return [g.get(n, "?") for n in names]


def main():
    m, _ = H.load_model("models/hybrid.pt")
    oof = easy_oof_auc(); out = {"min_cells": MIN_CELLS, "thresholds": "ID raw 0.5, attributes calibrated"}

    # ---------------------------------------------------------------- ISIC 2018 Task 2
    names = [n[:-4] for n in json.load(open("cache/isic18_names.json"))[:2594]]
    X = np.load("cache/isic18_img.npy", mmap_mode="r")
    A = np.load("cache/isic18_maskmaps.npy", mmap_mode="r"); L = np.load("cache/isic18_lesion.npy", mmap_mode="r")
    N = len(names); pm = np.zeros((N, 5, K + 5, 14, 14), np.float16)
    for s in range(0, N, 200):
        _, p = D.predict(m, np.ascontiguousarray(X[s:min(s + 200, N), :5]).reshape(-1, 224, 224, 3))
        pm[s:s + 200] = p.reshape(-1, 5, K + 5, 14, 14)
    les = H.pool14(np.asarray(L[:, :5])) > .5; zone = zone_of(les)
    gt_attr = H.pool14(np.asarray(A[:, :5])) >= .25                       # [N,5,5,14,14]
    gt_img = (gt_attr & zone[:, :, None]).sum((1, 3, 4)) >= 1               # image has the attribute (inside lesion+border)
    det, score = detect(pm.astype(np.float32), zone)
    sp = H.real_split(); split = np.empty(N, object)
    for k, v in sp.items():
        split[v] = k
    dx = np.array(dx_isic(names))
    R = {"n_images": N, "split_counts": {k: int((split == k).sum()) for k in sp}, "attr": {}, "id": {}}
    for a_i, a in enumerate(ATTRS):
        c = K + a_i; r = {}
        for sname, sel in (("train (seen)", split == "train"), ("cal+test (held out)", split != "train")):
            d, g = det[sel, c], gt_img[sel, a_i]
            r[sname] = dict(n=int(sel.sum()), gt_pos=int(g.sum()), detected=int(d.sum()), tp=int((d & g).sum()), fp=int((d & ~g).sum()),
                            fn=int((~d & g).sum()), sens=float((d & g).sum() / max(g.sum(), 1)), spec=float((~d & ~g).sum() / max((~g).sum(), 1)))
        R["attr"][a] = r
    for j, i in enumerate(ids):
        d = det[:, j]; a = id2attr.get(i)
        r = dict(name=name[i], n_det=int(d.sum()), share=float(d.mean()), kind="id->attr" if a else "id_new",
                 easy_validated=oof.get(i), mean_score_det=float(score[d, j].mean()) if d.any() else None)
        if a:
            g = gt_img[:, ATTRS.index(a)]
            r.update(attr=a, consistent=int((d & g).sum()), contradicts=int((d & ~g).sum()))
        dd = {}
        for k in sorted(set(dx[d])):
            dd[k] = int((dx[d] == k).sum())
        r["dx_of_detected"] = dd
        R["id"][i] = r
    R["dx_counts"] = {k: int((dx == k).sum()) for k in sorted(set(dx))}
    out["isic2018"] = R
    np.savez_compressed("cache/train_glyph_check_isic.npz", det=det, score=score.astype(np.float16), names=np.array(names), split=split.astype(str), dx=dx)

    # ---------------------------------------------------------------- EASY
    z = np.load("cache/easy_v3.npz"); en = [str(n) for n in z["names"]]; nrd = z["nrd"]; feats = [str(f) for f in z["feats"]]
    pe = np.load("cache/easy_pred_v3final.npz")["pm"][:, :5].astype(np.float32)   # same model (hybrid_v3_f0 == hybrid.pt)
    lz = zone_of(np.load("cache/mission_easy_lesion.npz")["cov"][:, :5] >= .5)
    dE, sE = detect(pe, lz)
    meta = {r["ISIC_id"]: r for r in csv.DictReader(open(EE.E + "metadata.csv"))}
    edx = np.array([meta[n]["diagnosis"] for n in en])
    dup = {d["easy"] for d in json.load(open("models/easy_isic2018_overlap.json"))["exact_duplicates"]}
    trained = set(); skipped = set(); f_of = {}
    for fi, f in enumerate(feats):
        mids, _ = EE.MAP[f]
        for mid in mids:
            f_of.setdefault(mid, []).append(fi)
        if f in H3.SKIP or len(mids) != 1:
            skipped.update(mids)
        else:
            trained.add(mids[0])
    skipped -= trained
    RE = {"n_images": len(en), "dup_with_isic2018_train": sorted(dup & set(en)), "id": {}, "attr": {}}
    for j, i in enumerate(ids):
        d = dE[:, j]
        kind = "easy_trained" if i in trained else "easy_skip" if i in skipped else "id_new"
        r = dict(name=name[i], n_det=int(d.sum()), share=float(d.mean()), kind=kind, easy_validated=oof.get(i),
                 mean_score_det=float(sE[d, j].mean()) if d.any() else None)
        if i in f_of:
            nr = nrd[:, f_of[i]].max(1)
            r.update(readers2=int((d & (nr >= 2)).sum()), readers1=int((d & (nr == 1)).sum()), readers0=int((d & (nr == 0)).sum()),
                     gt_pos=int((nr >= 2).sum()), missed=int((~d & (nr >= 2)).sum()))
        r["dx_of_detected"] = {k: int((edx[d] == k).sum()) for k in sorted(set(edx[d]))}
        RE["id"][i] = r
    for a_i, a in enumerate(ATTRS):
        fis = [fi for fi, f in enumerate(feats) if EE.MAP[f][1] == a]
        nr = nrd[:, fis].max(1) if fis else np.zeros(len(en))
        d = dE[:, K + a_i]
        RE["attr"][a] = dict(n_det=int(d.sum()), readers2=int((d & (nr >= 2)).sum()), readers1=int((d & (nr == 1)).sum()),
                             readers0=int((d & (nr == 0)).sum()), gt_pos=int((nr >= 2).sum()), missed=int((~d & (nr >= 2)).sum()))
    RE["dx_counts"] = {k: int((edx == k).sum()) for k in sorted(set(edx))}
    out["easy"] = RE
    np.savez_compressed("cache/train_glyph_check_easy.npz", det=dE, score=sE.astype(np.float16), names=np.array(en), dx=edx)
    json.dump(out, open("models/train_glyph_check.json", "w"), indent=1)
    report(out); print("geschrieben models/train_glyph_check.json, out/train_glyph_check.md")


def rates(r, nd, pos, neg):
    """detection rate in the malignant vs benign group (plausibility only) -> 'x% / y%'"""
    a = r["dx_of_detected"].get(pos, 0) / max(nd.get(pos, 0), 1); b = r["dx_of_detected"].get(neg, 0) / max(nd.get(neg, 0), 1)
    return f"{a * 100:.0f} % / {b * 100:.0f} %"


def report(o):
    I, E = o["isic2018"], o["easy"]; L = []
    L += ["# Glyphs the V3 model detects on its own real training images", "",
          f"Model `models/hybrid.pt` (V3 final). Detected = >= {o['min_cells']} patch cells in lesion + border above threshold "
          "(ID channels raw 0.5, ISIC attribute channels calibrated). Generated by `train_glyph_check.py`.", "",
          f"## ISIC 2018 Task 2 ({I['n_images']} images; train split {I['split_counts']['train']} seen, cal+test {I['split_counts']['cal'] + I['split_counts']['test']} held out)", "",
          "### 5 attribute channels (trained on the real masks)", "", "| attribute | split | GT+ | detected | sens | spec |", "|---|---|--:|--:|--:|--:|"]
    for a, r in I["attr"].items():
        for k, v in r.items():
            L.append(f"| {a} | {k} | {v['gt_pos']} | {v['detected']} | {v['sens']:.2f} | {v['spec']:.2f} |")
    nd = I["dx_counts"]
    L += ["", f"### ID channels that roll up to an ISIC attribute (detection vs that image's attribute mask)", "",
          "| ID | glyph | images | consistent | contradicts mask | EASY oof AUROC |", "|---|---|--:|--:|--:|--:|"]
    for i, r in sorted(I["id"].items(), key=lambda x: -x[1]["n_det"]):
        if r["kind"] == "id->attr" and r["n_det"]:
            L.append(f"| {i} | {r['name']} ({r['attr']}) | {r['n_det']} | {r['consistent']} | {r['contradicts']} | {r['easy_validated'] or '–' if r['easy_validated'] is None else round(r['easy_validated'], 2)} |")
    L += ["", f"### ID channels with no ISIC label (information the ISIC masks do not contain)", "",
          f"MEL / NV = detection rate among the {nd.get('MEL', 0)} melanomas / {nd.get('NV', 0)} nevi with an ISIC 2019 diagnosis (plausibility only).", "",
          "| ID | glyph | images | share | MEL / NV | EASY oof AUROC |", "|---|---|--:|--:|--:|--:|"]
    for i, r in sorted(I["id"].items(), key=lambda x: -x[1]["n_det"]):
        if r["kind"] == "id_new" and r["n_det"] >= 5:
            v = r["easy_validated"]
            L.append(f"| {i} | {r['name']} | {r['n_det']} | {r['share'] * 100:.1f} % | {rates(r, nd, 'MEL', 'NV')} | {'synthetic only' if v is None else round(v, 2)} |")
    nd = E["dx_counts"]
    L += ["", f"## EASY ({E['n_images']} images, all seen in training; {len(E['dup_with_isic2018_train'])} are duplicates of ISIC 2018 train images)", "",
          "Reader columns: detected images where >= 2 / exactly 1 / no reader marked the feature (in-sample, so agreement is expected; "
          "out-of-fold AUROC is the honest number).", "",
          "| ID | glyph | role | images | >=2 rd | 1 rd | 0 rd | missed (>=2 rd, not detected) | mel / nevus |", "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for i, r in sorted(E["id"].items(), key=lambda x: ({"easy_trained": 0, "easy_skip": 1, "id_new": 2}[x[1]["kind"]], -x[1]["n_det"])):
        if r["kind"] != "id_new" or r["n_det"] >= 3:
            g = lambda k: r.get(k, "–")
            L.append(f"| {i} | {r['name']} | {r['kind']} | {r['n_det']} | {g('readers2')} | {g('readers1')} | {g('readers0')} | {g('missed')} | {rates(r, nd, 'melanoma', 'nevus')} |")
    L += ["", "### 5 ISIC attribute channels on EASY (no EASY supervision; readers = any EASY feature mapped to the attribute)", "",
          "| attribute | detected | >=2 rd | 1 rd | 0 rd | missed |", "|---|--:|--:|--:|--:|--:|"]
    for a, r in E["attr"].items():
        L.append(f"| {a} | {r['n_det']} | {r['readers2']} | {r['readers1']} | {r['readers0']} | {r['missed']} |")
    open("out/train_glyph_check.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
