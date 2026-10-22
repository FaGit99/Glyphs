"""Mission synthetic-vs-mixed: capabilities C1-C6 per training regime -> models/mission_compare.json, out/mission_dose_curve.png.

  python mission_compare.py
Inputs: cache/mission_pred_<regime>.npz (mission_pred.py), cache/mission_easy_lesion.npz, cache/mission_ext.npz (mission_ext.py),
cache/easy_gt.npz (reader coverage, easy_eval.py masks), ISIC/Derm7pt caches.
EASY protocol = easy_eval.py: 6 views, 14x14 maps, image score = max over views and cells, positive >= 2 readers, negative = no reader,
exactly 1 reader excluded. Bootstrap over images (B = 1000, seed 0); differences are paired (same resamples for every regime).
"""
import csv, json, re
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import rankdata
import easy_eval as EE, hybrid as H, d7_data as DD
from hybrid3 import SKIP

B = 1000; RNG = np.random.default_rng(0)
IDS = H.id_list(); K = len(IDS); ATTRS = H.ATTRS; AMAP = H.attr_ids(IDS)
TAX = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
REG = ["R0", "R0img", "R0pipe", "R0pipe_s1", "R1", "R2", "R2ctrl", "R3", "R3_s1", "R5", "R4_25", "R4_50", "R4_75"]
P = {}
for r in REG:
    try:
        P[r] = dict(np.load(f"cache/mission_pred_{r}.npz"))
    except FileNotFoundError:
        print("missing predictions:", r)
REG = [r for r in REG if r in P]


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or np.isnan(s).any():
        return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_rows(y, S):
    """AUROC of every row of S [R, n] against y [n] (NaN rows -> NaN)"""
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    out = np.full(len(S), np.nan)
    if n1 == 0 or n0 == 0:
        return out
    ok = ~np.isnan(S).any(1)
    if ok.any():
        out[ok] = (rankdata(S[ok], axis=1)[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else [np.nan, np.nan]


def boot_auc(y, S, n_img=None, groups=None):
    """point AUROC per row and bootstrap matrix [B, R]; resampling over images (rows of y)"""
    n = len(y); bs = np.full((B, len(S)), np.nan)
    for b in range(B):
        ix = RNG.integers(0, n, n); bs[b] = auc_rows(y[ix], S[:, ix])
    return auc_rows(y, S), bs


# ------------------------------------------------------------------------------------------------ EASY (C1, C2, C3, C4)
g = np.load("cache/easy_gt.npz"); COV, NRD = g["cov"], g["nrd"]; FEATS = [str(f) for f in g["feats"]]; NE = len(NRD)
LES = np.load("cache/mission_easy_lesion.npz")["cov"] >= .5                       # [248,6,14,14] lesion cells
ZONE = np.stack([[binary_dilation(LES[i, v], iterations=1) for v in range(6)] for i in range(NE)])   # lesion + 1-cell border
ANYREG = COV.sum(1) >= .25                                                         # old proxy (any reader, any feature)


def easy_channels(f):
    mids, attr = EE.MAP[f]
    return [IDS.index(i) for i in mids], ([K + ATTRS.index(attr)] if attr else None)


def easy_scores(r, chans, head="map"):
    if head == "img" or r == "R0img":
        return P[r]["easy_pi"][:, :, chans].astype(np.float32).max((1, 2))
    return P[r]["easy_pm"][:, :, chans].astype(np.float32).max((1, 3, 4)).max(1)


IMGS = {r: (P[r]["easy_pi"].astype(np.float32).max(1) if r == "R0img" else P[r]["easy_pm"].astype(np.float32).max((1, 3, 4))) for r in REG}   # [248,76]
EASY_FEATS = [f for f in FEATS if (NRD[:, FEATS.index(f)] >= 2).sum() >= 10]


def easy_feature_table():
    """per feature x regime: AUROC + CI, rank among the 71 ID channels, generic control (mean of the 71 ID image scores)"""
    rows, boots = {}, {}
    for f in FEATS:
        fi = FEATS.index(f); pos = NRD[:, fi] >= 2; neg = NRD[:, fi] == 0; use = pos | neg; y = pos[use]
        idc, atc = easy_channels(f)
        S = np.stack([IMGS[r][use][:, idc].max(1) for r in REG])
        a, bs = boot_auc(y, S); boots[f] = (use, bs)
        G = np.stack([np.nanmean(IMGS[r][use][:, :K], 1) if not np.isnan(IMGS[r][:, :K]).all() else np.full(use.sum(), np.nan) for r in REG])
        gen = auc_rows(y, G)
        row = dict(n_pos=int(pos.sum()), n_neg=int(neg.sum()), ids=EE.MAP[f][0], trained_in_v3=f not in SKIP and len(EE.MAP[f][0]) == 1,
                   coarse=f in SKIP or len(EE.MAP[f][0]) > 1, regimes={})
        for k, r in enumerate(REG):
            if np.isnan(a[k]):
                continue
            allc = auc_rows(y, IMGS[r][use][:, :K].T)
            rank = int(1 + (allc > a[k]).sum()) if len(idc) == 1 else None
            row["regimes"][r] = dict(auroc=float(a[k]), ci=ci(bs[:, k]), rank_of_71=rank, generic_auroc=float(gen[k]),
                                     best_id=IDS[int(np.nanargmax(allc))], best_id_auroc=float(np.nanmax(allc)))
        if atc:                                              # ISIC attribute channel (task 2: R1 baseline on EASY)
            Sa = np.stack([IMGS[r][use][:, atc[0]] for r in REG]); aa, bsa = boot_auc(y, Sa)
            row["attr_channel"] = EE.MAP[f][1]
            row["attr"] = {r: dict(auroc=float(aa[k]), ci=ci(bsa[:, k])) for k, r in enumerate(REG) if not np.isnan(aa[k])}
        rows[f] = row
    return rows, boots


def mean_over(feats, boots, rows):
    """mean AUROC over features per regime, CI by the paired bootstrap (feature AUROCs share the image resamples)"""
    out = {}
    for k, r in enumerate(REG):
        if not all(r in rows[f]["regimes"] for f in feats):
            continue
        pt = float(np.mean([rows[f]["regimes"][r]["auroc"] for f in feats]))
        bm = np.mean([boots[f][1][:, k] for f in feats], 0)
        out[r] = dict(mean=pt, ci=ci(bm), n_features=len(feats), _boot=bm)
    return out


def diff(m, a, b):
    if a in m and b in m:
        d = m[a]["_boot"] - m[b]["_boot"]; return [m[a]["mean"] - m[b]["mean"], *ci(d)]
    return None


def easy_spatial(r, f):
    """lesion-referenced pointing game and plausibility on the positive images of feature f (channels: mapped IDs)"""
    if r == "R0img":
        return None
    fi = FEATS.index(f); idc, _ = easy_channels(f); pm = P[r]["easy_pm"]
    if np.isnan(pm[0, 0, idc[0], 0, 0]):
        return None
    hits, ch_les, ch_old, ys, ss, inside, outshare, outarea = [], [], [], [], [], [], [], []
    for n in np.nonzero(NRD[:, fi] >= 2)[0]:
        Pm = pm[n][:, idc].astype(np.float32).max(1)                              # [6,14,14]
        v = int(Pm.max((1, 2)).argmax()); cell = np.unravel_index(Pm[v].argmax(), (14, 14))
        gt = COV[n, fi, v] >= .25; z = ZONE[n, v]
        if gt.any() and z.any():
            hits.append(gt[cell]); ch_les.append((gt & z).sum() / z.sum()); inside.append(z[cell])
            if ANYREG[n, v].any():
                ch_old.append((gt & ANYREG[n, v]).sum() / ANYREG[n, v].sum())
        if z.any() and (~z).any():
            top = Pm[v] >= .5 * Pm[v].max(); outshare.append((top & ~z).sum() / top.sum()); outarea.append((~z).mean())
        for vv in range(6):
            g2 = COV[n, fi, vv] >= .5; z2 = ZONE[n, vv]
            if (g2 & z2).any() and (~g2 & z2).any():
                ys.append(g2[z2]); ss.append(pm[n, vv][idc].astype(np.float32).max(0)[z2])
    if not hits:
        return None
    return dict(pointing=float(np.mean(hits)), chance_lesion=float(np.mean(ch_les)), chance_old_proxy=float(np.mean(ch_old)) if ch_old else None,
                n=len(hits), argmax_in_lesion_zone=float(np.mean(inside)),
                cell_auroc_in_lesion=auc(np.concatenate(ys), np.concatenate(ss)) if ys else None,
                top_region_outside_share=float(np.mean(outshare)) if outshare else None, outside_area_share=float(np.mean(outarea)) if outarea else None)


# ------------------------------------------------------------------------------------------------ ISIC 2018 test blocks (C1, C3, C4)
def isic_eval():
    sp = H.real_split(); out = {}
    _, Ac, Lc = H.real_data(sp["cal"]); _, At, Lt = H.real_data(sp["test"])
    gc, gt = H.pool14(Ac) >= .25, H.pool14(At) >= .25; lt = H.pool14(Lt) > .5; inl = H.zones_inl(lt)
    nimg = len(At) // 5; img_of = np.arange(len(At)) // 5
    for k, a in enumerate(ATTRS):
        y_img = np.zeros(nimg, bool); np.logical_or.at(y_img, img_of, (At[:, k].astype(np.float32).mean((1, 2)) / 255. >= .01))
        has = np.nonzero(gt[:, k].reshape(len(gt), -1).any(1))[0]
        chance = float(np.mean([(gt[t, k] & inl[t]).sum() / max(inl[t].sum(), 1) for t in has]))
        S = []
        for r in REG:
            if "isic_test_pm" not in P[r]:
                S.append(np.full(nimg, np.nan)); continue
            s = P[r]["isic_test_pm"][:, K + k].astype(np.float32).reshape(len(At), -1).max(1)
            S.append(np.maximum.reduceat(s, np.arange(0, len(s), 5)))
        au, bs = boot_auc(y_img, np.stack(S)); res = {}
        for j, r in enumerate(REG):
            if np.isnan(au[j]):
                continue
            pc = P[r]["isic_cal_pm"][:, K + k].astype(np.float32); pt = P[r]["isic_test_pm"][:, K + k].astype(np.float32)
            thr = H.best_thr(gc[:, k].ravel(), pc.ravel()); pb = pt >= thr
            tp, fp, fn = [np.bincount(img_of, weights=v.reshape(len(pb), -1).sum(1), minlength=nimg) for v in (pb & gt[:, k], pb & ~gt[:, k], ~pb & gt[:, k])]
            io = [tp[ix].sum() / max((tp[ix] + fp[ix] + fn[ix]).sum(), 1) for ix in (RNG.integers(0, nimg, nimg) for _ in range(300))]
            point = float(np.mean([gt[t, k].ravel()[pt[t].ravel().argmax()] for t in has]))
            sh = []
            for p_, z in zip(pt, inl):
                if z.sum() >= 5:
                    m = p_ * (p_ >= thr)
                    if m.sum() > 0:
                        sh.append(m[~z].sum() / m.sum())
            res[r] = dict(image_auroc=float(au[j]), ci=ci(bs[:, j]), boot=bs[:, j], iou=float(tp.sum() / max((tp + fp + fn).sum(), 1)), iou_ci=ci(io), thr=float(thr),
                          pointing=point, outside_share=float(np.mean(sh)) if sh else None)
        out[a] = dict(n_pos=int(y_img.sum()), n_neg=int((~y_img).sum()), pointing_chance=chance, regimes=res)
    # per-ID evidence (C5): each ID channel mapped to an attribute, image-level AUROC against that attribute
    per_id = {}
    for k, a in enumerate(ATTRS):
        y_img = np.zeros(nimg, bool); np.logical_or.at(y_img, img_of, (At[:, k].astype(np.float32).mean((1, 2)) / 255. >= .01))
        for c in AMAP[a]:
            for r in REG:
                if "isic_test_pm" in P[r] and not np.isnan(P[r]["isic_test_pm"][0, c, 0, 0]):
                    s = np.maximum.reduceat(P[r]["isic_test_pm"][:, c].astype(np.float32).reshape(len(At), -1).max(1), np.arange(0, len(At), 5))
                    per_id.setdefault(r, {}).setdefault(IDS[c], []).append((f"ISIC:{a}", auc(y_img, s), int(y_img.sum())))
    return out, per_id


# ------------------------------------------------------------------------------------------------ Derm7pt test (C1, C6)
lab = json.load(open("cache/d7_labels.json")); SP = DD.split(); TE = SP["test"]; POOL = np.concatenate([SP["train"], SP["valid"]])
col = {i: TAX[i]["Derm7pt label"] for i in IDS}
VASC = {"DOT": "dotted", "HP": "hairpin", "LIR": "linear irregular", "COM": "comma", "ARB": "arborizing"}
D7MAP = {"network (any)": [i for i in IDS if "pigment_network" in col[i]], "network (atypical)": [i for i in IDS if "pigment_network" in col[i] and "ATP" in col[i] and "TYP" not in col[i]],
         "streaks": [i for i in IDS if col[i].startswith("streaks")], "dots/globules": [i for i in IDS if "dots_and_globules" in col[i]],
         "blue-whitish veil": [i for i in IDS if "blue_whitish_veil" in col[i]], "regression": [i for i in IDS if col[i].startswith("regression_structures")]}
for c_, n_ in VASC.items():
    D7MAP[f"vessels: {n_}"] = [i for i in IDS if re.search(rf"vascular_structures = {c_}\b", col[i])]
D7ATTR = {"network (any)": "pigment_network", "streaks": "streaks", "dots/globules": "globules"}
arr = {k: np.array(v) for k, v in lab.items()}
D7Y = {"network (any)": arr["pigment_network"] != "absent", "network (atypical)": arr["pigment_network"] == "atypical", "streaks": arr["streaks"] != "absent",
       "dots/globules": arr["dots_and_globules"] != "absent", "blue-whitish veil": arr["blue_whitish_veil"] == "present", "regression": arr["regression_structures"] != "absent"}
for c_, n_ in VASC.items():
    D7Y[f"vessels: {n_}"] = arr["vascular_structures"] == n_


def d7_eval():
    out, per_id = {}, {}
    for t, ids_t in D7MAP.items():
        y = D7Y[t][TE]
        if not ids_t or y.sum() < 10:
            continue
        c = [IDS.index(i) for i in ids_t]; S, Sa = [], []
        for r in REG:
            if "d7_pmax" not in P[r]:
                S.append(np.full(len(TE), np.nan)); Sa.append(np.full(len(TE), np.nan)); continue
            pm = P[r]["d7_pmax"][TE][:, :5]
            S.append(pm[:, :, c].max((1, 2)))
            Sa.append(pm[:, :, K + ATTRS.index(D7ATTR[t])].max(1) if t in D7ATTR else np.full(len(TE), np.nan))
            for cc in c:
                if not np.isnan(pm[0, 0, cc]):
                    per_id.setdefault(r, {}).setdefault(IDS[cc], []).append((f"Derm7pt:{t}", auc(y, pm[:, :, cc].max(1)), int(y.sum())))
        a, bs = boot_auc(y, np.stack(S)); aa, bsa = boot_auc(y, np.stack(Sa))
        out[t] = dict(n_pos=int(y.sum()), n_neg=int((~y).sum()), ids=ids_t, attr=D7ATTR.get(t),
                      regimes={r: dict(auroc=float(a[j]), ci=ci(bs[:, j]), boot=bs[:, j]) for j, r in enumerate(REG) if not np.isnan(a[j])},
                      attr_regimes={r: dict(auroc=float(aa[j]), ci=ci(bsa[:, j])) for j, r in enumerate(REG) if not np.isnan(aa[j])})
    return out, per_id


# ------------------------------------------------------------------------------------------------ downstream (C6)
def old_rule(Pr):
    """7-point checklist on glyph channels (external_check.old_rule): needs ID channels; no diagnosis training"""
    ix = lambda l: [IDS.index(i) for i in l]
    BM = {"network_atypical": ix(["DS003", "DS004"]), "veil": ix(["DS049"]), "streaks_irregular": ix(["DS008", "DS009", "DS024"]) + [K + 2],
          "dots_irregular": ix(["DS025", "DS037"]) + [K + 4], "regression_any": ix(["DS051"]), "vessels_dotted": ix(["DS060"]), "vessels_linear_irregular": ix(["DS065"])}
    gm = lambda n: Pr[:, BM[n]].max(1)
    return 2 * gm("network_atypical") + 2 * gm("veil") + gm("streaks_irregular") + gm("dots_irregular") + gm("regression_any") + 2 * np.maximum(gm("vessels_dotted"), gm("vessels_linear_irregular"))


def probe(Xtr, y, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = ~np.isnan(Xtr).any(0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight="balanced")).fit(Xtr[:, ok], y).decision_function(Xte[:, ok])


def downstream():
    meta = DD.meta(); mel = np.array([r["diagnosis"].startswith("melanoma") for r in meta]); nev = np.array(["nevus" in r["diagnosis"] for r in meta])
    regs = [r for r in REG if "d7_mil" in P[r]]; out = {}
    te = TE[mel[TE] | nev[TE]]; y = mel[te]; tr = POOL[mel[POOL] | nev[POOL]]
    try:
        ext = dict(np.load("cache/mission_ext.npz"))
    except FileNotFoundError:
        ext = None; print("no external features (mission_ext.py)")
    for kind in ("rule", "probe"):
        S = {}
        for r in regs:
            F = P[r]["d7_mil"].astype(np.float32)
            if kind == "rule":
                if np.isnan(F[:, :K]).all():
                    continue
                S[r] = {"Derm7pt test": old_rule(F[te])}
                if ext is not None and r in ext:
                    S[r]["external pooled"] = old_rule(ext[r]); S[r] |= {f"ext {s}": old_rule(ext[r][ext["source"] == s]) for s in np.unique(ext["source"])}
            else:
                S[r] = {"Derm7pt test": probe(F[tr], mel[tr], F[te])}
                if ext is not None and r in ext:
                    sc = probe(F[tr], mel[tr], ext[r]); S[r]["external pooled"] = sc
                    S[r] |= {f"ext {s}": sc[ext["source"] == s] for s in np.unique(ext["source"])}
        ys = {"Derm7pt test": y}
        if ext is not None:
            ys["external pooled"] = ext["group"] == "MEL"; ys |= {f"ext {s}": ext["group"][ext["source"] == s] == "MEL" for s in np.unique(ext["source"])}
        for ds, yy in ys.items():
            rr = [r for r in S if ds in S[r]]
            if not rr:
                continue
            a, bs = boot_auc(yy, np.stack([S[r][ds] for r in rr]))
            out.setdefault(kind, {})[ds] = dict(n_mel=int(yy.sum()), n_nev=int((~yy).sum()), regimes={r: dict(auroc=float(a[j]), ci=ci(bs[:, j]), boot=bs[:, j]) for j, r in enumerate(rr)})
    return out


# ------------------------------------------------------------------------------------------------ coverage (C5)
def coverage(rows, per_isic, per_d7):
    out = {}
    for r in REG:
        if r == "R0img" or r.startswith("R4") or r.endswith("_s1"):
            continue
        ev = {}
        for f, row in rows.items():
            if len(row["ids"]) == 1 and row["n_pos"] >= 10 and r in row["regimes"]:
                x = row["regimes"][r]; ev.setdefault(row["ids"][0], []).append((f"EASY:{f}", x["auroc"], row["n_pos"], x["ci"][0]))
        for src in (per_d7.get(r, {}), per_isic.get(r, {})):
            for i, l in src.items():
                ev.setdefault(i, []).extend((s, a, n, None) for s, a, n in l if n >= 10 and not np.isnan(a))
        def count(pred, srcs):
            return sorted(i for i, l in ev.items() if any(pred(e) for e in l if e[0].split(":")[0] in srcs))
        res = {}
        for name, srcs in (("EASY", {"EASY"}), ("EASY+Derm7pt", {"EASY", "Derm7pt"}), ("EASY+Derm7pt+ISIC", {"EASY", "Derm7pt", "ISIC"})):
            evaluable = count(lambda e: True, srcs); good = count(lambda e: e[1] >= .80, srcs)
            res[name] = dict(n_evaluable=len(evaluable), n_auroc_ge_080=len(good), ids_ge_080=good, evaluable=evaluable)
        res["EASY"]["n_lower_ci_gt_050"] = len(count(lambda e: e[3] is not None and e[3] > .5, {"EASY"}))
        res["per_id"] = {i: [dict(source=s, auroc=a, n_pos=n) for s, a, n, _ in l] for i, l in ev.items()}
        out[r] = res
    return out


# ------------------------------------------------------------------------------------------------ main
if __name__ == "__main__":
    rows, boots = easy_feature_table()
    trained = [f for f in EASY_FEATS if rows[f]["trained_in_v3"]]; coarse = [f for f in EASY_FEATS if rows[f]["coarse"]]
    m_tr, m_co = mean_over(trained, boots, rows), mean_over(coarse, boots, rows)
    spatial = {f: {r: easy_spatial(r, f) for r in REG if not r.startswith("R4") and not r.endswith("_s1")} for f in EASY_FEATS}
    isic, per_isic = isic_eval(); d7, per_d7 = d7_eval(); ds = downstream(); cov = coverage(rows, per_isic, per_d7)
    attr_feats = [f for f in EASY_FEATS if "attr" in rows[f]]

    def agg_spatial(feats, key):
        return {r: float(np.nanmean([spatial[f][r][key] for f in feats if spatial[f].get(r) and spatial[f][r][key] is not None])) for r in spatial[feats[0]] if any(spatial[f].get(r) for f in feats)}

    summary = {
        "C1_easy_trained25": {r: {k: v for k, v in x.items() if k != "_boot"} for r, x in m_tr.items()},
        "C1_easy_coarse4": {r: {k: v for k, v in x.items() if k != "_boot"} for r, x in m_co.items()},
        "C1_easy_differences": {f"{a}-{b}": diff(m_tr, a, b) for a, b in (("R3", "R0"), ("R3", "R2"), ("R2", "R0"), ("R3", "R5"), ("R5", "R0"), ("R2ctrl", "R0pipe"), ("R0pipe", "R0"),
                                                                           ("R0pipe_s1", "R0pipe"), ("R3_s1", "R3"), ("R4_25", "R2ctrl"), ("R4_50", "R4_25"), ("R4_75", "R4_50"), ("R3", "R4_75"))},
        "C1_easy_attr_channels": {r: float(np.mean([rows[f]["attr"][r]["auroc"] for f in attr_feats])) for r in REG if all(r in rows[f].get("attr", {}) for f in attr_feats)},
        "C2_easy_rank": {r: dict(median_rank=float(np.median([rows[f]["regimes"][r]["rank_of_71"] for f in trained])),
                                 share_rank1=float(np.mean([rows[f]["regimes"][r]["rank_of_71"] == 1 for f in trained])),
                                 share_top3=float(np.mean([rows[f]["regimes"][r]["rank_of_71"] <= 3 for f in trained])),
                                 mean_generic=float(np.mean([rows[f]["regimes"][r]["generic_auroc"] for f in trained])),
                                 mean_auroc_minus_generic=float(np.mean([rows[f]["regimes"][r]["auroc"] - rows[f]["regimes"][r]["generic_auroc"] for f in trained])))
                         for r in REG if all(r in rows[f]["regimes"] and rows[f]["regimes"][r]["rank_of_71"] for f in trained)},
        "C3_easy_pointing": agg_spatial(EASY_FEATS, "pointing"), "C3_easy_chance_lesion": agg_spatial(EASY_FEATS, "chance_lesion"),
        "C3_easy_chance_old_proxy": agg_spatial(EASY_FEATS, "chance_old_proxy"), "C3_easy_cell_auroc_in_lesion": agg_spatial(EASY_FEATS, "cell_auroc_in_lesion"),
        "C4_easy_top_region_outside_share": agg_spatial(EASY_FEATS, "top_region_outside_share"), "C4_easy_outside_area_share": agg_spatial(EASY_FEATS, "outside_area_share"),
        "C4_easy_argmax_in_lesion_zone": agg_spatial(EASY_FEATS, "argmax_in_lesion_zone"),
        "C1_isic_mean_image_auroc": {r: dict(mean=float(np.mean([isic[a]["regimes"][r]["image_auroc"] for a in ATTRS])), ci=ci(np.mean([isic[a]["regimes"][r]["boot"] for a in ATTRS], 0)))
                                     for r in REG if all(r in isic[a]["regimes"] for a in ATTRS)},
        "C3_isic_mean_iou": {r: float(np.mean([isic[a]["regimes"][r]["iou"] for a in ATTRS])) for r in REG if all(r in isic[a]["regimes"] for a in ATTRS)},
        "C3_isic_mean_pointing": {r: float(np.mean([isic[a]["regimes"][r]["pointing"] for a in ATTRS])) for r in REG if all(r in isic[a]["regimes"] for a in ATTRS)},
        "C3_isic_mean_pointing_chance": float(np.mean([isic[a]["pointing_chance"] for a in ATTRS])),
        "C4_isic_mean_outside_share": {r: float(np.mean([isic[a]["regimes"][r]["outside_share"] for a in ATTRS])) for r in REG if all(r in isic[a]["regimes"] for a in ATTRS)},
    }
    d7t = list(d7)
    summary["C1_d7_mean_auroc"] = {}
    for r in REG:
        if all(r in d7[t]["regimes"] for t in d7t):
            bm = np.mean([d7[t]["regimes"][r]["boot"] for t in d7t], 0)
            summary["C1_d7_mean_auroc"][r] = dict(mean=float(np.mean([d7[t]["regimes"][r]["auroc"] for t in d7t])), ci=ci(bm), n_criteria=len(d7t))
    easy_ids = {EE.MAP[f][0][0] for f in EE.MAP if f not in SKIP and len(EE.MAP[f][0]) == 1}      # ID channels with EASY supervision (R3/R5)
    for name, sel in (("with_easy_supervised_id", lambda t: any(i in easy_ids for i in d7[t]["ids"])), ("without_easy_supervised_id", lambda t: not any(i in easy_ids for i in d7[t]["ids"]))):
        tt = [t for t in d7t if sel(t)]; summary[f"C1_d7_{name}"] = {"criteria": tt}
        for r in REG:
            if all(r in d7[t]["regimes"] for t in tt):
                bm = np.mean([d7[t]["regimes"][r]["boot"] for t in tt], 0)
                summary[f"C1_d7_{name}"][r] = dict(mean=float(np.mean([d7[t]["regimes"][r]["auroc"] for t in tt])), ci=ci(bm))
    summary["C5_coverage"] = {r: {k: v for k, v in c.items() if k != "per_id"} for r, c in cov.items()}
    for kind in ds:
        for dset in ds[kind]:
            reg = ds[kind][dset]["regimes"]
            for a_, b_ in (("R3", "R0"), ("R3", "R5"), ("R3", "R2")):
                if a_ in reg and b_ in reg:
                    d_ = reg[a_]["boot"] - reg[b_]["boot"]; ds[kind][dset].setdefault("diff", {})[f"{a_}-{b_}"] = [reg[a_]["auroc"] - reg[b_]["auroc"], *ci(d_)]
    strip = lambda o: {k: strip(v) for k, v in o.items() if k != "boot"} if isinstance(o, dict) else o
    res = dict(protocol=__doc__, regimes=REG, summary=summary, easy_features=rows, easy_spatial=spatial, isic=strip(isic), derm7pt=strip(d7), downstream=strip(ds),
               coverage_per_id={r: c["per_id"] for r, c in cov.items()},
               dose={"x_images_per_fold": [0, 46, 93, 140, 186], "regimes": ["R2ctrl", "R4_25", "R4_50", "R4_75", "R3"]})
    json.dump(res, open("models/mission_compare.json", "w"), indent=1, ensure_ascii=False, default=float)

    # ---- console tables
    print("\nC1 EASY mean AUROC, 25 trained features | 4 coarse features")
    for r in REG:
        if r in m_tr:
            print(f"  {r:10s} {m_tr[r]['mean']:.3f} [{m_tr[r]['ci'][0]:.3f},{m_tr[r]['ci'][1]:.3f}]   {m_co[r]['mean']:.3f} [{m_co[r]['ci'][0]:.3f},{m_co[r]['ci'][1]:.3f}]" if r in m_co else "")
    for k, v in summary["C1_easy_differences"].items():
        if v:
            print(f"  {k:18s} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")
    for k in ("C1_easy_attr_channels", "C2_easy_rank", "C3_easy_pointing", "C3_easy_chance_lesion", "C3_easy_chance_old_proxy", "C3_easy_cell_auroc_in_lesion",
              "C4_easy_top_region_outside_share", "C4_easy_outside_area_share", "C1_isic_mean_image_auroc", "C3_isic_mean_iou", "C3_isic_mean_pointing",
              "C3_isic_mean_pointing_chance", "C4_isic_mean_outside_share", "C1_d7_mean_auroc", "C1_d7_with_easy_supervised_id", "C1_d7_without_easy_supervised_id"):
        print(k, json.dumps(summary[k], default=lambda x: round(float(x), 3)))
    for r, c in summary["C5_coverage"].items():
        print("C5", r, {k: (v["n_auroc_ge_080"], v["n_evaluable"]) for k, v in c.items()}, "EASY lowerCI>0.5:", c["EASY"]["n_lower_ci_gt_050"])
    for kind in ds:
        for dset, x in ds[kind].items():
            print("C6", kind, dset, {r: round(v["auroc"], 3) for r, v in x["regimes"].items()}, {k: [round(t, 3) for t in v] for k, v in x.get("diff", {}).items()})
    print(f"\n{'feature':52s} {'n+':>3s} " + " ".join(f"{r:>13s}" for r in ("R0", "R2", "R3", "R5")))
    for f in EASY_FEATS:
        row = rows[f]; cells = []
        for r in ("R0", "R2", "R3", "R5"):
            x = row["regimes"].get(r)
            cells.append(f"{x['auroc']:.2f} r{x['rank_of_71'] or '-'} g{x['generic_auroc']:.2f}" if x else "-")
        print(f"{f[:52]:52s} {row['n_pos']:3d} " + " ".join(f"{c:>13s}" for c in cells))

    # ---- dose curve figure (plain SVG; the venv has no matplotlib)
    xs = res["dose"]["x_images_per_fold"]; rr = res["dose"]["regimes"]
    if all(r in m_tr for r in rr):
        W, Hh, L, R_, T, Bm = 640, 400, 64, 20, 36, 56; y0, y1 = .40, 1.0
        X = lambda x: L + (W - L - R_) * x / 186; Y = lambda y: T + (Hh - T - Bm) * (y1 - y) / (y1 - y0)
        el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}" font-family="sans-serif" font-size="11">',
              f'<rect width="{W}" height="{Hh}" fill="white"/>', f'<text x="{W / 2}" y="20" text-anchor="middle" font-size="13">R4 dose curve: synthetic + ISIC + n EASY images (OOF, 25 trained features)</text>']
        for t in np.arange(.4, 1.01, .1):
            el.append(f'<line x1="{L}" x2="{W - R_}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e5e5e5"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.1f}</text>')
        for x in xs:
            el.append(f'<text x="{X(x):.1f}" y="{Hh - Bm + 16}" text-anchor="middle">{x}</text>')
        el.append(f'<text x="{(L + W - R_) / 2}" y="{Hh - 18}" text-anchor="middle">EASY expert-masked training images per fold (of 186)</text>')
        el.append(f'<text transform="translate(16,{(T + Hh - Bm) / 2}) rotate(-90)" text-anchor="middle">mean image AUROC on held-out EASY group</text>')
        for m_, lab_, c_, k_ in ((m_tr, "25 trained features", "#2a6fdb", 0), (m_co, "4 coarse-mapped features (never trained)", "#80868b", 1)):
            pts = [(X(x), Y(m_[r]["mean"])) for x, r in zip(xs, rr)]
            band = [(X(x), Y(m_[r]["ci"][1])) for x, r in zip(xs, rr)] + [(X(x), Y(m_[r]["ci"][0])) for x, r in zip(xs, rr)][::-1]
            el.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in band)}" fill="{c_}" fill-opacity=".15"/>')
            el.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{c_}" stroke-width="2"/>')
            el += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.5" fill="{c_}"/>' for a, b in pts]
            el.append(f'<line x1="{L + 12}" x2="{L + 32}" y1="{T + 14 + 16 * k_}" y2="{T + 14 + 16 * k_}" stroke="{c_}" stroke-width="2"/><text x="{L + 38}" y="{T + 18 + 16 * k_}">{lab_}</text>')
        for k_, (r, x, c_, lab_) in enumerate((("R0", 0, "#188038", "R0 synthetic only (dense_v1)"), ("R5", 186, "#d93025", "R5 ISIC + EASY, no synthetic"), ("R3_s1", 186, "#2a6fdb", "R3 second seed"))):
            if r in m_tr:
                a, (lo, hi) = m_tr[r]["mean"], m_tr[r]["ci"]; dx = {"R5": 8, "R3_s1": -8}.get(r, 6)
                el.append(f'<line x1="{X(x) + dx:.1f}" x2="{X(x) + dx:.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" stroke="{c_}"/>')
                el.append(f'<rect x="{X(x) + dx - 4:.1f}" y="{Y(a) - 4:.1f}" width="8" height="8" fill="{"white" if r == "R3_s1" else c_}" stroke="{c_}" stroke-width="1.5"/>')
                el.append(f'<rect x="{L + 18}" y="{T + 42 + 16 * k_ - 4}" width="8" height="8" fill="{"white" if r == "R3_s1" else c_}" stroke="{c_}" stroke-width="1.5"/><text x="{L + 38}" y="{T + 46 + 16 * k_}">{lab_}</text>')
        el.append(f'<text x="{W - R_}" y="{Hh - 4}" text-anchor="end" font-size="9" fill="#666">bands/bars: 95 % bootstrap CI over images; 0 = no-EASY control (R2ctrl), 186 = V3 (R3)</text></svg>')
        open("out/mission_dose_curve.svg", "w").write("\n".join(el)); print("wrote out/mission_dose_curve.svg")
