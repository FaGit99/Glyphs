"""End-to-end analysis of one dermoscopy image -> self-contained HTML report ("doctor view" prototype).

  python analyze.py <image> [out.html]

Pipeline: resize (short side 450 px = native pitch of the training data) -> lesion segmentation (whole image, 384 px)
-> overlapping 224 px tiles (stride 112) through the hybrid DenseViT -> stitched probability maps -> Platt calibration and
calibrated thresholds for the 5 validated ISIC attributes, raw probability >= 0.5 for the remaining synthetic ID channels
-> per structure: area, share inside lesion core / border (+-16 px) / outside, plausibility flags.
"""
import base64, csv, io, json, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F
import dense as D, hybrid as H, lesion_seg as LS, structures as S, diagnosis_map as DM, d7_data as DD, mil, direct_model as DMOD
try:
    EASYV = json.load(open("models/easy_validation.json"))   # ID -> Out-of-fold-AUROC auf den EASY-Expertenmasken (nur Bildebene)
except OSError:
    EASYV = {}

BAND = 16                      # px at 450-scale = one 14x14 cell, same as in the evaluation
TILE, STRIDE = 224, 112
tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
ATTR_DE = {"pigment_network": "Pigmentnetzwerk", "negative_network": "Negatives Netzwerk", "streaks": "Streaks / radiäre Streifen",
           "milia_like_cyst": "Milia-ähnliche Zysten", "globules": "Globuli / Dots"}
PALETTE = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15), (155, 89, 182), (230, 126, 34), (26, 188, 156),
           (236, 112, 99), (93, 173, 226), (88, 214, 141), (247, 220, 111), (175, 122, 197)]
_models = {}
DMAP = DM.load()



def models():
    if not _models:
        m, ck = H.load_model("models/hybrid.pt"); _models["det"], _models["ids"] = m, [str(i) for i in ck["ids"]]
        s = LS.SegViT(pretrained=False).to(D.dev); s.load_state_dict(torch.load("models/lesion_seg.pt")["state"]); s.eval(); _models["seg"] = s
        _models["cal"] = json.load(open("models/hybrid_calibration.json"))
        mm, _ = H.load_model("models/mel_final.pt"); _models["mel"] = mm
        _models["mel_cal"] = json.load(open("models/mel_calibration.json"))
        _models["direct"], _models["direct_size"] = DMOD.load("base_448_v17")   # read-only from the project folder
        _models["twopath"] = json.load(open("models/twopath.json"))
        _models["mel_ev"] = {}
        for k_, f_ in (("derm7pt", "models/melanoma_eval.json"), ("ham", "models/ham_check.json"), ("ext", "models/external_check.json")):
            try:
                _models["mel_ev"][k_] = json.load(open(f_))
            except Exception:
                pass
        try:
            _models["d7"] = json.load(open("models/transfer_dense_d7.json"))["hybrid"]
        except Exception:
            _models["d7"] = {}
    return _models


def lesion_mask(img450, seg):
    h, w = img450.shape[:2]
    p = LS.predict(seg, cv2.resize(img450, (LS.S, LS.S), interpolation=cv2.INTER_AREA)[None])[0]
    m = (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) > .5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    if cs:
        big = max(cv2.contourArea(c) for c in cs)
        cv2.drawContours(out, [c for c in cs if cv2.contourArea(c) >= .2 * big], -1, 1, -1)   # fills holes, drops specks
    return out, p


def positions(n):
    p = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    return p if p[-1] == n - TILE else p + [n - TILE]


@torch.no_grad()
def stitched_maps(img450, det):
    """probability maps for all channels at 450-scale: Gaussian-weighted average over overlapping tiles"""
    h, w = img450.shape[:2]
    pos = [(x, y) for y in positions(h) for x in positions(w)]
    tiles = np.stack([img450[y:y + TILE, x:x + TILE] for x, y in pos])
    _, pm = D.predict(det, tiles)                                              # [T,K,14,14]
    up = F.interpolate(torch.as_tensor(pm), size=(TILE, TILE), mode="bilinear", align_corners=False).numpy()
    g = np.exp(-(((np.arange(TILE) - TILE / 2 + .5) / (TILE / 3)) ** 2) / 2); win = (g[:, None] * g[None, :]).astype(np.float32) + 1e-3
    acc, wsum = np.zeros((pm.shape[1], h, w), np.float32), np.zeros((h, w), np.float32)
    for (x, y), u in zip(pos, up):
        acc[:, y:y + TILE, x:x + TILE] += u * win; wsum[y:y + TILE, x:x + TILE] += win
    return acc / wsum, len(pos), pos


def zone_masks(les):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1, 2 * BAND + 1))
    core, dil = cv2.erode(les, k) > 0, cv2.dilate(les, k) > 0
    return core, dil & ~core, ~dil


def png_b64(rgba):
    ok, b = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)); return base64.b64encode(b).decode()


def overlay(region, rgb):
    o = np.zeros(region.shape + (4,), np.uint8)
    o[region] = (*rgb, 85)
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, (*rgb, 255), 2)
    return o


SITE_DE = {"facial": "Gesicht", "volar": "Handfläche/Fußsohle", "nonfacial": "Körperstamm/Extremitäten (nicht Gesicht, nicht akral)"}
WEAK = .30                      # calibrated max probability below which a validated region is only a weak hint


D7_DE = {"network_any": "Pigmentnetzwerk (beliebig)", "network_atypical": "Atypisches Netzwerk", "streaks_any": "Streaks (beliebig)", "streaks_irregular": "Irreguläre Streaks", "dots_any": "Dots/Globuli (beliebig)",
         "dots_irregular": "Irreguläre Dots/Globuli", "veil": "Blau-weißer Schleier", "regression_any": "Regressionsstrukturen", "pigmentation_irregular": "Irreguläre Pigmentierung",
         "vessels_dotted": "Gefäße punktförmig", "vessels_hairpin": "Gefäße Hairpin", "vessels_linear_irregular": "Gefäße linear irregulär", "vessels_comma": "Gefäße Komma", "vessels_arborizing": "Gefäße arborisierend"}
CRIT = [("network_atypical", "Atypisches Pigmentnetzwerk", 2, ["network_atypical"]),
        ("veil", "Blau-weißer Schleier", 2, ["veil"]),
        ("vessels", "Atypische Gefäße (punktförmig / linear irregulär)", 2, ["vessels_dotted", "vessels_linear_irregular"]),
        ("streaks_irregular", "Irreguläre Streaks", 1, ["streaks_irregular"]),
        ("pigmentation_irregular", "Irreguläre Pigmentierung", 1, ["pigmentation_irregular"]),
        ("dots_irregular", "Irreguläre Punkte / Globuli", 1, ["dots_irregular"]),
        ("regression_any", "Regressionsstrukturen", 1, ["regression_any"])]
CRIT_RGB = [(255, 140, 0), (0, 160, 255), (255, 0, 140), (140, 255, 0), (160, 100, 255), (0, 220, 180), (255, 220, 0)]


def checklist(img, les, pos, M, render=True, keep=False):
    """7-point checklist from the Derm7pt-fine-tuned model: image-level probabilities (mean of the top-3 lesion tiles per channel),
    Platt-calibrated on out-of-fold predictions. Image-level only: the contributing tiles are shown, they are not a validated localisation."""
    h, w = les.shape
    cand = sorted([(float(les[y:y + TILE, x:x + TILE].mean()), x, y) for x, y in pos if les[y:y + TILE, x:x + TILE].mean() >= .15], reverse=True)[:DD.TMAX]
    if not cand:
        cand = [(0., (w - TILE) // 2, (h - TILE) // 2)]
    tiles = np.stack([img[y:y + TILE, x:x + TILE] for _, x, y in cand])
    pi, _ = D.predict(M["mel"], tiles); K0 = pi.shape[1] - len(DD.TNAMES)
    cal, order = {}, {}
    for c, nm in enumerate(DD.TNAMES):
        pt = pi[:, K0 + c]; order[nm] = np.argsort(-pt)[:mil.TOPK]
        agg = float(np.sort(pt)[::-1][:mil.TOPK].mean())
        a, b = M["mel_cal"][nm]["platt"]; lg = np.log(np.clip(agg, 1e-4, 1 - 1e-4) / (1 - np.clip(agg, 1e-4, 1 - 1e-4)))
        cal[nm] = float(1 / (1 + np.exp(-(a * lg + b))))
    rows, exp_score, pts = [], 0.0, 0
    for i, (key, label, wgt, chans) in enumerate(CRIT):
        p = max(cal[c] for c in chans); best = max(chans, key=lambda c: cal[c])
        rel = all(M["mel_cal"][c]["reliable"] for c in chans)
        ov = np.zeros((h, w, 4), np.uint8)
        if render and p >= .3:
            for t in order[best]:
                _, x, y = cand[int(t)]; cv2.rectangle(ov, (x + 2, y + 2), (x + TILE - 2, y + TILE - 2), (*CRIT_RGB[i], 255), 2)
        rows.append(dict(id="c_" + key, key=key, label=label, weight=wgt, prob=p, contrib=wgt * p, reliable=rel, auc=[M["mel_cal"][c]["auc_test"] for c in chans],
                         n_pos=min(M["mel_cal"][c]["n_pos"] for c in chans), rgb="rgb(%d,%d,%d)" % CRIT_RGB[i], png=png_b64(ov) if render else ""))
        exp_score += wgt * p; pts += wgt * (p >= .5)
    ev = M["mel_ev"]; info = {}
    try:
        info["derm7pt"] = ev["derm7pt"]["melanoma"]["R new glyph probabilities"]; info["ham"] = ev["ham"]["R 7-point rule, new glyph probabilities (no diagnosis training)"]
    except Exception:
        pass
    try:   # external check on 4 independent sources (models/external_check.json): melanoma vs nevus
        v = list(ev["ext"]["T1  melanoma vs nevus"].values())
        info["ext"] = dict(n_sets=len(v), auc=[min(x["R"] for x in v), max(x["R"] for x in v)],
                           sens3=[min(x["sens3"] for x in v), max(x["sens3"] for x in v)], spec3=[min(x["spec3"] for x in v), max(x["spec3"] for x in v)])
    except Exception:
        pass
    out = dict(rows=rows, expected=float(exp_score), points=int(pts), n_tiles=len(cand), info=info, extra={k: cal[k] for k in ("network_any", "streaks_any", "dots_any")})
    if keep:   # raw material for external evaluation (not JSON-serialisable; never passed to html())
        out.update(tiles=tiles, agg_raw=np.sort(pi, 0)[::-1][:mil.TOPK].mean(0), cal14=np.array([cal[n] for n in DD.TNAMES]))
    return out


@torch.no_grad()
def direct_logit(raw, M):
    """project's direct classifier on the ORIGINAL image: squash to 448x448 (INTER_LINEAR), CLIP normalisation, bf16"""
    x = torch.from_numpy(cv2.resize(raw, (M["direct_size"], M["direct_size"]), interpolation=cv2.INTER_LINEAR)).permute(2, 0, 1)[None].float().cuda() / 255.
    x = (x - DMOD.MEAN.cuda()) / DMOD.STD.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return float(M["direct"](x).float().item())


def twopath(raw, ck, M):
    """Two independent paths + disagreement flag (parameters fitted on the v17 validation split, see twopath_fit.py)."""
    P = M["twopath"]; g = P["glyph"]
    logit = direct_logit(raw, M)
    cols = np.array(g.get("cols", list(range(90))))                      # variant B: the 19 validated channels
    agg = np.asarray(ck["agg_raw"])[cols]
    z = (agg - np.array(g["mean"])) / np.array(g["scale"]); gscore = float(z @ np.array(g["coef"]) + g["intercept"])
    pct = float(np.interp(gscore, g["benign_quantiles"], np.linspace(0, 100, 101)))
    d_high = logit >= P["direct"]["thr_logit"]; g_high = gscore >= g["flag_thr"]
    # which channels drive the glyph score (probe contribution = standardised value x weight), named for the reader
    all_names = M["ids"] + ["ISIC:" + a for a in H.ATTRS] + ["D7:" + n for n in DD.TNAMES]; names = [all_names[k] for k in cols]
    contrib = z * np.array(g["coef"]); drivers = []
    for k in np.argsort(-contrib)[:12]:
        if contrib[k] < .1:
            break
        nm, pr = names[k], float(agg[k])
        if nm.startswith("D7:"):
            key = nm[3:]; cal = float(ck["cal14"][DD.TNAMES.index(key)])
            if cal < .5:
                continue          # only channels that are actually detected explain the score
            drivers.append(dict(name=D7_DE[key], detail=f"{cal * 100:.0f} % kalibriert, Bildebene, an Derm7pt geprüft", contrib=float(contrib[k])))
        else:
            if pr < .5:
                continue
            level = "hoch" if pr >= .85 else "mittel" if pr >= .65 else "niedrig"
            if nm.startswith("ISIC:"):
                drivers.append(dict(name=ATTR_DE[nm[5:]], detail=f"Bildwert {level}, Attribut mit echten Masken validiert", contrib=float(contrib[k])))
            else:
                t = tax[nm]; drivers.append(dict(name=t["Metaphoric term"] if not t["Metaphoric term"].startswith("(") else t["Descriptive term (IDS)"], detail=f"Bildwert {level}, nicht validiert", contrib=float(contrib[k])))
    state = "both_high" if d_high and g_high else "direct_only" if d_high else "flag" if g_high else "both_low"
    return dict(drivers=drivers, direct_prob=float(1 / (1 + np.exp(-logit))), direct_logit=logit, thr_prob=P["direct"]["thr_prob"], d_high=bool(d_high), g_score=gscore, g_pct=pct, g_high=bool(g_high), state=state,
                evidence=dict(direct=P["direct"], flag=P["flag"], agreement=P["agreement"], n_test=P["n_test"], fusion=P.get("fusion")))


def analyze(path, site=None):
    M = models(); ids = M["ids"]; amap = H.attr_ids(ids); t0 = time.time()
    raw = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    s = 450 / min(raw.shape[:2]); img = cv2.resize(raw, (int(round(raw.shape[1] * s)), int(round(raw.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    les, _ = lesion_mask(img, M["seg"]); core, border, outside = zone_masks(les)
    P, ntiles, pos = stitched_maps(img, M["det"]); K = len(ids)
    check = checklist(img, les, pos, M, keep=True)
    tp = twopath(raw, check, M)
    agg_raw = np.asarray(check["agg_raw"])             # image-level values of all 90 channels (kept for the group card)
    check = {k: v for k, v in check.items() if k not in ("tiles", "agg_raw", "cal14")}
    area_les = max(int(les.sum()), 1)
    warnings = []
    if les.sum() < .02 * h * w:
        warnings.append("Läsionssegmentierung unsicher: sehr kleine oder keine Läsion gefunden - Zonenangaben nicht verwertbar.")
    items, attr_max = [], {}
    # validated attribute channels
    for k, a in enumerate(H.ATTRS):
        p = P[K + k]; c = M["cal"][a]; reg = p >= c["thr"]
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); cal = 1 / (1 + np.exp(-(c["platt"][0] * lg + c["platt"][1])))
        attr_max[a] = float(cal.max())
        if reg.sum() < 40:
            continue
        items.append(dict(key=a, name=ATTR_DE[a], sub=a, status="region", prob=float(cal[reg].max()), p_region=float(cal[reg].mean()), reg=reg, expect="inside" if a != "streaks" else "periphery"))
    # synthetic-only ID channels (not validated on real masks)
    for j, sid in enumerate(ids):
        sp = S.SPEC[sid]
        if site and sp["site"] not in (None, site):
            continue
        reg = P[j] >= .5
        if reg.sum() < 40:
            continue
        d7 = None
        for nm, lst in {"network (any)": ["DS001", "DS002", "DS003", "DS004", "DS022"], "streaks": ["DS008"], "dots/globules": ["DS025", "DS037"], "blue-whitish veil": ["DS049"],
                        "vessels: dotted": ["DS060"], "vessels: hairpin": ["DS064"], "vessels: linear irregular": ["DS065"], "vessels: comma": ["DS067"], "vessels: arborizing": ["DS072"]}.items():
            if sid in lst and nm in M["d7"]:
                d7 = M["d7"][nm][0]
        items.append(dict(key=sid, name=tax[sid]["Metaphoric term"] if not tax[sid]["Metaphoric term"].startswith("(") else tax[sid]["Descriptive term (IDS)"], sub=tax[sid]["Descriptive term (IDS)"],
                          status="image" if (d7 or EASYV.get(sid, {}).get("auroc_oof", 0) >= .8) else "none", d7=d7, easy=EASYV.get(sid), prob=float(P[j][reg].max()), p_region=float(P[j][reg].mean()), reg=reg, expect=None))
    for it in items:
        r = it["reg"]; n = max(int(r.sum()), 1)
        it["area_px"] = int(r.sum()); it["lesion_cover"] = float((r & (les > 0)).sum() / area_les)
        it["zones"] = [float((r & core).sum() / n), float((r & border).sum() / n), float((r & outside).sum() / n)]
        flags = []
        sp = S.SPEC.get(it["key"])
        if sp and sp["site"] and not site:
            flags.append(f"Nur plausibel bei Lokalisation: {SITE_DE[sp['site']]} (Standort nicht angegeben).")
        if sp and sp["polarized"]:
            flags.append("Nur in polarisierter Dermoskopie beurteilbar.")
        it["weak"] = it["status"] == "region" and it["prob"] < WEAK
        if it["zones"][2] > .25:
            flags.append(f"{it['zones'][2]:.0%} der erkannten Fläche liegen außerhalb von Läsion und Randzone - Erkennung bitte kritisch prüfen.")
        if it["expect"] == "periphery" and it["zones"][0] > .7:
            flags.append("Streaks erwartet am Läsionsrand, Erkennung liegt überwiegend im Läsionskern.")
        it["flags"] = flags
    # --- diagnosis association (information only, from the IDS consensus in the taxonomy) ---
    for it in items:
        src_ids = [ids[j] for j in amap[it["key"]]] if it["key"] in amap else [it["key"]]
        entries = {n: gs for sid in src_ids for n, gs in DMAP[sid]["entries"]}
        it["diag"] = sorted(entries)
        it["groups"] = sorted({g for gs in entries.values() for g in gs})
        it["src_ids"] = src_ids
    det_ids = {sid for it in items if not it.get("weak") for sid in it["src_ids"]}
    wts = DM.weights(DMAP)
    # taxonomy IDs with a high IMAGE-level value but no region above the threshold: still named (labelled), so the group card is complete
    img_only = {sid: float(agg_raw[j]) for j, sid in enumerate(ids)
                if agg_raw[j] >= .5 and sid not in det_ids and sid in wts and not (site and S.SPEC[sid]["site"] not in (None, site))}
    det_ids = det_ids | set(img_only)
    group_rows = []
    for g, gname in DM.GROUPS.items():
        det = [it["name"] for it in items if g in it["groups"] and not it.get("weak")]
        nm0 = lambda sid: tax[sid]["Metaphoric term"] if not tax[sid]["Metaphoric term"].startswith("(") else tax[sid]["Descriptive term (IDS)"]
        det = det + [f"{nm0(sid)} (nur Bildebene, ohne Region, nicht validiert, Bildwert {'hoch' if p >= .85 else 'mittel'})" for sid, p in sorted(img_only.items(), key=lambda kv: -kv[1]) if g in wts[sid]]
        typical = sorted((sid for sid in wts if g in wts[sid] and sid not in det_ids and sid in ids), key=lambda sid: (-wts[sid][g], sid))
        nm = lambda sid: tax[sid]["Metaphoric term"] if not tax[sid]["Metaphoric term"].startswith("(") else tax[sid]["Descriptive term (IDS)"]
        if g == "MEL":
            det = det + [f"{r['label']} (7-Punkte-Kriterium, Bildebene, {r['prob'] * 100:.0f} %)" for r in check["rows"] if r["prob"] >= .5]
        weak = [f"{r['label']} ({r['prob'] * 100:.0f} %)" for r in check["rows"] if .3 <= r["prob"] < .5] if g == "MEL" else []
        group_rows.append(dict(key=g, name=gname, weak=weak, detected=det, missing=[nm(x) for x in typical[:6]], n_typical=sum(1 for sid in wts if g in wts[sid])))
    items.sort(key=lambda it: (it["status"] != "region" or it["weak"], -it["prob"]))
    for i, it in enumerate(items):
        it["rgb"] = PALETTE[i % len(PALETTE)]
    return dict(twopath=tp, check=check, group_rows=group_rows, attr_max=attr_max, site=site, img=img, les=les, core=core, border=border, outside=outside, items=items, warnings=warnings, tiles=ntiles, secs=time.time() - t0, name=path.split("/")[-1])


def html(R):
    img, les = R["img"], R["les"]; h, w = img.shape[:2]
    ok, jb = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); img_b64 = base64.b64encode(jb).decode()
    les_ov = np.zeros((h, w, 4), np.uint8)
    cs, _ = cv2.findContours(les, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(les_ov, cs, -1, (255, 255, 255, 255), 2)
    zo = np.zeros((h, w, 4), np.uint8); zo[R["border"]] = (255, 200, 0, 70); zo[R["outside"]] = (0, 0, 0, 35)
    layers = [dict(id="zones", label="Randzone (±16 px)", png=png_b64(zo), on=False, swatch="#ffc800"), dict(id="lesion", label="Läsionskontur", png=png_b64(les_ov), on=True, swatch="#ffffff")]
    rows = []
    for it in R["items"]:
        rows.append(dict(id="s_" + it["key"], key=it["key"], name=it["name"], sub=it["sub"], status=it["status"], d7=it.get("d7"), easy=it.get("easy"), prob=it["prob"], p_region=it["p_region"],
                         cover=it["lesion_cover"], area=it["area_px"], zones=it["zones"], flags=it["flags"], weak=it["weak"], diag=it["diag"], color="rgb(%d,%d,%d)" % it["rgb"], png=png_b64(overlay(it["reg"], it["rgb"]))))
    data = json.dumps(dict(w=w, h=h, name=R["name"], layers=layers, rows=rows, warnings=R["warnings"], secs=R["secs"], tiles=R["tiles"], lesion_frac=float(les.mean()), groups=R["group_rows"], check=R["check"], twopath=R["twopath"]), ensure_ascii=False)
    return TEMPLATE.replace("__DATA__", data).replace("__IMG__", img_b64)


TEMPLATE = r"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Struktur-Analyse</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2430;--mute:#667085;--line:#e3e6ec;--ok:#1a7f4b;--warn:#b25e09;--bad:#b42318;--okbg:#e6f4ec;--warnbg:#fdf0dd;--badbg:#fde8e6;--grey:#eef0f4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12161d;--card:#1a2029;--ink:#e8ecf3;--mute:#9aa4b5;--line:#2a3340;--ok:#5dd39e;--warn:#f0b35a;--bad:#ff8a80;--okbg:#17342a;--warnbg:#3a2c14;--badbg:#3d1d1a;--grey:#242c38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);font-size:13px}
.banner{background:var(--warnbg);color:var(--warn);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:12px 0;font-size:13.5px}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.stage{position:relative;width:100%;aspect-ratio:var(--ar);background:#000;border-radius:8px;overflow:hidden}.stage img{position:absolute;inset:0;width:100%;height:100%}
.stage img.ov{pointer-events:none;image-rendering:auto}
.row{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;padding:9px 0;border-top:1px solid var(--line);align-items:start}.row:first-of-type{border-top:0}
.sw{width:14px;height:14px;border-radius:4px;border:1px solid var(--line);margin-top:3px}.nm{font-weight:600}.d{font-size:12.5px;color:var(--mute)}
.badge{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:99px;margin-left:6px;white-space:nowrap}.b-region{background:var(--okbg);color:var(--ok)}.b-image{background:var(--warnbg);color:var(--warn)}.b-none{background:var(--grey);color:var(--mute)}
.bar{display:flex;height:9px;border-radius:5px;overflow:hidden;margin:4px 0 2px;background:var(--grey)}.bar i{display:block}.c0{background:#3b9c8f}.c1{background:#e0a21b}.c2{background:#c0504d}
.flag{background:var(--badbg);color:var(--bad);border-radius:8px;padding:5px 8px;margin-top:5px;font-size:12.5px}label{cursor:pointer}input[type=range]{width:160px}
h2{font-size:15px;margin:2px 0 8px}.legend{font-size:12px;color:var(--mute);margin-top:6px}.legend span{display:inline-flex;align-items:center;gap:5px;margin-right:12px}.legend b{width:10px;height:10px;border-radius:3px;display:inline-block}
details summary{cursor:pointer;color:var(--mute);font-size:13px;padding:8px 0}
</style></head><body><div class="wrap">
<h1>Struktur-Analyse <span class="sub" id="fname"></span></h1><div class="sub" id="meta"></div>
<div class="banner"><b>Forschungsprototyp, keine Diagnose.</b> Die Markierungen zeigen, wo das Modell dermoskopische Strukturen erkennt. Nur die grün gekennzeichneten Regionen wurden gegen echte Masken geprüft; alle anderen Markierungen stammen aus synthetischem Training und sind nicht validiert.</div>
<div id="warn"></div>
<div class="card" id="tp" style="margin-bottom:16px"></div>
<div class="grid"><div class="card"><div class="stage" id="stage"><img id="base" alt="Dermoskopiebild"></div>
<div style="display:flex;gap:16px;align-items:center;margin-top:10px;flex-wrap:wrap"><label>Deckkraft <input type="range" id="op" min="10" max="100" value="100"></label><span class="sub">Striche und Flächen pro Struktur im Panel rechts ein- und ausblenden</span></div>
<div class="legend"><span><b class="c0"></b>Läsionskern</span><span><b class="c1"></b>Randzone (±16 px)</span><span><b class="c2"></b>außerhalb</span></div></div>
<div><div class="card"><h2>Ebenen</h2><div id="layers"></div></div><div class="card" style="margin-top:12px"><h2>Erkannte Strukturen</h2><div id="main"></div>
<details open><summary id="moreSum"></summary><div id="more"></div></details></div>
<div class="card" style="margin-top:12px"><h2>7-Punkte-Checkliste <span class="badge b-image">experimentell, Bildebene</span></h2><div class="d" id="chkNote" style="margin-bottom:6px"></div><div id="chk"></div><div id="chkSum" style="margin-top:8px"></div><div class="d" id="chkInfo" style="margin-top:6px"></div></div>
<div class="card" style="margin-top:12px"><h2>Zuordnung zu Diagnosegruppen <span class="badge b-image">nur Information, kein Ranking</span></h2>
<div class="d" style="margin-bottom:6px">Die Zuordnung gibt an, bei welchen Diagnosen eine Struktur laut IDS-Konsens (Kittler 2016) vorkommt, nicht wie spezifisch sie ist. An 1011 Derm7pt-Fällen getestet, trennt eine daraus gebildete Rangfolge die Diagnosen nicht besser als der Zufall (AUC 0,46–0,61). Deshalb werden die Gruppen in fester Reihenfolge und ohne Bewertung gezeigt.</div><div id="groups"></div></div></div></div></div>
<script>
const D=__DATA__;const $=s=>document.querySelector(s);
$('#base').src="data:image/jpeg;base64,__IMG__";$('#stage').style.setProperty('--ar',D.w+'/'+D.h);$('#fname').textContent='· '+D.name;
$('#meta').textContent=`Bildgröße (nativer Maßstab) ${D.w}×${D.h} px · ${D.tiles} Kacheln · Läsion ${(D.lesion_frac*100).toFixed(0)} % des Bildes · ${D.secs.toFixed(1)} s`;
D.warnings.forEach(t=>{const e=document.createElement('div');e.className='flag';e.textContent=t;$('#warn').append(e)});
const imgs={};function mk(id,png,on){const i=document.createElement('img');i.className='ov';i.src='data:image/png;base64,'+png;i.style.display=on?'block':'none';$('#stage').append(i);imgs[id]=i}
function toggle(id,on){imgs[id].style.display=on?'block':'none'}
D.layers.forEach(l=>{mk(l.id,l.png,l.on);const r=document.createElement('div');r.className='row';r.innerHTML=`<span class="sw" style="background:${l.swatch}"></span><label><input type="checkbox" ${l.on?'checked':''}> ${l.label}</label>`;r.querySelector('input').onchange=e=>toggle(l.id,e.target.checked);$('#layers').append(r)});
const STAT={region:['Region validiert (ISIC-Masken)','b-region'],image:['nur Bildebene geprüft','b-image'],none:['nicht validiert','b-none']};
const topMore=new Set(D.rows.filter(r=>r.status!=='region'||r.weak).sort((p,q)=>q.prob-p.prob).slice(0,3).filter(r=>r.prob>=.85).map(r=>r.id));
const isOn=r=>(r.status==='region'&&!r.weak)||topMore.has(r.id);
function row(r){mk(r.id,r.png,isOn(r));const e=document.createElement('div');e.className='row';const st=STAT[r.status];const weak=r.weak?'<span class="badge b-image">schwacher Hinweis</span>':'';
const z=r.zones.map((v,i)=>`<i class="c${i}" style="width:${v*100}%"></i>`).join('');
const pr=r.status==='region'?`Wahrscheinlichkeit (kalibriert) max ${(r.prob*100).toFixed(0)} %`:`Modellsicherheit ${r.prob>=.85?'hoch':r.prob>=.65?'mittel':'niedrig'} (unkalibriert, keine Wahrscheinlichkeit)`;
const extra=(r.status==='image'&&r.d7?` · Derm7pt-AUC ${r.d7.toFixed(2)}`:'')+(r.easy?` · EASY-AUROC ${r.easy.auroc_oof.toFixed(2)}`:'');
e.innerHTML=`<span class="sw" style="background:${r.color}"></span><div><label><input type="checkbox" ${isOn(r)?'checked':''}> <span class="nm">${r.name}</span></label><span class="badge ${st[1]}">${st[0]}${extra}</span>${weak}<div class="d">${r.sub}</div>
<div class="d">${pr} · ${(r.cover*100).toFixed(0)} % der Läsionsfläche</div><div class="d">Laut IDS-Konsens vorkommend bei: ${r.diag.length?r.diag.join(', '):'keine spezifische Zuordnung in der Quelle'}</div><div class="bar" title="Kern / Rand / außerhalb">${z}</div>
<div class="d">Kern ${(r.zones[0]*100).toFixed(0)} % · Rand ${(r.zones[1]*100).toFixed(0)} % · außerhalb ${(r.zones[2]*100).toFixed(0)} %</div>${r.flags.map(f=>`<div class="flag">${f}</div>`).join('')}</div>`;
e.querySelector('input').onchange=x=>toggle(r.id,x.target.checked);return e}
const main=D.rows.filter(r=>r.status==='region'&&!r.weak),more=D.rows.filter(r=>r.status!=='region'||r.weak);
if(!main.length)$('#main').innerHTML='<div class="d">Keine validierte Struktur oberhalb des kalibrierten Schwellenwerts erkannt.</div>';
main.forEach(r=>$('#main').append(row(r)));more.forEach(r=>$('#more').append(row(r)));
$('#moreSum').textContent=`Weitere Hinweise: schwache Treffer und synthetisch trainierte Strukturen ohne Validierung an echten Masken (${more.length})`;if(!more.length)$('details').style.display='none';
D.groups.forEach(g=>{if(!g.detected.length&&!g.missing.length)return;const e=document.createElement('div');e.className='row';e.style.gridTemplateColumns='1fr';e.innerHTML=`<div><span class="nm">${g.name}</span><div class="d">${g.detected.length?'Erkannte Strukturen mit Bezug: '+g.detected.join(', '):'Keine erkannte Struktur mit Bezug zu dieser Gruppe'}</div>${g.weak&&g.weak.length?'<div class="d">Schwache Hinweise (30 bis 50 %, 7-Punkte-Kriterien): '+g.weak.join(', ')+'</div>':''}${g.missing.length?`<details><summary>Typische, aber nicht erkannte Strukturen (${g.n_typical} gelistet)</summary><div class="d">${g.missing.join(', ')}${g.n_typical>g.missing.length+g.detected.length?' …':''}</div></details>`:''}</div>`;$('#groups').append(e)});
(function(){const T=D.twopath,E=T.evidence,pc=x=>(x*100).toFixed(0)+' %';
const pos=Math.max(0,Math.min(100,T.direct_prob*100)),thr=T.thr_prob*100,gp=T.g_pct;
const msg={flag:['warn','Abweichung zwischen den Wegen – bitte zweiten Blick','Das Direktmodell liegt unter seiner Schwelle, die Strukturanalyse ist aber auffällig (mindestens 90. Perzentil der benignen Referenz). Im Testsplit ('+E.n_test+' Bilder) waren von '+E.flag.flagged+' solchen Fällen '+pc(E.flag.flagged_malignant)+' maligne (Grundrate bei Direkt-„benigne“: '+pc(E.flag.base_rate)+'); '+pc(1-E.flag.flagged_malignant)+' waren benigne. Der Hinweis erfasste '+E.flag.caught+' der '+E.flag.missed+' übersehenen malignen Läsionen; etwa '+E.flag.review_per_catch.toFixed(0)+' Hinweise pro gefundenem Fall.'],
both_high:['bad','Beide Wege auffällig','Direktmodell über der Schwelle und Strukturanalyse auffällig. Im Testsplit waren '+pc(E.agreement.malignant_rate_both_high)+' dieser Fälle maligne.'],
direct_only:['info','Nur das Direktmodell auffällig','Direktmodell über der Schwelle, die Strukturanalyse findet wenig typische Kriterien. Das ist häufig, weil die Strukturanalyse weniger trennscharf ist; im Testsplit waren '+pc(E.agreement.malignant_rate_direct_high_glyph_low)+' dieser Fälle maligne.'],
both_low:['ok','Beide Wege unauffällig','Kein Hinweis aus beiden Wegen. Das ist keine Entwarnung: im Testsplit waren '+pc(E.agreement.malignant_rate_both_low)+' dieser Fälle trotzdem maligne.']}[T.state];
const col={warn:['var(--warnbg)','var(--warn)'],bad:['var(--badbg)','var(--bad)'],info:['var(--grey)','var(--ink)'],ok:['var(--okbg)','var(--ok)']}[msg[0]];
$('#tp').innerHTML=`<h2>Zwei unabhängige Wege <span class="badge b-image">experimentell</span></h2>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px">
<div><div class="nm">Weg 1 · Direktmodell</div><div class="d">${E.direct.name}, trainiert auf maligne gegen benigne</div>
<div style="position:relative;height:12px;background:var(--grey);border-radius:6px;margin:10px 0 4px"><i style="position:absolute;left:0;top:0;height:12px;width:${pos}%;background:${T.d_high?'var(--bad)':'#3b9c8f'};border-radius:6px"></i><i style="position:absolute;left:${thr}%;top:-3px;width:2px;height:18px;background:var(--ink)"></i></div>
<div class="d">Modellscore ${T.direct_prob.toFixed(2)} (unkalibriert) · Schwelle ${T.thr_prob.toFixed(2)} (95 % Sensitivität auf Validierung) → <b>${T.d_high?'über':'unter'} der Schwelle</b></div>
<div class="d">Auf dem Testsplit: Sensitivität ${pc(E.direct.test_sens)}, Spezifität ${pc(E.direct.test_spec)}.</div></div>
<div><div class="nm">Weg 2 · Strukturanalyse</div><div class="d">Glyphen-Score aus 19 validierten Kanälen (5 ISIC-Attribute mit echten Masken, 14 Derm7pt-Kriterien)</div>
<div style="position:relative;height:12px;background:var(--grey);border-radius:6px;margin:10px 0 4px"><i style="position:absolute;left:0;top:0;height:12px;width:${gp}%;background:${T.g_high?'var(--warn)':'#3b9c8f'};border-radius:6px"></i><i style="position:absolute;left:90%;top:-3px;width:2px;height:18px;background:var(--ink)"></i></div>
<div class="d">Höher als ${gp.toFixed(0)} % der benignen Referenzbilder · Hinweisgrenze 90. Perzentil → <b>${T.g_high?'auffällig':'unauffällig'}</b></div>
<div class="d">Erwartungswert 7-Punkte-Checkliste: ${D.check.expected.toFixed(1)} von 9.</div><div class="d" style="margin-top:6px"><b>Hauptbeiträge zum Glyphen-Score</b> (nur Kanäle über 50 %):</div>${T.drivers.length?T.drivers.slice(0,5).map(x=>`<div class="d">· ${x.name} <span style="opacity:.8">(${x.detail})</span></div>`).join(''):'<div class="d">· kein einzelner Kanal über 50 %</div>'}</div></div>
<div style="background:${col[0]};color:${col[1]};border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-top:12px"><b>${msg[1]}</b><div style="font-size:13.5px;margin-top:3px">${msg[2]}</div></div>
<div class="d" style="margin-top:8px">Die beiden Wege sind unabhängig trainiert und werden bewusst nicht verrechnet: eine Fusion verbesserte die Genauigkeit des Direktmodells im Test praktisch nicht (AUROC ${E.fusion?E.fusion.direct_auc.toFixed(4).replace('.',',')+' → '+E.fusion.two_stage_auc.toFixed(4).replace('.',','):'0,9675 → 0,9691'}). Der Glyphenweg liefert die Erklärung und den Abweichungshinweis. Nur für Dermoskopiebilder; die Schwellen stammen aus dem v17-Datensatz.</div>`;})();
const C=D.check;
$('#chkNote').textContent=`Klassische Rechenregel (Argenziano) auf den vom Modell geschätzten Kriterien. Wahrscheinlichkeiten sind auf Derm7pt kalibriert und gelten für den Bildinhalt insgesamt (${C.n_tiles} Kacheln der Läsion); die markierten Bildbereiche zeigen nur, welche Kacheln am stärksten beitrugen, nicht eine geprüfte Lokalisierung.`;
const topIds=new Set([...C.rows].sort((p,q)=>q.prob-p.prob).slice(0,3).filter(r=>r.prob>=.05).map(r=>r.id));
C.rows.forEach(r=>{mk(r.id,r.png,topIds.has(r.id));const e=document.createElement('div');e.className='row';const rel=r.reliable?'':'<span class="badge b-none">wenige Validierungsfälle</span>';
const auc=r.auc.filter(x=>x!=null).length?` · Test-AUC ${Math.min(...r.auc.filter(x=>x!=null)).toFixed(2)}`:'';
e.innerHTML=`<span class="sw" style="background:${r.rgb}"></span><div><label><input type="checkbox" ${topIds.has(r.id)?'checked':''}> <span class="nm">${r.label}</span></label><span class="badge b-region">${r.weight} Punkt${r.weight>1?'e':''}</span>${rel}
<div class="bar" style="height:10px"><i style="width:${r.prob*100}%;background:${r.rgb}"></i></div><div class="d">Wahrscheinlichkeit (kalibriert) ${(r.prob*100).toFixed(0)} % · Beitrag ${r.contrib.toFixed(2)} Punkte${auc}</div></div>`;
e.querySelector('input').onchange=x=>toggle(r.id,x.target.checked);$('#chk').append(e)});
$('#chkSum').innerHTML=`<div class="nm">Erwartungswert ${C.expected.toFixed(1)} von 9 Punkten</div><div class="d">Bei Kriterien mit ≥ 50 %: ${C.points} Punkte. Die klassische Schwelle „≥ 3 Punkte“ wird hier bewusst nicht als Urteil gezeigt: sie übertrug sich nicht auf externe Daten (siehe unten).</div>`;
const iv=C.info;let t=(iv.derm7pt)?`Trennschärfe der Regel (Melanom gegen übrige): Derm7pt-Test AUC ${iv.derm7pt[0].toFixed(2)} (395 Bilder, gleiche Quelle wie das Training). `:'';
if(iv.ext)t+=`Externe Prüfung an ${iv.ext.n_sets} unabhängigen Datensätzen (Melanom gegen Nävus): AUC ${iv.ext.auc[0].toFixed(2)}–${iv.ext.auc[1].toFixed(2)}. Die Schwelle ≥ 3 Punkte erreichte dort nur ${(iv.ext.sens3[0]*100).toFixed(0)}–${(iv.ext.sens3[1]*100).toFixed(0)} % Sensitivität bei ${(iv.ext.spec3[0]*100).toFixed(0)}–${(iv.ext.spec3[1]*100).toFixed(0)} % Spezifität. `;
$('#chkInfo').textContent=t+'Kein Diagnoseersatz; die Werte hängen von Datensatz, Kamera und Melanomanteil ab.';
$('#op').oninput=e=>{document.querySelectorAll('.ov').forEach(i=>i.style.opacity=e.target.value/100)};
</script></body></html>"""

if __name__ == "__main__":
    site = next((a.split('=')[1] for a in sys.argv if a.startswith('--site=')), None)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    R = analyze(args[0], site); out = args[1] if len(args) > 1 else "out/report.html"
    open(out, "w", encoding="utf-8").write(html(R))
    print(f"{R['name']}: lesion {R['les'].mean():.0%} of image, {len(R['items'])} structures, {R['secs']:.1f}s -> {out}")
    for it in R["items"]:
        print(f"  {it['status']:6s} {it['name'][:28]:28s} p={it['prob']:.2f} zones core/border/out {it['zones'][0]:.2f}/{it['zones'][1]:.2f}/{it['zones'][2]:.2f}", *it["flags"][:1])
