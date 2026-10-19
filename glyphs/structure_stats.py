"""Statistics of detected structures (taxonomy IDs) on 1,000 random v17 images.

  python structure_stats.py extract   # sample + run the pipeline, cache models/structure_stats_1000.npz
  python structure_stats.py report    # tables -> out/structure_stats/ (CSV + HTML)

Sample: random 1,000 dermoscopic images from the whole v17 set (train + val + test), excluding images my glyph/lesion models were trained on
(ISIC 2018 Task 1-2, HAM10000 index-parity-0). Detection values per image:
  71 taxonomy IDs : image-level head output (mean of the top-3 lesion tiles)      [raw model output, not calibrated]
  14 Derm7pt criteria: calibrated probability                                      [image level]
   5 ISIC attributes: calibrated maximum of the stitched probability map            [as shown in the report]
 'clear' = value >= 0.5, 'weak (incl. clear)' = value >= 0.1. For the IDs a region (map p >= 0.5 over >= 40 px) is counted separately.
"""
import csv, glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
import analyze as A, direct_model as DM, d7_data as DD, diagnosis_map as DMAP, hybrid as H, external_check as EC, skin

SPL = "/mnt/samsung_data/mole_data/e_reports_v17/splits/v17/"
Dx = "/mnt/samsung_data/mole_data/"
N = 1000


def sample():
    names18 = {os.path.basename(f)[:-4] for sub in ("Training", "Validation", "Test") for f in glob.glob(Dx + f"data/raw/isic2018/ISIC2018_Task1-2_{sub}_Input/*.jpg")}
    ham0 = {os.path.basename(f)[:-4] for f in skin.ham_files(0)}
    rows = []
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(SPL + sp + ".csv")):
            if r["image_type"] == "dermoscopic" and r["isic_id"] not in names18 and r["isic_id"] not in ham0:
                r["split"] = sp; rows.append(r)
    rng = np.random.default_rng(2026)
    return len(rows), [rows[i] for i in rng.permutation(len(rows))[:N]]


def extract():
    n_all, rows = sample(); M = A.models(); ids = M["ids"]; K = len(ids); print(f"usable v17 images {n_all}; sample {len(rows)}", flush=True)
    out = {k: [] for k in ("id", "split", "target", "status", "idv", "idreg", "attr_cal", "attr_reg", "d7", "lesion")}; t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for n, (r, img) in enumerate(zip(rows, ex.map(lambda r: EC._safe(EC.load450, DM.ARCHIVE + r["image_relpath"]), rows))):
            if img is None:
                continue
            les, _ = A.lesion_mask(img, M["seg"]); h, w = les.shape
            P, nt, pos = A.stitched_maps(img, M["det"])
            ck = A.checklist(img, les, pos, M, render=False, keep=True)
            attr_cal, attr_reg = [], []
            for k, a in enumerate(H.ATTRS):
                p = P[K + k]; c = M["cal"][a]; lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                cal = 1 / (1 + np.exp(-(c["platt"][0] * lg + c["platt"][1]))); attr_cal.append(float(cal.max())); attr_reg.append(bool((p >= c["thr"]).sum() >= 40))
            for key, v in (("id", r["isic_id"]), ("split", r["split"]), ("target", int(float(r["target"]))), ("status", r["label_status"]), ("idv", ck["agg_raw"][:K]),
                           ("idreg", [(P[j] >= .5).sum() >= 40 for j in range(K)]), ("attr_cal", attr_cal), ("attr_reg", attr_reg), ("d7", ck["cal14"]), ("lesion", float(les.mean()))):
                out[key].append(v)
            if n % 250 == 249:
                print(f"  {n + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    np.savez("models/structure_stats_1000.npz", ids=np.array(ids), **{k: np.array(v) for k, v in out.items() if k != "ids"}, sample_ids=np.array(out["id"]), n_usable=n_all)
    print("done", len(out["id"]), "images")


if __name__ == "__main__" and sys.argv[1] == "extract":
    extract()


def report():
    import html as H_
    z = np.load("models/structure_stats_1000.npz", allow_pickle=True); ids = [str(i) for i in z["ids"]]
    tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}; dm = DMAP.load(); amap = H.attr_ids(ids)
    idv, idreg, tg, n = z["idv"], z["idreg"].astype(bool), z["target"].astype(int), len(z["target"])
    mal, ben = tg == 1, tg == 0
    name = lambda s: tax[s]["Metaphoric term"] if not tax[s]["Metaphoric term"].startswith("(") else tax[s]["Descriptive term (IDS)"]
    os.makedirs("out/structure_stats", exist_ok=True)
    clear, weak = idv >= .5, idv >= .1
    rows = []
    for j, sid in enumerate(ids):
        c, w = clear[:, j], weak[:, j]
        rows.append(dict(ID=sid, Kategorie=tax[sid]["Category"], Struktur=name(sid), Beschreibung=tax[sid]["Descriptive term (IDS)"], n_klar=int(c.sum()), pct_klar=100 * c.mean(), n_klar_Region=int((c & idreg[:, j]).sum()),
                         n_schwach=int(w.sum()), pct_schwach=100 * w.mean(), pct_klar_in_malignen=100 * c[mal].mean() if mal.any() else 0, pct_klar_in_benignen=100 * c[ben].mean(),
                         pct_schwach_in_malignen=100 * w[mal].mean(), pct_schwach_in_benignen=100 * w[ben].mean(), Diagnosegruppen=",".join(dm[sid]["groups"]) or "-"))
    rows.sort(key=lambda r: -r["n_klar"])
    def wcsv(fn, rs):
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w_.writeheader(); [w_.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}) for r in rs]
    wcsv("out/structure_stats/per_id.csv", rows)
    cats = []
    for cat in sorted({r["Kategorie"] for r in rows}):
        cj = [j for j, s in enumerate(ids) if tax[s]["Category"] == cat]
        cats.append(dict(Kategorie=cat, n_IDs=len(cj), Bilder_mit_mind_1_klar=int(clear[:, cj].any(1).sum()), pct_klar=100 * clear[:, cj].any(1).mean(), Bilder_mit_mind_1_schwach=int(weak[:, cj].any(1).sum()),
                         pct_schwach=100 * weak[:, cj].any(1).mean(), mittlere_Anzahl_klar=clear[:, cj].sum(1).mean(), mittlere_Anzahl_schwach=weak[:, cj].sum(1).mean()))
    cats.sort(key=lambda r: -r["pct_klar"]); wcsv("out/structure_stats/per_category.csv", cats)
    # validated channels
    val = []
    for k, a in enumerate(H.ATTRS):
        v = z["attr_cal"][:, k]; val.append(dict(Kanal=f"ISIC-Attribut {a}", Typ="Region validiert (echte Masken)", Wert="kalibriertes Kartenmaximum", n_klar=int((v >= .5).sum()), pct_klar=100 * (v >= .5).mean(), n_schwach=int((v >= .1).sum()), pct_schwach=100 * (v >= .1).mean(), n_Region=int(z["attr_reg"][:, k].sum())))
    for k, nm in enumerate(DD.TNAMES):
        v = z["d7"][:, k]; val.append(dict(Kanal=f"Derm7pt {nm}", Typ="Bildebene (Derm7pt geprüft)", Wert="kalibrierte Wahrscheinlichkeit", n_klar=int((v >= .5).sum()), pct_klar=100 * (v >= .5).mean(), n_schwach=int((v >= .1).sum()), pct_schwach=100 * (v >= .1).mean(), n_Region=""))
    wcsv("out/structure_stats/validated_channels.csv", val)
    nc, nw = clear.sum(1), weak.sum(1)
    dist = dict(n_images=n, malignant_share=100 * mal.mean(), clear_mean=nc.mean(), clear_median=float(np.median(nc)), clear_p90=float(np.percentile(nc, 90)), clear_zero_pct=100 * (nc == 0).mean(),
                weak_mean=nw.mean(), weak_median=float(np.median(nw)), weak_p90=float(np.percentile(nw, 90)), weak_zero_pct=100 * (nw == 0).mean(),
                clear_with_region_pct=100 * (clear & idreg).sum() / max(clear.sum(), 1), usable=int(z["n_usable"]))
    print(json_dump(dist))
    print("\nTop 15 structures (taxonomy IDs) by clear detections:")
    for r in rows[:15]:
        print(f"  {r['ID']} {r['Struktur'][:34]:34s} [{r['Kategorie'][:12]:12s}] clear {r['n_klar']:4d} ({r['pct_klar']:4.1f} %), with region {r['n_klar_Region']:3d} | >=0.1: {r['n_schwach']:4d} ({r['pct_schwach']:4.1f} %) | clear in malignant {r['pct_klar_in_malignen']:4.1f} % vs benign {r['pct_klar_in_benignen']:4.1f} %")
    print("\nIDs never detected clearly:", sum(r["n_klar"] == 0 for r in rows), "| never at >= 0.1:", sum(r["n_schwach"] == 0 for r in rows))
    print("\nBy category:"); [print(f"  {c['Kategorie']:20s} clear {c['pct_klar']:5.1f} % of images (mean {c['mittlere_Anzahl_klar']:.2f}) | >=0.1 {c['pct_schwach']:5.1f} % (mean {c['mittlere_Anzahl_schwach']:.2f})") for c in cats]
    print("\nValidated channels:"); [print(f"  {v['Kanal']:42s} clear {v['pct_klar']:5.1f} % | >=0.1 {v['pct_schwach']:5.1f} %") for v in val]
    # HTML
    esc = H_.escape
    def tbl(rs, cols, heads):
        return "<table><tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr>" + "".join("<tr>" + "".join(f"<td>{esc(str(round(r[c], 1)) if isinstance(r[c], float) else str(r[c]))}</td>" for c in cols) + "</tr>" for r in rs) + "</table>"
    page = f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Strukturstatistik, 1000 v17-Bilder</title><style>
body{{font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2430}}.w{{max-width:1250px;margin:0 auto;padding:16px}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:26px}}table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ec;margin-bottom:8px}}td,th{{padding:5px 8px;border-top:1px solid #e3e6ec;text-align:left;font-size:13px}}th{{color:#667085;font-weight:600}}.n{{background:#fdf0dd;color:#b25e09;border-radius:10px;padding:10px 12px;margin:12px 0;font-size:13.5px}}
@media(prefers-color-scheme:dark){{body{{background:#12161d;color:#e8ecf3}}table{{background:#1a2029;border-color:#2a3340}}td,th{{border-color:#2a3340}}.n{{background:#3a2c14;color:#f0b35a}}}}</style></head><body><div class='w'>
<h1>Erkannte Strukturen nach Taxonomie: 1 000 zufällige v17-Bilder</h1>
<div class='n'><b>Definition.</b> Stichprobe: 1 000 zufällige dermoskopische Bilder aus dem gesamten v17-Datensatz ({dist['usable']:,} nutzbar, ohne ISIC-2018-Task-1/2-Bilder und ohne die HAM-Hälfte der Synthese; {dist['malignant_share']:.0f} % maligne).
<b>Klar erkannt:</b> Bildwert ≥ 0,5. <b>Mit schwachen Signalen:</b> Bildwert ≥ 0,1 (schließt klare ein). Bildwert: bei den 71 Taxonomie-IDs die Ausgabe des Bildkopfs (nicht kalibriert, synthetisch trainiert, nicht an echten Masken validiert), bei den Derm7pt-Kriterien die kalibrierte Wahrscheinlichkeit, bei den ISIC-Attributen das kalibrierte Kartenmaximum. „Region“: Karte ≥ 0,5 über ≥ 40 px. Die Zahlen beschreiben, was das Modell meldet, nicht wie oft die Strukturen tatsächlich vorkommen.</div>
<h2>Strukturen pro Bild</h2>
<table><tr><th></th><th>Mittelwert</th><th>Median</th><th>90. Perzentil</th><th>Bilder ohne Struktur</th></tr><tr><td>klar erkannt (≥ 0,5)</td><td>{dist['clear_mean']:.1f}</td><td>{dist['clear_median']:.0f}</td><td>{dist['clear_p90']:.0f}</td><td>{dist['clear_zero_pct']:.1f} %</td></tr><tr><td>mit schwachen Signalen (≥ 0,1)</td><td>{dist['weak_mean']:.1f}</td><td>{dist['weak_median']:.0f}</td><td>{dist['weak_p90']:.0f}</td><td>{dist['weak_zero_pct']:.1f} %</td></tr></table>
<div style='font-size:13px'>Von allen klaren Erkennungen haben {dist['clear_with_region_pct']:.0f} % zusätzlich eine Region in der Karte.</div>
<h2>Nach Kategorie der Taxonomie</h2>{tbl(cats, ['Kategorie', 'n_IDs', 'Bilder_mit_mind_1_klar', 'pct_klar', 'Bilder_mit_mind_1_schwach', 'pct_schwach', 'mittlere_Anzahl_klar', 'mittlere_Anzahl_schwach'], ['Kategorie', 'IDs', 'Bilder mit ≥ 1 klar', '%', 'Bilder mit ≥ 1 (≥ 0,1)', '%', 'Ø klar pro Bild', 'Ø (≥ 0,1) pro Bild'])}
<h2>Validierte Kanäle (ISIC-Attribute, Derm7pt-Kriterien)</h2>{tbl(val, ['Kanal', 'Typ', 'Wert', 'n_klar', 'pct_klar', 'n_schwach', 'pct_schwach', 'n_Region'], ['Kanal', 'Prüfung', 'Wert', 'klar (n)', '%', '≥ 0,1 (n)', '%', 'mit Region (n)'])}
<h2>Alle 71 Taxonomie-IDs, sortiert nach klaren Erkennungen</h2>{tbl(rows, ['ID', 'Kategorie', 'Struktur', 'n_klar', 'pct_klar', 'n_klar_Region', 'n_schwach', 'pct_schwach', 'pct_klar_in_malignen', 'pct_klar_in_benignen', 'Diagnosegruppen'], ['ID', 'Kategorie', 'Struktur', 'klar (n)', '%', 'davon mit Region', '≥ 0,1 (n)', '%', '% klar in malignen', '% klar in benignen', 'Diagnosegruppen'])}
</div></body></html>"""
    open("out/structure_stats/index.html", "w", encoding="utf-8").write(page); print("\nwritten: out/structure_stats/index.html, per_id.csv, per_category.csv, validated_channels.csv")


def json_dump(d):
    import json
    return json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}, ensure_ascii=False)


if __name__ == "__main__" and sys.argv[1] == "report":
    report()
