"""Doctor-view reports for 5 clear melanomas and 5 unambiguous naevi from the clean v17 test split (random draw with a fixed seed inside a stated rule)."""
import base64, csv, json, os
import cv2, numpy as np
import fusion_val_test as FV, direct_model as DM, analyze as A

D = "/mnt/samsung_data/mole_data/data/metadata/"
dx = {}
for r in csv.DictReader(open(D + "bcn20000/metadata.csv")): dx[r["isic_id"]] = ("BCN20000", r["diagnosis"])
for r in csv.DictReader(open(D + "isic2020/ground_truth.csv")): dx.setdefault(r["image_name"], ("ISIC 2020", r["diagnosis"]))
for r in csv.DictReader(open(D + "HAM10000/HAM10000_metadata.csv")): dx.setdefault(r["image_id"], ("HAM10000", {"mel": "melanoma", "nv": "nevus"}.get(r["dx"], r["dx"])))
T = FV.load_split("test"); P = json.load(open("models/twopath.json")); gl = P["glyph"]
rows = {r["isic_id"]: r for r in DM.split_rows("test")}
cols = np.array(gl.get("cols", list(range(90))))
gs = ((T["G"][:, cols] - np.array(gl["mean"])) / np.array(gl["scale"])) @ np.array(gl["coef"]) + gl["intercept"]
prob = 1 / (1 + np.exp(-T["logit"])); pct = np.interp(gs, np.array(gl["benign_quantiles"]), np.linspace(0, 100, 101))
ok = lambda i, d, st: dx.get(T["id"][i], ("", ""))[1] == d and rows[T["id"][i]]["label_status"] == st
mel = [i for i in range(len(T["id"])) if ok(i, "melanoma", "malignant_confirmed") and prob[i] >= .95 and pct[i] >= 95]
nev = [i for i in range(len(T["id"])) if ok(i, "nevus", "benign_confirmed") and prob[i] <= .20 and pct[i] <= 50]
KEEP = {"Melanom": ["ISIC_0054755", "ISIC_0063769", "ISIC_0054760", "ISIC_0067843", "ISIC_0061434"], "Nävus": ["ISIC_1602561", "ISIC_0064370", "ISIC_0031031", "ISIC_0032811", "ISIC_7658533"]}
idx = {iid: i for i, iid in enumerate(T["id"])}
pick = [(k, idx[i]) for k, ids_ in KEEP.items() for i in ids_]   # same ten cases as with the 90-channel score (drawn by the rule with seed 11)
print(f"under the validated-19 score the rule would admit: melanoma {len(mel)}, nevus {len(nev)} candidates; the ten cases drawn earlier are kept")
out = []
for k, (kind, i) in enumerate(pick):
    iid = T["id"][i]; R = A.analyze(DM.ARCHIVE + rows[iid]["image_relpath"]); tp = R["twopath"]; c = R["check"]
    name = f"{'melanom' if kind == 'Melanom' else 'naevus'}_{k % 5 + 1}_{iid}.html"
    open(f"out/cases/{name}", "w", encoding="utf-8").write(A.html(R))
    th = cv2.resize(cv2.cvtColor(R["img"], cv2.COLOR_RGB2BGR), (160, int(160 * R["img"].shape[0] / R["img"].shape[1]))); b = base64.b64encode(cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    out.append(dict(kind=kind, id=iid, src=dx[iid][0], dx=dx[iid][1], file=name, thumb=b, direct=tp["direct_prob"], thr=tp["thr_prob"], g_pct=tp["g_pct"], state=tp["state"], expected=c["expected"],
                    items=[it["name"] for it in R["items"] if it["status"] == "region" and not it["weak"]]))
    print(f"{kind:7s} {iid} {dx[iid][0]:9s} direct {tp['direct_prob']:.2f}  glyph pct {tp['g_pct']:.0f}  state {tp['state']:11s} checklist {c['expected']:.1f}  structures: {out[-1]['items']}")
STATE = {"both_high": "beide Wege auffällig", "both_low": "beide Wege unauffällig", "flag": "Abweichung: zweiter Blick", "direct_only": "nur Direktmodell auffällig"}
rows_html = "".join(f"<tr><td><img src='data:image/jpeg;base64,{o['thumb']}' width='120'></td><td><b>{o['kind']}</b><br><span class=s>{o['id']}<br>{o['src']}, histologisch/bestätigt: {o['dx']}</span></td><td>{o['direct']:.2f}<br><span class=s>Schwelle {o['thr']:.2f}</span></td><td>{o['g_pct']:.0f}.</td><td>{o['expected']:.1f}</td><td>{STATE[o['state']]}</td><td>{', '.join(o['items']) or '–'}</td><td><a href='{o['file']}'>Bericht öffnen</a></td></tr>" for o in out)
open("out/cases/index.html", "w", encoding="utf-8").write(f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Zehn Beispielfälle</title><style>
body{{font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2430}}.w{{max-width:1180px;margin:0 auto;padding:16px}}h1{{font-size:20px}}table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ec}}td,th{{padding:8px 10px;border-top:1px solid #e3e6ec;text-align:left;vertical-align:top}}th{{font-size:13px;color:#667085}}.s{{font-size:12px;color:#667085}}.n{{background:#fdf0dd;color:#b25e09;border-radius:10px;padding:10px 12px;margin:12px 0;font-size:13.5px}}
@media(prefers-color-scheme:dark){{body{{background:#12161d;color:#e8ecf3}}table{{background:#1a2029;border-color:#2a3340}}td,th{{border-color:#2a3340}}a{{color:#8ab4ff}}.n{{background:#3a2c14;color:#f0b35a}}}}</style></head><body><div class='w'><h1>Zehn Beispielfälle: 5 klare Melanome, 5 eindeutige Nävi</h1>
<div class='n'><b>Auswahlregel (vorher festgelegt, danach zufällig mit festem Seed gezogen; die Fälle wurden mit dem früheren 90-Kanal-Score gezogen und unverändert beibehalten, die Anzeige nutzt jetzt den Score aus 19 validierten Kanälen):</b> sauberer v17-Testsplit, nur dermoskopische Bilder, die weder für das Glyphentraining noch für die Synthese genutzt wurden.
Melanom: histologisch bestätigt, Direktscore ≥ 0,95 und Glyphen-Score ≥ 95. Perzentil (mit dem früheren Score 218 Kandidaten). Nävus: bestätigter benigner Nävus, Direktscore ≤ 0,20 und Glyphen-Score ≤ 50. Perzentil (mit dem früheren Score 67 Kandidaten).
Die Fälle sind per Regel <b>leicht</b> und zeigen nicht die typische Leistung; schwierige Fälle (Abweichung, nur ein Weg auffällig) stehen in <code>out/twopath</code>. Forschungsprototyp, keine Diagnose. Die Bilder stammen aus öffentlichen Datensätzen (ISIC/HAM10000/BCN20000, CC BY-NC) und liegen nur lokal.</div>
<table><tr><th></th><th>Fall</th><th>Direktscore</th><th>Glyphen-Score (Perzentil der benignen Referenz)</th><th>7-Punkte (Erwartungswert)</th><th>Zustand</th><th>Validierte Strukturen im Bericht</th><th></th></tr>{rows_html}</table></div></body></html>""")
print("index: out/cases/index.html")
