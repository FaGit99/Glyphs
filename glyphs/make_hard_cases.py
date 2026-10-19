"""Doctor-view reports for 5 melanomas the direct model misses and 5 naevi it rates as clearly suspicious (clean v17 test split, random draw, fixed seed)."""
import base64, csv, json
import cv2, numpy as np
import fusion_val_test as FV, direct_model as DM, analyze as A

D = "/mnt/samsung_data/mole_data/data/metadata/"
dx = {}
for r in csv.DictReader(open(D + "bcn20000/metadata.csv")): dx[r["isic_id"]] = ("BCN20000", r["diagnosis"])
for r in csv.DictReader(open(D + "isic2020/ground_truth.csv")): dx.setdefault(r["image_name"], ("ISIC 2020", r["diagnosis"]))
for r in csv.DictReader(open(D + "HAM10000/HAM10000_metadata.csv")): dx.setdefault(r["image_id"], ("HAM10000", {"mel": "melanoma", "nv": "nevus"}.get(r["dx"], r["dx"])))
T = FV.load_split("test"); P = json.load(open("models/twopath.json")); gl = P["glyph"]; thr = P["direct"]["thr_prob"]
rows = {r["isic_id"]: r for r in DM.split_rows("test")}
cols = np.array(gl.get("cols", list(range(90))))
gs = ((T["G"][:, cols] - np.array(gl["mean"])) / np.array(gl["scale"])) @ np.array(gl["coef"]) + gl["intercept"]
prob = 1 / (1 + np.exp(-T["logit"])); pct = np.interp(gs, np.array(gl["benign_quantiles"]), np.linspace(0, 100, 101))
ok = lambda i, d, st: dx.get(T["id"][i], ("", ""))[1] == d and rows[T["id"][i]]["label_status"] == st
mel = [i for i in range(len(T["id"])) if ok(i, "melanoma", "malignant_confirmed") and prob[i] < thr]
nev = [i for i in range(len(T["id"])) if ok(i, "nevus", "benign_confirmed") and prob[i] >= .90]
rng = np.random.default_rng(23)
pick = [("Melanom (vom Direktmodell übersehen)", int(i)) for i in rng.choice(mel, 5, replace=False)] + [("Nävus (vom Direktmodell stark auffällig)", int(i)) for i in rng.choice(nev, 5, replace=False)]
print(f"candidates: missed melanomas {len(mel)} (glyph flag among them {int((pct[mel] >= 90).sum())}), conspicuous naevi {len(nev)} (glyph high among them {int((pct[nev] >= 90).sum())})")
STATE = {"both_high": "beide Wege auffällig", "both_low": "beide Wege unauffällig", "flag": "Abweichung: zweiter Blick", "direct_only": "nur Direktmodell auffällig"}
out = []
for k, (kind, i) in enumerate(pick):
    iid = T["id"][i]; R = A.analyze(DM.ARCHIVE + rows[iid]["image_relpath"]); tp = R["twopath"]; c = R["check"]
    name = f"{'melanom_uebersehen' if kind.startswith('Mel') else 'naevus_auffaellig'}_{k % 5 + 1}_{iid}.html"
    open(f"out/cases_hard/{name}", "w", encoding="utf-8").write(A.html(R))
    th = cv2.resize(cv2.cvtColor(R["img"], cv2.COLOR_RGB2BGR), (160, int(160 * R["img"].shape[0] / R["img"].shape[1]))); b = base64.b64encode(cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    out.append(dict(kind=kind, id=iid, src=dx[iid][0], dx=dx[iid][1], file=name, thumb=b, direct=tp["direct_prob"], thr=tp["thr_prob"], g_pct=tp["g_pct"], state=tp["state"], expected=c["expected"], drivers=[d["name"] for d in tp["drivers"][:3]]))
    print(f"{kind[:22]:22s} {iid} {dx[iid][0]:9s} direct {tp['direct_prob']:.2f}  glyph pct {tp['g_pct']:.0f}  state {tp['state']:11s} checklist {c['expected']:.1f}  drivers: {out[-1]['drivers']}")
rows_html = "".join(f"<tr><td><img src='data:image/jpeg;base64,{o['thumb']}' width='120'></td><td><b>{o['kind']}</b><br><span class=s>{o['id']}<br>{o['src']}, bestätigt: {o['dx']}</span></td><td>{o['direct']:.2f}<br><span class=s>Schwelle {o['thr']:.2f}</span></td><td>{o['g_pct']:.0f}.</td><td>{o['expected']:.1f}</td><td>{STATE[o['state']]}</td><td>{', '.join(o['drivers']) or '–'}</td><td><a href='{o['file']}'>Bericht öffnen</a></td></tr>" for o in out)
open("out/cases_hard/index.html", "w", encoding="utf-8").write(f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Zehn schwierige Fälle</title><style>
body{{font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2430}}.w{{max-width:1180px;margin:0 auto;padding:16px}}h1{{font-size:20px}}table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ec}}td,th{{padding:8px 10px;border-top:1px solid #e3e6ec;text-align:left;vertical-align:top}}th{{font-size:13px;color:#667085}}.s{{font-size:12px;color:#667085}}.n{{background:#fdf0dd;color:#b25e09;border-radius:10px;padding:10px 12px;margin:12px 0;font-size:13.5px}}
@media(prefers-color-scheme:dark){{body{{background:#12161d;color:#e8ecf3}}table{{background:#1a2029;border-color:#2a3340}}td,th{{border-color:#2a3340}}a{{color:#8ab4ff}}.n{{background:#3a2c14;color:#f0b35a}}}}</style></head><body><div class='w'><h1>Zehn schwierige Fälle: 5 übersehene Melanome, 5 auffällige Nävi</h1>
<div class='n'><b>Auswahlregel (vorab festgelegt, danach zufällig mit festem Seed gezogen, unabhängig vom Zustand des Glyphenwegs):</b> sauberer v17-Testsplit, nur dermoskopische Bilder, die weder für das Glyphentraining noch für die Synthese genutzt wurden, mit bestätigter Diagnose.
Melanom: Direktscore unter der Schwelle (0,40), also vom Direktmodell als benigne eingestuft ({len(mel)} Kandidaten). Nävus: Direktscore ≥ 0,90 ({len(nev)} Kandidaten).
Zum Vergleich der Anzeige mit den leichten Fällen siehe <code>out/cases</code>. Forschungsprototyp, keine Diagnose. Die Bilder stammen aus öffentlichen Datensätzen (ISIC/HAM10000/BCN20000, CC BY-NC) und liegen nur lokal.</div>
<table><tr><th></th><th>Fall</th><th>Direktscore</th><th>Glyphen-Score (Perzentil der benignen Referenz)</th><th>7-Punkte (Erwartungswert)</th><th>Zustand</th><th>Hauptbeiträge zum Glyphen-Score</th><th></th></tr>{rows_html}</table></div></body></html>""")
print("index: out/cases_hard/index.html")
