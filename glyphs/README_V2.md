# Glyphen-Pipeline (Dermoskopie) – Überblick (Version 2)

Zweipfad-Ansicht: Direktmodell (EVA-02 base 448 [9], v17) + Strukturanalyse („Glyphen“) mit Abweichungshinweis. Forschungsprototyp, keine Diagnose.
Ziel: eine unterstützende Information für die Differenzialdiagnose neben dem Klassifikationsmodell (erkannte Strukturen mit Lage und Validierungsstatus, 7-Punkte-Checkliste, laut Konsens assoziierte Diagnosen). Version 2 gleicht README und Paper ab, ergänzt Quellen und rechnet die Kennzahlen auch auf ausschließlich histologisch gesicherten Testbildern.

## Bericht für ein Bild erzeugen
```bash
cd glyphs
.venv/bin/python analyze.py <bild.jpg> out/report.html [--site=facial|volar|nonfacial]
```
Voraussetzungen: `models/hybrid.pt`, `models/mel_final.pt`, `models/lesion_seg.pt`, `models/*_calibration.json`, `models/twopath.json`
sowie das Direktmodell (nur lesend) unter `…/260826 try_cleaned_data/saved_models/base_448_v17/best_pauc.pt`.

## Aufbau
| Schritt | Dateien |
|---|---|
| Synthetische Strukturen (71 IDs der Taxonomie) | `geometry.py`, `structures.py`, `render.py`, `skin.py`, `compose.py`, `gen_dataset.py`, `taxonomy.csv`, `registry.csv` |
| Detektor mit Regionskarten (ViT-S/16 [11], 14×14-Karten pro ID) | `dense.py`, `hybrid.py` (Hybrid mit ISIC-2018-Masken, Kalibrierung), `train_dense.py` |
| Läsionssegmentierung (Zonen innen / Rand / außen) | `lesion_seg.py` |
| 7-Punkte-Kriterien (Derm7pt [3], Bildebene, MIL) | `d7_data.py`, `mil.py`, `train_melanoma.py`, `calibrate_mel.py` |
| Direktmodell (EVA-02) lesend aufrufen | `direct_model.py` (verifiziert: Test-AUROC 0,9675) |
| Zweipfad-Parameter | `twopath_fit.py` → `models/twopath.json` (Variante B); `validated_score.py` → `models/twopath_all90.json` (Variante A) |
| Histologische Nachrechnung (neu in V2) | `histo_eval.py` → `models/histo_eval.json`; `histo_external.py` → `models/histo_external.json` |
| Arztansicht (HTML) | `analyze.py` |

## Prüfungen (Skripte → Ergebnisse in `models/*.json`)
`realfake.py` (Echt-vs-synthetisch), `eval_dense.py`/`hybrid.py eval` (Lokalisierung), `pipeline_eval.py`, `eval_melanoma.py`, `external_check.py`
(4 externe Quellen), `fusion_check.py`, `direct_fusion.py`, `fusion_val_test.py` (Fusion Direktmodell + Glyphen), `check_integration.py`,
`histo_eval.py`, `histo_external.py` (Kennzahlen auf histologisch gesicherten Bildern).

## Welche Variante gilt?
Ausgeliefert (twopath.json, analyze.py, Paper V2) ist **Variante B**: Glyphen-Score aus den 19 an echten Daten validierten Kanälen (5 ISIC-Attribute, 14 Derm7pt-Kriterien).
**Variante A** (alle 90 Kanäle, `models/twopath_all90.json`) war in Paper V1 beschrieben und dient in V2 nur zum Vergleich. Der Docstring von `twopath_fit.py` beschreibt noch Variante A; der Code wurde nicht geändert.

## Wichtige Befunde (v17-Test, 8 054 nutzbare Bilder, 1 766 maligne)
- Regionen validiert nur für 5 ISIC-Attribute (Netzwerk, negatives Netzwerk, Streaks, Milia, Globuli); alle übrigen IDs sind synthetisch trainiert, nicht validiert.
- Keine Fusion: Direktmodell 0,9675 → zweistufig B 0,9688 AUROC (+0,0012 [−0,0006; +0,0034]); Variante A +0,0014 [+0,0002; +0,0028]. Praktisch kein Gewinn.
- Abweichungshinweis (Direktscore unter Schwelle, Glyphen-Score ≥ 90. Perzentil der benignen Validierungsbilder), Variante B:
  424 von 5 554 Bildern markiert laut `twopath.json` (Nachrechnung: 423), 5,7 % maligne (Grundrate 1,9 %), 24 von 105 übersehenen gefunden.
- Die klassische Schwelle „≥ 3 Punkte“ überträgt sich nicht auf externe Daten.
- Nur Dermoskopie; Schwellen stammen aus dem v17-Split.

## Nur histologisch gesicherte Testbilder (neu in V2)
Teilmenge über die Label-Status-Metadaten des v17-Splits: 3 143 Bilder (1 766 maligne, 1 377 benigne). Die übrigen 4 911 Benignen sind schwach gelabelt.

| Kennzahl | alle 8 054 | nur histologisch (3 143) |
|---|---|---|
| AUROC Direktmodell | 0,9675 [0,959; 0,974] | 0,899 [0,879; 0,915] |
| Spezifität bei 95 % Sensitivität, Direktmodell | 0,842 | 0,476 |
| Fusion B minus Direktmodell (AUROC) | +0,0012 [−0,0006; +0,0034] | −0,0003 [−0,0025; +0,0018] |
| Abweichungshinweis B: maligne unter Markierungen (Grundrate) | 5,7 % (1,9 %) | 14,1 % (12,4 %) |
| Anreicherung [95 %-Intervall] | ×3,0 [1,5; 4,5] | ×1,1 [0,5; 1,8] |
| 7-Punkte-Regel extern, Melanom gegen Naevus, gepoolt | 0,741 | 0,721 [0,700; 0,739] |

Auf histologisch gesicherten Bildern ist der Hinweis von der Grundrate nicht unterscheidbar. Nicht nachgerechnet: ImageNet-Vergleich und Fusion mit generischen Merkmalen auf der Teilmenge.

## V17-Daten (Kurzfassung; Details in Paper V2)
Roharchiv 144 636 Bilder, 96 059 im Split (Training 77 281 / Validierung 9 404 / Test 9 374), 48 577 ausgeschlossen (u. a. 33 897 nur klinisch eingeschätzte Bilder, 9 408 Sticker-Kohorte). Split nach Gruppen, Seed 20260901, stratifiziert nach Klasse × Zentrum.
Wichtig für die Auswertung: Alle schwach gelabelten Benignen stammen aus der Quelle ISIC_benign_only; die Quelle sagt im Test für 55 % der Bilder „benigne“ mit 100 % Treffsicherheit voraus (Runbook). Die Prüfung nach Quelle/Zentrum (E-m) war am 01.09.2026 offen; Hautton nicht bewertet (D1).

## Schwache Datenbasis des Glyphentrainings
- Echte Masken: nur 5 Attribute, 2 594 ISIC-2018-Bilder (Test-Positive: 224 / 23 / 9 / 34 / 70).
- Echte Strukturlabels: 1 011 Derm7pt-Fälle, nur Bildebene; Gefäßkanäle mit 8 bis 22 positiven Testfällen.
- 58 von 71 Strukturen ohne Zuordnung zu einem der fünf ISIC-Attribute: rein synthetisch trainiert, nicht validiert. Die übrigen 13 werden nur als Vereinigung je Attribut an echten Masken geprüft.
- Taxonomie → Diagnose allein: AUC 0,46 bis 0,61 (Probe 0,68 bis 0,92, Orakel 0,71 bis 0,96).

## Schlussfolgerung
Glyphenerkennung und Taxonomie liefern eine unterstützende, teilweise validierte Zusatzinformation neben dem Klassifikationsmodell. Ein messbarer Gewinn für die Differenzialdiagnose ist mit den vorliegenden Daten nicht belegt.
Die Verbesserung der Glyphen-Datenbasis sollte die Voraussetzungen schaffen, diese zweite „Meinung“ zum reinen Klassifikationsmodell zu verbessern; das ist eine Hypothese, die noch zu prüfen ist.

## Änderungen gegenüber Version 1 (Paper und README)
1. Paper V1 beschrieb Variante A (391 markiert, 6,9 %, 27 gefunden), ausgeliefert ist Variante B (424/423, 5,7 %, 24); Paper und Abbildung korrigiert.
2. Hinweiszeilen 95 %/98 % (A) in V1 (177 · 7,3 % · 13; 51 · 15,7 % · 8) nicht reproduzierbar; Nachrechnung 164 · 7,3 % · 12 und 42 · 19,0 % · 8.
3. Rundungsfehler korrigiert: 0,58→0,57 (Regression alt), 0,90→0,89 (Dots/Globuli neu; ImageNet-Probe Derm7pt), 0,88→0,87 (Dermatofibrom-Probe), 0,91→0,90 (BCC-Orakel), 0,76→0,75 (ImageNet-Probe ISIC 2019), 0,92→0,91 (Pigmentnetzwerk Pipeline).
4. Fusionsintervalle nachgerechnet (siehe oben); Reproduktion mit scikit-learn 1.7.2 nur näherungsweise.
5. „Rund 66 IDs“ ohne echte Masken war falsch (71 minus 5 Attributkanäle); richtig sind 58 von 71 IDs.
6. Histologische Nachrechnung, V17-Abschnitt, Quellen, Schlussfolgerung ergänzt.

## Quellen
[1] Kittler H et al. J Am Acad Dermatol 2016;74(6):1093–1106, doi:10.1016/j.jaad.2015.12.038 · [2] Argenziano G et al. Arch Dermatol 1998;134(12):1563–1570, doi:10.1001/archderm.134.12.1563 · [3] Kawahara J et al. IEEE J Biomed Health Inform 2019;23(2):538–546, doi:10.1109/JBHI.2018.2824327 · [4] Codella N et al. arXiv:1902.03368 · [5] Tschandl P et al. Sci Data 2018;5:180161, doi:10.1038/sdata.2018.161 · [6] Combalia M et al. arXiv:1908.02288 · [7] Rotemberg V et al. Sci Data 2021;8:34, doi:10.1038/s41597-021-00815-z · [8] https://challenge.isic-archive.com/data · [9] Fang Y et al. arXiv:2303.11331 · [10] Yan S et al. Nat Med 2025;31(8):2691–2702, doi:10.1038/s41591-025-03747-y · [11] Dosovitskiy A et al. arXiv:2010.11929 · [12] Platt J 1999 (Platt-Skalierung) · [13] Wightman R, timm, doi:10.5281/zenodo.4414861.
Projektquellen: [P1] V17-Split-Metadaten, [P2] 260820 DataqualityAnalysis, [P3] und [P4] 260826 try_cleaned_data, [P5] `models/*.json`, `histo_eval.py`, `histo_external.py`. Vollständige Angaben im Paper V2.

Python-Umgebung: `glyphs/.venv` (uv; torch, timm, scikit-learn, opencv). Das System-Python hat kein numpy.
