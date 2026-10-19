# Glyphen-Pipeline (Dermoskopie) – Überblick

Zweipfad-Ansicht: Direktmodell (EVA-02 base 448, v17) + Strukturanalyse („Glyphen“) mit Abweichungshinweis. Forschungsprototyp, keine Diagnose.

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
| Detektor mit Regionskarten (ViT-S/16, 14×14-Karten pro ID) | `dense.py`, `hybrid.py` (Hybrid mit ISIC-2018-Masken, Kalibrierung), `train_dense.py` |
| Läsionssegmentierung (Zonen innen / Rand / außen) | `lesion_seg.py` |
| 7-Punkte-Kriterien (Derm7pt, Bildebene, MIL) | `d7_data.py`, `mil.py`, `train_melanoma.py`, `calibrate_mel.py` |
| Direktmodell (EVA-02) lesend aufrufen | `direct_model.py` (verifiziert: Test-AUROC 0,9675) |
| Zweipfad-Parameter | `twopath_fit.py` → `models/twopath.json` |
| Arztansicht (HTML) | `analyze.py` |

## Prüfungen (Skripte → Ergebnisse in `models/*.json`)
`realfake.py` (Echt-vs-synthetisch), `eval_dense.py`/`hybrid.py eval` (Lokalisierung), `pipeline_eval.py`, `eval_melanoma.py`, `external_check.py`
(4 externe Quellen), `fusion_check.py`, `direct_fusion.py`, `fusion_val_test.py` (Fusion Direktmodell + Glyphen), `check_integration.py`.

## Wichtige Befunde
- Regionen validiert nur für 5 ISIC-Attribute (Netzwerk, negatives Netzwerk, Streaks, Milia, Globuli); alle übrigen IDs sind synthetisch trainiert, nicht validiert.
- Keine Fusion: Direktmodell 0,9675 → zweistufig 0,9687 AUROC (praktisch kein Gewinn). Der Glyphenweg liefert Erklärung und einen Abweichungshinweis
  (Direktscore unter Schwelle, Glyphen-Score ≥ 90. Perzentil der benignen Validierungsbilder: 424 von 5 554 Bildern markiert, 5,7 % maligne, 24 von 105 übersehenen).
  Der Glyphen-Score nutzt nur die 19 validierten Kanäle (5 ISIC-Attribute, 14 Derm7pt-Kriterien); die 90-Kanal-Variante (`models/twopath_all90.json`) war etwas trennschärfer, aber nicht belegbar (`validated_score.py`).
- Die klassische Schwelle „≥ 3 Punkte“ überträgt sich nicht auf externe Daten.
- Nur Dermoskopie; Schwellen stammen aus dem v17-Split.

Python-Umgebung: `glyphs/.venv` (uv; torch, timm, scikit-learn, opencv). Das System-Python hat kein numpy.
