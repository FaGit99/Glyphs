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
| Detektor mit Regionskarten (ViT-S/16, 14×14-Karten pro ID) | `dense.py`, `hybrid.py` (V2: Hybrid mit ISIC-2018-Masken, Kalibrierung), `hybrid3.py` (**V3**: zusätzlich EASY-Expertenmasken), `easy_data.py`, `train_dense.py` |
| Läsionssegmentierung (Zonen innen / Rand / außen) | `lesion_seg.py` |
| 7-Punkte-Kriterien (Derm7pt, Bildebene, MIL) | `d7_data.py`, `mil.py`, `train_melanoma.py`, `calibrate_mel.py` |
| Direktmodell (EVA-02) lesend aufrufen | `direct_model.py` (verifiziert: Test-AUROC 0,9675) |
| Zweipfad-Parameter | `twopath_fit.py` → `models/twopath.json` |
| Arztansicht (HTML) | `analyze.py` |

## Prüfungen (Skripte → Ergebnisse in `models/*.json`)
`realfake.py` (Echt-vs-synthetisch), `eval_dense.py`/`hybrid.py eval` (Lokalisierung), `pipeline_eval.py`, `eval_melanoma.py`, `external_check.py`
(4 externe Quellen), `easy_eval.py` / `easy_v3_pred.py` / `easy_v3_compare.py` (EASY, Out-of-fold), `fusion_check.py`, `direct_fusion.py`, `fusion_val_test.py` (Fusion Direktmodell + Glyphen), `check_integration.py`.

## Version V3 (seit 2026-10-19, aktiv)
`models/hybrid.pt` und `mel_*.pt` sind V3: Training mit synthetischen Daten + ISIC-2018-Masken + EASY-Expertenmasken (Liopyris et al. 2023, 248 Bilder, 24 ID-Kanäle;
positiv = ≥ 2 Leser, negativ = kein Leser, genau 1 Leser ignoriert; 4 grob zugeordnete Merkmale nicht trainiert), Derm7pt-Feintuning darauf neu.
**V2 ist eingefroren** in `frozen_V2/` (Modelle, JSON, READMEs, Beispielberichte, V2-Skripte); Details und Zahlen: `SITZUNG_2026-10-19_V3.md`, Vorgängerstand `README_V2.md`.
- EASY, Out-of-fold nach `group` (1–4): mittlerer AUROC über 25 trainierte Merkmale V2 0,785 → V3 0,896 (Kontrolle ohne EASY 0,781). Stärkste Gewinne: Cobblestone, Shiny white streaks, Peppering, Milky red, Linear irregular, Comma.
  Vorbehalt: Folds nicht nach Patient/Läsion getrennt, Gewinne können zu hoch sein. `models/easy_validation.json` speist die Badge „EASY-AUROC“ in der Arztansicht (nur Bildebene).
- ISIC-2018-Test, Derm7pt, extern (Melanom vs Naevus gepoolt 0,749 vs 0,741) und v17-Zweistufigkeit (0,9692 vs 0,9687) praktisch unverändert; Abweichungs-Flag jetzt 381 von 5 554, 7,1 % maligne (×3,7), 27 von 105.
- Neu erzeugen: `make_v3_dir.sh` baut `v3/` (Arbeitskopie gegen V3-Modelle); Training: `python hybrid3.py train <fold 1-4 | 0 | -1>`, Auswertung `python easy_v3_pred.py oof|noeasy|final` + `python easy_v3_compare.py`.
  Alle Trainingsdaten liegen für schnelles Training auf der GPU (Hybrid 8 s/Epoche).

## Wichtige Befunde
- Regionen validiert nur für 5 ISIC-Attribute (Netzwerk, negatives Netzwerk, Streaks, Milia, Globuli); alle übrigen IDs sind synthetisch trainiert, nicht validiert.
- Keine Fusion: Direktmodell 0,9675 → zweistufig 0,9692 AUROC (V3; V2 0,9687) (praktisch kein Gewinn). Der Glyphenweg liefert Erklärung und einen Abweichungshinweis
  (Direktscore unter Schwelle, Glyphen-Score ≥ 90. Perzentil der benignen Validierungsbilder: V3 381 von 5 554 Bildern markiert, 7,1 % maligne, 27 von 105 übersehenen; V2 424 / 5,7 % / 24).
  Der Glyphen-Score nutzt nur die 19 validierten Kanäle (5 ISIC-Attribute, 14 Derm7pt-Kriterien); die 90-Kanal-Variante (`models/twopath_all90.json`) war etwas trennschärfer, aber nicht belegbar (`validated_score.py`).
- Die klassische Schwelle „≥ 3 Punkte“ überträgt sich nicht auf externe Daten.
- Nur Dermoskopie; Schwellen stammen aus dem v17-Split.

Python-Umgebung: `glyphs/.venv` (uv; torch, timm, scikit-learn, opencv). Das System-Python hat kein numpy.
