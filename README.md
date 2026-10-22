# Glyphs: synthetic dermoscopic structures for an explainable second opinion

Code and results for the preprint

> F. Wolz. *Synthetic Glyphs for Dermatologist-Oriented Structure Detection: An Explainable Second Machine-Learning Opinion on Dermoscopic Lesions.* Preprint, version 8, 2026. Not peer reviewed.
> PDF: [`glyphs/Papers/Synthetic Glyphs for Dermatologist-Oriented Structure Detection V8.pdf`](glyphs/Papers/Synthetic%20Glyphs%20for%20Dermatologist-Oriented%20Structure%20Detection%20V8.pdf)

71 terms of the International Dermoscopy Society terminology are rendered as parametric *glyphs* on real lesion-free skin, which gives synthetic training scenes with exact region labels. A ViT-S/16 detector with an image head and a region head is trained on these scenes alone and mixed with real masks (ISIC 2018 Task 2, EASY expert annotations), and is evaluated on expert-annotated, checklist and external data. The detected structures are shown next to a direct classifier as a two-path report.

**Research prototype only. Not a medical device and not for diagnosis.**

## Repository layout

Everything lives in `glyphs/`. Supplementary Table S2 of the paper maps every result to the script and output file behind it.

| Step | Scripts | Outputs |
|---|---|---|
| Glyphs and synthetic scenes | `geometry.py`, `structures.py`, `render.py`, `skin.py`, `compose.py`, `gen_dataset.py`, `build_register.py`, `taxonomy.csv`, `registry.csv` | `data/synth_*.npz` (not included), `out/register/` |
| Real data caches | `prep_real.py`, `prep_real_masks.py`, `easy_data.py`, `d7_data.py` | `cache/` (not included) |
| Detector, hybrid and mixed model | `dense.py`, `hybrid.py`, `hybrid3.py` | `models/hybrid_eval*.json`, `models/hybrid_calibration*.json` |
| EASY evaluation | `easy_eval.py`, `easy_v3_pred.py`, `easy_v3_compare.py` | `models/easy_v3_compare.json` |
| Training regimes R0–R5 and image-subsample dose R4 | `mission_train.py`, `mission_pred.py`, `mission_ext.py`, `mission_compare.py`, `mission_figures.py`, `mission_report.py` | `models/mission_compare.json`, `MISSION_RESULTS.md`, `out/mission_fig_*.png` |
| Per-structure dose experiment R6 | `mission_train.py --tids --tdose --srcnorm`, `td_eval.py` | `models/td_dose.json`, `models/td_dose_A2.json` |
| Lesion segmentation | `lesion_seg.py`, `pipeline_eval.py` | `models/pipeline_eval.json` |
| Checklist channels (Derm7pt) | `mil.py`, `train_melanoma.py`, `calibrate_mel.py`, `eval_melanoma.py` | `models/melanoma_eval.json`, `models/mel_calibration.json` |
| External tests, direct classifier, two paths | `external_check.py`, `direct_model.py`, `fusion_val_test.py`, `twopath_fit.py` | `models/external_check.json`, `models/fusion_val_test.json`, `models/twopath.json` |
| Report | `analyze.py`, `check_integration.py` | HTML report |
| Paper | `Papers/build_v8/build_v8.py`, `Papers/build_v8/v8_figs.py` | the V8 PDF |

`frozen_V2/` holds the scripts, results and example reports of the earlier V2 model (without weights). `glyphs/README.md` and `glyphs/README_V2.md` are the original working notes (German).

## What is not included

- **Model weights** (`*.pt`) and the generated **synthetic scenes** (`data/`): several GB, above GitHub’s file-size limit. The scenes can be regenerated with `gen_dataset.py` using fixed seeds; the models can be retrained with the scripts above.
- **Real datasets** and their caches (`cache/`, `external/`). They are publicly available and must be obtained from their sources under their own licences: HAM10000 (doi:10.7910/DVN/DBW86T); ISIC 2018, ISIC 2019, ISIC 2020 and BCN20000 via the [ISIC Archive](https://challenge.isic-archive.com/data); [Derm7pt](https://derm.cs.sfu.ca); and the [EASY expert-annotation data](https://github.com/ISIC-Research/expert-annotation-agreement-data).
- The direct classifier (EVA-02) used for the two-path report, which belongs to a separate project.

Images under `out/register/` and in `frozen_V2/` contain skin windows from HAM10000 and example images from the ISIC Archive; these remain subject to their original licences.

## Setup

```bash
cd glyphs
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

A CUDA GPU is needed for training (one training run takes a few minutes on a recent workstation GPU).

## Licence

The code and the author’s own results and texts are released under the [MIT Licence](LICENSE). Images derived from HAM10000 or the ISIC Archive (in `glyphs/out/register/`, `glyphs/frozen_V2/` and `glyphs/Papers/build_v8/`) are not covered by it and remain subject to the licences of their source datasets.

## Use of AI tools

An AI assistant (Claude, Anthropic) was used throughout this work under the author’s direction; see the declarations in the paper.
