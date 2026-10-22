"""Markdown tables A and B for MISSION_RESULTS.md from models/mission_compare.json.

  python mission_report.py > out/mission_tables.md
"""
import json

d = json.load(open("models/mission_compare.json")); S = d["summary"]
COLS = ["R0", "R1", "R2", "R3", "R5"]
f3 = lambda x: "–" if x is None else f"{x:.3f}"
mc = lambda x: "–" if x is None else f"{x['mean']:.3f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"


def row(name, n, vals):
    print(f"| {name} | {n} | " + " | ".join(vals) + " |")


print("| Capability / metric | n | " + " | ".join(COLS) + " |\n|---|---|" + "---|" * len(COLS))
row("**C1** EASY mean image AUROC, 25 trained features (mapped ID channel)", "248 img, 25 feat", [mc(S["C1_easy_trained25"].get(r)) for r in COLS])
row("C1 EASY, 4 coarse-mapped features (never trained)", "248 img, 4 feat", [mc(S["C1_easy_coarse4"].get(r)) for r in COLS])
row("C1 EASY, ISIC-attribute channel of the mapped features", f"{sum('attr' in v for k, v in d['easy_features'].items() if v['n_pos'] >= 10)} feat", [f3(S["C1_easy_attr_channels"].get(r)) for r in COLS])
row("C1 ISIC 2018 test, image AUROC, mean of 5 attributes", "500 img", [mc(S["C1_isic_mean_image_auroc"].get(r)) for r in COLS])
row("C1 Derm7pt test, zero-shot criterion AUROC, mean", f"395 img, {S['C1_d7_mean_auroc']['R3']['n_criteria']} crit.", [mc(S["C1_d7_mean_auroc"].get(r)) for r in COLS])
row("  … criteria with an EASY-supervised ID", ", ".join(S["C1_d7_with_easy_supervised_id"]["criteria"]), [mc(S["C1_d7_with_easy_supervised_id"].get(r)) for r in COLS])
row("  … criteria without an EASY-supervised ID", ", ".join(S["C1_d7_without_easy_supervised_id"]["criteria"]), [mc(S["C1_d7_without_easy_supervised_id"].get(r)) for r in COLS])
c2 = S["C2_easy_rank"]
row("**C2** EASY median rank of mapped channel (of 71 IDs)", "25 feat", [f"{c2[r]['median_rank']:.0f}" if r in c2 else "–" for r in COLS])
row("C2 share of features where mapped channel is rank 1 / top 3", "25 feat", [f"{c2[r]['share_rank1']:.2f} / {c2[r]['share_top3']:.2f}" if r in c2 else "–" for r in COLS])
row("C2 generic-score control AUROC (mean of 71 ID scores) → AUROC − generic", "25 feat", [f"{c2[r]['mean_generic']:.3f} → {c2[r]['mean_auroc_minus_generic']:+.3f}" if r in c2 else "–" for r in COLS])
row("**C3** EASY pointing game (hit rate) vs lesion-referenced chance", "positives, 29 feat", [f"{S['C3_easy_pointing'][r]:.2f} vs {S['C3_easy_chance_lesion'][r]:.2f}" if r in S["C3_easy_pointing"] else "–" for r in COLS])
row("C3 EASY cell AUROC inside lesion + border", "29 feat", [f3(S["C3_easy_cell_auroc_in_lesion"].get(r)) for r in COLS])
row("C3 ISIC test patch IoU (threshold from cal blocks), mean 5 attr.", "500 img", [f3(S["C3_isic_mean_iou"].get(r)) for r in COLS])
row(f"C3 ISIC pointing (chance in lesion+rim {S['C3_isic_mean_pointing_chance']:.2f})", "tiles with GT", [f3(S["C3_isic_mean_pointing"].get(r)) for r in COLS])
row("**C4** EASY share of top region (≥ 50 % of map max) outside lesion + border", f"area outside ≈ {S['C4_easy_outside_area_share']['R3']:.2f}", [f3(S["C4_easy_top_region_outside_share"].get(r)) for r in COLS])
row("C4 ISIC share of thresholded mass outside lesion + border", "test tiles", [f3(S["C4_isic_mean_outside_share"].get(r)) for r in COLS])
cv = S["C5_coverage"]
for k in ("EASY", "EASY+Derm7pt", "EASY+Derm7pt+ISIC"):
    row(f"**C5** IDs with AUROC ≥ 0.80 / IDs evaluable ({k})" if k == "EASY" else f"C5 … ({k})", "of 71 IDs", [f"{cv[r][k]['n_auroc_ge_080']} / {cv[r][k]['n_evaluable']}" if r in cv else "–" for r in COLS])
ds = d["downstream"]
for kind, lab in (("rule", "zero-shot 7-point rule on glyph channels"), ("probe", "logistic probe on channel scores, fitted on Derm7pt train+valid")):
    for dset in ("Derm7pt test", "external pooled"):
        x = ds.get(kind, {}).get(dset)
        if x:
            row(f"**C6** melanoma vs nevus, {lab}: {dset}", f"{x['n_mel']} MEL / {x['n_nev']} NV",
                [f"{x['regimes'][r]['auroc']:.3f} [{x['regimes'][r]['ci'][0]:.2f}, {x['regimes'][r]['ci'][1]:.2f}]" if r in x["regimes"] else "–" for r in COLS])

print("\n\n| EASY feature | n+ / n− | R0 AUROC (rank, generic) | R2 | R3 (OOF) | R5 (OOF) | R3 − R0 | R5 − R3 |\n|---|---|---|---|---|---|---|---|")
for f, v in d["easy_features"].items():
    if v["n_pos"] < 10:
        continue
    cell = lambda r: (lambda x: f"{x['auroc']:.2f} ({x['rank_of_71'] or '–'}, {x['generic_auroc']:.2f})")(v["regimes"][r]) if r in v["regimes"] else "–"
    dd = lambda a, b: f"{v['regimes'][a]['auroc'] - v['regimes'][b]['auroc']:+.2f}"
    tag = " †" if v["coarse"] else ""
    print(f"| {f}{tag} | {v['n_pos']} / {v['n_neg']} | {cell('R0')} | {cell('R2')} | {cell('R3')} | {cell('R5')} | {dd('R3', 'R0')} | {dd('R5', 'R3')} |")

print("\n\n| EASY feature (ISIC channel) | R0 (union of IDs) | R1 real only | R2 | R3 | R5 |\n|---|---|---|---|---|---|")
for f, v in d["easy_features"].items():
    if v["n_pos"] >= 10 and "attr" in v:
        print(f"| {f} ({v['attr_channel']}) | " + " | ".join(f"{v['attr'][r]['auroc']:.2f}" if r in v["attr"] else "–" for r in ("R0", "R1", "R2", "R3", "R5")) + " |")
