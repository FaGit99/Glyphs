"""End-to-end evaluation of analyze.py on the ISIC test blocks (500 images never used for training or calibration).

Per attribute: whole-image AUC of the calibrated max probability vs. "mask covers >= 1 % of the image";
sensitivity / specificity of a (non-weak) detection; share of detected area outside lesion+border with the PREDICTED lesion
vs. the ground-truth lesion; lesion Dice of the segmentation inside the full pipeline; frequency of warnings.
"""
import json, time
import cv2, numpy as np
from sklearn.metrics import roc_auc_score
import analyze as A, hybrid as H

RAW = "/mnt/samsung_data/mole_data/data/raw/isic2018/"
names = json.load(open("cache/isic18_names.json")); sp = H.real_split()["test"]
rows = []; t0 = time.time()
for n, i in enumerate(sp):
    nm = names[i]; R = A.analyze(RAW + "ISIC2018_Task1-2_Training_Input/" + nm)
    h, w = R["les"].shape
    gt_les = (cv2.resize(cv2.imread(RAW + f"ISIC2018_Task1_Training_GroundTruth/{nm[:-4]}_segmentation.png", 0), (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    gz = A.zone_masks(gt_les)
    pres = []
    for a in H.ATTRS:
        m = cv2.imread(RAW + f"ISIC2018_Task2_Training_GroundTruth_v3/{nm[:-4]}_attribute_{a}.png", 0)
        pres.append(float((m > 127).mean()))
    inter = (R["les"] & gt_les).sum()
    row = dict(name=nm, dice=2 * inter / max(R["les"].sum() + gt_les.sum(), 1), pres=pres, attr_max=[R["attr_max"][a] for a in H.ATTRS],
               det={}, flags=sum(len(it["flags"]) > 0 for it in R["items"]), n_items=len(R["items"]))
    for it in R["items"]:
        if it["status"] == "region":
            r = it["reg"]; k = max(int(r.sum()), 1)
            row["det"][it["key"]] = dict(weak=bool(it["weak"]), out_pred=float((r & R["outside"]).sum() / k), out_gt=float((r & gz[2]).sum() / k), flagged=len(it["flags"]) > 0)
    rows.append(row)
    if n % 100 == 99:
        print(f"{n + 1}/{len(sp)} {time.time() - t0:.0f}s", flush=True)
d = np.array([r["dice"] for r in rows])
print(f"\nlesion Dice inside the pipeline: mean {d.mean():.3f}, median {np.median(d):.3f}; Dice<0.5 in {(d < .5).mean():.1%} of images")
print(f"{'attribute':17s} {'#pos/#neg':>10s} {'AUC':>6s} {'sens':>6s} {'spec':>6s} {'detected(non-weak)':>19s} {'outside-share pred/gt zones':>28s} {'warn-rate':>9s}")
res = {}
for k, a in enumerate(H.ATTRS):
    y = np.array([r["pres"][k] >= .01 for r in rows]); s = np.array([r["attr_max"][k] for r in rows])
    det = np.array([a in r["det"] and not r["det"][a]["weak"] for r in rows])
    ds = [r["det"][a] for r in rows if a in r["det"] and not r["det"][a]["weak"]]
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    sens = (det & y).sum() / max(y.sum(), 1); spec = (~det & ~y).sum() / max((~y).sum(), 1)
    op = np.mean([x["out_pred"] for x in ds]) if ds else float("nan"); og = np.mean([x["out_gt"] for x in ds]) if ds else float("nan")
    wr = np.mean([x["flagged"] for x in ds]) if ds else float("nan")
    print(f"{a:17s} {y.sum():4d}/{(~y).sum():<5d} {auc:6.3f} {sens:6.2f} {spec:6.2f} {det.sum():19d} {op:13.3f} /{og:7.3f} {wr:12.2f}")
    res[a] = dict(auc=float(auc), sens=float(sens), spec=float(spec), n_pos=int(y.sum()), n_det=int(det.sum()), out_pred=float(op), out_gt=float(og))
json.dump(dict(per_attr=res, lesion_dice_mean=float(d.mean()), lesion_dice_median=float(np.median(d))), open("models/pipeline_eval.json", "w"), indent=1)
