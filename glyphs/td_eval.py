"""Target-dose experiment: number of expert masks per structure vs EASY AUROC.
Arm A = synthetic + ISIC + EASY, arm B = ISIC + EASY without synthetic (same doses, seeds and mask draws -> paired);
arm A2 (A′) = as A, but loss averaged per source (synthetic rows / real rows, --srcnorm 1), so real rows weigh as in B.

  python td_eval.py [--arms A,B,A2] [--seeds 0,1,2] [--doses 0,2,4,8,all] [--tag _x] [--no-pred]
Models: models/mission_td{n}_s{seed}_f{1..4}.pt (arm A), mission_tdB{n}_s{seed}_f{1..4}.pt (arm B, --synth 0);
dose "all": arm A = R3 (seed 0), R3_s1, mission_tdall_s2; arm B = R5 (seed 0), mission_tdBall_s1, mission_tdBall_s2.
Predictions: cache/td_pred_<name>.npz (EASY only, OOF, mission_pred.easy_pred). Protocol = easy_eval.py / mission_compare.py.
Output: models/td_dose<tag>.json, out/mission_fig_td_dose<tag>.png/.pdf; with both arms also B - A per dose (paired bootstrap).

Dose variable = number of positive training images per fold whose mask for the target channel is kept (nominal n); the
effective number of image-level supervised positives (coverage >= 1 % in some view) is reported next to it.
Uncertainty: bootstrap over images (B = 1000, paired across doses) combined with resampling of the 3 seeds per dose.
Saturation fit per feature: AUROC(n) = a - b * exp(-n / tau); n90 = n reaching 90 % of the gain from n = 0 to the
full-mask AUROC (empirical, interpolated on the fit), CI from the same bootstrap.
"""
import argparse, json, os
import numpy as np
from scipy.optimize import curve_fit
import hybrid as H
from hybrid3 import easy_targets
from mission_train import easy_subset, target_withhold

TIDS = ["DS038", "DS006", "DS037"]
FEATS = {"Regression structures : Peppering / Granularity": "Peppering",
         "Shiny white structures : Shiny white streaks": "Shiny white structures",
         "Dots : Irregular": "Dots, irregular", "Dots : Regular": "Dots, regular"}
DOSES = [0, 2, 4, 8, 16, 32, "all"]; SEEDS = [0, 1, 2]
ARMS = {"A": dict(prefix="td", label="A  synthetic + ISIC + EASY", col="R3",
                  all={0: ("R3", "cache/mission_pred_R3.npz"), 1: ("R3_s1", "cache/mission_pred_R3_s1.npz"), 2: ("tdall_s2", None)}),
        "B": dict(prefix="tdB", label="B  ISIC + EASY, no synthetic", col="R5",
                  all={0: ("R5", "cache/mission_pred_R5.npz"), 1: ("tdBall_s1", None), 2: ("tdBall_s2", None)}),
        "A2": dict(prefix="tdS", label="A′  synthetic + ISIC + EASY, loss per source", col="R0",
                   all={0: ("tdSall_s0", None), 1: ("tdSall_s1", None)})}
B = 1000


def name(arm, n, s):
    return ARMS[arm]["all"][s][0] if n == "all" else f"{ARMS[arm]['prefix']}{n}_s{s}"


def predict(nm):
    out = f"cache/td_pred_{nm}.npz"
    if os.path.exists(out):
        return
    import mission_pred as MP
    z = np.load("cache/easy_v3.npz"); pi, pm = MP.easy_pred("oof:models/mission_" + nm + "_f{g}.pt", z["views"], z["group"])
    np.savez_compressed(out, easy_pi=pi, easy_pm=pm); print("wrote", out, flush=True)


def pred_path(arm, n, s):
    c = ARMS[arm]["all"][s][1]
    return c if n == "all" and c else f"cache/td_pred_{name(arm, n, s)}.npz"


def effective_n():
    """mean over folds of image-level supervised positives per target channel and nominal dose"""
    ids = H.id_list(); T, M, yi, vi, grp, V = easy_targets(ids); out = {}
    for n in DOSES:
        per = {t: [] for t in TIDS}
        for f in (1, 2, 3, 4):
            use = easy_subset(grp, f, 1.0); v = vi.copy()
            if n != "all":
                for c, wi in target_withhold(use, ids, TIDS, n, f, 0).items():
                    v[wi, :, c] = 0
            sup = ((yi[use] > 0) & (v[use] > 0)).any(1).sum(0)
            for t in TIDS:
                per[t].append(int(sup[ids.index(t)]))
        out[str(n)] = {t: float(np.mean(per[t])) for t in TIDS}
    return out


def sat(n, a, b, tau):
    return a - b * np.exp(-n / tau)


def fit_n90(x, y):
    """fit on dose points (x = effective n); n90 = n where fit reaches y0 + 0.9 * (y_full - y0)"""
    try:
        (a, b, tau), _ = curve_fit(sat, x, y, p0=(y[-1], max(y[-1] - y[0], 1e-3), 5.), bounds=([0, -1, .3], [1.5, 2, 500]), maxfev=5000)
    except RuntimeError:
        return np.nan, (np.nan,) * 3
    y0, yf = sat(0, a, b, tau), sat(x[-1], a, b, tau)
    if yf - y0 <= .01:                                   # no gain -> n90 undefined
        return np.nan, (a, b, tau)
    return float(-tau * np.log(1 - .9 * (1 - np.exp(-x[-1] / tau)))), (a, b, tau)


def main(a):
    global SEEDS, DOSES
    SEEDS = [int(x) for x in a.seeds.split(",")]; arms = a.arms.split(",")
    DOSES = [d if d == "all" else int(d) for d in a.doses.split(",")]
    import mission_compare as MC                         # EASY protocol, AUROC helpers, pointing game
    keys = [(arm, n, s) for arm in arms for n in DOSES for s in SEEDS]
    if not a.no_pred:
        for arm, n, s in keys:
            if pred_path(arm, n, s).startswith("cache/td_pred_"):
                predict(name(arm, n, s))
    eff = effective_n(); print("effective n:", json.dumps(eff), flush=True)
    IMG = {}
    for k in keys:
        z = np.load(pred_path(*k)); MC.P["|".join(map(str, k))] = dict(easy_pm=z["easy_pm"])
        IMG[k] = z["easy_pm"].astype(np.float32).max((1, 3, 4))                                 # [248,76]
    ref = {r: MC.IMGS[r] for r in ("R0", "R2ctrl") if r in MC.IMGS}
    rng = np.random.default_rng(0); res = {"effective_n": eff, "seeds": SEEDS, "arms": {arm: {} for arm in arms}, "diffs": {}}
    nA, nD, nS = len(arms), len(DOSES), len(SEEDS)
    for f, short in FEATS.items():
        fi = MC.FEATS.index(f); pos = MC.NRD[:, fi] >= 2; neg = MC.NRD[:, fi] == 0; use = pos | neg; y = pos[use]
        idc, _ = MC.easy_channels(f); t = MC.IDS[idc[0]]
        S = np.stack([IMG[k][use][:, idc].max(1) for k in keys])
        pt = MC.auc_rows(y, S).reshape(nA, nD, nS)
        bs = np.zeros((B, nA, nD, nS)); bseed = rng.integers(0, nS, (B, 1, nD, nS))             # same seed resample in both arms (paired draws)
        for b in range(B):
            ix = rng.integers(0, len(y), len(y)); bs[b] = MC.auc_rows(y[ix], S[:, ix]).reshape(nA, nD, nS)
        bm = np.take_along_axis(bs, np.broadcast_to(bseed, bs.shape), 3).mean(3)                  # [B, arms, doses]
        x = np.array([eff[str(n)][t] for n in DOSES])
        for ai, arm in enumerate(arms):
            m = pt[ai].mean(1); n90, par = fit_n90(x, m); bn90 = np.array([fit_n90(x, bm[b, ai])[0] for b in range(B)])
            gen, rank, sp = [], [], {}
            for n in DOSES:
                for s in SEEDS:
                    I = IMG[(arm, n, s)][use]; allc = MC.auc_rows(y, I[:, :MC.K].T); a_ = MC.auc(y, I[:, idc].max(1))
                    rank.append(int(1 + (allc > a_).sum())); gen.append(float(MC.auc(y, np.nanmean(I[:, :MC.K], 1))))
                pg = [MC.easy_spatial(f"{arm}|{n}|{s}", f) for s in SEEDS]
                sp[str(n)] = dict(pointing=float(np.mean([q["pointing"] for q in pg])), chance_lesion=pg[0]["chance_lesion"],
                                  cell_auroc_in_lesion=float(np.mean([q["cell_auroc_in_lesion"] for q in pg])))
            res["arms"][arm][f] = dict(
                short=short, channel=t, n_pos=int(pos.sum()), n_neg=int(neg.sum()), doses=[str(n) for n in DOSES], effective_n=x.tolist(),
                auroc_mean=m.tolist(), auroc_seeds=pt[ai].tolist(), ci=[MC.ci(bm[:, ai, j]) for j in range(nD)], seed_sd=pt[ai].std(1, ddof=1).tolist(),
                step_gain=[None] + [[float(m[j] - m[j - 1]), *MC.ci(bm[:, ai, j] - bm[:, ai, j - 1])] for j in range(1, nD)],
                rank_of_71_median=np.median(np.array(rank).reshape(nD, -1), 1).tolist(),
                generic_auroc_mean=np.array(gen).reshape(nD, -1).mean(1).tolist(),
                fit=dict(a=par[0], b=par[1], tau=par[2]), n90=n90, n90_ci=MC.ci(bn90), n90_undefined_share=float(np.isnan(bn90).mean()),
                spatial=sp, reference={r: float(MC.auc(y, ref[r][use][:, idc].max(1))) for r in ref})
            r_ = res["arms"][arm][f]
            print(f"{arm} {short:24s} n+={pos.sum():3d}  " + "  ".join(f"{d}:{v:.3f}" for d, v in zip(DOSES, m)) +
                  f"  n90={n90:.1f} [{r_['n90_ci'][0]:.1f}, {r_['n90_ci'][1]:.1f}]", flush=True)
        for p_, q_ in (("B", "A"), ("A2", "A"), ("A2", "B")):
            if p_ in arms and q_ in arms:
                ip, iq = arms.index(p_), arms.index(q_); d = pt[ip].mean(1) - pt[iq].mean(1)
                res["diffs"].setdefault(f"{p_}-{q_}", {})[f] = [[float(d[j]), *MC.ci(bm[:, ip, j] - bm[:, iq, j])] for j in range(nD)]
                print(f"  {p_}-{q_} {short:20s} " + "  ".join(f"{n}:{v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]"
                                                         for n, v in zip(DOSES, res["diffs"][f"{p_}-{q_}"][f])), flush=True)
    out = f"models/td_dose{a.tag}.json"; json.dump(res, open(out, "w"), indent=1, default=float); print("wrote", out)
    figure(res, a.tag)


def figure(res, tag=""):
    import matplotlib.pyplot as plt
    import mission_figures as MF
    feats = list(FEATS); arms = list(res["arms"])
    fig, axs = plt.subplots(1, len(feats), figsize=(12, 3.4), sharey=True)
    for ax, f in zip(axs, feats):
        MF.clean(ax, "y"); first = res["arms"][arms[0]][f]
        for ai, arm in enumerate(arms):
            r = res["arms"][arm][f]; c = MF.COL[ARMS[arm]["col"]]; xs = np.log2(np.array(r["effective_n"]) + 1)
            lo, hi = np.array(r["ci"]).T
            ax.fill_between(xs, lo, hi, color=c, alpha=.10, lw=0)
            for sd in np.array(r["auroc_seeds"]).T:
                ax.plot(xs, sd, "o", ms=3, color=c, alpha=.35, mew=0)
            ax.plot(xs, r["auroc_mean"], "-o", color=c, lw=2, ms=6, mec=MF.SURF, mew=1.5, zorder=3, label=ARMS[arm]["label"])
            if np.isfinite(r["n90"]):
                ax.plot([np.log2(r["n90"] + 1)], [.985 - .03 * ai], "v", ms=6, color=c, mec=MF.SURF, mew=1, clip_on=False)
                ax.text(np.log2(r["n90"] + 1) + .12, .985 - .03 * ai, f"n₉₀ ≈ {r['n90']:.0f}", fontsize=7, color=MF.INK2, va="center")
        for lab, key, ls in (("synthetic only (R0)", "R0", ":"), ("no EASY (R2ctrl)", "R2ctrl", "--")):
            if key in first["reference"]:
                ax.axhline(first["reference"][key], color=MF.MUTED, lw=.8, ls=ls, zorder=1, label=lab)
        tk = [0, 2, 4, 8, 16, 32, 64]; ax.set_xticks(np.log2(np.array(tk) + 1)); ax.set_xticklabels(tk)
        ax.set_title(f"{first['short']}  (n+ = {first['n_pos']})"); ax.set_xlabel("expert-masked positive images per fold")
        ax.set_ylim(.4, 1.0); ax.axhline(.5, color=MF.AXIS, lw=.8, zorder=1)
    axs[0].set_ylabel("EASY image AUROC (OOF)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(.5, 1.07), fontsize=7.5)
    fig.text(0, -.06, f"Line: mean of {len(res['seeds'])} seeds/draws (same mask draws in both arms); small dots: single seeds; band: 95 % CI "
             "(bootstrap over images and seeds); triangle: n₉₀ (90 % of the gain from 0 to all masks). Dose = images whose mask for the "
             "target channel is kept; all other EASY labels unchanged.", fontsize=7, color=MF.MUTED, wrap=True)
    fig.tight_layout(); MF.save(fig, "td_dose" + tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--no-pred", action="store_true"); p.add_argument("--arms", default="A")
    p.add_argument("--seeds", default="0,1,2"); p.add_argument("--doses", default="0,2,4,8,16,32,all"); p.add_argument("--tag", default=""); main(p.parse_args())
