"""Figures for MISSION_RESULTS.md from models/mission_compare.json (matplotlib) -> out/mission_fig_*.png and .pdf

  python mission_figures.py
Colour follows the regime in every figure: R3 blue, R5 orange, R0 aqua (first three slots of the reference categorical palette,
validated all-pairs in light mode); R1/R2 neutral grey. Text always in ink colours, never in series colours.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open("models/mission_compare.json")); S = d["summary"]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"R3": "#2a78d6", "R5": "#eb6834", "R0": "#1baf7a", "R2": "#9b9a94", "R1": "#c3c2b7", "R2ctrl": "#9b9a94"}
NAME = {"R0": "R0  synthetic only", "R1": "R1  ISIC only", "R2": "R2  synthetic + ISIC", "R3": "R3  synthetic + ISIC + EASY", "R5": "R5  ISIC + EASY, no synthetic"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": AXIS, "axes.linewidth": .8, "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": False, "legend.frameon": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.8, ls="-"); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"out/mission_fig_{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", f"out/mission_fig_{name}.png/.pdf")


def dot(ax, y, m, lo, hi, c, ms=7, z=3):
    ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=z)
    ax.plot([m], [y], "o", ms=ms, color=c, mec=SURF, mew=1, zorder=z + 1)


# ---------------------------------------------------------------- 1 dose curve
def fig_dose():
    rr = ["R2ctrl", "R4_25", "R4_50", "R4_75", "R3"]; xs = np.array([0, 46, 93, 140, 186])
    tr, co = S["C1_easy_trained25"], S["C1_easy_coarse4"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0)); clean(ax, "y")
    for m, c, lab in ((tr, COL["R3"], "25 trained features"), (co, MUTED, "4 coarse-mapped features (never trained)")):
        y = np.array([m[r]["mean"] for r in rr]); lo = [m[r]["ci"][0] for r in rr]; hi = [m[r]["ci"][1] for r in rr]
        ax.fill_between(xs, lo, hi, color=c, alpha=.10, lw=0)
        ax.plot(xs, y, "-", color=c, lw=2, solid_joinstyle="round"); ax.plot(xs, y, "o", ms=7, color=c, mec=SURF, mew=2, label=lab)
    for x, r in zip(xs, rr):
        ax.annotate(f"{tr[r]['mean']:.3f}", (x, tr[r]["mean"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    for r, x, mk, lab in (("R0", -9, "D", "R0 synthetic only (dense_v1), no real masks"), ("R5", 195, "s", "R5 ISIC + EASY, no synthetic"), ("R3_s1", 177, "o", "R3, second seed")):
        m = tr[r]; c = COL.get(r, COL["R3"])
        ax.plot([x, x], m["ci"], color=c, lw=2, solid_capstyle="round")
        ax.plot([x], [m["mean"]], mk, ms=7, color=SURF if r == "R3_s1" else c, mec=c if r == "R3_s1" else SURF, mew=2, label=lab)
    ax.annotate("", xy=(93, .866), xytext=(93, .80), arrowprops=dict(arrowstyle="-", color=AXIS, lw=.8))
    ax.text(97, .812, "mean curve flattens after ≈ 90 images\n(per structure: see R6)", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(xs, [f"{x}\n{p}" for x, p in zip(xs, ["no EASY", "25 %", "50 %", "75 %", "all (V3)"])])
    ax.set_xlim(-22, 208); ax.set_ylim(.40, 1.0); ax.axhline(.5, color=AXIS, lw=.8, zorder=1)
    ax.text(-20, .505, "chance", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlabel("EASY expert-masked training images per fold (synthetic + ISIC always included)")
    ax.set_ylabel("mean image AUROC, held-out EASY group")
    ax.set_title("How many expert-masked images are sufficient? (R4 dose curve)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, handletextpad=.4, borderaxespad=0)
    fig.text(0, -.09, "Bands and bars: 95 % bootstrap CI over images (1 000 resamples). Folds by EASY group, not by patient.", fontsize=7, color=MUTED)
    save(fig, "dose_curve")


# ---------------------------------------------------------------- 2 regimes x capabilities (small multiples)
def fig_regimes():
    regs = ["R0", "R1", "R2", "R3", "R5"]; ds = d["downstream"]
    panels = [
        ("EASY: 25 trained features\n", S["C1_easy_trained25"]),
        ("EASY: 4 coarse features\n(never trained)", S["C1_easy_coarse4"]),
        ("ISIC 2018 test: 5 attributes\n", S["C1_isic_mean_image_auroc"]),
        ("Derm7pt: criteria with an\nEASY-masked ID", S["C1_d7_with_easy_supervised_id"]),
        ("Derm7pt: no EASY-masked ID\n(streaks, regression, arborizing)", S["C1_d7_without_easy_supervised_id"]),
        ("Melanoma vs nevus, 4 external sets\n(frozen channels, Derm7pt-fitted probe)", {r: dict(mean=v["auroc"], ci=v["ci"]) for r, v in ds["probe"]["external pooled"]["regimes"].items()}),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.3), sharey=True); fig.subplots_adjust(wspace=.08, hspace=.62)
    ys = np.arange(len(regs))[::-1]
    for ax, (title, m) in zip(axs.ravel(), panels):
        clean(ax, "x"); ax.axvline(.5, color=AXIS, lw=.8, zorder=1)
        for y, r in zip(ys, regs):
            if r in m and isinstance(m[r], dict):
                dot(ax, y, m[r]["mean"], *m[r]["ci"], COL[r])
                ax.text(m[r]["ci"][1] + .015, y, f"{m[r]['mean']:.2f}", va="center", fontsize=7.5, color=INK2)
            else:
                ax.text(.52, y, "no channel", va="center", fontsize=7, color=MUTED)
        ax.set_xlim(.3, 1.0); ax.set_xticks([.4, .5, .6, .7, .8, .9, 1.0]); ax.set_title(title, fontsize=8.5)
        ax.set_yticks(ys, [NAME[r] for r in regs]); ax.set_ylim(-.6, len(regs) - .4)
    for ax in axs[1]:
        ax.set_xlabel("AUROC (dot = mean, line = 95 % CI)")
    fig.suptitle("Presence recognition by training regime: real masks lift recognition; synthesis matters where no mask exists", x=.0, ha="left", fontsize=10, fontweight="bold", y=1.0)
    save(fig, "regimes")


# ---------------------------------------------------------------- 3 per EASY feature
def fig_features():
    f = d["easy_features"]; rows = [(k, v) for k, v in f.items() if v["n_pos"] >= 10]
    rows.sort(key=lambda kv: (kv[1]["coarse"], kv[1]["regimes"]["R3"]["auroc"]))
    fig, ax = plt.subplots(figsize=(7.4, 8.2)); clean(ax, "x")
    lab = []
    for y, (k, v) in enumerate(rows):
        a = {r: v["regimes"][r]["auroc"] for r in ("R0", "R2", "R3", "R5")}
        if v["coarse"]:
            ax.axhspan(y - .5, y + .5, color="#f0efec", lw=0, zorder=0)
        ax.plot([a["R0"], a["R3"]], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=1)
        for r, mk in (("R2", "|"), ("R0", "o"), ("R5", "s"), ("R3", "o")):
            ax.plot([a[r]], [y], mk, ms=(9 if mk == "|" else 7), color=COL[r], mec=COL[r] if mk == "|" else SURF, mew=(2 if mk == "|" else 1.5), zorder=3 if r != "R3" else 4)
        grp, nm = k.split(" : ", 1)
        nm = nm.replace(" / Reticulation", "").replace("Pigment Network", "pigment network").replace(" / Polygons / Zig-zag pattern", " / polygons").replace(" : NOS", "")
        if nm in ("Regular", "Irregular") or grp == "Vessels":
            nm = f"{grp.split(' /')[0]}: {nm.lower()}"
        lab.append(f"{nm}{' †' if v['coarse'] else ''}  ({v['n_pos']})")
    ax.set_yticks(range(len(rows)), lab, fontsize=7.8); ax.set_ylim(-.7, len(rows) - .3)
    ax.axvline(.5, color=AXIS, lw=.8); ax.axvline(.8, color=GRID, lw=.8)
    ax.set_xlim(.25, 1.02); ax.set_xlabel("image AUROC on EASY (R3 and R5 out-of-fold)")
    h = [Line2D([], [], marker="o", ls="", ms=7, color=COL["R0"], mec=SURF, label=NAME["R0"]), Line2D([], [], marker="|", ls="", ms=9, mew=2, color=COL["R2"], label=NAME["R2"]),
         Line2D([], [], marker="o", ls="", ms=7, color=COL["R3"], mec=SURF, label=NAME["R3"]), Line2D([], [], marker="s", ls="", ms=7, color=COL["R5"], mec=SURF, label=NAME["R5"])]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5, handletextpad=.3, columnspacing=1)
    ax.set_title("Per EASY feature: synthetic only → with expert masks", pad=34)
    fig.text(0, .055, "(n) = positive images (≥ 2 readers). † shaded = coarse mapping, never trained with EASY. Grey line: R0 → R3 gain.", fontsize=7, color=MUTED)
    save(fig, "easy_features")


# ---------------------------------------------------------------- 4 coverage + localisation
def fig_coverage_loc():
    regs = ["R0", "R2", "R3", "R5"]; cv = S["C5_coverage"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1])); fig.subplots_adjust(wspace=.55)
    ys = np.arange(len(regs))[::-1]
    ax = axs[0]; clean(ax, "x")
    for y, r in zip(ys, regs):
        n, e = cv[r]["EASY+Derm7pt+ISIC"]["n_auroc_ge_080"], cv[r]["EASY+Derm7pt+ISIC"]["n_evaluable"]
        ax.barh(y, 71, height=.42, color="#f0efec", lw=0); ax.barh(y, e, height=.42, color="#e1e0d9", lw=0)
        ax.barh(y, n, height=.42, color=COL[r], lw=0); ax.text(n + 1, y, f"{n}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [NAME[r] for r in regs]); ax.set_xlim(0, 71); ax.set_xticks([0, 10, 20, 29, 40, 50, 60, 71])
    ax.set_xlabel("taxonomy IDs (of 71)\ncolour: AUROC ≥ 0.80 · light grey: evaluable (29) · lightest: no real data (42)", fontsize=8)
    ax.set_title("C5 coverage: IDs with real-data AUROC ≥ 0.80", fontsize=8.5)
    ax = axs[1]; clean(ax, "x")
    for y, r in zip(ys, regs):
        p, c = S["C3_easy_pointing"][r], S["C3_easy_chance_lesion"][r]
        ax.plot([c, p], [y, y], color=AXIS, lw=2, solid_capstyle="round")
        ax.plot([c], [y], "|", ms=10, mew=2, color=MUTED); ax.plot([p], [y], "o", ms=7, color=COL[r], mec=SURF, mew=2)
        ax.text(p + .02, y, f"{p:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, []); ax.set_xlim(.3, .9); ax.set_xlabel("hit rate\n(| = lesion-referenced chance)", fontsize=8)
    ax.set_title("C3 pointing game on EASY", fontsize=8.5)
    ax = axs[2]; clean(ax, "x")
    for y, r in zip(ys, regs + []):
        v = S["C3_isic_mean_iou"][r]; ax.barh(y, v, height=.42, color=COL[r], lw=0); ax.text(v + .008, y, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    v = S["C3_isic_mean_iou"]["R1"]; ax.axvline(v, color=MUTED, lw=.8)
    ax.set_yticks(ys, []); ax.set_xlim(0, .42); ax.set_xlabel(f"patch IoU, mean of 5 attributes\n(vertical line: R1 ISIC only, {v:.2f})", fontsize=8); ax.set_title("C3 boundaries on ISIC 2018 test", fontsize=8.5)
    save(fig, "coverage_localisation")


if __name__ == "__main__":
    fig_dose(); fig_regimes(); fig_features(); fig_coverage_loc()
