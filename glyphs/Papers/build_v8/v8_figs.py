"""Publication figures for paper V8 (no in-figure headlines, white background, serif text).
Run from glyphs/:  .venv/bin/python <this> <outdir>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

OUT = sys.argv[1]
d = json.load(open("models/mission_compare.json")); S = d["summary"]
INK, INK2, MUTED, GRID, AXIS = "#111111", "#444444", "#888888", "#e4e4e4", "#bdbdbd"
COL = {"R3": "#2a78d6", "R5": "#eb6834", "R0": "#1baf7a", "R2": "#8f8f8f"}
NAME = {"R0": "R0 synthetic only", "R2": "R2 synthetic + ISIC", "R3": "R3 synthetic + ISIC + EASY", "R5": "R5 ISIC + EASY, no synthetic"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "font.size": 9,
                     "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": .8,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.facecolor": "white"})


def clean(ax, grid):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=.7); ax.set_axisbelow(True); ax.tick_params(length=2)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_features():
    f = d["easy_features"]; rows = [(k, v) for k, v in f.items() if v["n_pos"] >= 10 and not v["coarse"]]
    rows.sort(key=lambda kv: kv[1]["regimes"]["R0"]["auroc"])
    fig, ax = plt.subplots(figsize=(6.3, 6.4)); clean(ax, "x")
    lab = []
    for y, (k, v) in enumerate(rows):
        a = {r: v["regimes"][r]["auroc"] for r in ("R0", "R2", "R3", "R5")}
        ax.plot([a["R0"], a["R3"]], [y, y], color=AXIS, lw=1.8, solid_capstyle="round", zorder=1)
        for r, mk in (("R2", "|"), ("R5", "s"), ("R0", "o"), ("R3", "o")):
            ax.plot([a[r]], [y], mk, ms=(8 if mk == "|" else 5.5), color=COL[r], mec=COL[r] if mk == "|" else "white",
                    mew=(1.8 if mk == "|" else .8), zorder=4 if r == "R3" else 3)
        grp, nm = k.split(" : ", 1)
        nm = nm.replace(" / Reticulation", "").replace("Pigment Network", "pigment network").replace(" / Polygons / Zig-zag pattern", " / polygons").replace(" : NOS", "")
        if nm in ("Regular", "Irregular") or grp == "Vessels":
            nm = f"{grp.split(' /')[0]}: {nm.lower()}"
        lab.append(f"{nm[0].upper() + nm[1:]} ({v['n_pos']})")
    ax.set_yticks(range(len(rows)), lab, fontsize=8); ax.set_ylim(-.7, len(rows) - .3)
    ax.axvline(.5, color=AXIS, lw=.8, ls="--"); ax.set_xlim(.35, 1.02)
    ax.set_xlabel("Image-level AUROC on EASY (out-of-fold for R3 and R5)")
    h = [Line2D([], [], marker="o", ls="", ms=5.5, color=COL["R0"], mec="white", label=NAME["R0"]),
         Line2D([], [], marker="|", ls="", ms=8, mew=1.8, color=COL["R2"], label=NAME["R2"]),
         Line2D([], [], marker="o", ls="", ms=5.5, color=COL["R3"], mec="white", label=NAME["R3"]),
         Line2D([], [], marker="s", ls="", ms=5.5, color=COL["R5"], mec="white", label=NAME["R5"])]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, handletextpad=.3, columnspacing=1.2)
    save(fig, "fig2_features")
    return len(rows)


def fig_dose():
    rr = ["R2ctrl", "R4_25", "R4_50", "R4_75", "R3"]; xs = np.array([0, 46, 93, 140, 186])
    tr = S["C1_easy_trained25"]
    fig, ax = plt.subplots(figsize=(6.3, 3.4)); clean(ax, "y")
    y = np.array([tr[r]["mean"] for r in rr]); lo = [tr[r]["ci"][0] for r in rr]; hi = [tr[r]["ci"][1] for r in rr]
    ax.fill_between(xs, lo, hi, color=COL["R3"], alpha=.13, lw=0)
    ax.plot(xs, y, "-o", color=COL["R3"], lw=1.8, ms=5.5, mec="white", mew=.8, label="R2 → R4 → R3 (synthetic + ISIC + n EASY images)")
    for x, v in zip(xs, y):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    for r, ls, txt in (("R0", ":", "R0 synthetic only"), ("R5", "--", "R5 real masks only, no synthetic")):
        m = tr[r]["mean"]; ax.axhline(m, color=COL[r], lw=1, ls=ls)
        ax.text(188, m + .004, f"{txt} ({m:.3f})", ha="right", va="bottom", fontsize=8, color=INK2)
    s1 = tr["R3_s1"]; ax.plot([179], [s1["mean"]], "o", ms=5.5, mfc="white", mec=COL["R3"], mew=1.2, label=f"R3, second seed ({s1['mean']:.3f})")
    ax.set_xticks(xs, ["0", "46\n(25 %)", "93\n(50 %)", "140\n(75 %)", "186\n(100 %)"])
    ax.set_xlim(-8, 194); ax.set_ylim(.70, .93)
    ax.set_xlabel("Dermatologist-masked EASY training images per fold")
    ax.set_ylabel("Mean image-level AUROC\n(25 EASY features)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, .12), fontsize=8)
    save(fig, "fig4_dose")


def fig_regions():
    im = Image.open("out/hybrid_examples_v3final.png").convert("RGB"); h = im.height // 10
    rows = [im.crop((0, i * h, im.width, (i + 1) * h)) for i in range(10)]
    gap = 24; W = im.width * 2 + gap
    out = Image.new("RGB", (W, h * 5), "white")
    for i in range(5):
        out.paste(rows[2 * i], (0, i * h)); out.paste(rows[2 * i + 1], (im.width + gap, i * h))
    out.save(f"{OUT}/fig3_regions.png", dpi=(300, 300))


def fig_target_dose():
    """Figure 5: per-structure dose experiment R6 (arms A, B from td_dose.json, 3 seeds; A′ from td_dose_A2.json, 2 seeds)"""
    T = json.load(open("models/td_dose.json")); T2 = json.load(open("models/td_dose_A2.json"))
    feats = list(T["arms"]["A"])
    arms = [(T["arms"]["A"], COL["R3"], "-", "o", COL["R3"], "A  synthetic + ISIC + EASY (3 seeds)"),
            (T["arms"]["B"], COL["R5"], "-", "s", COL["R5"], "B  ISIC + EASY, no synthetic (3 seeds)"),
            (T2["arms"]["A2"], COL["R3"], "--", "o", "white", "A′  as A, loss averaged per source (2 seeds)")]
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6), sharey=True, sharex=True); axs = axs.ravel()
    for ax, f in zip(axs, feats):
        clean(ax, "y")
        for k, (r, c, ls, mk, mfc, lab) in enumerate(arms):
            r = r[f]; xs = np.log2(np.array(r["effective_n"]) + 1)
            if k < 2:
                lo, hi = np.array(r["ci"]).T; ax.fill_between(xs, lo, hi, color=c, alpha=.10, lw=0)
            ax.plot(xs, r["auroc_mean"], ls=ls, marker=mk, color=c, lw=1.5, ms=4.5, mfc=mfc, mec=c if mfc == "white" else "white",
                    mew=1.0 if mfc == "white" else .6, label=lab, zorder=3 + k)
        ref = T["arms"]["A"][f]["reference"]["R0"]
        ax.axhline(ref, color=COL["R0"], lw=1, ls=":", label="R0 synthetic only")
        ax.axhline(.5, color=AXIS, lw=.8)
        tk = [0, 2, 4, 8, 16, 32, 64]; ax.set_xticks(np.log2(np.array(tk) + 1), [str(t) for t in tk])
        a = T["arms"]["A"][f]; ax.set_title(f"{a['short']} (n+ = {a['n_pos']})", fontsize=9, loc="left")
        ax.set_ylim(.40, 1.0)
    fig.supylabel("Image-level AUROC on EASY (out-of-fold)", fontsize=9, x=.0)
    fig.supxlabel("Dermatologist-masked positive images of the structure per training fold", fontsize=9, y=.01)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(.5, 1.07), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(.02, .03, 1, .99)); save(fig, "fig5_target_dose")


if __name__ == "__main__":
    print("features:", fig_features()); fig_dose(); fig_regions(); fig_target_dose()
