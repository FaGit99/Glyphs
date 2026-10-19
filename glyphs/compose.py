"""Scene composition: skin + lesion + structures -> image, instance masks, labels.

Placement rules (see META region): vessels and zones live inside the lesion,
streaks sit on its rim, crown vessels ring it. Skin outside stays glyph-free.
Layer order: zone -> network -> streaks -> vessels -> imaging artefacts.
"""
import json
import os
import sys
import cv2
import numpy as np
from scipy.spatial import cKDTree

import geometry as G
import render as R
import skin
import structures as S

PAL = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "palette.npz"))
_RATIO = PAL["inner"].mean(1) / PAL["skin"].mean(1)

SOFTEN_SIGMA = (.8, 1.4)  # px at output resolution; structures are softer than my vector drawings
SOFTEN_W = .9
VIGNETTE_P = .25  # probability of a dermatoscope vignette (off in the real-vs-synthetic check: real crops are centre crops)


noise = S.noise


def lesion_mask(rng, size, pigmented=True):
    """Irregular border: warped ellipse + multi-octave noise, soft edge."""
    gy, gx = np.mgrid[:size, :size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, size * .1, 2)
    # native-resolution crops of real lesions: ~25 % small lesions, otherwise lesion fills most of the frame
    a = (rng.uniform(.26, .4) if rng.random() < .25 else rng.uniform(.4, .9)) * size
    b = a * rng.uniform(.7, 1.0)
    th = rng.uniform(0, np.pi)
    x, y = gx - cx, gy - cy
    u = (x * np.cos(th) + y * np.sin(th)) / a
    v = (-x * np.sin(th) + y * np.cos(th)) / b
    f = np.sqrt(u ** 2 + v ** 2)
    f = f + noise(rng, size, size / 7, .12) + noise(rng, size, size / 25, .05) + noise(rng, size, 3, .015)
    m = (f < 1).astype(np.float32)
    return np.clip(cv2.GaussianBlur(m, (0, 0), rng.uniform(1.2, 2.2) if pigmented else rng.uniform(3, 5)), 0, 1)


def lesion_colour(rng, canvas, lesion, pale, skin_mean):
    """Ratio-transfer a real lesion's colours (core + rim) onto this skin tone."""
    size = lesion.shape[0]
    pool = np.nonzero(_RATIO > .82)[0] if pale else np.nonzero(_RATIO < .75)[0]
    i = rng.choice(pool)
    ratio = lambda k: PAL[k][i] / np.maximum(PAL["skin"][i], 1)
    inner = np.clip(skin_mean * ratio("inner"), 0, 255)
    rim = np.clip(skin_mean * ratio("rim"), 0, 255)
    ys, xs = np.nonzero(lesion > .5)
    cy, cx = ys.mean(), xs.mean()
    gy, gx = np.mgrid[:size, :size]
    r = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) / max(np.sqrt(len(xs) / np.pi), 1)
    t = np.clip(r / 1.0, 0, 1)[..., None] ** 1.5
    col = inner[None, None, :] * (1 - t) + rim[None, None, :] * t
    k = .4 if pale else 1.0
    col = col + (noise(rng, size, 30, 16) + noise(rng, size, 10, 9) + noise(rng, size, 3, 3))[..., None] * k
    # hue variation, kept inside the real brown palette (yellow-tan <-> red-brown); no blue shift
    hue = noise(rng, size, 22, 6) + noise(rng, size, 6, 3) + noise(rng, size, 1.5, 2)
    col = col + hue[..., None] * np.array([.4, .15, -.3]) * k
    a = .55 if pale else .96
    w = (lesion * a)[..., None]
    canvas[:] = canvas * (1 - w) + col * w
    return inner, rim


def artefacts(rng, canvas, size):
    """Unlabelled distractors: hairs, bubbles, vignette."""
    for _ in range(rng.integers(0, 4)):
        p0 = rng.uniform(0, size, 2)
        a = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0, 1, 40)
        L = rng.uniform(.3, .8) * size
        bend = rng.normal(0, .5)
        pts = np.stack([p0[0] + L * t * np.cos(a + bend * t), p0[1] + L * t * np.sin(a + bend * t)], 1)
        g = G.Glyph([(pts - 0, np.ones(40))])
        m = R.draw_glyph(g, size, 0, 0, 1, 0, rng.uniform(1.0, 2.0), rng)
        col = (rng.choice([[40, 30, 25], [120, 90, 60], [200, 190, 180]]))
        R.blend(canvas, R.soften(m, 1.1), col, alpha=rng.uniform(.35, .7))
    for _ in range(rng.integers(0, 5)):
        c = rng.uniform(.1, .9, 2) * size
        rad = rng.uniform(3, 8)
        m = np.zeros((size, size), np.float32)
        cv2.circle(m, (int(c[0]), int(c[1])), int(rad), 1, 1, cv2.LINE_AA)
        R.blend(canvas, R.soften(m, .8), (255, 255, 255), alpha=.5)
    if rng.random() < VIGNETTE_P:
        gy, gx = np.mgrid[:size, :size]
        rr = np.sqrt((gx - size / 2) ** 2 + (gy - size / 2) ** 2) / (size / 2)
        vig = np.clip((rr - rng.uniform(.9, 1.0)) * 3, 0, 1) ** 1.5
        canvas *= (1 - vig[..., None])


def compose(rng, ids, work=448, out=224, prefer_real=True, skin_files=None):
    """ids: taxonomy IDs to include (incompatible body sites are dropped).
    Geometry is rendered at `work` px, skin stays at native `out` resolution; the result is
    area-downsampled to `out` px. Returns (uint8 RGB, labels, masks, skin source, lesion mask)."""
    ids, site = S.compatible(list(ids))
    size = work
    bg_native, src = skin.make_skin(rng, out, prefer_real, skin_files)
    canvas = cv2.resize(bg_native, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    skin_canvas = canvas.copy()
    pale = bool(ids) and all(S.SPEC[i]["pale"] for i in ids)  # structure-free scenes get a pigmented lesion
    lesion = lesion_mask(rng, size, not pale)
    skin_mean = bg_native.reshape(-1, 3).mean(0)
    inner, rim = lesion_colour(rng, canvas, lesion, pale, skin_mean)
    volar = None
    if site == "volar":  # palmoplantar skin: parallel ridge pattern modulates everything
        volar = S.make_volar(rng, size)
        canvas *= (1 + .10 * (.5 + .5 * np.cos(volar["phase"]) - .5))[..., None]
    ctx = S.Ctx(rng, canvas, lesion, inner, rim, skin_mean, skin_canvas, out,
                polarized=any(S.SPEC[i]["polarized"] for i in ids), volar=volar)
    for sid in sorted(ids, key=lambda i: S.SPEC[i]["layer"]):
        sp = S.SPEC[sid]
        sp["fn"](ctx, sid, **sp["params"])
    artefacts(rng, canvas, size)

    # native resolution: area downsample, then transplant real skin micro-texture into synthetic areas
    img = cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA)
    les_out = cv2.resize(lesion, (out, out), interpolation=cv2.INTER_AREA)
    # real structures are soft: defocus / optics blur inside the lesion area, before the real texture goes back on
    soft = cv2.GaussianBlur(img, (0, 0), rng.uniform(*SOFTEN_SIGMA))
    w = les_out[..., None] * SOFTEN_W
    img = img * (1 - w) + soft * w
    hp = bg_native.astype(np.float32) - cv2.GaussianBlur(bg_native.astype(np.float32), (0, 0), 1.5)
    hp = np.roll(hp, tuple(rng.integers(0, out, 2)), (0, 1))[:, ::int(rng.choice([-1, 1]))]
    img += hp * (les_out[..., None] * rng.uniform(1.1, 1.8))
    return R.finish(img, rng), ctx.labels, ctx.masks, src, les_out


def random_scene_ids(rng, k=(1, 4), exclude_sites=("volar",), pool=None):
    pool = pool or [i for i, sp in S.SPEC.items() if sp["site"] not in exclude_sites]
    ids = [str(x) for x in rng.choice(pool, size=int(rng.integers(*k)), replace=False)]
    return S.compatible(ids)[0]


def save_scene(path, img, labels, masks, src, overlay=True):
    cv2.imwrite(path + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(path + ".json", "w") as f:
        json.dump(dict(skin_source=src, labels=labels), f, indent=1)
    np.savez_compressed(path + "_masks.npz", **masks)
    if overlay:
        ov = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        for lab in labels:
            x0, y0, x1, y1 = lab["bbox"]
            cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.putText(ov, lab["id"][2:], (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
        cv2.imwrite(path + "_overlay.png", ov)


def glyph_sheet(path, cell=96):
    """Geometry review sheet: every vessel glyph on plain background, 2 random seeds each."""
    ids = sorted(G.GLYPHS)
    sheet = np.full((cell * 2 + 16, cell * len(ids), 3), 245, np.uint8)
    for col, cid in enumerate(ids):
        for row in range(2):
            rng = np.random.default_rng(100 + row)
            cv = np.ones((cell, cell, 3), np.float32) * 245
            m = R.draw_glyph(G.GLYPHS[cid](rng), cell, cell / 2, cell / 2, cell * .42, 0, 2.2, rng)
            R.blend(cv, R.soften(m, .5), R.COLORS["red"], alpha=.9)
            sheet[row * cell + 16:(row + 1) * cell + 16, col * cell:(col + 1) * cell] = cv.astype(np.uint8)
        cv2.putText(sheet, cid, (col * cell + 4, 11), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


SCENES = [
    ["DS001", "DS008"],
    ["DS001", "DS049"],
    ["DS049", "DS008"],
    ["DS060", "DS064", "DS067"],
    ["DS072", "DS065", "DS062"],
    ["DS070", "DS063", "DS073"],
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    os.makedirs(out, exist_ok=True)
    glyph_sheet(os.path.join(out, "glyph_sheet.png"))
    for seed, ids in enumerate(SCENES):
        rng = np.random.default_rng(seed + 10)
        img, labels, masks, src, _ = compose(rng, ids)
        save_scene(os.path.join(out, f"scene{seed}"), img, labels, masks, src)
        print(seed, ids, src, len(labels), "instances")
    ims = [cv2.imread(os.path.join(out, f"scene{i}.png")) for i in range(len(SCENES))]
    cv2.imwrite(os.path.join(out, "contact.png"), np.vstack([np.hstack(ims[:3]), np.hstack(ims[3:])]))
