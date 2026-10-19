"""Structure generators (one per family) and the ID registry SPEC.

Every generator has the signature fn(ctx, sid, **params), draws onto ctx.canvas and
records one label per structure (a union mask for textures / clod fields, one mask per
instance for vessels). SPEC maps each taxonomy ID to (generator, params, layer, ...).
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

import geometry as G
import render as R

SHIFT = 2  # sub-pixel precision for cv2 drawing


def noise(rng, size, sigma, amp=1.0):
    n = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-6) * amp


def blend_arr(canvas, alpha, colour):
    """Blend a per-pixel colour array (or a single colour) with an alpha map."""
    a = np.clip(alpha, 0, 1)[..., None]
    canvas[:] = canvas * (1 - a) + np.asarray(colour, np.float32) * a


class Painter:
    """Collect AA-drawn shapes with per-shape colours, then blend once."""

    def __init__(self, size):
        self.layer = np.zeros((size, size, 3), np.uint8)
        self.mask = np.zeros((size, size), np.uint8)

    def _col(self, c):
        return tuple(int(v) for v in c)

    def ellipse(self, c, axes, angle, colour, arc=(0, 360), thick=-1):
        ctr = (int(round(c[0] * 4)), int(round(c[1] * 4)))
        ax = (max(int(round(axes[0] * 4)), 2), max(int(round(axes[1] * 4)), 2))
        for img, col in ((self.layer, self._col(colour)), (self.mask, 255)):
            cv2.ellipse(img, ctr, ax, angle, arc[0], arc[1], col, thick, cv2.LINE_AA, SHIFT)

    def line(self, p0, p1, w, colour):
        a = (int(round(p0[0] * 4)), int(round(p0[1] * 4)))
        b = (int(round(p1[0] * 4)), int(round(p1[1] * 4)))
        for img, col in ((self.layer, self._col(colour)), (self.mask, 255)):
            cv2.line(img, a, b, col, max(int(round(w)), 1), cv2.LINE_AA, SHIFT)

    def finish(self, ctx, alpha=1.0, blur=.6, clip=None):
        L = self.layer.astype(np.float32)
        M = self.mask.astype(np.float32) / 255
        if blur > 0:
            L, M = cv2.GaussianBlur(L, (0, 0), blur), cv2.GaussianBlur(M, (0, 0), blur)
        col = np.clip(L / np.maximum(M, 1e-3)[..., None], 0, 255)
        a = M * alpha
        if clip is not None:
            a = a * clip
        blend_arr(ctx.canvas, a, col)
        return a


class Ctx:
    def __init__(self, rng, canvas, lesion, inner, rim, skin_mean, skin_canvas, out, polarized=False, volar=None):
        self.rng, self.canvas, self.lesion, self.inner, self.rim = rng, canvas, lesion, inner, rim
        self.skin, self.skin_canvas, self.out, self.polarized, self.volar = skin_mean, skin_canvas, out, polarized, volar
        self.size = lesion.shape[0]
        ys, xs = np.nonzero(lesion > .5)
        self.centroid = np.array([xs.mean(), ys.mean()])
        self.r_eq = float(np.sqrt(len(xs) / np.pi))
        self.labels, self.masks, self.k = [], {}, 0
        self._rim = None

    # -- colours / geometry helpers -------------------------------------
    def col(self, name):
        if not isinstance(name, str):
            name = name[self.rng.integers(len(name))]
        v = {"brown": self.inner * .55, "dark_brown": self.inner * .35, "light_brown": self.rim * 1.15,
             "tan": (172, 128, 88), "black": (30, 24, 24), "gray": (122, 122, 132), "blue_gray": (105, 118, 150),
             "blue": (72, 96, 152), "white": (240, 236, 230), "red": (205, 52, 66), "pink": (238, 165, 172),
             "purple": (128, 52, 98), "yellow_orange": (205, 155, 78), "skin": self.skin,
             "skin_light": self.skin * 1.08}[name]
        return np.clip(np.asarray(v, float), 0, 255)

    def patch(self, cover):
        """Soft sub-region of the lesion covering ~`cover` of its area."""
        if cover >= 1:
            return self.lesion
        n = noise(self.rng, self.size, self.size / 7)
        t = np.quantile(n[self.lesion > .5], 1 - cover) if (self.lesion > .5).any() else 0
        return self.lesion * (1 / (1 + np.exp(-(n - t) / .2)))

    def pts_inside(self, n, min_gap=0.0, mask=None, margin_erode=9):
        m = self.lesion > .8 if mask is None else mask > .5
        m = cv2.erode(m.astype(np.uint8), np.ones((margin_erode, margin_erode), np.uint8)).astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            ys, xs = np.nonzero((self.lesion if mask is None else mask) > .5)
        if len(xs) == 0:
            return np.zeros((0, 2))
        pts = np.zeros((0, 2))
        for i in self.rng.permutation(len(xs))[: max(n * 25, 50)]:
            p = np.array([xs[i], ys[i]], float)
            if min_gap and len(pts) and (np.hypot(*(pts - p).T) < min_gap).any():
                continue
            pts = np.vstack([pts, p])
            if len(pts) >= n:
                break
        return pts

    def rim_contour(self):
        if self._rim is None:
            cs, _ = cv2.findContours((self.lesion > .5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=len)[:, 0, :].astype(np.float32)
            self._rim = c[((c > 3) & (c < self.size - 4)).all(1)]  # border pieces are crop artefacts
        return self._rim

    def rim_arc(self, span=(.2, .6), step=(5, 9), inward=0.0, full=False):
        """Points along an arc of the rim (moved inward by `inward` px) with outward unit vectors."""
        c = self.rim_contour()
        if len(c) < 40:
            return []
        n = len(c)
        start = int(self.rng.integers(n))
        length = n if full else int(n * self.rng.uniform(*span))
        stp = int(self.rng.integers(int(step[0]), int(step[1]) + 1))
        out = []
        for i in range(start, start + length, stp):
            p = c[i % n]
            v = p - self.centroid
            v = v / (np.hypot(*v) + 1e-6)
            out.append((p - v * inward, v))
            if self.rng.random() < .1:
                start += int(self.rng.integers(6, 20))
        return out

    def blob(self, c, rx, ry, rot=0.0, wob=.2, sigma=3.0):
        gy, gx = np.mgrid[:self.size, :self.size].astype(np.float32)
        x, y = gx - c[0], gy - c[1]
        u = (x * np.cos(rot) + y * np.sin(rot)) / rx
        v = (-x * np.sin(rot) + y * np.cos(rot)) / ry
        f = np.sqrt(u ** 2 + v ** 2) + noise(self.rng, self.size, self.size / 9, wob)
        return cv2.GaussianBlur((f < 1).astype(np.float32), (0, 0), sigma)

    def record(self, sid, mask, region):
        sel = cv2.resize(np.asarray(mask, np.float32), (self.out, self.out), interpolation=cv2.INTER_AREA) > .25
        if sel.sum() < 3:
            return
        py, px = np.nonzero(sel)
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([px, py], 1).astype(np.float32))
        sp = SPEC[sid]
        self.labels.append(dict(id=sid, parent=sp["parent"], instance=self.k, region=region, site=sp["site"],
                                polarized_only=sp["polarized"],
                                bbox=[int(px.min()), int(py.min()), int(px.max()), int(py.max())],
                                rbox=[float(cx), float(cy), float(w), float(h), float(ang)], area_px=int(sel.sum())))
        self.masks[f"{sid}_{self.k}"] = sel
        self.k += 1


# --- network / mesh-like lines ----------------------------------------------

def _mesh_edge(ctx, region, cell, warp):
    rng, size = ctx.rng, ctx.size
    inside = region > .5
    ys, xs = np.nonzero(inside)
    n = max(int(inside.sum() / cell ** 2), 40)
    idx = rng.integers(len(xs), size=n)
    seeds = np.stack([xs[idx], ys[idx]], 1).astype(np.float32) + rng.normal(0, .8, (n, 2))
    gy, gx = np.mgrid[:size, :size].astype(np.float32)
    wx = gx + (noise(rng, size, 6, warp) if warp else 0)
    wy = gy + (noise(rng, size, 6, warp) if warp else 0)
    d, ix = cKDTree(seeds).query(np.stack([wx.ravel(), wy.ravel()], 1), k=2)
    return (d[:, 1] - d[:, 0]).reshape(size, size), ix[:, 0].reshape(size, size), n


def g_network(ctx, sid, cell=(15, 22), thr=(1.6, 2.3), alpha=.6, line="brown", cover=1.0, irregular=0.0,
              blur=1.3, var=None, warp=2.2, patchy=True):
    rng, size = ctx.rng, ctx.size
    region = ctx.patch(cover)
    if (region > .5).sum() < 300:
        return
    edge, _, _ = _mesh_edge(ctx, region, rng.uniform(*cell), warp)
    t = rng.uniform(*thr) * (np.clip(1 + irregular * noise(rng, size, 16), .35, 2.5) if irregular else 1.0)
    m = np.clip(1 - edge / t, 0, 1)
    if patchy:
        m = m * np.clip(.65 + noise(rng, size, 14, .45), .1, 1)
    inner = cv2.GaussianBlur(cv2.erode((region > .5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 5)
    m = m * inner
    c0 = ctx.col(line)
    if var:  # atypical: colour drifts along the network
        w = 1 / (1 + np.exp(-noise(rng, size, 20, 1.5)))
        colour = c0[None, None, :] * (1 - w[..., None]) + ctx.col(var)[None, None, :] * w[..., None]
    else:
        colour = c0
    a = cv2.GaussianBlur(m, (0, 0), blur * .6) * alpha
    blend_arr(ctx.canvas, a, colour)
    ctx.record(sid, m * (region > .5), "inside")


def g_cobblestone(ctx, sid, cell=(30, 44), alpha=.75, cover=.85):
    rng = ctx.rng
    region = ctx.patch(cover)
    if (region > .5).sum() < 400:
        return
    edge, idx, n = _mesh_edge(ctx, region, rng.uniform(*cell), 0)
    bright = rng.uniform(.85, 1.2, n)
    col = ctx.col("brown")[None, None, :] * 1.3 * bright[idx][..., None]
    a = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.2) * alpha
    blend_arr(ctx.canvas, a, col)
    blend_arr(ctx.canvas, np.clip(1 - edge / 1.3, 0, 1) * region * .3, ctx.col("skin_light"))
    ctx.record(sid, region * (region > .5), "inside")


def g_polygons(ctx, sid, rhomb=False, alpha=.7, cover=.7):
    """Rhomboids (two oblique line families) or angulated polygons (straight Voronoi edges)."""
    rng, size = ctx.rng, ctx.size
    region = ctx.patch(cover)
    if rhomb:
        gy, gx = np.mgrid[:size, :size].astype(np.float32)
        base, d, m = rng.uniform(0, np.pi), rng.uniform(14, 20), np.zeros((size, size), np.float32)
        for sgn in (-1, 1):
            t = base + sgn * np.radians(rng.uniform(25, 38))
            u = (gx * np.cos(t) + gy * np.sin(t)) / d
            dist = np.abs(((u + .5) % 1) - .5) * d
            m = np.maximum(m, np.clip(1 - dist / 1.2, 0, 1))
        m = m * region
    else:
        edge, _, _ = _mesh_edge(ctx, region, rng.uniform(26, 40), 0)
        m = np.clip(1 - edge / 1.6, 0, 1) * region
    blend_arr(ctx.canvas, cv2.GaussianBlur(m, (0, 0), .6) * alpha, ctx.col(["dark_brown", "gray"]))
    ctx.record(sid, m, "inside")


# --- curvy line patterns --------------------------------------------------------

def make_volar(rng, size, period=(11, 15), warp=4.0, curvature=.002):
    th = rng.uniform(0, np.pi)
    P = rng.uniform(*period)
    gy, gx = np.mgrid[:size, :size].astype(np.float32)
    x, y = gx - size / 2, gy - size / 2
    v = -x * np.sin(th) + y * np.cos(th)
    u = (x * np.cos(th) + y * np.sin(th) + noise(rng, size, 45, warp) + noise(rng, size, 15, warp * .3) + curvature * v ** 2) / P * 2 * np.pi
    return dict(phase=u, period=P, theta=th)


def _wrap(phase):  # distance (rad) to nearest ridge crest
    return np.abs(((phase + np.pi) % (2 * np.pi)) - np.pi)


def g_isolines(ctx, sid, kind="cerebriform", cover=.8):
    """Cerebriform fissures / crypts as iso-lines of smooth noise."""
    rng, size = ctx.rng, ctx.size
    n = noise(rng, size, rng.uniform(8, 11))
    line = np.exp(-(n / .32) ** 2) * ctx.patch(cover)
    blend_arr(ctx.canvas, cv2.GaussianBlur(line, (0, 0), .8) * .65, ctx.col("dark_brown"))
    ctx.record(sid, line, "inside")
    if kind == "crypts":
        pts = ctx.pts_inside(int(rng.integers(8, 22)), min_gap=18, mask=(line < .1) * ctx.lesion)
        P = Painter(size)
        for p in pts:
            P.ellipse(p, (rng.uniform(7, 12), rng.uniform(5, 8)), rng.uniform(0, 180), ctx.col("dark_brown"))
        a = P.finish(ctx, .85, .8, ctx.lesion > .3)
        ctx.record(sid, a, "inside")


def g_fingerprint(ctx, sid, cover=.6):
    rng = ctx.rng
    vf = make_volar(rng, ctx.size, period=(7, 9), warp=6.0, curvature=.003)
    f = np.exp(-((_wrap(vf["phase"] + np.pi)) / .45) ** 2)  # thin brown parallel lines
    m = f * ctx.patch(cover)
    blend_arr(ctx.canvas, cv2.GaussianBlur(m, (0, 0), .6) * .5, ctx.col("brown"))
    ctx.record(sid, m, "inside")


def g_acral(ctx, sid, mode="furrow", cover=.85):
    rng, size = ctx.rng, ctx.size
    v = ctx.volar
    if v is None:
        return
    ph = v["phase"]
    region = ctx.patch(cover)
    if mode == "furrow":     # thin pigment in the furrows
        m = np.exp(-(_wrap(ph + np.pi) / .3) ** 2)
    elif mode == "ridge":    # thick pigment on the ridges
        m = np.exp(-(_wrap(ph) / 1.5) ** 2)
    elif mode == "lattice":  # furrow lines + lines crossing the ridges
        gy, gx = np.mgrid[:size, :size].astype(np.float32)
        th = v["theta"] + np.pi / 2
        u2 = ((gx - size / 2) * np.cos(th) + (gy - size / 2) * np.sin(th)) / (v["period"] * 1.5) * 2 * np.pi
        cross = np.exp(-(_wrap(u2) / .3) ** 2) * np.exp(-(_wrap(ph) / 1.4) ** 2)
        m = np.maximum(np.exp(-(_wrap(ph + np.pi) / .42) ** 2), cross)
    else:                    # fibrillar: short oblique strokes crossing the ridges
        P = Painter(size)
        for p in ctx.pts_inside(int(rng.integers(45, 85)), min_gap=9, mask=region):
            a = v["theta"] + rng.uniform(-.55, .55)
            d = v["period"] * rng.uniform(.7, 1.3) * np.array([np.cos(a), np.sin(a)])
            P.line(p - d, p + d, 2.0, ctx.col("dark_brown"))
        a = P.finish(ctx, .7, .6, region)
        ctx.record(sid, a, "inside")
        return
    m = m * region
    blend_arr(ctx.canvas, cv2.GaussianBlur(m, (0, 0), .6) * .72, ctx.col(["dark_brown", "black"]))
    ctx.record(sid, m, "inside")


# --- rim structures --------------------------------------------------------------

def _branched(rng):
    g = G.Glyph([G._stroke(G._curve(lambda t: (2 * t - 1, 0 * t), 20), (1, .15))])
    for s in (-1, 1):
        a = s * rng.uniform(.45, .8)
        t = np.linspace(0, 1, 12)
        g.strokes.append(G._stroke(np.stack([.1 + .7 * t * np.cos(a), .7 * t * np.sin(a)], 1), (.8, .1)))
    return g


def g_streaks(ctx, sid, span=(.2, .6), L=(5, 15), step=(5, 9), width=(2.2, 3.4), arcs=(1, 3), full=False,
              colour="rim", kind="streak", alpha=.7, bulbs=0.0, blur=1.0):
    rng, size = ctx.rng, ctx.size
    for _ in range(1 if full else int(rng.integers(*arcs))):
        pts = ctx.rim_arc(span, step, full=full)
        if not pts:
            return
        m = np.zeros((size, size), np.float32)
        for p, v in pts:
            rot = np.arctan2(v[1], v[0]) + rng.normal(0, .22)
            ln = rng.uniform(*L) * (1.8 if rng.random() > .8 else 1.0)
            if kind == "pseudopod" or rng.random() < bulbs:
                g, sc = G.pseudopod(rng), ln / 1.2
                ctr = p + np.array([np.cos(rot), np.sin(rot)]) * (ln / 2 - 2)
            elif kind == "branched":
                g, sc = _branched(rng), ln / 2
                ctr = p + np.array([np.cos(rot), np.sin(rot)]) * (ln / 2 - 3)
            else:
                g, sc = G.streak(rng), ln / 2
                ctr = p + np.array([np.cos(rot), np.sin(rot)]) * (ln / 2 - 3)
            m = np.maximum(m, R.draw_glyph(g, size, ctr[0], ctr[1], sc, rot, rng.uniform(*width), rng))
        c = np.clip(ctx.rim * .7, 0, 255) if colour == "rim" else ctx.col(colour)
        R.blend(ctx.canvas, R.soften(m, blur), c, alpha=alpha)
        ctx.record(sid, m, "periphery")


def g_border(ctx, sid):
    """Moth-eaten border: skin-coloured scallops biting into the rim."""
    rng, size = ctx.rng, ctx.size
    pts = ctx.rim_arc(step=(10, 16), full=True)
    if not pts:
        return
    P = Painter(size)
    for p, v in pts:
        r = rng.uniform(6, 11)
        q = p - v * r * .35
        c = ctx.skin_canvas[int(np.clip(q[1], 0, size - 1)), int(np.clip(q[0], 0, size - 1))]
        P.ellipse(q, (r, r * rng.uniform(.8, 1.1)), rng.uniform(0, 180), c)
    a = P.finish(ctx, .95, 1.2)
    ctx.record(sid, a, "periphery")


def g_leaf(ctx, sid, groups=(1, 3)):
    rng, size = ctx.rng, ctx.size
    P = Painter(size)
    for _ in range(int(rng.integers(*groups))):
        arc = ctx.rim_arc(span=(.02, .03), step=(1, 2))
        if arc:
            p, v = arc[0]
        else:  # lesion rim outside the frame: anchor inside instead
            pts = ctx.pts_inside(1)
            if len(pts) == 0:
                return
            p, a0 = pts[0], rng.uniform(0, 6.3)
            v = np.array([np.cos(a0), np.sin(a0)])
        base_dir = np.arctan2(-v[1], -v[0])  # pointing inward
        for a in np.linspace(-.9, .9, int(rng.integers(4, 8))):
            ln, w = rng.uniform(45, 80), rng.uniform(14, 22)
            ang = base_dir + a + rng.normal(0, .08)
            c = p + np.array([np.cos(ang), np.sin(ang)]) * ln / 2
            P.ellipse(c, (ln / 2, w / 2), np.degrees(ang), ctx.col(["blue_gray", "gray", "brown"]))
    a = P.finish(ctx, .75, 1.0, ctx.lesion > .3)
    ctx.record(sid, a, "periphery")


def g_spoke(ctx, sid, n=(1, 3), concentric=False, colours=("brown", "blue_gray")):
    rng, size = ctx.rng, ctx.size
    P = Painter(size)
    for c in ctx.pts_inside(int(rng.integers(*n)), min_gap=60):
        if concentric:  # clod within a clod
            r = rng.uniform(14, 24)
            colour = ctx.col(list(colours))
            P.ellipse(c, (r, r * rng.uniform(.8, 1)), rng.uniform(0, 180), colour)
            P.ellipse(c, (r * .45, r * .4), rng.uniform(0, 180), colour * .5)
            continue
        for a in np.linspace(0, 2 * np.pi, int(rng.integers(7, 11)), endpoint=False) + rng.uniform(0, 1):
            ln = rng.uniform(30, 50)
            P.line(c, c + ln * np.array([np.cos(a), np.sin(a)]), rng.uniform(2.4, 3.4), ctx.col(["brown", "gray"]))
        P.ellipse(c, (5.5, 5.5), 0, ctx.col("dark_brown"))
    a = P.finish(ctx, .85, .7, ctx.lesion > .3)
    ctx.record(sid, a, "inside")


def g_white_lines(ctx, sid):
    """Shiny white streaks: short parallel lines in small groups, orthogonal to each other."""
    rng, size = ctx.rng, ctx.size
    P = Painter(size)
    th = rng.uniform(0, np.pi)
    for c in ctx.pts_inside(int(rng.integers(3, 8)), min_gap=44):
        th2 = th + (np.pi / 2 if rng.random() < .5 else 0)
        d = np.array([np.cos(th2), np.sin(th2)])
        nrm = np.array([-d[1], d[0]])
        ln = rng.uniform(22, 36)
        for k in range(int(rng.integers(3, 6))):
            o = nrm * (k - 2) * 5.5
            P.line(c + o - d * ln / 2, c + o + d * ln / 2, 2.2, ctx.col("white"))
    a = P.finish(ctx, .85, .5, ctx.lesion > .3)
    ctx.record(sid, a, "inside")


def g_shiny_blotches(ctx, sid):
    rng, size = ctx.rng, ctx.size
    P = Painter(size)
    for c in ctx.pts_inside(int(rng.integers(3, 8)), min_gap=20):
        P.ellipse(c, (rng.uniform(8, 16), rng.uniform(6, 11)), rng.uniform(0, 180), ctx.col("white"))
    a = P.finish(ctx, .55, 1.6, ctx.lesion > .3)
    P2 = Painter(size)
    for c in ctx.pts_inside(int(rng.integers(2, 6)), min_gap=25):
        a0 = rng.uniform(0, 6.3)
        pts = [c]
        for _ in range(int(rng.integers(5, 9))):
            a0 += rng.normal(0, .35)
            pts.append(pts[-1] + 8 * np.array([np.cos(a0), np.sin(a0)]))
        for i in range(len(pts) - 1):
            P2.line(pts[i], pts[i + 1], 2.4, ctx.col("white"))
    a2 = P2.finish(ctx, .8, .5, ctx.lesion > .3)
    ctx.record(sid, np.maximum(a, a2), "inside")


# --- clods / globules / dots / circles ----------------------------------------------

def g_blobs(ctx, sid, colour="brown", r=(3.5, 8), n=(15, 60), elong=(1, 1.4), place="random", cover=1.0,
            min_gap=.6, var=.12, alpha=.9, blur=.7, core=None, rim_span=(.5, 1.0), inward=7, rim_step=(5, 8)):
    rng, size = ctx.rng, ctx.size
    region = ctx.patch(cover)
    nn = int(rng.integers(*n))
    if place == "rim":
        pts = [p for p, _ in ctx.rim_arc(rim_span, rim_step, inward=inward, full=rim_span[0] >= 1)][:nn] if rim_span else []
    else:
        pts = ctx.pts_inside(nn, min_gap=r[0] * 2 * min_gap, mask=region)
    if len(pts) == 0:
        return
    P = Painter(size)
    for p in pts:
        rr, e, ang = rng.uniform(*r), rng.uniform(*elong), rng.uniform(0, 180)
        c = ctx.col(colour) * rng.uniform(1 - var, 1 + var)
        P.ellipse(p, (rr * e, rr), ang, c)
        if core:
            P.ellipse(p, (rr * e * .45, rr * .45), ang, ctx.col(core) if isinstance(core, (str, list)) else c * .5)
    a = P.finish(ctx, alpha, blur, ctx.lesion > .3)
    ctx.record(sid, a, "inside" if place != "rim" else "periphery")


def g_dots(ctx, sid, colour="gray", r=(2.6, 4.4), n=(100, 300), arr="random", cover=1.0, alpha=.9, blur=.5, halo=None):
    rng, size = ctx.rng, ctx.size
    region = ctx.patch(cover)
    P = Painter(size)
    ok = False
    if arr == "random":
        for p in ctx.pts_inside(int(rng.integers(*n)), mask=region, margin_erode=3):
            P.ellipse(p, (rng.uniform(*r),) * 2, 0, ctx.col(colour) * rng.uniform(.85, 1.1)); ok = True
    elif arr == "mixed":  # granularity: dots of any colour
        for p in ctx.pts_inside(int(rng.integers(*n)), mask=region, margin_erode=3):
            P.ellipse(p, (rng.uniform(*r),) * 2, 0, ctx.col(["brown", "gray", "blue_gray", "dark_brown", "black"])); ok = True
    elif arr == "rings":  # annular-granular: gray dots arranged as rings + random granules
        for c in ctx.pts_inside(int(rng.integers(5, 14)), min_gap=18, mask=region):
            rr = rng.uniform(10, 14)
            for a in np.linspace(0, 2 * np.pi, int(rng.integers(8, 13)), endpoint=False):
                P.ellipse(c + rr * np.array([np.cos(a), np.sin(a)]), (2.6, 2.6), 0, ctx.col("gray")); ok = True
        for p in ctx.pts_inside(int(rng.integers(40, 120)), mask=region, margin_erode=3):
            P.ellipse(p, (2.4, 2.4), 0, ctx.col("gray"))
    elif arr == "rows":   # linear dots at the periphery
        for row in range(int(rng.integers(1, 3))):
            for p, _ in ctx.rim_arc((.3, .7), (9, 13), inward=8 + 8 * row):
                P.ellipse(p, (rng.uniform(*r),) * 2, 0, ctx.col(colour)); ok = True
    elif arr == "rosette":  # four dots in a square
        for c in ctx.pts_inside(int(rng.integers(4, 12)), min_gap=24, mask=region):
            s, th = rng.uniform(10, 15), rng.uniform(0, np.pi / 2)
            for k in range(4):
                a = th + k * np.pi / 2 + np.pi / 4
                P.ellipse(c + s / 1.414 * np.array([np.cos(a), np.sin(a)]), (rng.uniform(3, 4.4),) * 2, 0, ctx.col("white")); ok = True
    elif arr == "target":   # dot inside a halo (hypopigmented space)
        for c in ctx.pts_inside(int(rng.integers(*n)), min_gap=22, mask=region):
            P.ellipse(c, (rng.uniform(8, 11),) * 2, 0, ctx.col(halo or "skin_light"))
            P.ellipse(c, (rng.uniform(*r),) * 2, 0, ctx.col(colour)); ok = True
    if not ok:
        return
    a = P.finish(ctx, alpha, blur, ctx.lesion > .3)
    ctx.record(sid, a, "periphery" if arr == "rows" else "inside")


def g_rings(ctx, sid, kind="concentric", colour="gray", n=(3, 8)):
    rng, size = ctx.rng, ctx.size
    P = Painter(size)
    pts = ctx.pts_inside(int(rng.integers(*n)), min_gap=44)
    if len(pts) == 0:
        return
    for c in pts:
        if kind == "concentric":
            r0, st = rng.uniform(10, 16), rng.uniform(5, 7)
            for k in range(int(rng.integers(2, 4))):
                P.ellipse(c, (r0 + k * st,) * 2, 0, ctx.col(colour), thick=3)
        elif kind == "incomplete":
            a0 = rng.uniform(0, 360)
            P.ellipse(c, (rng.uniform(12, 18),) * 2, 0, ctx.col("brown"), arc=(a0, a0 + rng.uniform(180, 300)), thick=4)
        else:
            P.ellipse(c, (rng.uniform(8, 14),) * 2, 0, ctx.col("white"), thick=3)
    a = P.finish(ctx, .85, .6, ctx.lesion > .3)
    ctx.record(sid, a, "inside")


# --- structureless zones ---------------------------------------------------------

def g_zone(ctx, sid, colour, alpha=.6, size_f=(.3, .6), offset=.25, sigma=4.0, wob=.2, ecc=False, cover=False):
    rng = ctx.rng
    if ecc is None:
        ecc = rng.random() < .5
    if cover:
        m = cv2.GaussianBlur(ctx.lesion, (0, 0), sigma) * ctx.patch(.9)
    else:
        d = rng.uniform(0, 2 * np.pi)
        c = ctx.centroid + (ctx.r_eq * .55 * np.array([np.cos(d), np.sin(d)]) if ecc else rng.normal(0, ctx.r_eq * offset, 2))
        rx = ctx.r_eq * rng.uniform(*size_f)
        m = ctx.blob(c, rx, rx * rng.uniform(.6, 1.0), rng.uniform(0, 6.3), wob, sigma) * ctx.lesion
    m = m / max(m.max(), 1e-6)
    m = m * np.clip(.82 + noise(rng, ctx.size, 7, .12) + noise(rng, ctx.size, 2, .04), .45, 1.05)
    col = ctx.col(colour)[None, None, :] + noise(rng, ctx.size, 9, 7)[..., None]
    blend_arr(ctx.canvas, m * alpha, col)
    ctx.record(sid, m, "inside")


def g_rainbow(ctx, sid):
    rng, size = ctx.rng, ctx.size
    t = 1 / (1 + np.exp(-noise(rng, size, 24, 1.3)))
    c0, c1, c2 = np.array([190, 92, 98.]), np.array([112, 122, 168.]), np.array([196, 168, 112.])
    k = t[..., None]
    col = np.where(k < .5, c0 * (1 - 2 * k) + c1 * 2 * k, c1 * (2 - 2 * k) + c2 * (2 * k - 1))
    m = cv2.GaussianBlur(ctx.lesion, (0, 0), 2) * ctx.patch(.85)
    blend_arr(ctx.canvas, m * .38, col)
    ctx.record(sid, m, "inside")


def g_homog(ctx, sid):
    rng = ctx.rng
    colour = ctx.col(["brown", "blue_gray", "pink", "gray", "tan"]) * rng.uniform(.95, 1.05)
    m = cv2.GaussianBlur(ctx.lesion, (0, 0), 2.5)
    blend_arr(ctx.canvas, m * .8, colour)
    ctx.record(sid, m, "inside")


def g_follicular(ctx, sid, mode="strawberry"):
    """Strawberry (red zone) / pseudonetwork (brown zone) interrupted by regular follicular openings."""
    rng, size = ctx.rng, ctx.size
    region = cv2.GaussianBlur(ctx.lesion, (0, 0), 2) * ctx.patch(.92)
    if mode == "strawberry":
        blend_arr(ctx.canvas, region * .5, ctx.col("red"))
        sp, hole, ha = rng.uniform(19, 27), "skin_light", .6
    else:
        blend_arr(ctx.canvas, region * .5, ctx.col("brown"))
        sp, hole, ha = rng.uniform(16, 23), "skin_light", .55
    P = Painter(size)
    for gy in np.arange(0, size, sp):
        for gx in np.arange(0, size, sp):
            p = np.array([gx, gy]) + rng.normal(0, sp * .28, 2)
            if region[int(np.clip(p[1], 0, size - 1)), int(np.clip(p[0], 0, size - 1))] > .6:
                P.ellipse(p, (rng.uniform(3.2, 5.5),) * 2, 0, ctx.col(hole))
    a = P.finish(ctx, ha, .8, region)
    ctx.record(sid, np.maximum(region * (region > .5), a), "inside")


# --- vessels (geometry in geometry.py) ---------------------------------------------

def g_vessel(ctx, sid, gid, n=(1, 1), ring=False):
    rng, size = ctx.rng, ctx.size
    meta = G.META[gid]
    reps = []
    if ring:
        reps.append((ctx.centroid, ctx.r_eq * rng.uniform(.8, 1.0), "ring"))
    else:
        for c in ctx.pts_inside(int(rng.integers(n[0], n[1] + 1))):
            reps.append((c, min(rng.uniform(*meta["extent_px"]) * 2.2, ctx.r_eq * 1.4) / 2, "inside"))
    for c, scale, region in reps:
        m = R.draw_glyph(G.GLYPHS[gid](rng), size, c[0], c[1], scale, rng.uniform(0, 2 * np.pi), rng.uniform(1.3, 2.0) * 1.6, rng)
        if region == "inside":
            m = m * (ctx.lesion > .3)
        R.vessel_look(ctx.canvas, m, rng)
        ctx.record(sid, m, region)


def g_vessel_set(ctx, sid, mode="mono"):
    """Monomorphous: one vessel type repeated. Polymorphous: three or more types."""
    rng, size = ctx.rng, ctx.size
    types = [t for t in sorted(G.GLYPHS) if t not in ("DS070", "DS061", "DS060", "DS073")]
    chosen = [types[rng.integers(len(types))]] if mode == "mono" else list(rng.choice(types, size=int(rng.integers(3, 5)), replace=False))
    m_all = np.zeros((size, size), np.float32)
    for gid in chosen:
        for c in ctx.pts_inside(int(rng.integers(2, 4)) if mode == "poly" else int(rng.integers(5, 9)), min_gap=18):
            scale = min(rng.uniform(*G.META[gid]["extent_px"]) * 1.6, ctx.r_eq * 1.2) / 2
            m = R.draw_glyph(G.GLYPHS[gid](rng), size, c[0], c[1], scale, rng.uniform(0, 6.3), rng.uniform(1.3, 2.0) * 1.6, rng) * (ctx.lesion > .3)
            R.vessel_look(ctx.canvas, m, rng)
            m_all = np.maximum(m_all, m)
    ctx.record(sid, m_all, "inside")


# --- registry ------------------------------------------------------------------------------

SPEC = {}


def reg(sid, fn, layer, parent=None, site=None, polarized=False, pale=False, **params):
    SPEC[sid] = dict(fn=fn, params=params, layer=layer, parent=parent or sid, site=site, polarized=polarized, pale=pale)


# zones (layer 0)
reg("DS048", g_zone, 0, colour=["dark_brown", "black"], alpha=.8, size_f=(.25, .5), sigma=6, wob=.35, ecc=None)
reg("DS049", g_zone, 0, colour="blue_gray", alpha=.6, size_f=(.35, .6), sigma=9)
reg("DS050", g_zone, 0, colour="pink", alpha=.6, size_f=(.3, .6), sigma=6, pale=True)
reg("DS051", g_zone, 0, colour="white", alpha=.7, size_f=(.3, .55), sigma=4, wob=.4)
reg("DS052", g_zone, 0, colour="white", alpha=.7, size_f=(.2, .35), offset=.08, sigma=4, wob=.3)
reg("DS053", g_rainbow, 0)
reg("DS054", g_follicular, 0, site="facial", pale=True, mode="strawberry")
reg("DS055", g_homog, 0)
reg("DS056", g_follicular, 0, site="facial", mode="pseudonet")
reg("DS057", g_zone, 0, colour="tan", alpha=.6, size_f=(.3, .55), ecc=True, sigma=7, wob=.35)
reg("DS059", g_border, 0)
# network family & line patterns (layer 1)
reg("DS001", g_network, 1, parent="network")
reg("DS022", g_network, 1, parent="network", cell=(15, 20), thr=(1.5, 1.9), patchy=False)
reg("DS002", g_network, 1, parent="network", cell=(12, 16), thr=(1.0, 1.3), alpha=.4, blur=1.0)
reg("DS003", g_network, 1, parent="network", cell=(16, 23), thr=(2.8, 3.6), alpha=.7)
reg("DS004", g_network, 1, parent="network", cell=(14, 24), thr=(2.4, 3.6), alpha=.7, irregular=.9, var="black")
reg("DS005", g_network, 1, parent="negative_network", cell=(12, 18), thr=(2.0, 2.6), alpha=.6, line="skin_light", cover=.5)
reg("DS021", g_network, 1, cell=(18, 26), thr=(1.1, 1.5), alpha=.55, line="white", cover=.5)
reg("DS012", g_isolines, 1, kind="cerebriform")
reg("DS014", g_isolines, 1, kind="crypts")
reg("DS013", g_fingerprint, 1)
reg("DS019", g_polygons, 1, site="facial", rhomb=True)
reg("DS020", g_polygons, 1, site="nonfacial", rhomb=False)
reg("DS029", g_cobblestone, 1)
reg("DS015", g_acral, 1, site="volar", mode="fibrillar")
reg("DS016", g_acral, 1, site="volar", mode="ridge")
reg("DS017", g_acral, 1, site="volar", mode="lattice")
reg("DS018", g_acral, 1, site="volar", mode="furrow")
# clods (layer 2)
reg("DS025", g_blobs, 2, parent="globules", colour="brown", r=(6, 13), n=(12, 45), blur=1.0)
reg("DS026", g_blobs, 2, parent="globules", colour="brown", r=(6, 10), n=(14, 30), place="rim", rim_span=(1.0, 1.0), rim_step=(12, 17), inward=11, blur=1.0)
reg("DS027", g_blobs, 2, colour=["yellow_orange", "tan", "brown"], r=(9, 18), n=(3, 10), core="dark_brown", blur=1.0)
reg("DS028", g_spoke, 2, concentric=True, colours=("blue_gray", "brown"), n=(2, 6))
reg("DS035", g_spoke, 2, concentric=True, colours=("brown", "blue"), n=(2, 5))
reg("DS030", g_blobs, 2, colour="blue_gray", r=(18, 34), n=(2, 6), elong=(1.1, 1.6), cover=.5, alpha=.75, blur=2.0)
reg("DS031", g_blobs, 2, colour="blue", r=(6, 11), n=(6, 22), alpha=.85, blur=1.1)
reg("DS033", g_blobs, 2, colour="pink", r=(5, 10), n=(5, 18), alpha=.8, blur=1.3, pale=True)
reg("DS034", g_blobs, 2, colour=["red", "purple"], r=(12, 26), n=(3, 7), elong=(1, 1.6), blur=.8, pale=True)
reg("DS040", g_blobs, 2, colour="white", r=(4, 10), n=(5, 20), cover=.6, blur=.8)
reg("DS023", g_streaks, 2, parent="streaks", full=True, step=(5, 7), L=(8, 16), bulbs=.4)
reg("DS011", g_spoke, 2)
reg("DS010", g_leaf, 2)
# circles
reg("DS045", g_rings, 2, kind="concentric", colour="gray")
reg("DS046", g_rings, 2, kind="incomplete")
reg("DS047", g_rings, 2, kind="white", pale=True)
# dots (layer 3)
reg("DS037", g_dots, 3, parent="globules", arr="mixed", n=(80, 300))
reg("DS038", g_dots, 3, colour="gray", r=(2.2, 3.6), n=(120, 420), cover=.4)
reg("DS039", g_dots, 3, arr="rings")
reg("DS041", g_dots, 3, arr="rosette", polarized=True, pale=True)
reg("DS042", g_dots, 3, arr="rows", colour="brown", r=(2.6, 3.6))
reg("DS043", g_dots, 3, arr="target", colour="brown", r=(3, 4.2), n=(10, 35), halo="skin_light")
# rim structures & white structures & vessels (layer 4)
reg("DS008", g_streaks, 4, parent="streaks")
reg("DS009", g_streaks, 4, parent="streaks", span=(.12, .3), L=(16, 32), step=(4, 7), width=(3, 4))
reg("DS007", g_streaks, 4, parent="streaks", kind="branched", L=(28, 46), step=(14, 22), width=(2.4, 3.4))
reg("DS024", g_streaks, 4, parent="streaks", kind="pseudopod", L=(10, 20), step=(12, 20))
reg("DS006", g_white_lines, 4, polarized=True)
reg("DS032", g_shiny_blotches, 4, polarized=True, pale=True)
for _sid, _gid, _n in (("DS060", "DS060", (2, 4)), ("DS061", "DS061", (1, 1)), ("DS062", "DS062", (2, 5)),
                       ("DS063", "DS063", (1, 1)), ("DS064", "DS064", (2, 5)), ("DS065", "DS065", (1, 1)),
                       ("DS066", "DS066", (1, 1)), ("DS067", "DS067", (2, 5)), ("DS071", "DS071", (1, 1)),
                       ("DS072", "DS072", (1, 1)), ("DS073", "DS073", (1, 1))):
    reg(_sid, g_vessel, 4, parent="vessels", pale=True, gid=_gid, n=_n)
reg("DS070", g_vessel, 4, parent="vessels", pale=True, gid="DS070", ring=True)
reg("DS068", g_vessel_set, 4, parent="vessels", pale=True, mode="mono")
reg("DS069", g_vessel_set, 4, parent="vessels", pale=True, mode="poly")
reg("DS074", g_dots, 4, parent="vessels", pale=True, arr="target", colour="red", r=(3, 4.2), n=(10, 30), halo="pink")


def compatible(ids):
    """Drop IDs that cannot share a body site (volar vs facial vs non-facial/non-acral)."""
    sites = {SPEC[i]["site"] for i in ids} - {None}
    if "volar" in sites:
        keep = "volar"
    elif "facial" in sites:
        keep = "facial"
    else:
        keep = "nonfacial" if "nonfacial" in sites else None
    return [i for i in ids if SPEC[i]["site"] in (None, keep)], keep
