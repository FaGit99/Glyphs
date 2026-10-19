"""Parametric glyph geometry in a unit frame (extent ~[-1, 1]^2, origin = centre).

A glyph is geometry only: tapered strokes and filled blobs. Colour, blur and
blending belong to render.py. Stroke widths are relative profiles (0..1); the
renderer applies an absolute pixel width so lines do not thicken with scale.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Glyph:
    strokes: list = field(default_factory=list)  # (pts Nx2, profile N)
    blobs: list = field(default_factory=list)    # (cx, cy, rx, ry, angle)


def _curve(fn, n=60):
    t = np.linspace(0, 1, n)
    return np.stack(fn(t), axis=1)


def _stroke(pts, taper=(1.0, 1.0)):
    pts = np.asarray(pts, float)
    return pts, np.linspace(taper[0], taper[1], len(pts))


def _smooth(pts, n=60):
    """Catmull-Rom through control points."""
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(p) - 2):
        for t in np.linspace(0, 1, n // (len(pts) - 1), endpoint=False):
            out.append(0.5 * ((2 * p[i]) + (-p[i - 1] + p[i + 1]) * t
                              + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t ** 2
                              + (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t ** 3))
    out.append(pts[-1])
    return np.array(out)


# --- vessels ---------------------------------------------------------------

def dotted(rng):  # DS060
    g = Glyph()
    for _ in range(rng.integers(5, 10)):
        r = rng.uniform(0.9, 1.0) ** 0.5 * 0.9
        a = rng.uniform(0, 2 * np.pi)
        g.blobs.append((r * np.cos(a), r * np.sin(a), 0.15, 0.15, 0))
    return g


def clods(rng):  # DS061 red-purple lacunes
    g = Glyph()
    for _ in range(rng.integers(3, 6)):
        c = rng.uniform(-0.55, 0.55, 2)
        g.blobs.append((c[0], c[1], rng.uniform(.22, .36), rng.uniform(.16, .28), rng.uniform(0, 180)))
    return g


def linear(rng):  # DS062
    b = rng.uniform(-0.25, 0.25)
    return Glyph([_stroke(_curve(lambda t: (2 * t - 1, b * np.sin(np.pi * t))), (.9, .9))])


def coiled(rng):  # DS063 glomerular
    turns = rng.uniform(2.2, 3.2)
    def f(t):
        th = t * turns * 2 * np.pi
        r = 0.18 + 0.55 * t
        return r * np.cos(th), r * np.sin(th)
    tail = _curve(lambda t: (0.73 + 0.27 * t, -0.0 * t), 12)
    return Glyph([_stroke(np.vstack([_curve(f, 90), tail]), (1, 1))])


def hairpin(rng):  # DS064
    w, h = rng.uniform(.22, .36), rng.uniform(.4, .7)
    th = np.linspace(np.pi, 0, 20)
    top = np.stack([w * np.cos(th), h + w * np.sin(th)], 1)
    left = np.stack([np.full(10, -w), np.linspace(-1, h, 10)], 1)
    right = np.stack([np.full(10, w), np.linspace(h, -1, 10)], 1)
    pts = np.vstack([left, top, right])
    pts[:, 1] -= 0.0
    return Glyph([_stroke(pts, (1, 1))])


def serpentine(rng):  # DS065
    k, a = rng.uniform(1.5, 2.5), rng.uniform(.25, .4)
    ph = rng.uniform(0, np.pi)
    return Glyph([_stroke(_curve(lambda t: (2 * t - 1, a * np.sin(k * np.pi * t + ph)), 90), (.9, .9))])


def corkscrew(rng):  # DS066 helical
    w = rng.uniform(14, 20)
    return Glyph([_stroke(_curve(lambda t: (1.6 * t - .8 + .12 * np.cos(w * t), .22 * np.sin(w * t)), 160), (1, .9))])


def comma(rng):  # DS067
    arc = rng.uniform(1.2, 1.9)
    return Glyph([_stroke(_curve(lambda t: (.6 * np.sin(arc * t) - .3, .6 * (1 - np.cos(arc * t)) - .3), 40), (1.2, .15))])


def crown(rng):  # DS070 ring of short vessels pointing to the centre, not crossing it
    g = Glyph()
    n = rng.integers(7, 11)
    off = rng.uniform(0, 2 * np.pi)
    for i in range(n):
        a = off + 2 * np.pi * i / n + rng.normal(0, .08)
        r = np.linspace(1.0, rng.uniform(.5, .65), 20)
        bend = rng.uniform(-.3, .3)
        g.strokes.append(_stroke(np.stack([r * np.cos(a + bend * (1 - r)), r * np.sin(a + bend * (1 - r))], 1), (.9, .5)))
    return g


def string_of_pearls(rng):  # DS071
    g = Glyph()
    k, a, ph = rng.uniform(1, 1.8), rng.uniform(.25, .4), rng.uniform(0, np.pi)
    for x in np.linspace(-.9, .9, rng.integers(9, 14)):
        g.blobs.append((x, a * np.sin(k * np.pi * (x + 1) / 2 + ph), .07, .07, 0))
    return g


def arborizing(rng, depth=3):  # DS072
    g = Glyph()

    def branch(p, ang, ln, w0, d):
        steps = 14
        pts = [p]
        a = ang
        for _ in range(steps):
            a += rng.normal(0, .06)
            pts.append(pts[-1] + ln / steps * np.array([np.cos(a), np.sin(a)]))
        pts = np.array(pts)
        g.strokes.append((pts, np.linspace(w0, w0 * .55, len(pts))))
        if d > 0:
            for s in (-1, 1):
                if rng.random() < .85:
                    q = pts[rng.integers(7, steps)]
                    branch(q, a + s * rng.uniform(.4, .8), ln * .6, w0 * .65, d - 1)

    branch(np.array([-.9, 0.]), rng.normal(0, .15), 1.2, 1.1, depth)
    return g


def clustered(rng):  # DS073
    g = Glyph()
    c = rng.uniform(-.3, .3, 2)
    for _ in range(rng.integers(6, 10)):
        p = c + rng.normal(0, .22, 2)
        if rng.random() < .5:
            g.blobs.append((p[0], p[1], .1, .1, 0))
        else:
            a = rng.uniform(0, 2 * np.pi)
            d = .09 * np.array([np.cos(a), np.sin(a)])
            g.strokes.append(_stroke(np.vstack([p - d, p + d]), (1, 1)))
    return g


# --- non-vessel structures (placed relative to a lesion, see compose.py) ---

def streak(rng):  # one radial streak, base at -1 end, tip at +1
    ln = rng.uniform(.8, 1.0)
    return Glyph([_stroke(_curve(lambda t: (ln * (2 * t - 1), .05 * np.sin(3 * t)), 20), (1.0, .1))])


def pseudopod(rng):  # DS024 bulbous projection
    g = Glyph([_stroke(_curve(lambda t: (t - .5, 0 * t), 10), (.6, .6))])
    g.blobs.append((.5, 0, .22, .18, 0))
    return g


GLYPHS = {
    "DS060": dotted, "DS061": clods, "DS062": linear, "DS063": coiled,
    "DS064": hairpin, "DS065": serpentine, "DS066": corkscrew, "DS067": comma,
    "DS070": crown, "DS071": string_of_pearls, "DS072": arborizing, "DS073": clustered,
}
# colour / rotation / anchoring metadata; Typ A glyphs only
META = {
    "DS060": dict(rotation="free", extent_px=(14, 30), kind="vessel"),
    "DS061": dict(rotation="free", extent_px=(14, 34), kind="vessel"),
    "DS062": dict(rotation="free", extent_px=(24, 60), kind="vessel"),
    "DS063": dict(rotation="free", extent_px=(12, 26), kind="vessel"),
    "DS064": dict(rotation="free", extent_px=(14, 30), kind="vessel"),
    "DS065": dict(rotation="free", extent_px=(30, 70), kind="vessel"),
    "DS066": dict(rotation="free", extent_px=(26, 56), kind="vessel"),
    "DS067": dict(rotation="free", extent_px=(14, 28), kind="vessel"),
    "DS070": dict(rotation="free", extent_px=(50, 110), kind="vessel"),
    "DS071": dict(rotation="free", extent_px=(36, 80), kind="vessel"),
    "DS072": dict(rotation="free", extent_px=(40, 90), kind="vessel"),
    "DS073": dict(rotation="free", extent_px=(24, 50), kind="vessel"),
}
