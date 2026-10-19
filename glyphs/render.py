"""Appearance layer: turns geometry into soft, coloured alpha layers.

All draw_* functions return a float mask in [0, 1] of shape (S, S); `blend`
composites a colour layer onto the canvas. Instance masks come for free.
"""
import cv2
import numpy as np

SS = 4  # supersampling factor

COLORS = {  # RGB by semantic colour role, not per glyph
    "red": (208, 58, 72),
    "brown": (85, 52, 38),
    "blue_gray": (105, 118, 150),
    "white": (240, 235, 230),
}


def _xf(pts, x, y, scale, rot):
    c, s = np.cos(rot), np.sin(rot)
    r = np.array([[c, -s], [s, c]])
    return (np.asarray(pts) * scale) @ r.T + np.array([x, y])


def draw_glyph(g, size, x, y, scale, rot, width_px, rng):
    """Rasterise one glyph instance -> alpha mask (size x size)."""
    big = np.zeros((size * SS, size * SS), np.uint8)
    for pts, prof in g.strokes:
        p = _xf(pts, x, y, scale, rot) * SS
        for i in range(len(p) - 1):
            w = max(1, int(round(width_px * SS * 0.5 * (prof[i] + prof[i + 1]))))
            cv2.line(big, tuple(np.round(p[i]).astype(int)), tuple(np.round(p[i + 1]).astype(int)), 255, w, cv2.LINE_AA)
    for cx, cy, rx, ry, ang in g.blobs:
        c = _xf([[cx, cy]], x, y, scale, rot)[0] * SS
        cv2.ellipse(big, (int(c[0]), int(c[1])), (max(1, int(rx * scale * SS)), max(1, int(ry * scale * SS))),
                    np.degrees(rot) + ang, 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def soften(mask, sigma):
    return cv2.GaussianBlur(mask, (0, 0), sigma) if sigma > 0 else mask


def blend(canvas, mask, color, alpha=1.0, shade=0.0):
    """canvas float RGB 0..255 modified in place. shade darkens the mask core."""
    col = np.array(color, np.float32)[None, None, :]
    layer = np.ones_like(canvas) * col
    if shade:
        core = cv2.erode(mask, np.ones((3, 3), np.uint8))
        layer = layer * (1 - shade * core[..., None])
    a = np.clip(mask * alpha, 0, 1)[..., None]
    canvas[:] = canvas * (1 - a) + layer * a


def vessel_look(canvas, mask, rng):
    """Red vessels: slightly blurred, translucent, brighter rim."""
    m = soften(mask, rng.uniform(.6, 1.0))
    blend(canvas, m, COLORS["red"], alpha=rng.uniform(.82, .95), shade=0)


def finish(img, rng):
    """Sensor-level effects at native resolution: faint grain, contrast jitter."""
    img = img + rng.normal(0, rng.uniform(.4, 1.2), img.shape)
    img = (img - 128) * rng.uniform(.95, 1.08) + 128
    return np.clip(img, 0, 255).astype(np.uint8)
