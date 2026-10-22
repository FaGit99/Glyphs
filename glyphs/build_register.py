"""Register of all drawn glyph IDs. Per ID: (a) isolated elementary unit rendered by the real generator on a flat
neutral background, with the count forced to one unit; (b) two training-style scenes (ID alone, full composition);
generator, parameters, units per scene, and frequency in the synthetic training set.
Output: out/register/{register.csv, register.html, overview.png, elements/, scenes/}."""
import csv, os, base64
import cv2, numpy as np
import compose, structures as S, skin

OUT = "out/register"
for d in ("elements", "scenes"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
tax = {r["ID"]: r for r in csv.DictReader(open("taxonomy.csv", encoding="utf-8"))}
files = skin.ham_files(0)
tr = np.load("data/synth_train.npz"); ids = [str(i) for i in tr["ids"]]; L = tr["labels"]
png = lambda im: base64.b64encode(cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]).decode()
SIZE, OUTR, CELL = 448, 224, 150


def isolated(sid, seed=7):
    """Real generator on a flat background; ctx placement helpers return exactly one anchor (the lesion centre)."""
    sp = S.SPEC[sid]; rng = np.random.default_rng(seed)
    pale = sp["pale"]
    skin_c = np.array([222, 184, 166], float) if pale else np.array([214, 170, 148], float)
    inner = np.array([196, 150, 140], float) if pale else np.array([150, 98, 82], float)
    rim = inner * .9
    canvas = np.ones((SIZE, SIZE, 3), np.float32) * skin_c
    gy, gx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    r = np.hypot(gx - SIZE / 2, gy - SIZE / 2)
    lesion = np.clip((SIZE * .42 - r) / 3, 0, 1)
    w = lesion[..., None] * (.5 if pale else .95)
    canvas = canvas * (1 - w) + inner * w
    volar = None
    if sp["site"] == "volar":
        volar = S.make_volar(rng, SIZE); canvas *= (1 + .10 * (.5 + .5 * np.cos(volar["phase"]) - .5))[..., None]
    ctx = S.Ctx(rng, canvas, lesion, inner, rim, skin_c, canvas.copy(), OUTR, polarized=sp["polarized"], volar=volar)
    centre = ctx.centroid[None, :].copy()
    ctx.pts_inside = lambda n, **kw: centre if n >= 1 else centre[:0]
    full_rim = ctx.rim_arc
    # one anchor on the rim, pointing outwards, so a single streak / rim dot / rim clod results
    c0 = np.array([SIZE / 2 + SIZE * .42 - 12, SIZE / 2], float)
    ctx.rim_arc = lambda *a, **k: [(c0, np.array([1., 0.]))]
    p = dict(sp["params"])
    fn = sp["fn"].__name__
    if fn in ("g_blobs", "g_dots", "g_rings", "g_spoke", "g_vessel"): p["n"] = (1, 2)
    if fn == "g_leaf": p["groups"] = (1, 2)
    sp["fn"](ctx, sid, **p)
    img = cv2.resize(canvas, (OUTR, OUTR), interpolation=cv2.INTER_AREA)
    img = np.clip(cv2.GaussianBlur(img, (0, 0), .7), 0, 255).astype(np.uint8)
    if not ctx.labels:
        return img[56:168, 56:168], "no unit rendered"
    x0, y0, x1, y1 = ctx.labels[0]["bbox"]
    if (x1 - x0) * (y1 - y0) > .22 * OUTR ** 2:      # area texture: show a central patch, not one 'unit'
        c = img[56:168, 56:168]; note = "texture patch (area structure)"
    else:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; h = max((max(x1 - x0, y1 - y0) + 24) // 2, 26)
        c = img[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]; note = "single unit"
    return cv2.resize(c, (CELL, CELL), interpolation=cv2.INTER_CUBIC), note


def count_units(masks, sid):
    m = np.zeros((OUTR, OUTR), np.uint8)
    for k, v in masks.items():
        if k.startswith(sid + "_"): m |= v.astype(np.uint8)
    return cv2.connectedComponents(m)[0] - 1


rows, N = [], 12
for sid in sorted(S.SPEC):
    sp = S.SPEC[sid]; cnt, comp = [], []
    ex = []
    for s in range(N):
        rng = np.random.default_rng(1000 + int(sid[2:]) * 31 + s)
        img, labels, masks, _, _ = compose.compose(rng, [sid], SIZE, OUTR, skin_files=files)
        cnt.append(len(labels)); comp.append(count_units(masks, sid))
        if s < 2: ex.append(cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_CUBIC))
    elem, note = isolated(sid)
    for im, nm in ((elem, f"elements/{sid}.png"), (np.hstack(ex), f"scenes/{sid}.png")):
        cv2.imwrite(f"{OUT}/{nm}", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    j = ids.index(sid)
    rows.append(dict(ID=sid, metaphoric_term=tax[sid]["Metaphoric term"], descriptive_term=tax[sid]["Descriptive term (IDS)"],
                     generator=sp["fn"].__name__, layer=sp["layer"], parent=sp["parent"], site=sp["site"] or "",
                     params=";".join(f"{k}={v}" for k, v in sp["params"].items()), unit_note=note,
                     label_instances_per_scene=f"{min(cnt)}/{np.mean(cnt):.1f}/{max(cnt)}",
                     connected_units_per_scene=f"{min(comp)}/{np.mean(comp):.1f}/{max(comp)}",
                     train_images_with_id=int(L[:, j].sum()), train_share_pct=round(100 * L[:, j].mean(), 1),
                     _a=png(elem), _b=png(np.hstack(ex))))
    print(sid, note, rows[-1]["connected_units_per_scene"], flush=True)

cols = [k for k in rows[0] if not k.startswith("_")]
with open(f"{OUT}/register.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
h = ["<meta charset=utf-8><title>Glyph register</title><style>body{font:13px sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px;vertical-align:top}img{display:block}.e{width:150px}.s{width:300px}</style>",
     f"<h2>Register of elementary glyphs ({len(rows)} IDs; training set {len(L)} images)</h2>"
     "<p>Elementary unit = the real generator on a flat neutral background, count forced to one unit (area structures: central patch). "
     "Scenes = two training-style compositions with only this ID. Units per scene are min/mean/max over 12 test scenes "
     "(label instances = records written by the generator; connected units = connected components of the instance mask, approximate).</p>"
     "<table><tr><th>elementary unit<th>in training scenes<th>ID / term<th>generator, parameters<th>layer<th>label instances<th>connected units<th>train images with ID"]
for r in rows:
    h.append(f"<tr><td><img class=e src='data:image/png;base64,{r['_a']}'><small>{r['unit_note']}</small>"
             f"<td><img class=s src='data:image/png;base64,{r['_b']}'>"
             f"<td><b>{r['ID']}</b> {r['metaphoric_term']}<br>{r['descriptive_term']}"
             f"<td>{r['generator']}<br><small>{r['params']}</small><td>{r['layer']}<td>{r['label_instances_per_scene']}"
             f"<td>{r['connected_units_per_scene']}<td>{r['train_images_with_id']} ({r['train_share_pct']} %)")
open(f"{OUT}/register.html", "w", encoding="utf-8").write("\n".join(h) + "</table>")
cells = []
for r in rows:
    e = cv2.imread(f"{OUT}/elements/{r['ID']}.png"); cap = np.full((18, CELL, 3), 255, np.uint8)
    cv2.putText(cap, r["ID"], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1); cells.append(np.vstack([cap, e]))
while len(cells) % 9: cells.append(np.full_like(cells[0], 255))
cv2.imwrite(f"{OUT}/overview.png", np.vstack([np.hstack(cells[i:i + 9]) for i in range(0, len(cells), 9)]))
