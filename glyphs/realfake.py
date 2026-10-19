"""Real-vs-synthetic check at native resolution.

Real  : centre 224 px crops of HAM10000 images (parity 1 files, native pixels).
Synth : compose() scenes at 224 px (skin backgrounds from parity 0 files only).
Zones : whole image, centre 96 px (lesion zone), 64 px corner (skin zone) - shows WHERE they differ.
"""
import sys, time, os
import cv2, numpy as np
import compose, geometry as G, skin

compose.VIGNETTE_P = 0.0
S = 224
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = "out/realfake"; os.makedirs(OUT, exist_ok=True)


def jpeg(img, q):
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)


def make_synth(n):
    files = skin.ham_files(0)
    out = []
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        ids = compose.random_scene_ids(rng, k=(2, 6))  # all registry IDs except volar (acral) ones
        img, *_ = compose.compose(rng, ids, 448, S, skin_files=files)
        out.append(img)  # no JPEG re-encode: it adds a detectable artefact (see floor_test.py)
    return out


def load_real(n):
    files = skin.ham_files(1)
    rng = np.random.default_rng(0)
    out = []
    for f in rng.choice(files, n, replace=False):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]; y, x = (h - S) // 2, (w - S) // 2
        out.append(im[y:y + S, x:x + S])
    return out


def feats(im):
    n = min(im.shape[:2]); im = im[:n, :n]
    f, names = [], []
    def add(k, v):
        v = np.atleast_1d(v); f.extend(v.tolist()); names.extend([f"{k}{i}" if len(v) > 1 else k for i in range(len(v))])
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for k, nm in enumerate("LAB"): add(f"lab_{nm}_mean", lab[..., k].mean()); add(f"lab_{nm}_std", lab[..., k].std())
    for k, nm in enumerate("HSV"): add(f"hsv_{nm}_mean", hsv[..., k].mean()); add(f"hsv_{nm}_std", hsv[..., k].std())
    for k, nm in enumerate("LAB"): add(f"hist_{nm}_", np.histogram(lab[..., k], 8, (0, 255))[0] / g.size)
    add("lap_var", cv2.Laplacian(g, cv2.CV_32F).var())
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)); add("grad_pct", np.percentile(mag, [50, 90, 99]))
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    add("noise_std", res.std()); add("noise_kurt", ((res - res.mean()) ** 4).mean() / (res.var() ** 2 + 1e-6))
    b = n // 8; blk = g[:n // b * b, :n // b * b].reshape(n // b, b, n // b, b).std(axis=(1, 3)); add("blockstd_pct", np.percentile(blk, [10, 50, 90]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))); yy, xx = np.indices(F.shape); r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    rad = (np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[:n // 2]
    add("spec_slope", np.polyfit(np.log(np.arange(2, n // 2)), np.log(rad[2:] + 1e-6), 1)[0]); add("hf_share", rad[n // 4:].sum() / rad.sum())
    add("sat_p99", np.percentile(hsv[..., 1], 99))
    return np.array(f, np.float32), names


def zone(im, name):
    c = S // 2
    return {"whole 224": im, "centre 96 (lesion)": im[c - 48:c + 48, c - 48:c + 48], "corner 64 (skin)": im[:64, :64]}[name]


if __name__ == "__main__":
    t = time.time()
    syn, real = make_synth(N), load_real(N)
    print(f"images ready {time.time() - t:.0f}s", flush=True)
    for i in range(8):
        cv2.imwrite(f"{OUT}/pair{i}.png", cv2.cvtColor(np.hstack([syn[i], real[i]]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{OUT}/pairs.png", np.vstack([np.hstack([cv2.imread(f"{OUT}/pair{i}.png") for i in (j, j + 1)]) for j in (0, 2, 4, 6)]))
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    y = np.array([1] * N + [0] * N); cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for zn in ("whole 224", "centre 96 (lesion)", "corner 64 (skin)"):
        X, names = zip(*[feats(zone(im, zn)) for im in syn + real]); X = np.stack(X); names = names[0]
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        print(f"\n[{zn}] 5-fold AUC = {roc_auc_score(y, p):.3f}")
        clf.fit(X, y)
        for i in np.argsort(clf.feature_importances_)[::-1][:4]:
            print(f"   {names[i]:14s} imp={clf.feature_importances_[i]:.2f}  synth={X[y == 1, i].mean():8.2f}  real={X[y == 0, i].mean():8.2f}")
