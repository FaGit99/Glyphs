"""Lesion/skin colour statistics from real HAM10000 images (ratio-transfer palette for synthetic lesions)."""
import glob, os, numpy as np, cv2
R = "/mnt/samsung_data/mole_data/data/raw/"
SEG = "/mnt/samsung_data/mole_data/data/Segmentation/HAM10000_segmentations_lesion_tschandl/"
files = sorted(glob.glob(R + "HAM10000_images_part_1/*.jpg") + glob.glob(R + "HAM10000_images_part_2/*.jpg"))
rng = np.random.default_rng(0)
rec = []
for f in rng.permutation(files)[:1500]:
    img = cv2.imread(f); m = cv2.imread(SEG + os.path.basename(f)[:-4] + "_segmentation.png", 0)
    if img is None or m is None: continue
    m = (m > 127).astype(np.uint8)
    if not (.03 < m.mean() < .6): continue
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    inner = cv2.erode(m, k(int(2 * np.sqrt(m.sum() / np.pi) * .5)))          # core: shrunk to ~half the diameter
    rim = m.astype(bool) & ~cv2.erode(m, k(int(2 * np.sqrt(m.sum() / np.pi) * .25))).astype(bool)
    skin = ~cv2.dilate(m, k(41)).astype(bool)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); skin &= g > 80
    if inner.sum() < 200 or rim.sum() < 200 or skin.sum() < 2000: continue
    rgb = img[..., ::-1].astype(np.float32)
    hp = g.astype(np.float32) - cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
    rec.append(np.concatenate([rgb[inner.astype(bool)].mean(0), rgb[rim].mean(0), rgb[skin].mean(0), [hp[m.astype(bool)].std(), hp[skin].std()]]))
rec = np.array(rec, np.float32)
np.savez("palette.npz", inner=rec[:, 0:3], rim=rec[:, 3:6], skin=rec[:, 6:9], hf_lesion=rec[:, 9], hf_skin=rec[:, 10])
c = rec[:, 0:3].mean(1) / rec[:, 6:9].mean(1)
print(len(rec), "records; inner/skin brightness ratio percentiles 5/25/50/75/95:", np.percentile(c, [5, 25, 50, 75, 95]).round(2))
print("pale (>0.82):", (c > .82).sum(), " pigmented (<0.75):", (c < .75).sum())
