"""Lesion segmentation (ViT-S/16 at 384 px + small conv decoder).

  python lesion_seg.py prep            # cache 384x384 images + masks (HAM subset, ISIC Task 1 train)
  python lesion_seg.py train [epochs]
  python lesion_seg.py eval

Train: HAM10000 subset + ISIC train blocks; validation: ISIC cal blocks; test: ISIC test blocks (same blocks as hybrid.py).
Input is the whole image squashed to 384x384 (aspect is restored when the mask is resized back).
"""
import glob, os, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from multiprocessing import Pool
import dense as D
from hybrid import real_split

S = 384
RAW = "/mnt/samsung_data/mole_data/data/raw/"
SEG_HAM = "/mnt/samsung_data/mole_data/data/Segmentation/HAM10000_segmentations_lesion_tschandl/"
cv2.setNumThreads(1)


def load_pair(a):
    ip, mp = a
    im, m = cv2.imread(ip), cv2.imread(mp, 0)
    if im is None or m is None:
        return None
    return (cv2.cvtColor(cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB),
            (cv2.resize(m, (S, S), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8))


def prep():
    t = time.time()
    names = sorted(glob.glob(RAW + "isic2018/ISIC2018_Task1-2_Training_Input/*.jpg"))[:2594]
    isic = [(f, RAW + "isic2018/ISIC2018_Task1_Training_GroundTruth/" + os.path.basename(f)[:-4] + "_segmentation.png") for f in names]
    ham_f = sorted(glob.glob(RAW + "HAM10000_images_part_1/*.jpg") + glob.glob(RAW + "HAM10000_images_part_2/*.jpg"))
    ham_f = list(np.random.default_rng(0).permutation(ham_f)[:4000])
    ham = [(f, SEG_HAM + os.path.basename(f)[:-4] + "_segmentation.png") for f in ham_f]
    with Pool(16) as p:
        ri = p.map(load_pair, isic, chunksize=16); rh = p.map(load_pair, ham, chunksize=16)
    assert all(r is not None for r in ri), "missing ISIC pair"
    rh = [r for r in rh if r is not None]
    np.savez("cache/seg_isic.npz", x=np.stack([r[0] for r in ri]), y=np.stack([r[1] for r in ri]), names=np.array([os.path.basename(f) for f, _ in isic]))
    np.savez("cache/seg_ham.npz", x=np.stack([r[0] for r in rh]), y=np.stack([r[1] for r in rh]))
    print(f"ISIC {len(ri)}, HAM {len(rh)}, {time.time() - t:.0f}s")


class SegViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.b = timm.create_model(D.BACKBONE, pretrained=pretrained, num_classes=0, img_size=S)
        d = self.b.num_features
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())
        self.dec = nn.Sequential(blk(d, 192), nn.Upsample(scale_factor=2, mode="bilinear"), blk(192, 96),
                                 nn.Upsample(scale_factor=2, mode="bilinear"), blk(96, 48), nn.Upsample(scale_factor=2, mode="bilinear"), nn.Conv2d(48, 1, 1))

    def forward(self, x):
        f = self.b.forward_features(x)[:, 1:]
        g = int(f.shape[1] ** .5)
        y = self.dec(f.transpose(1, 2).reshape(x.shape[0], -1, g, g))      # stride 2
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear")[:, 0]


def aug(x, y):
    k = torch.randint(0, 4, (x.shape[0],), device=D.dev)
    for r in range(1, 4):
        m = k == r
        if m.any():
            x[m] = torch.rot90(x[m], r, (2, 3)); y[m] = torch.rot90(y[m], r, (1, 2))
    fl = torch.rand(x.shape[0], device=D.dev) < .5
    x[fl] = x[fl].flip(3); y[fl] = y[fl].flip(2)
    x = x * (1 + .15 * torch.randn(x.shape[0], 1, 1, 1, device=D.dev)) + 10 * torch.randn(x.shape[0], 3, 1, 1, device=D.dev)
    return x, y


def norm(x):
    return (x.clamp(0, 255) - D.MEAN) / D.STD


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.as_tensor(np.ascontiguousarray(X[i:i + bs])).to(D.dev).permute(0, 3, 1, 2).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(norm(x)).float()).cpu())
    return torch.cat(out).numpy()


def metrics(p, y):
    pb = p > .5; inter = (pb & (y > 0)).sum((1, 2)); uni = (pb | (y > 0)).sum((1, 2)); s = pb.sum((1, 2)) + (y > 0).sum((1, 2))
    return 2 * inter / np.maximum(s, 1), inter / np.maximum(uni, 1)


def data():
    a, b = np.load("cache/seg_isic.npz"), np.load("cache/seg_ham.npz"); sp = real_split()
    return a, b, sp


def train(epochs):
    a, b, sp = data()
    tr = np.concatenate([a["x"][sp["train"]], b["x"]]); ty = np.concatenate([a["y"][sp["train"]], b["y"]])
    vx, vy = a["x"][sp["cal"]], a["y"][sp["cal"]]
    Xt, Yt = torch.from_numpy(tr), torch.from_numpy(ty)
    model = SegViT().to(D.dev); bs = 32; per = len(Xt) // bs
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4}, {"params": model.dec.parameters(), "lr": 5e-4}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 5e-4], total_steps=epochs * per, pct_start=.1)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(per):
            idx = perm[i * bs:(i + 1) * bs]
            x = Xt[idx].to(D.dev).permute(0, 3, 1, 2).float(); y = Yt[idx].to(D.dev).float()
            x, y = aug(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(norm(x))
            p = torch.sigmoid(lg.float())
            loss = F.binary_cross_entropy_with_logits(lg.float(), y) + (1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        d, j = metrics(predict(model, vx), vy)
        print(f"epoch {ep + 1}/{epochs} loss {tl / per:.4f}  val(ISIC cal) Dice {d.mean():.3f} IoU {j.mean():.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict()}, "models/lesion_seg.pt")


def evaluate():
    a, _, sp = data()
    model = SegViT(pretrained=False).to(D.dev); model.load_state_dict(torch.load("models/lesion_seg.pt")["state"])
    p = predict(model, a["x"][sp["test"]]); d, j = metrics(p, a["y"][sp["test"]])
    print(f"ISIC test blocks ({len(d)} images): Dice {d.mean():.3f} (median {np.median(d):.3f}), IoU {j.mean():.3f}; images with Dice<0.8: {(d < .8).mean():.1%}, <0.5: {(d < .5).mean():.1%}")
    # boundary band quality: IoU of the +-16px (at 450 px scale ~ 14 px at 384) contour band
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)); bi = []
    for pp, yy in zip(p, a["y"][sp["test"]]):
        pb = (pp > .5).astype(np.uint8); band = lambda m: cv2.dilate(m, k) - cv2.erode(m, k)
        b1, b2 = band(pb) > 0, band(yy) > 0; bi.append((b1 & b2).sum() / max((b1 | b2).sum(), 1))
    print(f"contour-band (+-14 px) IoU {np.mean(bi):.3f}")


if __name__ == "__main__":
    {"prep": prep, "train": lambda: train(int(sys.argv[2]) if len(sys.argv) > 2 else 20), "eval": evaluate}[sys.argv[1]]()
