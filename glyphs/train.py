"""Fine-tune a ViT-S/16 on the synthetic dataset (multi-label, one sigmoid per taxonomy ID).

Usage: python train.py <tag> [epochs] [--scratch]
"""
import sys, time, json
import numpy as np, torch, timm, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

tag = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 12
dev = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255


def load(name):
    d = np.load(f"data/synth_{name}.npz")
    return torch.from_numpy(d["images"]), torch.from_numpy(d["labels"]).float(), list(d["ids"])


def prep(xb, train):
    x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    if train:  # dermoscopy is rotation/flip invariant; mild photometric + blur/noise jitter
        k = torch.randint(0, 4, (x.shape[0],), device=dev)
        for r in range(1, 4):
            m = k == r
            if m.any():
                x[m] = torch.rot90(x[m], r, (2, 3))
        fl = torch.rand(x.shape[0], device=dev) < .5
        x[fl] = x[fl].flip(3)
        x = x * (1 + .1 * torch.randn(x.shape[0], 1, 1, 1, device=dev)) + 8 * torch.randn(x.shape[0], 3, 1, 1, device=dev)
        x = x + torch.randn_like(x) * (torch.rand(x.shape[0], 1, 1, 1, device=dev) * 3)
    return (x.clamp(0, 255) - MEAN) / STD


def build(n_out, scratch=False):
    m = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=not scratch, num_classes=n_out)
    return m.to(dev)


@torch.no_grad()
def predict(model, imgs, bs=256):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(prep(imgs[i:i + bs], False)).float()).cpu())
    return torch.cat(out).numpy()


def mean_auc(y, p):
    a = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a)), len(a)


if __name__ == "__main__":
    Xtr, Ytr, ids = load("train"); Xva, Yva, _ = load("val")
    model = build(len(ids), scratch="--scratch" in sys.argv)
    bs, steps = 128, epochs * (len(Xtr) // 128)
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not n.startswith("head")], "lr": 1e-4},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=.1)
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x, y = prep(Xtr[idx], True), Ytr[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item(); step += 1
        auc, n = mean_auc(Yva.numpy(), predict(model, Xva))
        print(f"epoch {ep + 1}/{epochs} loss {tot / (len(perm) // bs):.4f}  synthetic-val mean AUC {auc:.3f} ({n} IDs)  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "ids": [str(i) for i in ids]}, f"models/{tag}.pt")
