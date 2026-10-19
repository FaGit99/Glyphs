"""ViT-S/16 with an image-level head (CLS) and a patch head (one probability map per taxonomy ID, 14x14)."""
import numpy as np, timm, torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
BACKBONE = "vit_small_patch16_224.augreg_in21k_ft_in1k"


class DenseViT(nn.Module):
    def __init__(self, n_out, pretrained=True):
        super().__init__()
        self.b = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        d = self.b.num_features
        self.img = nn.Linear(d, n_out)
        self.pat = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(self, x):
        f = self.b.forward_features(x)                     # [B, 1+196, D]
        cls, tok = f[:, 0], f[:, 1:]
        B, N, D = tok.shape
        g = int(N ** .5)
        return self.img(cls), self.pat(tok).transpose(1, 2).reshape(B, -1, g, g)


def prep(xb, maps=None, train=False):
    """xb uint8 [B,H,W,3]; maps optional float [B,K,28,28]; same flips/rotations applied to both."""
    x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    if maps is not None:
        maps = maps.to(dev, non_blocking=True)
    if train:
        k = torch.randint(0, 4, (x.shape[0],), device=dev)
        for r in range(1, 4):
            m = k == r
            if m.any():
                x[m] = torch.rot90(x[m], r, (2, 3))
                if maps is not None:
                    maps[m] = torch.rot90(maps[m], r, (2, 3))
        fl = torch.rand(x.shape[0], device=dev) < .5
        x[fl] = x[fl].flip(3)
        if maps is not None:
            maps[fl] = maps[fl].flip(3)
        x = x * (1 + .1 * torch.randn(x.shape[0], 1, 1, 1, device=dev)) + 8 * torch.randn(x.shape[0], 3, 1, 1, device=dev)
        x = x + torch.randn_like(x) * (torch.rand(x.shape[0], 1, 1, 1, device=dev) * 3)
    x = (x.clamp(0, 255) - MEAN) / STD
    return x if maps is None else (x, maps)


@torch.no_grad()
def predict(model, imgs, bs=256):
    """imgs uint8 numpy/tensor [N,224,224,3] -> (image probs [N,K], map probs [N,K,14,14])."""
    model.eval(); pi, pm = [], []
    for i in range(0, len(imgs), bs):
        xb = torch.as_tensor(np.ascontiguousarray(imgs[i:i + bs]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, m = model(prep(xb))
        pi.append(torch.sigmoid(a.float()).cpu()); pm.append(torch.sigmoid(m.float()).cpu())
    return torch.cat(pi).numpy(), torch.cat(pm).numpy()
