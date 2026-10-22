"""Fine-tune the hybrid model with Derm7pt 7-point-checklist labels as image-level (MIL) targets.

  python train_melanoma.py fold <k>      # hold out fold k of train+valid (5 folds); writes out-of-fold predictions
  python train_melanoma.py final         # train on train+valid; writes test predictions
Extra steps arg: python train_melanoma.py fold 0 800
Batch per step: 64 synthetic scenes + 32 real ISIC tiles (as in hybrid.py) + 8 Derm7pt images x 12 lesion tiles.
Derm7pt channels: image logit = mean of the top-3 tile logits (multiple-instance pooling).
"""
import sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import dense as D, hybrid as H, d7_data as DD, mil

mode = sys.argv[1]; fold = int(sys.argv[2]) if mode == "fold" else -1
steps = int(sys.argv[-1]) if sys.argv[-1].isdigit() and len(sys.argv) > (3 if mode == "fold" else 2) else 800
NI, NR, ND, NT = 64, 32, 8, 12        # synthetic, ISIC tiles, Derm7pt images, tiles per Derm7pt image
LAM_D7 = 1.0
torch.manual_seed(7 + max(fold, 0)); rng = np.random.default_rng(7 + max(fold, 0))

sp = DD.split(); pool = np.concatenate([sp["train"], sp["valid"]])
if mode == "fold":
    perm = np.random.default_rng(0).permutation(pool); folds = np.array_split(perm, 5)
    held = folds[fold]; train_idx = np.setdiff1d(pool, held)
else:
    held, train_idx = sp["test"], pool
Y7 = torch.from_numpy(DD.labels()); tiles, cnt = mil.load_tiles()

# model: copy the 76 hybrid outputs, add the 14 Derm7pt channels
base, ck = H.load_model("models/hybrid.pt"); K = ck["n_out"]; n_new = len(DD.TARGETS)
model = D.DenseViT(K + n_new, pretrained=False).to(D.dev)
sd = base.state_dict()
for k_, v in model.state_dict().items():
    if k_ in sd and sd[k_].shape == v.shape:
        v.copy_(sd[k_])
    elif k_ in sd:                                   # head rows: copy the old outputs, keep new ones at init
        v[:sd[k_].shape[0]] = sd[k_]
ids = [str(i) for i in ck["ids"]]; nid = len(ids); amap = H.attr_ids(ids)

# data sources
d = np.load("data/synth_train.npz"); dv = D.dev                                                                    # Trainingsdaten auf der GPU
Xs, Rs, Ys = torch.from_numpy(d["images"]).to(dv), torch.from_numpy(d["regions"]).to(dv), torch.from_numpy(d["labels"]).float().to(dv)
As, ys = H.synth_attr(d["regions"], d["labels"], amap); As, ys = torch.from_numpy(As).to(dv), torch.from_numpy(ys).float().to(dv)
sp_r = H.real_split(); Xr, Ar, _ = H.real_data(sp_r["train"]); Xr, Ar = torch.from_numpy(Xr).to(dv), torch.from_numpy(Ar).to(dv)
yr = (Ar.float().mean((2, 3)) / 255. >= .01).float()
ytr = Y7[train_idx]; posw = ((1 - ytr.mean(0)) / ytr.mean(0).clamp(min=.02)).clamp(1, 6).to(D.dev)

opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 5e-5}, {"params": list(model.img.parameters()) + list(model.pat.parameters()), "lr": 5e-4}], weight_decay=.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 5e-4], total_steps=steps, pct_start=.1)
pw = torch.tensor(H.POS_W, device=D.dev); t0 = time.time(); K76 = K


def d7_batch():
    imgs = rng.choice(train_idx, ND, replace=False); x = []
    for j in imgs:
        sel = rng.choice(int(cnt[j]), NT, replace=int(cnt[j]) < NT); x.append(np.asarray(tiles[j][sel]))
    return torch.from_numpy(np.concatenate(x)), Y7[imgs]


for step in range(steps):
    model.train()
    si = torch.randint(0, len(Xs), (NI,), device=dv); ri = torch.randint(0, len(Xr), (NR,), device=dv); xd, yd = d7_batch(); xd = xd.to(dv)
    x = torch.cat([Xs[si], Xr[ri], xd]); B = len(x)
    maps = torch.zeros(NI + NR + len(xd), K, 28, 28, device=dv); maps[:NI, :nid] = Rs[si].float(); maps[:NI, nid:K] = As[si].float(); maps[NI:NI + NR, nid:K] = Ar[ri].float()
    x, maps = D.prep(x, maps, True)
    yimg = torch.zeros(B, K, device=dv); yimg[:NI, :nid] = Ys[si]; yimg[:NI, nid:K] = ys[si]; yimg[NI:NI + NR, nid:K] = yr[ri]
    mask = torch.zeros(B, K, device=dv); mask[:NI] = 1; mask[NI:NI + NR, nid:K] = 1
    yimg, mask = yimg.to(D.dev), mask.to(D.dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        li, lm = model(x)
    li = li.float(); lm = lm.float()
    n_ = NI + NR
    tgt = F.avg_pool2d(maps[:n_] / 255., 2)
    loss = H.masked_bce(li[:n_, :K], yimg[:n_], mask[:n_]) + H.MAP_W * H.masked_bce(lm[:n_, :K], tgt, mask[:n_, :, None, None].expand_as(tgt), pw)
    z = li[n_:, K:].reshape(ND, NT, n_new).topk(mil.TOPK, dim=1).values.mean(1)         # MIL pooling of tile logits
    l_d7 = F.binary_cross_entropy_with_logits(z, yd.to(D.dev), pos_weight=posw)
    loss = loss + LAM_D7 * l_d7
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if (step + 1) % 100 == 0:
        print(f"step {step + 1}/{steps} loss {loss.item():.3f} (d7 {l_d7.item():.3f})  {time.time() - t0:.0f}s", flush=True)

pred = mil.agg_predict(model, tiles, cnt, held)
tag = f"mel_fold{fold}" if mode == "fold" else "mel_final"
np.savez(f"models/{tag}_pred.npz", pred=pred, idx=held)
torch.save({"state": model.state_dict(), "ids": ids, "n_out": K + n_new}, f"models/{tag}.pt")
from sklearn.metrics import roc_auc_score
ho = Y7[held].numpy(); a = [roc_auc_score(ho[:, c], pred[:, K + c]) if 0 < ho[:, c].sum() < len(ho) else float("nan") for c in range(n_new)]
print(f"{tag}: held-out {len(held)} images; 7-point-channel AUC mean {np.nanmean(a):.3f}: " + ", ".join(f"{n}:{x:.2f}" for n, x in zip(DD.TNAMES, a)))
