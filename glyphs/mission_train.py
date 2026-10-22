"""Mission synthetic-vs-mixed: training regimes on the hybrid3.py pipeline (same architecture, folds, steps, batch layout).

  python mission_train.py <name> <fold> --synth 0|1 --isic 0|1 --easy <frac> [--seed s] [--epochs 16]
    fold 1..4: EASY group <fold> held out (OOF); fold 0: all EASY images; --easy 0: no EASY at all
    --easy <frac>: share of the training-fold EASY images used, stratified by exemplar feature (nested: 25 % subset of 50 % of 75 %)
    --tids DS038,DS006,DS037 --tdose <n>: per target channel only n positive EASY images keep their mask; the other positives of
      that channel are ignored for that channel (image stays in training for all other channels, negatives unchanged).
      Independent draw per channel, nested over n within a seed. --tdose -1: no withholding.
    --srcnorm 1: loss = loss on synthetic rows + loss on ISIC/EASY rows, each averaged over its own entries, so real rows carry the
      same weight as with --synth 0 (default 0: one average over the whole batch, as in hybrid3.py).
Output: models/mission_<name>_f<fold>.pt

Differences to hybrid3.py are only the switches: a disabled source is removed from the batch (its rows are not replaced), so the
number of optimiser steps (16 epochs x 166) and the per-step count of the remaining sources stay identical to V3.
"""
import argparse, csv, time
import numpy as np, torch, torch.nn.functional as F
import dense as D
import hybrid as H
import easy_eval as EE
from hybrid3 import easy_targets, SKIP


def easy_subset(grp, fold, frac, seed=0):
    """training-fold EASY image indices; per exemplar feature a fixed permutation, the first round(frac*n) are taken"""
    names = [str(n) for n in np.load("cache/easy_v3.npz")["names"]]
    ex = {r["ISIC_id"]: r["exemplar"] for r in csv.DictReader(open(EE.E + "metadata.csv"))}
    pool = np.nonzero(grp != fold)[0] if fold > 0 else np.arange(len(grp))
    if frac >= 1:
        return pool
    rng = np.random.default_rng(1000 + 10 * fold + seed)
    feats = sorted({ex[names[i]] for i in pool})
    sel = {f: (lambda s: s[rng.permutation(len(s))])(np.array([i for i in pool if ex[names[i]] == f])) for f in feats}
    tie = rng.random(len(feats))                                         # fixed tie-break, identical for every frac
    want = np.array([frac * len(sel[f]) for f in feats]); k = np.floor(want).astype(int)
    rest = int(round(frac * len(pool))) - k.sum()                        # largest remainder -> exact total
    k[np.lexsort((tie, -(want - k)))[:rest]] += 1
    return np.array(sorted(i for f, n in zip(feats, k) for i in sel[f][:n]))


def target_withhold(use, ids, tids, n, fold, seed):
    """-> {channel: training-image indices whose positive mask for that channel is withheld}"""
    z = np.load("cache/easy_v3.npz"); nrd, feats = z["nrd"], [str(f) for f in z["feats"]]
    out = {}
    for k, t in enumerate(tids):
        fis = [fi for fi, f in enumerate(feats) if f not in SKIP and EE.MAP[f][0] == [t]]
        pos = use[(nrd[use][:, fis] >= 2).any(1)]
        perm = pos[np.random.default_rng(5000 + 100 * fold + 10 * seed + k).permutation(len(pos))]
        out[ids.index(t)] = perm[n:]
        print(f"target {t}: {len(pos)} positive training images, mask kept for {min(n, len(pos))}", flush=True)
    return out


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ids = H.id_list(); K = len(ids); amap = H.attr_ids(ids); C = K + 5; dv = D.dev
    d = np.load("data/synth_train.npz")
    per_epoch = len(d["labels"]) // 96                                   # 166 steps/epoch as in hybrid3.py, for every regime
    nS = 96 if a.synth else 0; nR = 32 if a.isic else 0; nE = 16 if a.easy > 0 else 0
    if nS:
        Xs, Rs, Ys = torch.from_numpy(d["images"]).to(dv), torch.from_numpy(d["regions"]).to(dv), torch.from_numpy(d["labels"]).float().to(dv)
        As, ys = H.synth_attr(d["regions"], d["labels"], amap); As, ys = torch.from_numpy(As).to(dv), torch.from_numpy(ys).float().to(dv)
    del d
    if nR:
        sp = H.real_split(); Xr, Ar, _ = H.real_data(sp["train"])
        Xr = torch.from_numpy(Xr).to(dv); Ar = torch.from_numpy(Ar).to(dv); yr = (Ar.float().mean((2, 3)) / 255. >= .01).float()
    if nE:
        T, M, yi, vi, grp, V = easy_targets(ids)
        use = easy_subset(grp, a.fold, a.easy)
        if a.tids and a.tdose >= 0:
            for c, wi in target_withhold(use, ids, a.tids.split(","), a.tdose, a.fold, a.seed).items():
                M[wi, :, c] = 0; vi[wi, :, c] = 0
        Xe = torch.from_numpy(np.ascontiguousarray(V[use].reshape(-1, 224, 224, 3))).to(dv)
        Te = torch.from_numpy(T[use].reshape(-1, C, 28, 28)).to(dv); Me = torch.from_numpy(M[use].reshape(-1, C, 28, 28)).to(dv)
        ye = torch.from_numpy(yi[use].reshape(-1, C)).to(dv); ve = torch.from_numpy(vi[use].reshape(-1, C)).to(dv)
        print("EASY training images:", len(use), "tiles:", len(Xe), flush=True)
    print(f"sources: synth {nS}, ISIC {nR}, EASY {nE} per step; {a.epochs} x {per_epoch} steps; seed {a.seed}", flush=True)
    model = D.DenseViT(C).to(dv)
    steps = a.epochs * per_epoch
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4},
                             {"params": list(model.img.parameters()) + list(model.pat.parameters()), "lr": 1e-3}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=.1)
    pw = torch.tensor(H.POS_W, device=dv); t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xs), device=dv) if nS else None; tl = 0
        for it in range(per_epoch):
            B = nS + nR + nE; xs = []
            maps = torch.zeros(B, C, 28, 28, device=dv); mm = torch.full((B, C, 28, 28), 255., device=dv)
            yimg = torch.zeros(B, C, device=dv); mask = torch.ones(B, C, device=dv)
            if nS:
                si = perm[it * nS:(it + 1) * nS]; xs.append(Xs[si])
                maps[:nS, :K] = Rs[si].float(); maps[:nS, K:] = As[si].float(); yimg[:nS, :K] = Ys[si]; yimg[:nS, K:] = ys[si]
            if nR:
                ri = torch.randint(0, len(Xr), (nR,), device=dv); xs.append(Xr[ri]); o = nS
                maps[o:o + nR, K:] = Ar[ri].float(); mm[o:o + nR, :K] = 0; yimg[o:o + nR, K:] = yr[ri]; mask[o:o + nR, :K] = 0
            if nE:
                ei = torch.randint(0, len(Xe), (nE,), device=dv); xs.append(Xe[ei]); o = nS + nR
                maps[o:] = Te[ei].float(); mm[o:] = Me[ei].float(); mm[o:, K:] = 0
                yimg[o:] = ye[ei]; mask[o:] = ve[ei]; mask[o:, K:] = 0
            x, both = D.prep(torch.cat(xs), torch.cat([maps, mm], 1), True)
            maps, mm = both[:, :C], both[:, C:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                li, lm = model(x)
            tgt = F.avg_pool2d(maps / 255., 2); mcell = (F.avg_pool2d(mm / 255., 2) > .99).float()
            lossf = lambda sl: H.masked_bce(li[sl].float(), yimg[sl], mask[sl]) + H.MAP_W * H.masked_bce(lm[sl].float(), tgt[sl], (mcell * mask[:, :, None, None])[sl], pw)
            if a.srcnorm and nS and (nR or nE):          # synthetic loss + real loss, each normalised on its own rows (real part = --synth 0 loss)
                loss = lossf(slice(0, nS)) + lossf(slice(nS, None))
            else:
                loss = lossf(slice(None))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {tl / per_epoch:.4f}  {time.time() - t0:.0f}s", flush=True)
    out = f"models/mission_{a.name}_f{a.fold}.pt"
    torch.save({"state": model.state_dict(), "ids": ids, "n_out": C, "mode": "mission", "fold": a.fold, "args": vars(a)}, out); print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name"); p.add_argument("fold", type=int)
    p.add_argument("--synth", type=int, default=1); p.add_argument("--isic", type=int, default=1); p.add_argument("--easy", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--tids", default=""); p.add_argument("--srcnorm", type=int, default=0); p.add_argument("--tdose", type=int, default=-1)
    train(p.parse_args())
