"""V3-Training: Hybrid (synthetisch + ISIC-2018-Masken) plus EASY-Reader-Masken für direkt zugeordnete ID-Kanäle.

  python hybrid3.py train <fold> [epochs]   # fold 1..4: EASY-Gruppe <fold> wird nicht trainiert (Out-of-fold-Test);
                                            # fold 0: alle EASY-Bilder (Endmodell); fold -1: ohne EASY (Kontrolle, gleiche Pipeline/Seed)
Ausgabe: models/hybrid_v3_f<fold>.pt  (fold -1 -> hybrid_v3_noeasy.pt)

EASY-Ziele je ID-Kanal (nur Kanäle mit genau einem zugeordneten EASY-Merkmal bzw. einer Gruppe von Merkmalen mit derselben ID):
  Bild positiv = >=2 Leser markieren (irgendein zugeordnetes Merkmal); negativ = kein Leser markiert (wie in V2: 'kein Leser' = negativ);
  genau 1 Leser (und kein Merkmal mit >=2) = ganzer Kanal ignoriert. Zelle: >=2 Leser -> 1; 1 Leser -> ignoriert; 0 -> 0.
ISIC-Attributkanäle erhalten aus EASY keine Supervision.
"""
import sys, time
import numpy as np, torch, torch.nn.functional as F
import dense as D
import hybrid as H
import easy_eval as EE

# Merkmale, deren Zuordnung zu grob/mehrdeutig ist, werden nicht zum Training benutzt (Auswertung weiter für alle)
SKIP = {"Pattern : Homogeneous : NOS", "Structureless : Structureless brown (tan)", "Regression structures : Scarlike depigmentation",
        "Lines : Angulated lines / Polygons / Zig-zag pattern"}


def easy_targets(ids):
    """-> T [N,5,K+5,28,28] uint8 (0/255 Ziel), M (gültig), yimg [N,5,K+5], vimg [N,5,K+5] (Bild-Ebene gültig)"""
    z = np.load("cache/easy_v3.npz"); cnt, nrd, feats = z["cnt"], z["nrd"], [str(f) for f in z["feats"]]
    N, K = len(nrd), len(ids)
    T = np.zeros((N, 5, K + 5, 28, 28), np.uint8); M = np.zeros_like(T)
    yimg = np.zeros((N, 5, K + 5), np.float32); vimg = np.zeros((N, 5, K + 5), np.float32)
    byid = {}
    for fi, f in enumerate(feats):
        mids, _ = EE.MAP[f]
        if f in SKIP or len(mids) != 1 or mids[0] not in ids:
            continue
        byid.setdefault(ids.index(mids[0]), []).append(fi)
    for c, fis in byid.items():
        pos = (nrd[:, fis] >= 2).any(1); neg = (nrd[:, fis] == 0).all(1)
        for i in range(N):
            if not (pos[i] or neg[i]):
                continue                                                  # mehrdeutig: ganzer Kanal ignoriert
            cc = cnt[i][fis][:, :5]                                       # [nf,5,28,28]
            ok = (cc >= 2).any(0) if pos[i] else np.zeros((5, 28, 28), bool)
            amb = (cc == 1).any(0) & ~ok
            T[i, :, c] = ok * 255; M[i, :, c] = (~amb) * 255
            cov = ok.reshape(5, -1).mean(1); ambcov = amb.reshape(5, -1).mean(1)
            yimg[i, :, c] = cov >= .01
            vimg[i, :, c] = ((cov >= .01) | (ambcov < .01)).astype(np.float32)
    return T, M, yimg, vimg, z["group"], z["views"][:, :5]


def train(fold, epochs):
    torch.manual_seed(0); np.random.seed(0)
    ids = H.id_list(); K = len(ids); amap = H.attr_ids(ids)
    sp = H.real_split(); Xr, Ar, _ = H.real_data(sp["train"])
    Xr = torch.from_numpy(Xr).to(D.dev); Ar = torch.from_numpy(Ar).to(D.dev)
    yr = (Ar.float().mean((2, 3)) / 255. >= .01).float()
    d = np.load("data/synth_train.npz")
    dv = D.dev                                                         # alle Trainingsdaten liegen auf der GPU (Batches ohne CPU-Arbeit)
    Xs, Rs, Ys = torch.from_numpy(d["images"]).to(dv), torch.from_numpy(d["regions"]).to(dv), torch.from_numpy(d["labels"]).float().to(dv)
    As, ys = H.synth_attr(d["regions"], d["labels"], amap); As, ys = torch.from_numpy(As).to(dv), torch.from_numpy(ys).float().to(dv)
    nS, nR, nE, per_epoch = 96, 32, (16 if fold >= 0 else 0), len(Xs) // 96
    if nE:
        T, M, yi, vi, grp, V = easy_targets(ids)
        use = np.nonzero(grp != fold)[0] if fold > 0 else np.arange(len(grp))
        Xe = torch.from_numpy(np.ascontiguousarray(V[use].reshape(-1, 224, 224, 3))).to(D.dev)
        Te = torch.from_numpy(T[use].reshape(-1, K + 5, 28, 28)).to(D.dev); Me = torch.from_numpy(M[use].reshape(-1, K + 5, 28, 28)).to(D.dev)
        ye = torch.from_numpy(yi[use].reshape(-1, K + 5)).to(D.dev); ve = torch.from_numpy(vi[use].reshape(-1, K + 5)).to(D.dev)
        print("EASY-Trainingsbilder:", len(use), "Kacheln:", len(Xe), "Kanäle mit EASY-Ziel:", int((Me.reshape(len(Me), K + 5, -1).amax(2) > 0).any(0).sum()), flush=True)
    model = D.DenseViT(K + 5).to(D.dev)
    steps = epochs * per_epoch
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4},
                             {"params": list(model.img.parameters()) + list(model.pat.parameters()), "lr": 1e-3}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=.1)
    pw = torch.tensor(H.POS_W, device=D.dev); t0 = time.time(); C = K + 5
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xs), device=D.dev); tl = 0
        for it in range(per_epoch):
            ri = torch.randint(0, len(Xr), (nR,), device=D.dev); si = perm[it * nS:(it + 1) * nS]
            B = nS + nR + nE
            maps = torch.zeros(B, C, 28, 28, device=dv); mm = torch.full((B, C, 28, 28), 255., device=dv)
            maps[:nS, :K] = Rs[si].float(); maps[:nS, K:] = As[si].float(); maps[nS:nS + nR, K:] = Ar[ri].float()
            mm[nS:nS + nR, :K] = 0
            yimg = torch.zeros(B, C, device=dv); yimg[:nS, :K] = Ys[si]; yimg[:nS, K:] = ys[si]; yimg[nS:nS + nR, K:] = yr[ri]
            mask = torch.ones(B, C, device=dv); mask[nS:nS + nR, :K] = 0
            x = torch.cat([Xs[si], Xr[ri]])
            if nE:
                ei = torch.randint(0, len(Xe), (nE,), device=dv); x = torch.cat([x, Xe[ei]])
                maps[nS + nR:] = Te[ei].float(); mm[nS + nR:] = Me[ei].float(); mm[nS + nR:, K:] = 0
                yimg[nS + nR:] = ye[ei]; mask[nS + nR:] = ve[ei]; mask[nS + nR:, K:] = 0
            x, both = D.prep(x, torch.cat([maps, mm], 1), True)
            maps, mm = both[:, :C], both[:, C:]
            yimg, mask = yimg.to(D.dev), mask.to(D.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                li, lm = model(x)
            tgt = F.avg_pool2d(maps / 255., 2); mcell = (F.avg_pool2d(mm / 255., 2) > .99).float()
            loss = H.masked_bce(li.float(), yimg, mask) + H.MAP_W * H.masked_bce(lm.float(), tgt, mcell * mask[:, :, None, None], pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        print(f"epoch {ep + 1}/{epochs} loss {tl / per_epoch:.4f}  {time.time() - t0:.0f}s", flush=True)
    out = "models/hybrid_v3_noeasy.pt" if fold < 0 else f"models/hybrid_v3_f{fold}.pt"
    torch.save({"state": model.state_dict(), "ids": ids, "n_out": K + 5, "mode": "hybrid", "fold": fold}, out); print("gespeichert", out)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16)
