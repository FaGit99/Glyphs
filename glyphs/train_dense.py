"""Train DenseViT on the synthetic set. Usage: python train_dense.py <tag> [epochs]"""
import sys, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import dense as D

tag = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MAP_W, POS_W = 5.0, 3.0   # weight of the map loss, positive weight inside it


def load(name):
    d = np.load(f"data/synth_{name}.npz")
    return (torch.from_numpy(d["images"]), torch.from_numpy(d["labels"]).float(),
            torch.from_numpy(d["regions"]), [str(i) for i in d["ids"]])


def targets14(maps):  # [B,K,28,28] float 0-255 -> [B,K,14,14] coverage 0-1
    return F.avg_pool2d(maps / 255., 2)


if __name__ == "__main__":
    Xtr, Ytr, Rtr, ids = load("train"); Xva, Yva, Rva, _ = load("val")
    model = D.DenseViT(len(ids)).to(D.dev)
    bs, steps = 128, epochs * (len(Xtr) // 128)
    heads = ["img", "pat"]
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4},
                             {"params": [p for h in heads for p in getattr(model, h).parameters()], "lr": 1e-3}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=.1)
    pw = torch.tensor(POS_W, device=D.dev)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr)); tl = tm = 0; n = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x, r = D.prep(Xtr[idx], Rtr[idx].float(), True)
            y = Ytr[idx].to(D.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                li, lm = model(x)
            l_img = F.binary_cross_entropy_with_logits(li.float(), y)
            l_map = F.binary_cross_entropy_with_logits(lm.float(), targets14(r), pos_weight=pw)
            loss = l_img + MAP_W * l_map
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += l_img.item(); tm += l_map.item(); n += 1
        pi, pm = D.predict(model, Xva)
        gt = targets14(Rva.float()).numpy()
        ai = [roc_auc_score(Yva.numpy()[:, j], pi[:, j]) for j in range(len(ids)) if 0 < Yva[:, j].sum() < len(Yva)]
        gtb = (gt >= .25).reshape(len(Xva), len(ids), -1).transpose(1, 0, 2).reshape(len(ids), -1)
        pp = pm.reshape(len(Xva), len(ids), -1).transpose(1, 0, 2).reshape(len(ids), -1)
        am = [roc_auc_score(gtb[j], pp[j]) for j in range(len(ids)) if 0 < gtb[j].sum() < gtb.shape[1]]
        print(f"epoch {ep + 1}/{epochs} img-loss {tl / n:.4f} map-loss {tm / n:.4f}  val AUC image {np.mean(ai):.3f} patch {np.mean(am):.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "ids": ids}, f"models/{tag}.pt")
