"""Run the project's direct classifier (read-only use of its checkpoints).

Arm base_448_v17: timm eva02_base_patch14_448.mim_in22k_ft_in1k, img_size 448, head Linear(.,1); input = cv2 image -> RGB -> squash-resize to
448x448 (INTER_LINEAR) -> /255 -> OpenAI-CLIP mean/std (as in the project's datasets_gpu.py / augmentations_gpu_balanced.py); bf16; prob = sigmoid(logit).

  python direct_model.py verify              # reproduce the stored v17 test predictions (AUROC 0.9675)
  python direct_model.py predict <split>     # val | test  -> models/direct_base448_<split>.npz
"""
import csv, os, sys, time
import cv2, numpy as np, timm, torch, torch.nn as nn

ROOT = "/mnt/windows_data/Documents and Settings/Fabian/Eigene Dateien/GitHub/260826 try_cleaned_data/saved_models/"
ARMS = {"base_448_v17": ("eva02_base_patch14_448.mim_in22k_ft_in1k", 448), "small_224_v17": ("eva02_small_patch14_224.mim_in22k", 224)}
SPLITS = "/mnt/samsung_data/mole_data/e_reports_v17/splits/v17/"
ARCHIVE = "/mnt/samsung_data/mole_data/data/raw/ISICall_v17/"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
cv2.setNumThreads(1)


class ISICModel(nn.Module):
    def __init__(self, name, img_size):
        super().__init__()
        self.model = timm.create_model(name, pretrained=False, img_size=img_size)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        return self.model(x)


def load(arm="base_448_v17"):
    name, size = ARMS[arm]
    m = ISICModel(name, size); sd = torch.load(ROOT + arm + "/best_pauc.pt", map_location="cpu")
    print("load_state_dict:", m.load_state_dict(sd, strict=True)); return m.cuda().eval(), size


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.p, self.s = paths, size

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = cv2.imread(self.p[i])
        if im is None:
            return torch.zeros(3, self.s, self.s, dtype=torch.uint8), 0
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (self.s, self.s), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(im).permute(2, 0, 1).contiguous(), 1


@torch.no_grad()
def predict_paths(model, size, paths, bs=48, workers=12):
    dl = torch.utils.data.DataLoader(DS(paths, size), batch_size=bs, num_workers=workers, shuffle=False)
    out, ok, t0 = [], [], time.time()
    for k, (x, good) in enumerate(dl):
        x = (x.cuda().float() / 255. - MEAN.cuda()) / STD.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float().squeeze(1).cpu()); ok.append(good)
        if k % 40 == 39:
            print(f"  {(k + 1) * bs}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return torch.cat(out).numpy(), torch.cat(ok).numpy().astype(bool)


def split_rows(split):
    return list(csv.DictReader(open(SPLITS + f"{split}.csv")))


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    model, size = load()
    if sys.argv[1] == "verify":
        ref = list(csv.DictReader(open("/mnt/samsung_data/mole_data/audit_reports/x_swap/X5_results/X5_BEST_epoch_013_auroc_0.9734__test__resize__20260924T082542Z_predictions.csv")))
        rows = split_rows("test"); assert [r["isic_id"] for r in rows] == [r["isic_id"] for r in ref], "row order differs"
        logit, ok = predict_paths(model, size, [ARCHIVE + r["image_relpath"] for r in rows])
        prob = 1 / (1 + np.exp(-logit)); pref = np.array([float(r["prob"]) for r in ref]); y = np.array([float(r["target"]) for r in rows])
        print(f"unreadable images: {int((~ok).sum())}; AUROC reproduced {roc_auc_score(y, prob):.4f} (stored {roc_auc_score(y, pref):.4f}); prob difference vs stored: max {np.abs(prob - pref).max():.4f}, mean {np.abs(prob - pref).mean():.5f}; correlation {np.corrcoef(prob, pref)[0, 1]:.5f}")
        np.savez("models/direct_base448_test.npz", ids=np.array([r["isic_id"] for r in rows]), logit=logit, y=y)
    else:
        sp = sys.argv[2]; rows = split_rows(sp)
        logit, ok = predict_paths(model, size, [ARCHIVE + r["image_relpath"] for r in rows])
        y = np.array([float(r["target"]) for r in rows]); print(f"{sp}: unreadable {int((~ok).sum())}, AUROC {roc_auc_score(y, logit):.4f}")
        np.savez(f"models/direct_base448_{sp}.npz", ids=np.array([r["isic_id"] for r in rows]), logit=logit, y=y)
