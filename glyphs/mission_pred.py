"""Mission synthetic-vs-mixed: predictions of every regime on EASY, ISIC 2018 (cal/test blocks) and Derm7pt, one channel layout.

  python mission_pred.py [regime ...]      # default: all regimes whose checkpoints exist -> cache/mission_pred_<regime>.npz

Channel layout for every regime: 71 taxonomy IDs + 5 ISIC attributes (76). Missing channels are NaN
(real_only has no ID channels; dense_v1 has no attribute channels -> union (max) of the mapped IDs, as in hybrid.py eval).
OOF regimes: EASY predictions from the fold model that did not see the image's group; ISIC/Derm7pt from the fold-0 model (all EASY).
"""
import os, sys
import numpy as np, torch, timm
import dense as D, hybrid as H, mil

M = "models/"
REGIMES = {   # name: (EASY source, ISIC/Derm7pt source); "oof:<pattern with {g}>" = fold models per EASY group
    "R0": (M + "dense_v1.pt", M + "dense_v1.pt"),
    "R0pipe": (M + "mission_r0pipe_f0.pt", M + "mission_r0pipe_f0.pt"),
    "R0pipe_s1": (M + "mission_r0pipes1_f0.pt", M + "mission_r0pipes1_f0.pt"),
    "R1": (M + "real_only.pt", M + "real_only.pt"),
    "R2": ("frozen_V2/models_full/hybrid.pt", "frozen_V2/models_full/hybrid.pt"),
    "R2ctrl": (M + "hybrid_v3_noeasy.pt", M + "hybrid_v3_noeasy.pt"),
    "R3": ("oof:" + M + "hybrid_v3_f{g}.pt", M + "hybrid_v3_f0.pt"),
    "R3_s1": ("oof:" + M + "mission_r3s1_f{g}.pt", None),
    "R5": ("oof:" + M + "mission_r5_f{g}.pt", M + "mission_r5_f0.pt"),
    "R4_25": ("oof:" + M + "mission_dose25_f{g}.pt", None),
    "R4_50": ("oof:" + M + "mission_dose50_f{g}.pt", None),
    "R4_75": ("oof:" + M + "mission_dose75_f{g}.pt", None),
}
IDS = H.id_list(); K = len(IDS); AMAP = H.attr_ids(IDS)


def load(path):
    m, ck = H.load_model(path); n = ck.get("n_out", len(ck["ids"]))
    assert [str(i) for i in ck["ids"]] == IDS
    return m, n


def to76(x, n):
    """model output [..., n, ...] on axis 1 -> [..., 76, ...]"""
    if n == K + 5:
        return x
    sh = list(x.shape); sh[1] = K + 5; out = np.full(sh, np.nan, np.float32)
    if n == K:
        out[:, :K] = x
        for j, a in enumerate(H.ATTRS):
            out[:, K + j] = x[:, AMAP[a]].max(1)
    elif n == 5:
        out[:, K:] = x
    return out


def run(m, n, tiles):
    pi, pm = D.predict(m, tiles)
    return to76(pi, n), to76(pm, n)


def easy_pred(src, V, grp):
    pi = np.zeros((len(V), 6, K + 5), np.float32); pm = np.zeros((len(V), 6, K + 5, 14, 14), np.float16)
    for g, path in ([(g, src[4:].format(g=g)) for g in (1, 2, 3, 4)] if src.startswith("oof:") else [(None, src)]):
        sel = grp == g if g else np.ones(len(V), bool)
        m, n = load(path); a, b = run(m, n, V[sel].reshape(-1, 224, 224, 3))
        pi[sel] = a.reshape(-1, 6, K + 5); pm[sel] = b.reshape(-1, 6, K + 5, 14, 14)
    return pi, pm


def main(names):
    z = np.load("cache/easy_v3.npz"); V, grp = z["views"], z["group"]
    sp = H.real_split(); Xc = H.real_data(sp["cal"])[0]; Xt = H.real_data(sp["test"])[0]
    d7 = np.load("cache/d7_img.npy", mmap_mode="r"); tiles7, cnt7 = mil.load_tiles()
    for r in names:
        es, fs = REGIMES[r]
        paths = [es[4:].format(g=g) for g in (1, 2, 3, 4)] if es.startswith("oof:") else [es]
        if not all(os.path.exists(p) for p in paths + ([fs] if fs else [])):
            print("skip (checkpoint missing):", r); continue
        out = {}
        out["easy_pi"], out["easy_pm"] = easy_pred(es, V, grp)
        if fs:
            m, n = load(fs)
            for k, X in (("isic_cal", Xc), ("isic_test", Xt)):
                a, b = run(m, n, X); out[k + "_pi"] = a.astype(np.float16); out[k + "_pm"] = b.astype(np.float16)
            pi7, pm7 = [], []
            for i in range(0, len(d7), 128):
                a, b = run(m, n, np.ascontiguousarray(d7[i:i + 128]).reshape(-1, 224, 224, 3))
                pi7.append(a.reshape(-1, 6, K + 5)); pm7.append(b.reshape(-1, 6, K + 5, 196).max(-1))
            out["d7_pi"], out["d7_pmax"] = np.concatenate(pi7), np.concatenate(pm7)
            out["d7_mil"] = to76(mil.agg_predict(m, tiles7, cnt7, np.arange(len(cnt7))), n)    # top-3 mean of image head over lesion tiles
        np.savez(f"cache/mission_pred_{r}.npz", **out); print("wrote", r, {k: v.shape for k, v in out.items()}, flush=True)
        torch.cuda.empty_cache()


def synth_v1_easy():
    """R0 image-level classifier (timm ViT, 71 sigmoid outputs; train.py) on the 6 EASY views -> cache/mission_pred_R0img.npz"""
    import train as T
    ck = torch.load(M + "synth_v1.pt", map_location="cpu", weights_only=False); assert [str(i) for i in ck["ids"]] == IDS
    net = T.build(K, scratch=True); net.load_state_dict(ck["state"]); net.eval()
    V = np.load("cache/easy_v3.npz")["views"]
    p = T.predict(net, torch.from_numpy(np.ascontiguousarray(V.reshape(-1, 224, 224, 3)))).reshape(len(V), 6, K)
    np.savez("cache/mission_pred_R0img.npz", easy_pi=to76(p.transpose(0, 2, 1), K).transpose(0, 2, 1)); print("wrote R0img")


def easy_lesion():
    """lesion_seg.pt on the 248 EASY images (whole image at short side 450, as analyze.lesion_mask) -> lesion coverage per view and
    14x14 cell: cache/mission_easy_lesion.npz (cov [248,6,14,14] float, frac_image [248])"""
    import cv2, easy_eval as EE, lesion_seg as LS
    s = LS.SegViT(pretrained=False).to(D.dev); s.load_state_dict(torch.load(M + "lesion_seg.pt", weights_only=False)["state"]); s.eval()
    names = [str(n) for n in np.load("cache/easy_v3.npz")["names"]]; cov, fr = [], []
    for n in names:
        img = cv2.cvtColor(EE.resize_min450(cv2.imread(f"{EE.E}images/{n}.jpg"), cv2.INTER_AREA), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
        p = LS.predict(s, cv2.resize(img, (LS.S, LS.S), interpolation=cv2.INTER_AREA)[None])[0]
        m = (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) > .5).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); les = np.zeros_like(m)
        if cs:
            big = max(cv2.contourArea(c) for c in cs)
            cv2.drawContours(les, [c for c in cs if cv2.contourArea(c) >= .2 * big], -1, 1, -1)
        cov.append(np.stack([cv2.resize(v, (14, 14), interpolation=cv2.INTER_AREA) for v in EE.views(les.astype(np.float32), cv2.INTER_AREA)]))
        fr.append(les.mean())
    np.savez("cache/mission_easy_lesion.npz", cov=np.stack(cov), frac_image=np.array(fr)); print("wrote mission_easy_lesion.npz; lesion share of image: median", np.median(fr))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["lesion"]:
        easy_lesion()
    elif args == ["R0img"]:
        synth_v1_easy()
    else:
        main(args or list(REGIMES))

