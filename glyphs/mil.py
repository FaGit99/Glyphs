"""Helpers: tile-level model outputs aggregated to image level (mean of the top-3 tiles per channel)."""
import numpy as np, torch
import dense as D

TOPK = 3


def load_tiles():
    return np.load("cache/d7_tiles.npy", mmap_mode="r"), np.load("cache/d7_tile_count.npy")


@torch.no_grad()
def agg_predict(model, tiles, cnt, idx, bs_img=16, topk=TOPK, n_out=None):
    """image-level aggregated probabilities [len(idx), K]: top-k mean over the lesion tiles of each image"""
    model.eval(); out = []
    for i in range(0, len(idx), bs_img):
        sel = idx[i:i + bs_img]
        x = np.concatenate([np.asarray(tiles[j, :cnt[j]]) for j in sel]); lens = [int(cnt[j]) for j in sel]
        pi, _ = D.predict(model, x)
        o = 0
        for L in lens:
            p = np.sort(pi[o:o + L], axis=0)[::-1][:topk].mean(0); out.append(p); o += L
    return np.stack(out)
