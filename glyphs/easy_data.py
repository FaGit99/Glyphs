"""EASY-Daten für das Training (V3): 6 Ansichten je Bild (224 px, wie prep_real.py/easy_eval.py) und Leser-Zählkarten auf 28x28.

  python easy_data.py build     # -> cache/easy_v3.npz  (views, cnt, nrd, group, names, feats)
  python easy_data.py leak      # Überschneidung mit v17/ISIC2018 (nur Bericht)

cnt[i, f, v, :, :] = Anzahl Leser, die Merkmal f in Ansicht v dieser Zelle markiert haben (0..6, uint8, 28x28 Bruchteil -> gerundet
auf Leserzahl nach Mittelung: Zelle gilt als markiert, wenn >=50 % der Zellfläche maskiert ist).
Trainingsregeln (siehe train_targets): positiv = >=2 Leser, negativ = kein Leser, genau 1 Leser = ignoriert.
"""
import csv, json, os, sys
import cv2, numpy as np
from multiprocessing import Pool
import easy_eval as EE

E = EE.E


def mask28(path):
    m = cv2.imread(path, 0)
    if m is None:
        return None
    m = EE.resize_min450((m > 127).astype(np.float32), cv2.INTER_AREA)
    return np.stack([cv2.resize(v, (28, 28), interpolation=cv2.INTER_AREA) >= .5 for v in EE.views(m, cv2.INTER_AREA)]).astype(np.uint8)


def img_views(n):
    img = cv2.imread(f"{E}images/{n}.jpg")
    img = cv2.cvtColor(EE.resize_min450(img, cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.stack(EE.views(img, cv2.INTER_AREA))


def build():
    names = [l.strip() for l in open(E + "image_list.txt") if l.strip()]
    j = json.load(open(E + "markups.json")); feats = EE.FEATS
    meta = {r["ISIC_id"]: r for r in csv.DictReader(open(E + "metadata.csv"))}
    group = np.array([int(meta[n]["group"]) for n in names], np.int8)
    jobs, keys = [], []
    for ii, n in enumerate(names):
        for rd, fs in j[n].items():
            for f in fs:
                jobs.append(f"{E}masks/{n}_{rd}_{EE.fname(f)}.png"); keys.append((ii, feats.index(f)))
    cnt = np.zeros((len(names), len(feats), 6, 28, 28), np.uint8); nrd = np.zeros((len(names), len(feats)), np.int16)
    with Pool(8) as p:
        for (ii, fi), r in zip(keys, p.imap(mask28, jobs, chunksize=4)):
            if r is not None:
                cnt[ii, fi] += r; nrd[ii, fi] += 1
        views = np.stack(p.map(img_views, names, chunksize=4))
    np.savez("cache/easy_v3.npz", views=views, cnt=cnt, nrd=nrd, group=group, names=np.array(names), feats=np.array(feats))
    print("geschrieben", views.shape, cnt.shape, "Gruppen", np.bincount(group))


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
