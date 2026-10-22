"""Vorhersagen auf EASY für V3-Modelle -> cache/easy_pred_<tag>.npz (Format wie easy_pred.npz: pm [N,6,76,14,14], ids).

  python easy_v3_pred.py oof        # jedes Bild vom Modell, das seine Gruppe nicht gesehen hat (hybrid_v3_f<group>.pt) -> _v3oof
  python easy_v3_pred.py noeasy     # Kontrolle ohne EASY-Training (hybrid_v3_noeasy.pt, alle Bilder) -> _v3noeasy
  python easy_v3_pred.py final      # Endmodell hybrid_v3_f0.pt (EASY im Training, nur zum Vergleich in-sample) -> _v3final
  python easy_v3_pred.py v2check    # hybrid.pt (V2) mit diesem Code -> _v2check (muss easy_pred_gpu.npz entsprechen)
"""
import sys
import numpy as np
import dense as D, hybrid as H


def run(path, V):
    m, ck = H.load_model(path)
    _, pm = D.predict(m, V.reshape(-1, 224, 224, 3))
    return pm.reshape(len(V), 6, -1, 14, 14), [str(i) for i in ck["ids"]]


if __name__ == "__main__":
    mode = sys.argv[1]; z = np.load("cache/easy_v3.npz"); V, grp = z["views"], z["group"]
    if mode == "oof":
        pm = None
        for g in (1, 2, 3, 4):
            p, ids = run(f"models/hybrid_v3_f{g}.pt", V[grp == g])
            if pm is None:
                pm = np.zeros((len(V),) + p.shape[1:], np.float16)
            pm[grp == g] = p
        tag = "_v3oof"
    else:
        path, tag = {"noeasy": ("models/hybrid_v3_noeasy.pt", "_v3noeasy"), "final": ("models/hybrid_v3_f0.pt", "_v3final"),
                     "v2check": ("models/hybrid.pt", "_v2check")}[mode]
        pm, ids = run(path, V); pm = pm.astype(np.float16)
    np.savez(f"cache/easy_pred{tag}.npz", pm=pm, ids=np.array(ids)); print("geschrieben", tag, pm.shape)
