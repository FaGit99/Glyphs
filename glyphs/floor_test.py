"""Attribution of the skin-zone gap: real lesion-free windows vs the same pipeline's skin, with factors toggled."""
import numpy as np, cv2, sys
import skin, compose, realfake as RF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
N = 300; S = 224
real_files = skin.ham_files(1); syn_files = skin.ham_files(0)
def real_skin():  # lesion-free native windows from held-out files (same selection rule as backgrounds)
    out = []
    for i in range(N):
        out.append(cv2.cvtColor(skin.real_patch(np.random.default_rng(5000 + i), S, files=real_files)[0], cv2.COLOR_RGB2BGR)[..., ::-1][..., ::-1][:, :, ::-1])
    return out
real = [np.ascontiguousarray(x) for x in real_skin()]
def synth_skin(vignette, jpeg_q, grain):
    out = []
    orig_art = compose.artefacts
    if not vignette:
        compose.artefacts = lambda rng, canvas, size: None
    for i in range(N):
        rng = np.random.default_rng(7000 + i)
        img, *_ = compose.compose(rng, [], 448, S, skin_files=syn_files)  # no structures: pale lesion only, take corner
        if jpeg_q: img = RF.jpeg(img, jpeg_q)
        out.append(img)
    compose.artefacts = orig_art
    return out
y = np.array([1] * N + [0] * N); cv = StratifiedKFold(5, shuffle=True, random_state=0)
def auc(syn):
    X = np.stack([RF.feats(im[:64, :64])[0] for im in syn + real])
    p = cross_val_predict(GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
print("A  full pipeline (vignette+artefacts, jpeg 85-98):", round(auc(synth_skin(True, 90, True)), 3))
print("B  no artefacts/vignette, jpeg 90               :", round(auc(synth_skin(False, 90, True)), 3))
print("C  no artefacts/vignette, no jpeg re-encode     :", round(auc(synth_skin(False, 0, True)), 3))
print("D  control: real vs real (held-out halves)      :", end=" ")
half = [np.ascontiguousarray(skin.real_patch(np.random.default_rng(9000 + i), S, files=real_files)[0]) for i in range(N)]
X = np.stack([RF.feats(im[:64, :64])[0] for im in half + real]); p = cross_val_predict(GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0), X, y, cv=cv, method="predict_proba")[:, 1]
print(round(roc_auc_score(y, p), 3))
