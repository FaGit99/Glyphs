import cv2, numpy as np, compose, skin
files = skin.ham_files(0); cells = []
for seed in range(12):
    rng = np.random.default_rng(300 + seed)
    ids = compose.random_scene_ids(rng, k=(2, 6), exclude_sites=())
    img, labels, masks, src, _ = compose.compose(rng, ids, 448, 224, skin_files=files)
    big = cv2.cvtColor(cv2.resize(img, (336, 336), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    cap = np.full((22, 336, 3), 255, np.uint8)
    cv2.putText(cap, " ".join(i[2:] for i in sorted(set(l["id"] for l in labels))), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    cells.append(np.vstack([cap, big]))
cv2.imwrite("out/mixed.png", np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, 12, 3)]))
