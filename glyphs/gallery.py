"""Render every registered ID alone (and report taxonomy coverage)."""
import csv, os, sys, time, traceback
import cv2, numpy as np
import compose, structures as S

OUT = "out/gallery"; os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open("taxonomy.csv", encoding="utf-8")))
term = {r["ID"]: (r["Metaphoric term"], r["Descriptive term (IDS)"]) for r in rows}
cat = {r["ID"]: r["Category"] for r in rows}

missing = [i for i in term if i not in S.SPEC]
print(f"registry: {len(S.SPEC)} IDs drawn; taxonomy has {len(term)}")
print("not drawn:", ", ".join(f"{i} ({cat[i]}: {term[i][0]})" for i in missing))
extra = [i for i in S.SPEC if i not in term]
print("registry IDs missing in taxonomy:", extra)

ids = sorted(S.SPEC) if len(sys.argv) < 2 else sys.argv[1:]
cells, fails, t0 = [], [], time.time()
for sid in ids:
    t = time.time()
    try:
        rng = np.random.default_rng(int(sid[2:]) * 7 + 3)
        img, labels, masks, src, _ = compose.compose(rng, [sid], 448, 224, skin_files=compose.skin.ham_files(0))
        if not labels:
            fails.append((sid, "no label recorded"))
        big = cv2.resize(img, (336, 336), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{OUT}/{sid}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cap = np.full((38, 336, 3), 255, np.uint8)
        cv2.putText(cap, f"{sid} {term[sid][0][:34]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
        cv2.putText(cap, f"{term[sid][1][:46]}", (3, 31), cv2.FONT_HERSHEY_SIMPLEX, .36, (80, 80, 80), 1)
        cells.append(np.vstack([cap, cv2.cvtColor(big, cv2.COLOR_RGB2BGR)]))
    except Exception as e:
        fails.append((sid, repr(e))); traceback.print_exc(limit=3)
        cells.append(np.zeros((374, 336, 3), np.uint8))
    if time.time() - t > 4: print(f"  slow: {sid} {time.time() - t:.1f}s")
print(f"rendered {len(ids)} in {time.time() - t0:.0f}s; failures: {fails}")
per = 3
for k in range(0, len(cells), per * 2):
    chunk = cells[k:k + per * 2]
    while len(chunk) % per: chunk.append(np.zeros_like(cells[0]))
    cv2.imwrite(f"{OUT}/sheet_{k // (per * 2):02d}.png", np.vstack([np.hstack(chunk[i:i + per]) for i in range(0, len(chunk), per)]))
