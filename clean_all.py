#!/usr/bin/env python3
"""Batch cleaner v2 -- no self-verify rename (that detector is unreliable).

Per file: restore pristine original from git -> iterative LaMa removal with
the band-sweep mask -> save IN PLACE keeping the f_ prefix. Then emit
before/after contact sheets so a human (or vision model) verifies by eye.
Nothing is renamed automatically.
"""
import subprocess, os, glob, json, sys
import cv2
import numpy as np
import dewatermark_lama as dl

REV = "3d6fa23"
files = sorted(glob.glob("assets/f_*"))
print(f"processing {len(files)} files", flush=True)
log = []
for i, fp in enumerate(files, 1):
    name = os.path.basename(fp); orig = name[2:]
    try:
        raw = subprocess.run(["git", "show", f"{REV}:assets/{orig}"],
                             capture_output=True).stdout
        if raw:
            open(fp, "wb").write(raw)
        out, status, _, _ = dl.remove(fp)
        if status == "unreadable":
            log.append((name, status)); print(f"[{i}/{len(files)}] UNREADABLE {name}", flush=True); continue
        if status == "clean":
            log.append((name, "nodetect")); print(f"[{i}/{len(files)}] NODETECT {name}", flush=True); continue
        cv2.imwrite(fp, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.append((name, status)); print(f"[{i}/{len(files)}] {name}  {status}", flush=True)
    except Exception as e:
        log.append((name, f"error:{e}")); print(f"[{i}/{len(files)}] ERROR {name}: {e}", flush=True)

# Contact sheets: before(original from git) | after(on disk), 10 rows each.
def strip(img, W=760):
    h, w = img.shape[:2]; y0 = int(h * 0.76); c = img[y0:, :]
    s = W / w; return cv2.resize(c, (W, int(c.shape[0] * s)))

files = sorted(glob.glob("assets/f_*"))
os.makedirs("/tmp/sheets", exist_ok=True)
for d in glob.glob("/tmp/sheets/*"): os.remove(d)
for ci in range(0, len(files), 10):
    chunk = files[ci:ci+10]; rows = []
    for fp in chunk:
        name = os.path.basename(fp); orig = name[2:]
        raw = subprocess.run(["git", "show", f"{REV}:assets/{orig}"], capture_output=True).stdout
        b = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        a = cv2.imread(fp)
        bc, ac = strip(b), strip(a)
        sep = np.full((bc.shape[0], 8, 3), (0, 0, 255), np.uint8)
        lab = np.full((bc.shape[0], 150, 3), 60, np.uint8)
        cv2.putText(lab, orig[:13], (4, bc.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(np.hstack([lab, bc, sep, ac]))
    hh = max(r.shape[0] for r in rows)
    rows = [np.vstack([r, np.zeros((hh-r.shape[0], r.shape[1], 3), np.uint8)]) for r in rows]
    g = np.full((6, rows[0].shape[1], 3), (0, 255, 0), np.uint8)
    out = rows[0]
    for r in rows[1:]: out = np.vstack([out, g, r])
    cv2.imwrite(f"/tmp/sheets/sheet_{ci//10}.png", out)
print(f"\n--- processed {len(log)}; contact sheets in /tmp/sheets/ ---", flush=True)
json.dump(log, open("/tmp/clean_all.json", "w"), indent=2)
