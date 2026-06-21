#!/usr/bin/env python3
"""Watermark removal v2 -- LaMa fill instead of cv2.inpaint(TELEA).

Reuses dewatermark.py's OCR detector to *locate* the WWW.CHUNMINGARTS.ORG
line and build a stroke mask, then reconstructs the background with the LaMa
deep-inpainting model.  LaMa fills textured/painted backgrounds far better
than Telea, which is what failed on white-text-over-busy-artwork cases.

Robustness for truncated / smudged / faint marks (the user flagged these):
  * masking does NOT require a full watermark -- any detected fragment seeds
    the band, and the mask is dilated generously to swallow partial/smeared
    glyphs and anti-aliased halos that glyph-precise masking left behind.
  * the mask still stays inside the detected watermark band/x-range, so the
    artist's red seal and calligraphy elsewhere are untouched.

Usage:  python3 dewatermark_lama.py --apply assets/f_*.jpg
        python3 dewatermark_lama.py --preview assets/f_foo.jpg   (-> /tmp/lama_preview)
"""
import sys, os, argparse
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image
import dewatermark as dw      # reuse find_line / build_mask / ocr detection

_LAMA = None
def lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()          # downloads weights once, runs on CPU
    return _LAMA


def _white_tophat_tokens(img):
    """Aggressive fallback for faint/letter-spaced WHITE watermarks that the
    default OCR misses: a white top-hat isolates thin bright strokes from the
    (darker) painting, then OCR the upscaled bottom strip at a low conf gate.
    Returns dewatermark-style token boxes in full-image coords."""
    import pytesseract, re
    H, W = img.shape[:2]
    y0 = int(H * 0.78)
    strip = img[y0:H, :]
    if strip.shape[0] < 8 or strip.shape[1] < 8:
        return []
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    out = []
    for width in (1800, 2600):
        s = max(1.0, float(width) / strip.shape[1])
        up = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        kw = max(9, int(up.shape[0] * 0.5)) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kw))
        th = cv2.morphologyEx(up, cv2.MORPH_TOPHAT, kern)   # bright thin marks
        th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
        for v in (th, cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            for psm in (11, 6, 7):
                data = pytesseract.image_to_data(
                    v, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
                for i in range(len(data["text"])):
                    t = data["text"][i].strip()
                    if not t or float(data["conf"][i]) < 8 or not dw.TOKEN_RE.search(t):
                        continue
                    out.append(dict(
                        t=t, conf=float(data["conf"][i]),
                        strong=bool(dw.STRONG_RE.search(t)),
                        x=int(data["left"][i] / s), y=int(y0 + data["top"][i] / s),
                        w=int(data["width"][i] / s), h=int(data["height"][i] / s)))
    return out


def geometric_mask(img):
    """OCR-free fallback: locate the white, letter-spaced watermark as a thin
    horizontal run of bright strokes in the bottom strip. For the *known*
    watermarked corpus this is more reliable than reading faint text over busy
    paintings. Returns a binary mask (0/255) or None."""
    H, W = img.shape[:2]
    ys = int(H * 0.74)
    strip = img[ys:H, :]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    sh = strip.shape[0]
    # White top-hat: keep thin bright strokes, suppress broad bright areas.
    kw = max(9, int(sh * 0.5)) | 1
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kw)))
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    # Watermark is near-white: require both brightness (top-hat) and low colour
    # saturation, so coloured highlights in the art are rejected.
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    stroke = (th > max(40, int(np.percentile(th, 99)))) & (sat < 70)
    stroke = stroke.astype(np.uint8) * 255
    if stroke.sum() == 0:
        return None
    # Join glyphs of one text line horizontally; keep lines thin vertically.
    joined = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 25), 3)))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    best = None
    for i in range(1, nlab):
        x, y, w, h, area = stats[i]
        if w < W * 0.18:                       # a watermark line spans real width
            continue
        if h > sh * 0.5 or h < 2:              # thin: text height, not a blob
            continue
        if w / max(h, 1) < 5:                  # wide aspect = a text line
            continue
        if best is None or w * (1.0 / (h + 1)) > best[0]:
            best = (w * (1.0 / (h + 1)), x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    pad = max(3, h // 2)
    x0, x1 = max(0, x - pad), min(strip.shape[1], x + w + pad)
    y0, y1 = max(0, y - pad), min(sh, y + h + pad)
    m = np.zeros((H, W), np.uint8)
    # Mask the actual bright strokes inside the detected line box, dilated, so
    # art between glyphs is largely spared while every stroke is covered.
    region = stroke[y0:y1, x0:x1]
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, h) | 1,) * 2))
    m[ys + y0:ys + y1, x0:x1] = region
    return m if m.any() else None


def band_sweep_mask(img, y0, y1):
    """Given the watermark's thin vertical band, mask every white text stroke
    across the FULL image width inside it. This is what catches the letter-
    spaced ends (WWW. / .ORG) that glyph-precise detection under-reaches -- the
    whole 'WWW.CHUNMINGARTS.ORG' line sits in one band, so sweeping the band
    grabs all of it. A saturation gate spares the (coloured) art and red seal."""
    H, W = img.shape[:2]
    y0, y1 = max(0, y0), min(H, y1)
    if y1 - y0 < 3:
        return None
    band = img[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bh = band.shape[0]
    kw = max(9, int(bh * 0.9)) | 1
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kw)))
    sat = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[:, :, 1]
    thr = max(28, float(np.percentile(th, 97)))
    stroke = ((th > thr) & (sat < 85) & (gray > 110)).astype(np.uint8) * 255
    if stroke.sum() == 0:
        return None
    stroke = cv2.dilate(stroke, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, bh // 2) | 1,) * 2))
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, :] = stroke
    return m


def solid_band_mask(img):
    """Guaranteed-coverage mask via row/column PROJECTION of white-stroke energy.
    The watermark is a horizontal line of white letters in the bottom strip, so
    it shows up as the row-band with the most thin-bright-stroke energy -- robust
    even when OCR and connected-components both fail to read it. Mask that band
    solid; LaMa rebuilds the strip so no stroke survives."""
    H, W = img.shape[:2]
    ys = int(H * 0.72)
    strip = img[ys:, :]
    sh = strip.shape[0]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    kw = max(9, int(sh * 0.35)) | 1
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kw)))
    # White, thin strokes: bright top-hat, low saturation, high value.
    white = ((th > max(22, float(np.percentile(th, 96)))) &
             (sat < 95) & (val > 115)).astype(np.float32)
    rowsum = white.sum(axis=1)
    if rowsum.max() < W * 0.012:          # no text-like concentration -> give up
        return None
    on = np.where(rowsum > rowsum.max() * 0.28)[0]
    y0, y1 = int(on.min()), int(on.max()) + 1
    band = white[y0:y1, :]
    colsum = band.sum(axis=0)
    con = np.where(colsum > max(1.0, colsum.max() * 0.12))[0]
    if len(con) == 0:
        return None
    x0, x1 = int(con.min()), int(con.max()) + 1
    h = max(6, y1 - y0); span = x1 - x0
    # Pad vertically; extend x to catch letter-spaced ends, snap to left edge
    # when the mark starts there (it usually does).
    xL = max(0, x0 - int(span * 0.12) - h)
    xR = min(W, x1 + int(span * 0.15) + h)
    if x0 < W * 0.14:
        xL = 0
    pad = max(3, int(h * 0.7))
    Y0 = max(0, ys + y0 - pad); Y1 = min(H, ys + y1 + pad)
    m = np.zeros((H, W), np.uint8)
    m[Y0:Y1, xL:xR] = 255
    return m


def detect_mask(img):
    """Return a dilated binary mask (uint8 0/255) over the watermark, or None.
    Built from dewatermark's detector but grown to cover smudged/partial marks."""
    if os.environ.get("DEWM_SOLID"):
        return solid_band_mask(img)
    line = dw.find_line(img)
    if line is None:
        # Fallback 1: faint/letter-spaced white mark the default detector missed.
        toks = _white_tophat_tokens(img)
        line = dw.pick_line(toks, img.shape[0]) if toks else None
        if line:
            yc = np.median([d["y"] + d["h"] / 2 for d in line])
            if yc <= img.shape[0] * 0.72:
                line = None
    geo = geometric_mask(img)              # OCR-free white-stroke detector
    if line is None:
        # Fallback 2: geometric detection + a band sweep over its rows so the
        # full-width letter-spaced mark (incl. ends) is covered, not just the
        # widest connected blob.
        if geo is None:
            return None
        ys, ye = np.where(geo.any(axis=1))[0][[0, -1]]
        h = ye - ys
        sweep = band_sweep_mask(img, ys - h, ye + h)
        return cv2.bitwise_or(geo, sweep) if sweep is not None else geo
    built = dw.build_mask(img, line)
    if built is None:
        return geo
    mask = built[0]
    if not mask.any():
        return geo
    # Grow the stroke mask: smudged/truncated watermarks leave faint halos that
    # the tight color-gate misses; LaMa tolerates a generous mask, so dilate to
    # guarantee the whole mark (incl. anti-aliased edges) is covered.
    hmed = int(np.median([d["h"] for d in line]))
    k = max(5, hmed // 2) | 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Union with the geometric detector: each catches marks the other misses,
    # so together they cover the full watermark line more consistently.
    if geo is not None:
        mask = cv2.bitwise_or(mask, geo)
    # Full-width band sweep over the watermark's y-rows: grabs the letter-spaced
    # ends (WWW. / .ORG) that glyph-precise masking under-reaches.
    tops = [d["y"] for d in line]; bots = [d["y"] + d["h"] for d in line]
    sweep = band_sweep_mask(img, min(tops) - hmed, max(bots) + hmed)
    if sweep is not None:
        mask = cv2.bitwise_or(mask, sweep)
    return mask


def _lama_fill(img, mask):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = lama()(Image.fromarray(rgb), Image.fromarray(mask).convert("L"))
    out = cv2.cvtColor(np.array(res.convert("RGB")), cv2.COLOR_RGB2BGR)
    # LaMa may pad/resize to multiples of 8; restore exact original size.
    if out.shape[:2] != img.shape[:2]:
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return out


def remove(path, max_iter=5):
    """Detect -> LaMa-fill -> RE-DETECT, looping until no watermark is found
    (clean) or max_iter. The re-detection is the per-file 'check it again'
    step: residue left by one pass is caught and removed by the next."""
    img0 = cv2.imread(path)
    if img0 is None:
        return None, "unreadable", None, None
    img = img0
    masks = []
    for _ in range(max_iter):
        mask = detect_mask(img)
        if mask is None or not mask.any():
            break                      # re-check found nothing -> clean
        img = _lama_fill(img, mask)
        masks.append(mask)
    if not masks:
        return img0, "clean", None, None
    union = masks[0]
    for m in masks[1:]:
        union = cv2.bitwise_or(union, m)
    cov = float((union > 0).mean()) * 100
    return img, f"watermarked iters={len(masks)} cov={cov:.2f}%", img0, union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--outdir", default="/tmp/lama_preview")
    args = ap.parse_args()
    if args.preview:
        os.makedirs(args.outdir, exist_ok=True)
    n_wm = n_clean = 0
    for p in args.paths:
        out, status, orig, mask = remove(p)
        name = os.path.basename(p)
        if status in ("clean", "unreadable"):
            n_clean += status == "clean"
            print(f"{status:11} {name}", flush=True)
            continue
        n_wm += 1
        print(f"WM          {name}  {status}", flush=True)
        if args.preview:
            ov = orig.copy(); ov[mask > 0] = (0, 0, 255)
            sep = np.full((orig.shape[0], 6, 3), (0, 0, 255), np.uint8)
            m = np.hstack([orig, sep, ov, sep, out])
            if m.shape[0] > 600:
                s = 600 / m.shape[0]; m = cv2.resize(m, None, fx=s, fy=s)
            cv2.imwrite(os.path.join(args.outdir, "cmp_" + name + ".jpg"), m)
        if args.apply:
            ext = os.path.splitext(p)[1].lower()
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
            cv2.imwrite(p, out, params)
    print(f"\n--- {n_wm} watermarked, {n_clean} clean ---")


if __name__ == "__main__":
    main()
