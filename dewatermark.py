#!/usr/bin/env python3
"""
Remove the "WWW.CHUNMINGARTS.ORG" watermark from yuchunming.art images.

Pipeline per image:
  1. Multi-variant OCR (gray/inverted/color-channels/CLAHE) to find the
     watermark text tokens and their boxes.  Keyed on the literal tokens
     (CHUNMING / ARTS / ORG / WWW) so the artist's red seal and Chinese
     calligraphy are never touched.
  2. Group token boxes into the single horizontal watermark line; derive a
     thin y-band and an x-range (extended when OCR only read part of it).
  3. Sample the watermark's actual color from a detected token, then inside
     the band mask only pixels matching that color -> captures every glyph
     even when OCR missed some, while sparing the surrounding artwork.
  4. Inpaint the masked strokes (cv2.inpaint, TELEA).

Modes:
  --preview  : write before|after montages to /tmp/wm_preview, don't modify.
  --report   : just classify (watermark? / tokens) without writing anything.
  --apply    : overwrite the image in place (originals recoverable via git).
"""
import sys, os, re, argparse, json
# Single-thread each tesseract call so multiprocessing (not OMP) provides the
# parallelism -- avoids workers oversubscribing the cores and stalling.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import cv2
cv2.setNumThreads(1)
import numpy as np
import pytesseract

STRONG_RE = re.compile(r"(chunm|hunmi|nming|mingart|ingarts)", re.I)
TOKEN_RE  = re.compile(r"(chunm|hunmi|nming|mingart|ingarts|arts\.|\.org|^org$|^www$|www\.)", re.I)
END_RE    = re.compile(r"org", re.I)          # right end of the line (".ORG")
START_RE  = re.compile(r"www", re.I)          # left end of the line ("WWW.")


def _variants(img):
    """Single-channel preprocessings that expose differently-coloured text:
    the colour channels separate red/white marks, CLAHE lifts faint ones, and
    the adaptive threshold binarises low-contrast text on busy backgrounds."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 41, 12)
    return [("gray", gray), ("inv", 255 - gray),
            ("green", g), ("blue", b), ("red", r),
            ("clahe", clahe), ("clahe_inv", 255 - clahe),
            ("adap", adap), ("adap_inv", 255 - adap)]


def _ocr_on(img, psms, x0=0, y0=0, sx=1.0, sy=1.0):
    """OCR every variant of `img`; map boxes back to parent coords via offset
    (x0,y0) and scale (sx,sy). Returns watermark-token box dicts."""
    out = []
    for _, vimg in _variants(img):
        for psm in psms:
            data = pytesseract.image_to_data(
                vimg, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
            for i in range(len(data["text"])):
                t = data["text"][i].strip()
                if not t or float(data["conf"][i]) < 15 or not TOKEN_RE.search(t):
                    continue
                out.append(dict(
                    t=t, conf=float(data["conf"][i]), strong=bool(STRONG_RE.search(t)),
                    x=int(x0 + data["left"][i] / sx), y=int(y0 + data["top"][i] / sy),
                    w=int(data["width"][i] / sx), h=int(data["height"][i] / sy)))
    return out


def ocr_tokens(img):
    """Return watermark-token boxes in full-resolution `img` coords. Combines a
    size-normalised full-image pass with an upscaled bottom-strip pass taken
    from the *original* resolution -- the strip pass makes small/faint marks
    legible and is what catches the otherwise-missed watermarks."""
    H, W = img.shape[:2]
    # Full pass on a size-normalised copy (boxes mapped back to img coords).
    full, fscale = scaled_for_ocr(img)
    found = _ocr_on(full, (11, 6), sx=fscale, sy=fscale)
    # Bottom-strip passes at original resolution. The watermark always sits
    # here; upscaling makes the text large for OCR. Two target widths are used
    # because watermark text size varies -- a fixed scale catches some marks but
    # renders others too small or too large for the recogniser.
    y0 = int(H * 0.74)
    strip = img[y0:H, :]
    if strip.shape[0] >= 8 and strip.shape[1] >= 8:
        for width in (1800, 2400):
            s = max(1.0, float(width) / strip.shape[1])
            up = cv2.resize(strip, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            found += _ocr_on(up, (11, 6, 7), x0=0, y0=y0, sx=s, sy=s)
    return found


def pick_line(tokens, H):
    """Group tokens into the dominant watermark line; return its members."""
    if not tokens:
        return []
    # Cluster by vertical center; pick the cluster whose best token has the
    # highest confidence, preferring clusters that contain a STRONG token.
    toks = sorted(tokens, key=lambda d: d["y"] + d["h"] / 2)
    clusters, cur = [], [toks[0]]
    for d in toks[1:]:
        ref = cur[-1]
        if abs((d["y"] + d["h"] / 2) - (ref["y"] + ref["h"] / 2)) <= max(ref["h"], d["h"]) * 1.2:
            cur.append(d)
        else:
            clusters.append(cur); cur = [d]
    clusters.append(cur)

    def score(c):
        return (any(d["strong"] for d in c), max(d["conf"] for d in c), len(c))
    return max(clusters, key=score)


def watermark_color(img, line):
    """Estimate the watermark stroke color (BGR) from detected token boxes."""
    samples = []
    boxes = sorted(line, key=lambda d: -d["conf"])[:3]
    for d in boxes:
        x0, y0 = max(0, d["x"]), max(0, d["y"])
        x1, y1 = min(img.shape[1], d["x"] + d["w"]), min(img.shape[0], d["y"] + d["h"])
        roi = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if len(roi) < 10:
            continue
        med = np.median(roi, axis=0)                       # background
        dist = np.linalg.norm(roi - med, axis=1)
        text = roi[dist > np.percentile(dist, 70)]         # outlier = strokes
        if len(text):
            samples.append(np.median(text, axis=0))
    if not samples:
        return None
    return np.median(np.array(samples), axis=0)


def build_mask(img, line):
    H, W = img.shape[:2]
    color = watermark_color(img, line)
    if color is None:
        return None
    # y-band from the line, with padding.
    tops = [d["y"] for d in line]
    bots = [d["y"] + d["h"] for d in line]
    hmed = int(np.median([d["h"] for d in line]))
    pad = max(3, int(hmed * 0.4))
    y0 = max(0, min(tops) - pad)
    y1 = min(H, max(bots) + pad)
    # x-range: union of tokens, extended toward the missing side(s).
    lefts = [d["x"] for d in line]
    rights = [d["x"] + d["w"] for d in line]
    xL, xR = min(lefts), max(rights)
    span = xR - xL
    has_start = any(START_RE.search(d["t"]) for d in line)
    has_end = any(END_RE.search(d["t"]) and not START_RE.search(d["t"]) for d in line)
    has_strong = any(d["strong"] for d in line)
    # Full watermark width ~ several x the detected fragment; extend generously
    # toward the missing side(s). Nearest-centroid masking below keeps the
    # over-extension safe (only watermark-colored strokes get picked up).
    if not has_start:
        ext = max(span * 2, hmed * 16) if not has_strong else max(span, hmed * 6)
        xL = max(0, xL - int(ext))
    if not has_end:
        ext = max(span * 2, hmed * 12) if not has_strong else max(span, hmed * 5)
        xR = min(W, xR + int(ext))
    xL = max(0, xL - pad); xR = min(W, xR + pad)

    band = img[y0:y1, xL:xR].astype(np.float32)
    # Background colour: sample the band's outer rows (above/below the glyphs),
    # so a bold watermark can't bias the estimate toward its own colour.
    bh = band.shape[0]
    edge = max(1, bh // 5)
    bg_pixels = np.vstack([band[:edge].reshape(-1, 3), band[-edge:].reshape(-1, 3)])
    bg = np.median(bg_pixels, axis=0)
    if np.linalg.norm(color - bg) < 12:
        return None
    d_wm = np.linalg.norm(band - color.reshape(1, 1, 3), axis=2)
    d_bg = np.linalg.norm(band - bg.reshape(1, 1, 3), axis=2)
    # Colour gate: pixel is closer to the watermark colour than to the band
    # background -- captures full strokes incl. faint/anti-aliased edges.
    color_m = (d_wm < d_bg) & (d_wm < 120)
    # Stroke locator: a top-hat on a "watermark-ness" score responds to thin
    # glyph strokes but NOT to broad uniform regions (e.g. a light margin that
    # merely sits near the watermark colour). We use it to find *where* the text
    # is, then dilate into a local text zone.
    score = np.clip(d_bg - d_wm, 0, 255).astype(np.uint8)
    k = max(7, hmed * 2) | 1                      # kernel > glyph, < broad bg
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(score, cv2.MORPH_TOPHAT, kern)
    stroke_core = tophat > max(6, int(0.12 * float(tophat.max())))
    if stroke_core.sum() == 0:
        return None
    zk = max(5, hmed) | 1                          # grow strokes to cover glyphs
    zone = cv2.dilate(stroke_core.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zk, zk)))
    # Final mask: full watermark-coloured strokes, but only inside the text zone
    # -- so a broad near-colour background outside the text is never smeared.
    m = ((color_m & (zone > 0)).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, xL:xR] = m
    coverage = float((mask > 0).mean())
    return mask, coverage, (y0, y1, xL, xR)


def scaled_for_ocr(img):
    """Return (work, scale): upscale tiny images so text is legible, downscale
    huge ones so the OCR passes stay fast. The mask is scaled back afterward."""
    H, W = img.shape[:2]
    if max(H, W) < 1100:
        s = 1100.0 / max(H, W)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s
    if max(H, W) > 1500:
        s = 1500.0 / max(H, W)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s
    return img, 1.0


def find_line(work):
    """Detect the watermark text line in `work`, applying the accept gate.
    Returns the list of token dicts forming the line, or None if no watermark."""
    line = pick_line(ocr_tokens(work), work.shape[0])
    if not line:
        return None
    # The watermark always sits in the bottom of the image. Requiring that
    # rejects false matches on the artist's name appearing in body text higher
    # up (e.g. poster captions that mention "Chunming").
    yc = np.median([d["y"] + d["h"] / 2 for d in line])
    if yc <= work.shape[0] * 0.60:
        return None
    if any(d["strong"] for d in line):
        return line
    # No CHUNMING-ish read. Accept only a high-confidence watermark token
    # (WWW/ORG) sitting in the bottom strip, where the mark always lives --
    # in this corpus those tokens only appear as the watermark.
    bottom = [d for d in line if (d["y"] + d["h"] / 2) > work.shape[0] * 0.72]
    has_pair = (any(START_RE.search(d["t"]) for d in line) and
                any(END_RE.search(d["t"]) for d in line) and
                all(d["conf"] > 60 for d in line))
    has_strong_corner = any(d["conf"] >= 85 for d in bottom)
    return line if (has_pair or has_strong_corner) else None


def process(path, max_passes=3):
    img = cv2.imread(path)
    if img is None:
        return None, "unreadable"
    H, W = img.shape[:2]
    out = img
    acc_mask = np.zeros((H, W), np.uint8)
    all_tokens = []
    # Iterate: inpaint, then re-OCR the result. Wide letter-spaced marks often
    # leave an isolated trailing ".ORG" after pass 1; pass 2 finds and clears
    # it. Converges quickly; clean images exit on the first pass.
    for _ in range(max_passes):
        line = find_line(out)
        if line is None:
            break
        built = build_mask(out, line)
        if built is None:
            break
        mask = built[0]
        if not mask.any():
            break
        out = cv2.inpaint(out, mask, 4, cv2.INPAINT_TELEA)
        acc_mask = cv2.bitwise_or(acc_mask, mask)
        all_tokens.extend(d["t"] for d in line)
    if not acc_mask.any():
        return img, "clean"
    coverage = float((acc_mask > 0).mean())
    info = f"watermarked coverage={coverage*100:.2f}% tokens={all_tokens}"
    return out, info, img, acc_mask


def handle_one(p, preview, apply, report, outdir):
    """Process a single path; returns (name, status, info)."""
    res = process(p)
    info = res[1]
    name = os.path.basename(p)
    if info in ("clean", "unreadable"):
        return name, info, info
    out = res[0]
    if not report:
        orig, mask = res[2], res[3]
        if preview:
            h = orig.shape[0]
            sep = np.full((h, 6, 3), (0, 0, 255), np.uint8)
            overlay = orig.copy(); overlay[mask > 0] = (0, 0, 255)
            montage = np.hstack([orig, sep, overlay, sep, out])
            mh = 700
            if montage.shape[0] > mh:
                s = mh / montage.shape[0]
                montage = cv2.resize(montage, None, fx=s, fy=s)
            cv2.imwrite(os.path.join(outdir, "cmp_" + name + ".jpg"), montage)
        if apply:
            ext = os.path.splitext(p)[1].lower()
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg", ".JPG") else []
            cv2.imwrite(p, out, params)
    return name, "watermarked", info


def main():
    from functools import partial
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outdir", default="/tmp/wm_preview")
    ap.add_argument("--summary", default="/tmp/wm_summary.json")
    args = ap.parse_args()
    if args.preview:
        os.makedirs(args.outdir, exist_ok=True)
    summary = {"watermarked": [], "clean": [], "unreadable": []}
    worker = partial(handle_one, preview=args.preview, apply=args.apply,
                     report=args.report, outdir=args.outdir)

    def record(name, status, info):
        summary[status].append(name)
        tag = {"clean": "clean     ", "unreadable": "UNREADABLE",
               "watermarked": "WM        "}[status]
        line = f"{tag} {name}"
        if status == "watermarked":
            line += f"  {info}"
        print(line, flush=True)

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for name, status, info in pool.imap_unordered(worker, args.paths, chunksize=2):
                record(name, status, info)
    else:
        for p in args.paths:
            record(*worker(p))

    print("\n--- summary ---")
    print(json.dumps({k: len(v) for k, v in summary.items()}))
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
