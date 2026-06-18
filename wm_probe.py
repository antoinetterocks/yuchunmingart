#!/usr/bin/env python3
"""Probe v2: OCR across several preprocessing variants to robustly find the
CHUNMINGARTS watermark line. Reports which variant caught it + boxes."""
import sys, re
import cv2
import numpy as np
import pytesseract

STRONG_RE = re.compile(r"(chunm|nming|mingart|ingarts)", re.I)
TOKEN_RE  = re.compile(r"(chunm|nming|mingart|ingarts|^arts?$|arts\.|\.org|^org$|^www$|www\.)", re.I)

def variants(img):
    """Yield (name, single-channel uint8) images to OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield "gray", gray
    yield "inv",  255 - gray
    yield "green", g          # red text -> dark here on light bg
    yield "blue",  b
    yield "clahe", clahe
    yield "clahe_inv", 255 - clahe

def ocr_boxes(g, psm):
    data = pytesseract.image_to_data(g, output_type=pytesseract.Output.DICT,
                                     config=f"--psm {psm}")
    out = []
    n = len(data["text"])
    for i in range(n):
        t = data["text"][i].strip()
        if not t:
            continue
        conf = float(data["conf"][i])
        if conf < 15:
            continue
        out.append((t, conf, data["left"][i], data["top"][i],
                    data["width"][i], data["height"][i]))
    return out

def probe(path):
    img = cv2.imread(path)
    if img is None:
        print(f"{path}: UNREADABLE"); return
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) < 1100:
        scale = 1100.0 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    hits = []
    for vname, g in variants(img):
        for psm in (11, 6):
            for (t, conf, x, y, bw, bh) in ocr_boxes(g, psm):
                if TOKEN_RE.search(t):
                    hits.append((vname, psm, t, round(conf),
                                 round(x/W, 2), round(y/H, 2), round(bw/W, 2),
                                 bool(STRONG_RE.search(t))))
    print(f"\n=== {path}  ({w}x{h}) scale={scale:.2f}")
    strong = [hh for hh in hits if hh[-1]]
    if not hits:
        print("  (NOTHING)")
    else:
        print(f"  strong={len(strong)} total={len(hits)}")
        for hh in hits[:8]:
            print("   ", hh)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
