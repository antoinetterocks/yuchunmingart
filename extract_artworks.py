#!/usr/bin/env python3
"""
Extract artwork metadata from the mirrored gallery pages into artworks.csv.

Metadata lives in the original filename preserved in each <img alt="...">, e.g.:
  "001 oil 聚居-山寨晨 Kinship-Dwelling-Mountain Village's Morning (2002)-30X40inch.JPG"
We parse out index, medium, Chinese title, English title, year and physical
dimensions where present, and always keep the raw title. We also record the
image's pixel dimensions and on-disk file size.
"""
import os
import re
import csv
import html

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES_BASE = "https://antoinetterocks.github.io/yuchunmingart"

# page file -> (Medium, Era/Series) per the site navigation
GALLERIES = {
    "oil2020s.html":            ("Oil",        "2020s"),
    "oil2010s.html":            ("Oil",        "2010s"),
    "2000s.html":               ("Oil",        "2000s"),
    "1990s.html":               ("Oil",        "1990s"),
    "new-page-1.html":          ("Oil",        "1980s"),
    "new-page.html":            ("Oil",        "Religious Leader Series"),
    "watercolor2020s.html":     ("Watercolor", "2020s"),
    "watercolor2010s.html":     ("Watercolor", "1990-2000s"),
    "1980s.html":               ("Watercolor", "1980s"),
    "us-university-series.html":("Watercolor", "U.S. University Series"),
}

CJK = r"　-〿㐀-䶿一-鿿＀-￯"

IMG_RE = re.compile(r"<img\b[^>]*>", re.I)
ATTR_RE = lambda name: re.compile(rf'{name}="([^"]*)"', re.I)
ALT_RE = ATTR_RE("alt")
DIMS_RE = ATTR_RE("data-image-dimensions")
SRC_RE = ATTR_RE("data-image")


def attrs(tag):
    alt = ""
    # take the first non-empty alt in the tag
    for m in re.finditer(r'alt="([^"]*)"', tag, re.I):
        if m.group(1).strip():
            alt = m.group(1)
            break
    dims = DIMS_RE.search(tag)
    src = SRC_RE.search(tag)
    return (html.unescape(alt),
            dims.group(1) if dims else "",
            src.group(1) if src else "")


def parse_title(raw):
    """Best-effort structured fields from the original filename/title."""
    name = re.sub(r"\.(jpe?g|png|gif|webp)$", "", raw, flags=re.I).strip()

    index = ""
    m = re.match(r"^\s*(\d{1,4})\s+", name)
    if m:
        index = m.group(1)
        name = name[m.end():]

    medium = ""
    m = re.match(r"^\s*(oils?|watercolou?rs?|ink|acrylic)\b\.?\s*", name, re.I)
    if m:
        medium = m.group(1)
        name = name[m.end():]

    year = ""
    m = re.search(r"\(?\b((?:19|20)\d{2})\b\)?", name)
    if m:
        year = m.group(1)

    dimensions = ""
    m = re.search(r"(\d{1,3}\s*[xX×]\s*\d{1,3}\s*(?:inch|in|cm)?)", name)
    if m:
        dimensions = m.group(1).replace(" ", "")

    # title text = name minus the year-parenthetical and trailing dimension token
    title = name
    title = re.sub(r"\(?(?:19|20)\d{2}\)?", "", title)
    title = re.sub(r"-?\s*\d{1,3}\s*[xX×]\s*\d{1,3}\s*(?:inch|in|cm)?", "", title)
    title = title.strip(" -_")

    chinese = "".join(re.findall(rf"[{CJK}\s\-·、，。]+", title)).strip(" -_、，")
    english = re.sub(rf"[{CJK}]", "", title)
    english = re.sub(r"\s{2,}", " ", english).strip(" -_")

    return index, medium, title, chinese, english, year, dimensions


rows = []
for page, (medium_default, era) in GALLERIES.items():
    path = os.path.join(ROOT, page)
    if not os.path.exists(path):
        continue
    text = open(path, encoding="utf-8", errors="replace").read()
    seen = {}  # local src -> best (longest alt) record
    for tag in IMG_RE.findall(text):
        alt, pix, src = attrs(tag)
        if not src or not src.startswith("assets/"):
            continue
        prev = seen.get(src)
        if prev is None or len(alt) > len(prev[0]):
            seen[src] = (alt, pix, src)
    for src, (alt, pix, _) in seen.items():
        fpath = os.path.join(ROOT, src)
        size_kb = round(os.path.getsize(fpath) / 1024) if os.path.exists(fpath) else ""
        idx, med, title, zh, en, year, dims = parse_title(alt)
        rows.append({
            "Gallery": page.replace(".html", ""),
            "Medium": medium_default,
            "Era / Series": era,
            "Index": idx,
            "Title (raw)": alt,
            "Title (English)": en,
            "Title (Chinese)": zh,
            "Year": year,
            "Physical dimensions": dims,
            "Image pixels (WxH)": pix,
            "File size (KB)": size_kb,
            "Local file": src,
            "Live URL": f"{PAGES_BASE}/{src}",
        })

# sort: medium, era, then index/title
def sort_key(r):
    try:
        i = int(r["Index"])
    except (ValueError, TypeError):
        i = 9999
    return (r["Medium"], r["Era / Series"], i, r["Title (raw)"])

rows.sort(key=sort_key)

cols = ["Gallery", "Medium", "Era / Series", "Index", "Title (raw)",
        "Title (English)", "Title (Chinese)", "Year", "Physical dimensions",
        "Image pixels (WxH)", "File size (KB)", "Local file", "Live URL"]

out = os.path.join(ROOT, "artworks.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} artworks -> {out}")
# quick coverage stats
def pct(field):
    n = sum(1 for r in rows if r[field])
    return f"{n}/{len(rows)} ({100*n//max(len(rows),1)}%)"
for fld in ["Year", "Physical dimensions", "Title (English)", "Title (Chinese)"]:
    print(f"  has {fld}: {pct(fld)}")
print("\nby medium/era:")
from collections import Counter
for k, c in sorted(Counter((r["Medium"], r["Era / Series"]) for r in rows).items()):
    print(f"  {k[0]:11} {k[1]:24} {c}")
