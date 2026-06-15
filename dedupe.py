#!/usr/bin/env python3
"""
Dedupe assets/ in place: keep ONE copy of each unique image, delete byte-identical
duplicates, and rewrite every reference in the .html pages to the kept copy.

No unique image is ever lost — only exact byte-for-byte duplicates are removed.
"""
import os
import re
import hashlib
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")


def md5(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


# 1) Group files by content hash.
by_hash = {}
files = [f for f in os.listdir(ASSETS) if os.path.isfile(os.path.join(ASSETS, f))]
print(f"hashing {len(files)} files...")
for name in files:
    by_hash.setdefault(md5(os.path.join(ASSETS, name)), []).append(name)

# 2) Pick a canonical name per hash (shortest, then alphabetical for stability).
rename = {}          # duplicate name -> canonical name
removed_bytes = 0
removed_count = 0
for h, names in by_hash.items():
    canonical = sorted(names, key=lambda n: (len(n), n))[0]
    for n in names:
        if n != canonical:
            rename[n] = canonical

print(f"unique images: {len(by_hash)}")
print(f"duplicate files to remove: {len(rename)}")

# 3) Rewrite all HTML pages: replace any reference to a duplicate with its canonical.
html_pages = glob.glob(os.path.join(ROOT, "*.html"))
total_refs = 0
for page in html_pages:
    with open(page, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    orig = text
    for dup, canon in rename.items():
        if dup in text:
            text = text.replace("assets/" + dup, "assets/" + canon)
            text = text.replace(dup, canon)  # catch any non-prefixed refs
    if text != orig:
        total_refs += 1
        with open(page, "w", encoding="utf-8") as f:
            f.write(text)
print(f"rewrote references in {total_refs} html pages")

# 4) Delete the duplicate files.
for dup in rename:
    p = os.path.join(ASSETS, dup)
    if os.path.exists(p):
        removed_bytes += os.path.getsize(p)
        os.remove(p)
        removed_count += 1

print(f"removed {removed_count} duplicate files, reclaimed {removed_bytes/1e9:.2f} GB")
print(f"assets now: {len(os.listdir(ASSETS))} files")
