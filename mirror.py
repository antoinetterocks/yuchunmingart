#!/usr/bin/env python3
"""
Mirror https://yuchunming.art into the current folder for offline viewing.

Stdlib only — no pip installs needed.

What it does:
  - Crawls every page on the yuchunming.art domain (following internal links).
  - Saves each HTML page to a local file (index.html, exhibitions-.html, ...).
  - Downloads every referenced asset: images (incl. Squarespace CDN), CSS, JS, fonts.
  - For Squarespace CDN images it grabs a high-resolution version (?format=2500w).
  - Rewrites all URLs in the saved HTML/CSS to point at the local copies,
    so the mirror opens offline by double-clicking index.html.

Usage:
    python3 mirror.py
"""

import os
import re
import sys
import time
import gzip
import hashlib
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit, unquote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

START_URL = "https://yuchunming.art/"
DOMAIN = "yuchunming.art"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = "assets"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"

# Pages we know exist from the site navigation (seed list; crawler also auto-discovers).
SEED_PATHS = [
    "/", "/exhibitions-", "/oil2020s", "/oil2010s", "/2000s", "/1990s",
    "/new-page-1", "/new-page", "/watercolor2020s", "/watercolor2010s",
    "/1980s", "/us-university-series", "/new-page-2", "/chronology",
    "/publications", "/videos", "/contact",
]

visited_pages = set()      # normalized page URLs already crawled
queued = []                # pages still to crawl
downloaded_assets = {}     # remote asset URL -> local relative path
page_filenames = {}        # normalized page URL -> local html filename


def log(msg):
    print(msg, flush=True)


def fetch(url, binary=False, tries=3):
    """Fetch a URL, transparently handling gzip. Returns bytes or str."""
    last_err = None
    for attempt in range(tries):
        try:
            req = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Accept-Encoding": "gzip",
            })
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                if binary:
                    return data, resp.headers.get("Content-Type", "")
                charset = "utf-8"
                ct = resp.headers.get("Content-Type", "")
                m = re.search(r"charset=([\w-]+)", ct)
                if m:
                    charset = m.group(1)
                return data.decode(charset, errors="replace"), ct
        except (URLError, HTTPError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def is_internal(url):
    netloc = urlparse(url).netloc.lower()
    return netloc == "" or netloc == DOMAIN or netloc == "www." + DOMAIN


def normalize_page(url):
    """Strip fragment/query for page identity, drop trailing slash."""
    s = urlsplit(url)
    path = s.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((s.scheme or "https", s.netloc or DOMAIN, path or "/", "", ""))


def page_to_filename(url):
    """Map a page URL to a local .html filename."""
    if url in page_filenames:
        return page_filenames[url]
    path = urlsplit(url).path
    if path in ("", "/"):
        name = "index.html"
    else:
        name = path.strip("/").replace("/", "_")
        if not name.endswith(".html"):
            name += ".html"
    page_filenames[url] = name
    return name


def asset_local_path(url, content_type=""):
    """Choose a stable local path under assets/ for an asset URL."""
    if url in downloaded_assets:
        return downloaded_assets[url]
    s = urlsplit(url)
    base = os.path.basename(unquote(s.path)) or "file"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    # Ensure an extension based on content type when missing.
    if "." not in base:
        ext = {
            "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
            "image/webp": ".webp", "text/css": ".css",
            "application/javascript": ".js", "text/javascript": ".js",
        }.get(content_type.split(";")[0].strip(), "")
        base += ext
    # Disambiguate with a short hash of the full URL (handles CDN params/dupes).
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    name, ext = os.path.splitext(base)
    local = f"{ASSET_DIR}/{name}-{h}{ext}"
    downloaded_assets[url] = local
    return local


def upgrade_squarespace(url):
    """Request a high-res version of Squarespace CDN images."""
    if "squarespace-cdn.com" in url or "squarespace.com" in url:
        s = urlsplit(url)
        if "format=" in s.query:
            q = re.sub(r"format=[^&]*", "format=2500w", s.query)
        else:
            q = (s.query + "&" if s.query else "") + "format=2500w"
        return urlunsplit((s.scheme, s.netloc, s.path, q, ""))
    return url


fetch_to_local = {}  # upgraded fetch URL -> local path (dedupes thumbnail variants)


def download_asset(url):
    """Download one asset (image/css/js/font). Returns local relative path or None."""
    fetch_url = upgrade_squarespace(url)
    if url in downloaded_assets:
        return downloaded_assets[url]
    # Many srcset thumbnails of the same image upgrade to one 2500w URL —
    # download/store it once and point every referrer at the same file.
    if fetch_url in fetch_to_local:
        downloaded_assets[url] = fetch_to_local[fetch_url]
        return fetch_to_local[fetch_url]
    try:
        data, ct = fetch(fetch_url, binary=True)
    except Exception as e:
        log(f"  ! asset failed {url}: {e}")
        return None
    local = asset_local_path(fetch_url, ct)
    fetch_to_local[fetch_url] = local
    downloaded_assets[url] = local
    full = os.path.join(OUT_DIR, local)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)
    # If it's CSS, recurse into url(...) and @import references.
    if ct.split(";")[0].strip() == "text/css" or local.endswith(".css"):
        try:
            text = data.decode("utf-8", errors="replace")
            text = process_css(text, url)
            with open(full, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
    log(f"  + {local}  ({len(data)//1024} KB)")
    return local


def process_css(text, css_url):
    """Download assets referenced inside a CSS file and rewrite to local paths."""
    def repl(m):
        raw = m.group(1).strip("'\"")
        if raw.startswith("data:") or not raw:
            return m.group(0)
        abs_url = urljoin(css_url, raw)
        local = download_asset(abs_url)
        if not local:
            return m.group(0)
        # CSS lives in assets/, assets live in assets/ -> relative within same dir
        return f"url({os.path.basename(local)})"
    text = re.sub(r"url\(([^)]+)\)", repl, text)

    def imp(m):
        raw = m.group(1).strip("'\"")
        abs_url = urljoin(css_url, raw)
        local = download_asset(abs_url)
        if not local:
            return m.group(0)
        return f'@import "{os.path.basename(local)}"'
    text = re.sub(r'@import\s+["\']([^"\']+)["\']', imp, text)
    return text


# Attributes that may carry URLs we want to localize.
URL_ATTRS = ["src", "href", "data-src", "data-image", "poster", "content"]


def process_html(html, page_url):
    """Download referenced assets, rewrite links to local files. Returns new HTML."""
    page_file = page_to_filename(normalize_page(page_url))

    # 1) Handle srcset specially (comma-separated url size pairs).
    def srcset_repl(m):
        attr, val = m.group(1), m.group(2)
        parts = []
        for chunk in val.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            bits = chunk.split()
            u = bits[0]
            abs_url = urljoin(page_url, u)
            local = download_asset(abs_url)
            if local:
                bits[0] = local
            parts.append(" ".join(bits))
        return f'{attr}="{", ".join(parts)}"'
    html = re.sub(r'(srcset|data-srcset)="([^"]*)"', srcset_repl, html)

    # 2) Generic attribute URL rewriting.
    def attr_repl(m):
        attr, quote, val = m.group(1), m.group(2), m.group(3)
        if not val or val.startswith(("data:", "mailto:", "tel:", "javascript:", "#")):
            return m.group(0)
        abs_url = urljoin(page_url, val)
        if not abs_url.startswith("http"):
            return m.group(0)

        # Internal HTML page link -> local html file.
        if is_internal(abs_url) and attr in ("href",):
            norm = normalize_page(abs_url)
            path = urlsplit(norm).path
            # Treat as a page if it has no file extension (Squarespace routes).
            if "." not in os.path.basename(path) or path.endswith(".html"):
                fname = page_to_filename(norm)
                if norm not in visited_pages and norm not in queued:
                    queued.append(norm)
                frag = urlsplit(abs_url).fragment
                return f'{attr}={quote}{fname}{("#" + frag) if frag else ""}{quote}'

        # Otherwise treat as an asset (images, css, js, fonts).
        if attr == "content" and not re.search(r"\.(jpg|jpeg|png|gif|webp|svg|css|js)", abs_url, re.I):
            return m.group(0)  # don't pull arbitrary meta content URLs
        local = download_asset(abs_url)
        if local:
            return f'{attr}={quote}{local}{quote}'
        return m.group(0)

    html = re.sub(r'(\b(?:src|href|data-src|data-image|poster|content))=("|\')([^"\']*)\2', attr_repl, html)

    # 3) Inline <style> blocks.
    def style_repl(m):
        return f"<style{m.group(1)}>{process_css(m.group(2), page_url)}</style>"
    html = re.sub(r"<style([^>]*)>(.*?)</style>", style_repl, html, flags=re.S)

    return html


def crawl():
    os.makedirs(os.path.join(OUT_DIR, ASSET_DIR), exist_ok=True)

    # Seed the queue.
    for p in SEED_PATHS:
        queued.append(normalize_page(urljoin(START_URL, p)))

    while queued:
        page = queued.pop(0)
        if page in visited_pages:
            continue
        visited_pages.add(page)
        log(f"\n== PAGE {page}")
        try:
            html, ct = fetch(page)
        except Exception as e:
            log(f"  ! page failed: {e}")
            continue
        if "html" not in ct:
            continue
        new_html = process_html(html, page)
        fname = page_to_filename(page)
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(new_html)
        log(f"  saved -> {fname}")

    log(f"\nDone. {len(visited_pages)} pages, {len(downloaded_assets)} assets.")
    log(f"Open: {os.path.join(OUT_DIR, 'index.html')}")


if __name__ == "__main__":
    crawl()
