#!/usr/bin/env python3
"""
gpic_review.py - Triage the near-duplicate clusters found by gpic_dedup.py.

It reads the duplicates.csv produced by gpic_dedup.py (no re-hashing of the
whole library) and, for every cluster, decides how confident we are that the
images really are the same photo, so you only have to eyeball the uncertain
few instead of all 9000+ groups.

For each cluster it computes:
  * SHA-256 of the file bytes         -> byte-identical copies (EXACT).
  * perceptual-hash Hamming distance  -> structural closeness.
  * SSIM (structural similarity)      -> pixel-level similarity.
and assigns a confidence label:
  EXACT   - byte-identical or pixel-identical; 100% safe, no review needed.
  HIGH    - SSIM >= 0.97 and Hamming <= 3; almost certainly the same photo.
  MEDIUM  - SSIM >= 0.90 or Hamming <= 5; probably same, quick glance advised.
  LOW     - everything else; please look.

Outputs:
  * keep_candidates.csv  - one row per image, marked KEEP or CANDIDATE
                           (KEEP = highest resolution, then largest file, then
                           earliest date). Nothing is deleted.
  * review_gallery*.html - visual gallery, sorted worst-confidence-first, with
                           embedded thumbnails for the clusters that need eyes.

This tool never deletes or moves files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import os
import sys
from collections import deque
from dataclasses import dataclass, field

# Confidence thresholds (tunable via CLI).
DEFAULT_HIGH_SSIM = 0.97
DEFAULT_HIGH_HAMMING = 3
DEFAULT_MEDIUM_SSIM = 0.90
DEFAULT_MEDIUM_HAMMING = 5

CONFIDENCE_ORDER = {"EXACT": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
CONFIDENCE_COLOR = {
    "EXACT": "#2e7d32", "HIGH": "#558b2f", "MEDIUM": "#ef6c00", "LOW": "#c62828",
    "UNREADABLE": "#6a1b9a",
}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".heic", ".heif", ".tif", ".tiff",
}


def maybe_register_heif() -> None:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Img:
    filename: str
    url: str
    path: str = ""
    width: int = 0
    height: int = 0
    filesize: int = 0
    date: str = ""
    sha256: str = ""
    pixels_sha: str = ""
    phash: int | None = None
    gray: object = None          # 256x256 float ndarray for SSIM
    thumb_b64: str = ""
    readable: bool = False
    role: str = ""               # KEEP / CANDIDATE


@dataclass
class ClusterResult:
    cluster_id: str
    images: list = field(default_factory=list)
    confidence: str = "LOW"
    max_hamming: int = 0
    min_ssim: float = 1.0


# --------------------------------------------------------------------------- #
# Path resolution: map basenames in the CSV to real files under the root
# --------------------------------------------------------------------------- #
def build_path_index(root: str) -> dict:
    index: dict[str, deque] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                index.setdefault(name, deque()).append(os.path.join(dirpath, name))
    return index


def resolve_path(index: dict, filename: str) -> str:
    paths = index.get(filename)
    if not paths:
        return ""
    # Multiple files can share a basename (Takeout repeats photos across albums);
    # hand out distinct paths on repeated lookups, reusing the last once exhausted.
    if len(paths) > 1:
        return paths.popleft()
    return paths[0]


# --------------------------------------------------------------------------- #
# Per-image analysis
# --------------------------------------------------------------------------- #
def _exif_date(pil_img) -> str:
    try:
        exif = pil_img.getexif()
        # 36867 DateTimeOriginal, 306 DateTime
        for tag in (36867, 306):
            val = exif.get(tag)
            if val:
                return str(val)
    except Exception:
        pass
    return ""


def analyze_image(img: Img) -> None:
    from PIL import Image
    import imagehash
    import numpy as np

    if not img.path or not os.path.isfile(img.path):
        return
    try:
        img.filesize = os.path.getsize(img.path)
        with open(img.path, "rb") as fh:
            data = fh.read()
        img.sha256 = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = im.convert("RGB")
            img.width, img.height = rgb.size
            img.pixels_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            img.date = _exif_date(im) or ""
            img.phash = int(str(imagehash.phash(rgb)), 16)
            gray = rgb.convert("L").resize((256, 256))
            img.gray = np.asarray(gray, dtype=np.float64) / 255.0
        if not img.date:
            img.date = _mtime(img.path)
        img.readable = True
    except Exception:
        img.readable = False


def make_thumbnail(img: Img, thumb_size: int) -> None:
    """Build the embedded JPEG thumbnail; only needed for clusters we display."""
    from PIL import Image

    if not img.readable or not img.path:
        return
    try:
        with Image.open(img.path) as im:
            im.load()
            thumb = im.convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=70)
            img.thumb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass


def _mtime(path: str) -> str:
    import datetime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ssim(a, b) -> float:
    from scipy.ndimage import gaussian_filter
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = gaussian_filter(a, 1.5)
    mu_b = gaussian_filter(b, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = gaussian_filter(a * a, 1.5) - mu_a2
    var_b = gaussian_filter(b * b, 1.5) - mu_b2
    cov = gaussian_filter(a * b, 1.5) - mu_ab
    smap = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
    return float(smap.mean())


# --------------------------------------------------------------------------- #
# Cluster classification and KEEP selection
# --------------------------------------------------------------------------- #
def classify_cluster(cluster: ClusterResult, thresholds: dict) -> None:
    readable = [im for im in cluster.images if im.readable]
    if len(readable) < len(cluster.images):
        # Any unreadable member -> can't auto-trust; force a look.
        cluster.confidence = "LOW"
    if len(readable) < 2:
        cluster.max_hamming = 0
        cluster.min_ssim = 1.0
        if cluster.confidence != "LOW":
            cluster.confidence = "LOW"
        _assign_keep(cluster)
        return

    max_ham = 0
    min_s = 1.0
    all_identical = True
    for i in range(len(readable)):
        for j in range(i + 1, len(readable)):
            a, b = readable[i], readable[j]
            identical = (a.sha256 == b.sha256) or _pixels_equal(a, b)
            if not identical:
                all_identical = False
            ham = hamming(a.phash, b.phash) if (a.phash is not None and b.phash is not None) else 64
            s = 1.0 if identical else ssim(a.gray, b.gray)
            max_ham = max(max_ham, 0 if identical else ham)
            min_s = min(min_s, s)

    cluster.max_hamming = max_ham
    cluster.min_ssim = min_s

    if all_identical:
        label = "EXACT"
    elif min_s >= thresholds["high_ssim"] and max_ham <= thresholds["high_hamming"]:
        label = "HIGH"
    elif min_s >= thresholds["medium_ssim"] or max_ham <= thresholds["medium_hamming"]:
        label = "MEDIUM"
    else:
        label = "LOW"

    # Never upgrade above LOW if an unreadable member forced review.
    if cluster.confidence == "LOW" and len(readable) < len(cluster.images):
        label = "LOW"
    cluster.confidence = label
    _assign_keep(cluster)


def _pixels_equal(a: Img, b: Img) -> bool:
    # Full-resolution pixel identity: catches re-encodes whose bytes differ but
    # decoded pixels match, without the false positives of a downscaled compare.
    return bool(a.pixels_sha and a.pixels_sha == b.pixels_sha)


def _assign_keep(cluster: ClusterResult) -> None:
    def sort_key(im: Img):
        area = im.width * im.height
        date = im.date or "9999-99-99"
        return (-area, -im.filesize, date)
    ordered = sorted(cluster.images, key=sort_key)
    for idx, im in enumerate(ordered):
        im.role = "KEEP" if idx == 0 else "CANDIDATE"


# --------------------------------------------------------------------------- #
# CSV parsing / output
# --------------------------------------------------------------------------- #
def read_clusters(csv_path: str) -> list:
    clusters: list = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filenames = row.get("filenames", "").split("|") if row.get("filenames") else []
            urls = row.get("urls", "").split("|") if row.get("urls") else []
            if len(urls) < len(filenames):
                urls += [""] * (len(filenames) - len(urls))
            imgs = [Img(filename=f, url=u) for f, u in zip(filenames, urls)]
            clusters.append(ClusterResult(cluster_id=row.get("cluster_id", ""), images=imgs))
    return clusters


def write_keep_candidates(clusters: list, out_path: str) -> int:
    rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "cluster_id", "confidence", "max_hamming", "min_ssim", "role",
            "filename", "path", "url", "width", "height", "filesize_bytes", "date",
        ])
        for c in clusters:
            for im in c.images:
                writer.writerow([
                    c.cluster_id, c.confidence, c.max_hamming, f"{c.min_ssim:.4f}",
                    im.role or ("UNREADABLE" if not im.readable else ""),
                    im.filename, im.path, im.url, im.width, im.height,
                    im.filesize, im.date,
                ])
                rows += 1
    return rows


# --------------------------------------------------------------------------- #
# HTML gallery
# --------------------------------------------------------------------------- #
_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f4f4f6; color: #222; }
header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
         padding: 12px 16px; z-index: 10; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
h1 { font-size: 18px; margin: 0 0 6px; }
.counts span { display: inline-block; margin-right: 10px; font-size: 13px; }
.filters button { margin-right: 6px; padding: 5px 10px; border: 1px solid #bbb;
         background: #fff; border-radius: 4px; cursor: pointer; font-size: 13px; }
.filters button.active { background: #222; color: #fff; border-color: #222; }
.cluster { background: #fff; margin: 14px; border-radius: 8px; padding: 10px 12px;
         box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.badge { display: inline-block; color: #fff; padding: 2px 8px; border-radius: 10px;
         font-size: 12px; font-weight: 600; }
.cluster h2 { font-size: 14px; margin: 6px 0 10px; font-weight: 500; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; }
.tile { width: 210px; border: 1px solid #e2e2e2; border-radius: 6px; padding: 6px;
        background: #fafafa; }
.tile img { width: 100%; height: auto; border-radius: 4px; background: #eee; }
.tile.keep { border-color: #2e7d32; box-shadow: 0 0 0 2px #2e7d3233; }
.role { font-size: 11px; font-weight: 700; padding: 1px 6px; border-radius: 8px; }
.role.KEEP { background: #2e7d32; color: #fff; }
.role.CANDIDATE { background: #b0bec5; color: #222; }
.meta { font-size: 11px; color: #555; margin-top: 4px; line-height: 1.4; word-break: break-all; }
.meta a { color: #1565c0; text-decoration: none; }
.noimg { font-size: 12px; color: #999; padding: 30px 0; text-align: center; }
.pager { padding: 14px; text-align: center; }
.pager a { margin: 0 6px; }
"""

_JS = """
function applyFilter(level, btn) {
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.cluster').forEach(c => {
    c.style.display = (level === 'ALL' || c.dataset.confidence === level) ? '' : 'none';
  });
}
"""


def _tile_html(im: Img, embed: bool) -> str:
    role_cls = im.role or "CANDIDATE"
    keep_cls = " keep" if im.role == "KEEP" else ""
    if embed and im.thumb_b64:
        img_tag = f'<img src="data:image/jpeg;base64,{im.thumb_b64}" loading="lazy">'
    elif not im.readable:
        img_tag = '<div class="noimg">unreadable<br>(HEIC plugin?)</div>'
    else:
        img_tag = '<div class="noimg">(thumb omitted)</div>'
    url_html = (f'<a href="{html.escape(im.url)}" target="_blank">open in Google Photos</a>'
                if im.url else '<span>(no url)</span>')
    dims = f"{im.width}x{im.height}" if im.width else "?"
    size_kb = f"{im.filesize/1024:.0f} KB" if im.filesize else "?"
    return (
        f'<div class="tile{keep_cls}">'
        f'<span class="role {role_cls}">{html.escape(im.role or "?")}</span> '
        f'{img_tag}'
        f'<div class="meta">{html.escape(im.filename)}<br>'
        f'{dims} &middot; {size_kb}<br>{html.escape(im.date)}<br>{url_html}</div>'
        f'</div>'
    )


def _cluster_html(c: ClusterResult, embed_levels: set) -> str:
    embed = c.confidence in embed_levels
    color = CONFIDENCE_COLOR.get(c.confidence, "#555")
    note = {
        "EXACT": "byte/pixel identical - safe to dedupe",
        "HIGH": "almost certainly the same photo",
        "MEDIUM": "probably same - quick glance advised",
        "LOW": "please look",
    }.get(c.confidence, "")
    tiles = "".join(_tile_html(im, embed) for im in c.images)
    return (
        f'<div class="cluster" data-confidence="{c.confidence}">'
        f'<span class="badge" style="background:{color}">{c.confidence}</span>'
        f'<h2>Cluster {html.escape(str(c.cluster_id))} &middot; {len(c.images)} images '
        f'&middot; max Hamming {c.max_hamming} &middot; SSIM {c.min_ssim:.3f} '
        f'&middot; {note}</h2>'
        f'<div class="tiles">{tiles}</div></div>'
    )


def write_gallery(clusters: list, out_dir: str, embed_levels: set, per_page: int) -> list:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EXACT": 3}
    ordered = sorted(clusters, key=lambda c: (order.get(c.confidence, 0),
                                              -c.max_hamming, c.min_ssim))
    counts = {}
    for c in clusters:
        counts[c.confidence] = counts.get(c.confidence, 0) + 1
    counts_html = "".join(
        f'<span style="color:{CONFIDENCE_COLOR.get(k,"#555")}">{k}: {counts.get(k,0)}</span>'
        for k in ("LOW", "MEDIUM", "HIGH", "EXACT") if counts.get(k))

    pages = [ordered[i:i + per_page] for i in range(0, len(ordered), per_page)] or [[]]
    written = []
    for pnum, page in enumerate(pages, 1):
        nav = ""
        if len(pages) > 1:
            links = " ".join(
                (f'<b>{i}</b>' if i == pnum else f'<a href="review_gallery_{i}.html">{i}</a>')
                for i in range(1, len(pages) + 1))
            nav = f'<div class="pager">Page: {links}</div>'
        body = "".join(_cluster_html(c, embed_levels) for c in page)
        doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Duplicate Review Gallery</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>Duplicate Review Gallery &middot; {len(clusters)} clusters</h1>
  <div class="counts">{counts_html}</div>
  <div class="filters">
    <button class="active" onclick="applyFilter('ALL', this)">All</button>
    <button onclick="applyFilter('LOW', this)">LOW</button>
    <button onclick="applyFilter('MEDIUM', this)">MEDIUM</button>
    <button onclick="applyFilter('HIGH', this)">HIGH</button>
    <button onclick="applyFilter('EXACT', this)">EXACT</button>
  </div>
</header>
{nav}
{body}
{nav}
<script>{_JS}</script>
</body></html>"""
        path = os.path.join(out_dir, f"review_gallery_{pnum}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(csv_path: str, root: str, out_dir: str, thumb_size: int,
        embed_levels: set, per_page: int, thresholds: dict) -> None:
    maybe_register_heif()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading clusters from: {csv_path}")
    clusters = read_clusters(csv_path)
    print(f"  {len(clusters)} clusters.")

    print(f"Indexing files under: {root}")
    index = build_path_index(root)
    print(f"  {sum(len(v) for v in index.values())} image files indexed.")

    print("\nAnalyzing clusters (SHA-256 + pHash + SSIM)...")
    total = len(clusters)
    for i, c in enumerate(clusters, 1):
        for im in c.images:
            im.path = resolve_path(index, im.filename)
            analyze_image(im)
        classify_cluster(c, thresholds)
        if c.confidence in embed_levels:
            for im in c.images:
                make_thumbnail(im, thumb_size)
        if i % 200 == 0 or i == total:
            print(f"  analyzed {i}/{total}")

    kc_path = os.path.join(out_dir, "keep_candidates.csv")
    rows = write_keep_candidates(clusters, kc_path)
    pages = write_gallery(clusters, out_dir, embed_levels, per_page)

    counts: dict[str, int] = {}
    for c in clusters:
        counts[c.confidence] = counts.get(c.confidence, 0) + 1

    print("\nSummary")
    for label in ("EXACT", "HIGH", "MEDIUM", "LOW"):
        if counts.get(label):
            print(f"  {label:<7}: {counts[label]}")
    need = counts.get("MEDIUM", 0) + counts.get("LOW", 0)
    print(f"  -> clusters needing a human glance (MEDIUM+LOW): {need}")
    print(f"  keep_candidates.csv rows : {rows}")
    print(f"  keep_candidates.csv      : {os.path.abspath(kc_path)}")
    print(f"  gallery page(s)          : {len(pages)}")
    for p in pages:
        print(f"    {os.path.abspath(p)}")
    print("\nOpen the gallery in any browser. Review is sorted worst-first;")
    print("EXACT/HIGH are auto-trusted. Nothing has been deleted.")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile
    import numpy as np
    from PIL import Image

    print("Running gpic_review self-test...\n")
    tmp = tempfile.mkdtemp(prefix="review_selftest_")
    photos = os.path.join(tmp, "Takeout", "GP")
    os.makedirs(photos)

    def scene(shift=0, noise=0, seed=0, size=(256, 256)):
        x = np.linspace(0, 255, 256)
        arr = np.tile(x, (256, 1))
        arr[30:120, 40:150] = 255
        arr = arr + shift
        if noise:
            rng = np.random.default_rng(seed)
            arr = arr + rng.normal(0, noise, arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "L").convert("RGB")
        return im.resize(size)

    def other():
        y = np.linspace(0, 255, 256)
        arr = np.tile(y.reshape(-1, 1), (1, 256))
        arr[150:230, 120:220] = 0
        return Image.fromarray(arr.astype("uint8"), "L").convert("RGB")

    # Cluster 1: exact byte copy -> EXACT
    scene().save(os.path.join(photos, "exact_a.jpg"))
    import shutil as _sh
    _sh.copyfile(os.path.join(photos, "exact_a.jpg"), os.path.join(photos, "exact_b.jpg"))
    # Cluster 2: same photo, resized/re-saved -> HIGH
    scene().save(os.path.join(photos, "high_a.jpg"))
    scene(size=(200, 200)).save(os.path.join(photos, "high_b.jpg"))
    # Cluster 3: mild edit/noise -> MEDIUM/HIGH
    scene().save(os.path.join(photos, "med_a.jpg"))
    scene(shift=12, noise=6, seed=2).save(os.path.join(photos, "med_b.jpg"))
    # Cluster 4: false positive (structurally different) -> LOW
    scene().save(os.path.join(photos, "low_a.jpg"))
    other().save(os.path.join(photos, "low_b.jpg"))

    csv_path = os.path.join(tmp, "duplicates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "representative_phash", "image_count", "filenames", "urls"])
        w.writerow(["1", "0", 2, "exact_a.jpg|exact_b.jpg", "http://x/EXA|http://x/EXB"])
        w.writerow(["2", "0", 2, "high_a.jpg|high_b.jpg", "http://x/HIA|http://x/HIB"])
        w.writerow(["3", "0", 2, "med_a.jpg|med_b.jpg", "http://x/MEA|http://x/MEB"])
        w.writerow(["4", "0", 2, "low_a.jpg|low_b.jpg", "http://x/LOA|http://x/LOB"])

    out_dir = os.path.join(tmp, "review_out")
    thresholds = {"high_ssim": DEFAULT_HIGH_SSIM, "high_hamming": DEFAULT_HIGH_HAMMING,
                  "medium_ssim": DEFAULT_MEDIUM_SSIM, "medium_hamming": DEFAULT_MEDIUM_HAMMING}
    run(csv_path, os.path.join(tmp, "Takeout"), out_dir, 180,
        {"LOW", "MEDIUM"}, 400, thresholds)

    labels = {}
    with open(os.path.join(out_dir, "keep_candidates.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[row["cluster_id"]] = row["confidence"]
    print("\nCluster labels:", labels)

    ok = True
    if labels.get("1") != "EXACT":
        print("FAIL: identical copies should be EXACT."); ok = False
    else:
        print("PASS: identical copies -> EXACT.")
    if labels.get("4") != "LOW":
        print("FAIL: the different image pair should be LOW."); ok = False
    else:
        print("PASS: structurally different pair -> LOW.")
    if labels.get("2") not in ("HIGH", "EXACT"):
        print(f"FAIL: resized copy should be HIGH (got {labels.get('2')})."); ok = False
    else:
        print("PASS: resized copy -> HIGH/EXACT.")
    gallery = os.path.join(out_dir, "review_gallery_1.html")
    if os.path.isfile(gallery) and os.path.getsize(gallery) > 0:
        print("PASS: gallery HTML generated.")
    else:
        print("FAIL: gallery HTML missing."); ok = False

    print(f"\nSelf-test {'SUCCEEDED' if ok else 'FAILED'}.")
    print(f"(Artifacts at: {out_dir})")
    if not ok:
        sys.exit(1)
