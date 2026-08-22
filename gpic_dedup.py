#!/usr/bin/env python3
"""
gpic_dedup.py - Find near-duplicate photos in a Google Takeout export.

Pipeline:
  1. Detect OS and verify required libraries are installed (guides the user if not).
  2. Recursively scan a Takeout root folder for images.
  3. For each image, match its Google Photos JSON sidecar to recover the unique
     `url` (the https://photos.google.com/photo/... link).
  4. Compute a perceptual hash (pHash) for each image.
  5. Cluster near-duplicates using Hamming distance via an in-script BK-tree
     + union-find (no external BK-tree dependency).
  6. Write the clusters to a CSV.

Cross-platform: uses only pathlib/os, so it runs on Windows and Linux alike.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import sys
from dataclasses import dataclass, field

# Third-party libraries are imported lazily inside functions so that the
# dependency check can run and give guidance even when they are missing.

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".heic", ".heif", ".tif", ".tiff",
}

REQUIRED_LIBS = {
    # import name -> pip package name
    "PIL": "Pillow",
    "imagehash": "imagehash",
}

# Optional but strongly recommended: decode Apple HEIC/HEIF photos.
OPTIONAL_LIBS = {
    "pillow_heif": "pillow-heif",
}

_HEIF_REGISTERED = None


def maybe_register_heif() -> bool:
    """Register the HEIF/HEIC opener with Pillow if the plugin is present."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED is None:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            _HEIF_REGISTERED = True
        except Exception:
            _HEIF_REGISTERED = False
    return _HEIF_REGISTERED


# --------------------------------------------------------------------------- #
# Step 1: OS detection and dependency checking
# --------------------------------------------------------------------------- #
def detect_os() -> str:
    system = platform.system()
    if system == "Windows":
        return "Windows"
    if system == "Linux":
        return "Linux"
    if system == "Darwin":
        return "macOS"
    return system or "Unknown"


def pip_install_command(missing_pkgs: list[str]) -> str:
    """Return the platform-appropriate pip command to install missing packages."""
    exe = "python" if detect_os() == "Windows" else "python3"
    return f"{exe} -m pip install --user {' '.join(missing_pkgs)}"


def check_dependencies(verbose: bool = True) -> list[str]:
    """Return the list of missing pip package names (empty if all present)."""
    import importlib.util

    missing: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for import_name, pip_name in REQUIRED_LIBS.items():
        present = importlib.util.find_spec(import_name) is not None
        rows.append((import_name, pip_name, "OK" if present else "MISSING"))
        if not present:
            missing.append(pip_name)

    if verbose:
        print(f"Operating system : {detect_os()} ({platform.platform()})")
        print(f"Python           : {platform.python_version()} ({sys.executable})")
        print("Library check (required):")
        for import_name, pip_name, status in rows:
            print(f"  {import_name:<12} ({pip_name:<12}) : {status}")
        print("Library check (optional):")
        for import_name, pip_name in OPTIONAL_LIBS.items():
            present = importlib.util.find_spec(import_name) is not None
            note = "" if present else "  <- needed to read HEIC/HEIF (.heic) photos"
            status = "OK" if present else "MISSING"
            print(f"  {import_name:<12} ({pip_name:<12}) : {status}{note}")

    return missing


def ensure_dependencies_or_exit() -> None:
    missing = check_dependencies(verbose=True)
    if missing:
        print()
        print("Some required libraries are missing. Please install them by running:")
        print()
        print(f"    {pip_install_command(missing)}")
        print()
        print("Then re-run this script.")
        sys.exit(1)
    print("All required libraries are installed.\n")


# --------------------------------------------------------------------------- #
# Step 3: JSON sidecar matching (robust to Takeout naming quirks)
# --------------------------------------------------------------------------- #
def _read_url_from_json(json_path: str) -> str:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ""
    url = data.get("url", "")
    return url if isinstance(url, str) else ""


def find_sidecar(image_path: str) -> str:
    """
    Locate the Google Photos JSON sidecar for an image.

    Takeout sidecar naming is inconsistent. We try, in order:
      1. <image>.json
      2. <image>.supplemental-metadata.json  (and any .supplemental-*.json)
      3. the '-edited' variant mapped back to the original's sidecar
      4. duplicate-counter forms, e.g. name(1).jpg -> name.jpg(1).json
      5. a truncated-prefix match within the same directory
    Returns the sidecar path, or "" if none found.
    """
    directory = os.path.dirname(image_path)
    base = os.path.basename(image_path)

    # 1. exact
    cand = image_path + ".json"
    if os.path.isfile(cand):
        return cand

    # 2. supplemental-metadata variants
    for cand in glob.glob(glob.escape(image_path) + ".supplemental*.json"):
        if os.path.isfile(cand):
            return cand

    # 3. '-edited' variant: strip the '-edited' token and retry the original
    stem, ext = os.path.splitext(base)
    for token in ("-edited", "-EDITED", "-modifie", "-bearbeitet"):
        if stem.endswith(token):
            original = os.path.join(directory, stem[: -len(token)] + ext)
            sc = find_sidecar(original)
            if sc:
                return sc

    # 4. duplicate counter: name(1).jpg -> name.jpg(1).json
    if stem.endswith(")") and "(" in stem:
        head, counter = stem.rsplit("(", 1)
        cand = os.path.join(directory, f"{head}{ext}({counter}.json")
        if os.path.isfile(cand):
            return cand

    # 5. truncated-prefix fallback: Takeout may truncate long JSON names.
    prefix = base[:40]
    for cand in sorted(glob.glob(os.path.join(glob.escape(directory), glob.escape(prefix) + "*.json"))):
        if os.path.isfile(cand):
            return cand

    return ""


# --------------------------------------------------------------------------- #
# Step 4: perceptual hashing
# --------------------------------------------------------------------------- #
def compute_phash_int(image_path: str):
    """Return (phash_int, "") on success, or (None, reason) on failure."""
    from PIL import Image
    import imagehash

    try:
        with Image.open(image_path) as img:
            img.load()
            h = imagehash.phash(img)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return int(str(h), 16), ""


# --------------------------------------------------------------------------- #
# Step 5: BK-tree + union-find clustering
# --------------------------------------------------------------------------- #
def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    """Minimal BK-tree over integers using Hamming distance."""

    def __init__(self) -> None:
        self._root = None
        # node value -> {distance: child value}
        self._children: dict[int, dict[int, int]] = {}
        # value -> list of item indices sharing that exact hash
        self._items: dict[int, list[int]] = {}

    def add(self, value: int, index: int) -> None:
        if value in self._items:
            self._items[value].append(index)
            return
        self._items[value] = [index]
        if self._root is None:
            self._root = value
            self._children[value] = {}
            return
        node = self._root
        while True:
            d = hamming(value, node)
            kids = self._children[node]
            if d in kids:
                node = kids[d]
            else:
                kids[d] = value
                self._children[value] = {}
                return

    def query(self, value: int, threshold: int) -> list[int]:
        """Return indices of all items within `threshold` Hamming distance."""
        if self._root is None:
            return []
        result: list[int] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming(value, node)
            if d <= threshold:
                result.extend(self._items[node])
            lo, hi = d - threshold, d + threshold
            for edge, child in self._children[node].items():
                if lo <= edge <= hi:
                    stack.append(child)
        return result


class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    path: str
    filename: str
    phash: int
    url: str = ""


@dataclass
class Cluster:
    cluster_id: int
    representative_phash: int
    records: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Step 2 + 3 + 4: scanning and building records
# --------------------------------------------------------------------------- #
def scan_images(root: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def build_records(image_paths: list[str]) -> list:
    from collections import Counter

    maybe_register_heif()
    records: list = []
    total = len(image_paths)
    skipped = 0
    reason_by_type: Counter = Counter()
    skipped_by_ext: Counter = Counter()
    for i, path in enumerate(image_paths, 1):
        phash, reason = compute_phash_int(path)
        if phash is None:
            skipped += 1
            reason_by_type[reason.split(":", 1)[0]] += 1
            skipped_by_ext[os.path.splitext(path)[1].lower()] += 1
            if skipped <= 10:  # show a few concrete examples, then aggregate
                print(f"  [skip] {reason} :: {path}")
            continue
        sidecar = find_sidecar(path)
        url = _read_url_from_json(sidecar) if sidecar else ""
        records.append(ImageRecord(path=path, filename=os.path.basename(path),
                                    phash=phash, url=url))
        if i % 200 == 0 or i == total:
            print(f"  hashed {i}/{total}")
    if skipped:
        _report_skips(skipped, reason_by_type, skipped_by_ext)
    return records


def _report_skips(skipped, reason_by_type, skipped_by_ext) -> None:
    print(f"\n  Skipped {skipped} file(s) that could not be read.")
    print("  By error type:")
    for reason, count in reason_by_type.most_common():
        print(f"    {count:>6}  {reason}")
    print("  By file extension:")
    for ext, count in skipped_by_ext.most_common():
        print(f"    {count:>6}  {ext or '(no extension)'}")
    heic = skipped_by_ext.get(".heic", 0) + skipped_by_ext.get(".heif", 0)
    if heic:
        print(f"\n  {heic} of the skipped files are HEIC/HEIF (Apple) images.")
        if not maybe_register_heif():
            print("  Pillow cannot decode HEIC/HEIF without a plugin. Install it and")
            print("  re-run to include these photos:")
            print(f"      {pip_install_command(['pillow-heif'])}")
        else:
            print("  The HEIC plugin is installed but these still failed -- the files")
            print("  may be corrupt or truncated in the export.")


def cluster_records(records: list, threshold: int) -> list:
    tree = BKTree()
    for idx, rec in enumerate(records):
        tree.add(rec.phash, idx)

    uf = UnionFind(len(records))
    for idx, rec in enumerate(records):
        for neighbor in tree.query(rec.phash, threshold):
            if neighbor != idx:
                uf.union(idx, neighbor)

    groups: dict[int, list[int]] = {}
    for idx in range(len(records)):
        groups.setdefault(uf.find(idx), []).append(idx)

    clusters: list = []
    for cid, (_root, members) in enumerate(sorted(groups.items()), 1):
        recs = [records[m] for m in members]
        clusters.append(Cluster(cluster_id=cid,
                                 representative_phash=recs[0].phash,
                                 records=recs))
    return clusters


# --------------------------------------------------------------------------- #
# Step 6: CSV output
# --------------------------------------------------------------------------- #
def write_csv(clusters: list, output_path: str, only_duplicates: bool) -> int:
    rows_written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["cluster_id", "representative_phash", "image_count", "filenames", "urls"]
        )
        for cluster in clusters:
            if only_duplicates and len(cluster.records) < 2:
                continue
            filenames = "|".join(r.filename for r in cluster.records)
            urls = "|".join(r.url for r in cluster.records)
            writer.writerow([
                cluster.cluster_id,
                f"{cluster.representative_phash:016x}",
                len(cluster.records),
                filenames,
                urls,
            ])
            rows_written += 1
    return rows_written


def safe_write_csv(clusters: list, output_path: str, only_duplicates: bool):
    """Write the CSV, recovering interactively from permission/path errors.

    Returns (rows_written, final_output_path). Never loses the computed
    clusters: on failure the user can fix permissions and retry, or pick a
    new path, without re-running the whole pipeline.
    """
    while True:
        # A folder was given instead of a file: write a CSV inside it.
        if os.path.isdir(output_path):
            new_path = os.path.join(output_path, "duplicates.csv")
            print(f"Output path is a folder; writing to: {new_path}")
            output_path = new_path
        try:
            rows = write_csv(clusters, output_path, only_duplicates)
            return rows, output_path
        except (PermissionError, OSError) as exc:
            print(f"\nCould not write the CSV to: {output_path}")
            print(f"  Reason: {exc}")
            print("This is almost always one of:")
            print("  - The folder is read-only / your user lacks write permission.")
            print("    On Windows: right-click the folder > Properties > Security >")
            print("    Edit, select your user, tick 'Modify' and 'Write', then Apply.")
            print("  - The CSV is currently open in another program (e.g. Excel). Close it.")
            print("  - The path points to a protected location; pick another folder.")
            print()
            choice = input(
                "Fix it, then press Enter to retry -- or type a new output path "
                "(q to abort): "
            ).strip().strip('"').strip("'")
            if choice.lower() == "q":
                print("Aborted: the CSV was not written (computed data was not saved).")
                return 0, output_path
            if choice:
                output_path = choice


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(root: str, output_path: str, threshold: int,
                 only_duplicates: bool) -> None:
    print(f"\nScanning for images under: {root}")
    image_paths = scan_images(root)
    print(f"Found {len(image_paths)} image file(s).")
    if not image_paths:
        print("Nothing to do.")
        return None

    print("\nComputing perceptual hashes and matching metadata...")
    records = build_records(image_paths)
    print(f"Successfully hashed {len(records)} image(s).")

    print(f"\nClustering near-duplicates (Hamming threshold = {threshold})...")
    clusters = cluster_records(records, threshold)
    dup_groups = sum(1 for c in clusters if len(c.records) >= 2)

    rows, output_path = safe_write_csv(clusters, output_path, only_duplicates)

    print("\nSummary")
    print(f"  images processed : {len(records)}")
    print(f"  total clusters   : {len(clusters)}")
    print(f"  duplicate groups : {dup_groups}")
    print(f"  rows written     : {rows}")
    print(f"  output CSV       : {os.path.abspath(output_path)}")
    return output_path


# --------------------------------------------------------------------------- #
# Self-test / dry run
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Generate a synthetic Takeout tree, run the pipeline, and verify output."""
    import tempfile
    import io
    import numpy as np
    from PIL import Image

    print("Running self-test (dry run) on synthetic data...\n")
    tmp = tempfile.mkdtemp(prefix="gpic_selftest_")
    photos_dir = os.path.join(tmp, "Takeout", "Google Photos", "2024")
    os.makedirs(photos_dir, exist_ok=True)

    # pHash is based on luminance *structure*, so test images must contain real
    # structure (gradients/blocks), not flat colors, to be meaningful.
    def pattern_a(shift=0, noise=0, seed=0):
        x = np.linspace(0, 255, 256)
        arr = np.tile(x, (256, 1))          # horizontal gradient
        arr[30:120, 40:150] = 255           # bright block, top-left
        arr = arr + shift
        if noise:
            rng = np.random.default_rng(seed)
            arr = arr + rng.normal(0, noise, arr.shape)
        arr = np.clip(arr, 0, 255).astype("uint8")
        return Image.fromarray(arr, mode="L").convert("RGB")

    def pattern_b():
        y = np.linspace(0, 255, 256)
        arr = np.tile(y.reshape(-1, 1), (1, 256))   # vertical gradient
        arr[150:230, 120:220] = 0                    # dark block, bottom-right
        arr = np.clip(arr, 0, 255).astype("uint8")
        return Image.fromarray(arr, mode="L").convert("RGB")

    def save(img, name):
        img.save(os.path.join(photos_dir, name))

    def make_sidecar(image_name, url, style="plain"):
        if style == "plain":
            jp = os.path.join(photos_dir, image_name + ".json")
        else:
            jp = os.path.join(photos_dir, image_name + ".supplemental-metadata.json")
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump({"title": image_name, "url": url}, fh)

    # A, a noisy near-copy of A, and an '-edited' brightness-shifted A -> one cluster.
    save(pattern_a(), "scene.jpg")
    make_sidecar("scene.jpg", "https://photos.google.com/photo/RED1", "plain")

    save(pattern_a(noise=4, seed=1), "scene_copy.jpg")
    make_sidecar("scene_copy.jpg", "https://photos.google.com/photo/RED2", "supplemental")

    save(pattern_a(shift=10), "scene-edited.jpg")  # no own sidecar on purpose

    # A structurally different image -> its own cluster.
    save(pattern_b(), "other.jpg")
    make_sidecar("other.jpg", "https://photos.google.com/photo/BLUE1", "plain")

    out_csv = os.path.join(tmp, "duplicates.csv")
    run_pipeline(os.path.join(tmp, "Takeout"), out_csv, threshold=5, only_duplicates=False)

    print("\nCSV contents:")
    with open(out_csv, encoding="utf-8") as fh:
        content = fh.read()
    print(content)

    reader = list(csv.DictReader(io.StringIO(content)))
    red_row = next((r for r in reader if "RED1" in r["urls"]), None)
    blue_row = next((r for r in reader if "BLUE1" in r["urls"]), None)

    ok = True
    if red_row is None or int(red_row["image_count"]) < 3:
        print("FAIL: the three near-duplicate images did not cluster together.")
        ok = False
    else:
        print(f"PASS: near-duplicate cluster has {red_row['image_count']} images "
              f"(includes the -edited variant and both sidecar styles).")
    if red_row and "RED1" in red_row["urls"] and "RED2" in red_row["urls"]:
        print("PASS: both plain and supplemental sidecar URLs were recovered.")
    else:
        print("FAIL: sidecar URL recovery incomplete.")
        ok = False
    if blue_row is None or blue_row is red_row:
        print("FAIL: the distinct image was not separated.")
        ok = False
    else:
        print("PASS: the structurally different image is in its own cluster.")

    print(f"\nSelf-test {'SUCCEEDED' if ok else 'FAILED'}.")
    print(f"(Temp data left at: {tmp})")
    if not ok:
        sys.exit(1)
