#!/usr/bin/env python3
"""
gpic.py - Single entry point for the Google Photos duplicate toolkit.

All user interaction lives here. The two workhorses are used as libraries:
  * gpic_dedup  - scans a Google Takeout photo folder and produces a
                  duplicates CSV (perceptual-hash clustering).
  * gpic_review - triages an existing duplicates CSV into confidence-ranked
                  clusters plus a visual HTML gallery.

Modes (choose exactly one):
  --build                     build a duplicates CSV from a photo folder.
  --review-csv-path <csv>     review an existing duplicates CSV.

If no mode is given, an interactive menu asks what to do and prompts for the
parameters. Ctrl+C exits cleanly without a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys

import gpic_dedup as dedup
import gpic_review as review

DEF_OUTPUT = "duplicates.csv"
DEF_THRESHOLD = 5
DEF_REVIEW_OUT = "review_out"
DEF_EMBED = "LOW,MEDIUM"
DEF_PER_PAGE = 400
DEF_THUMB = 220


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #
def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    resp = input(f"{text}{suffix}: ").strip().strip('"').strip("'")
    return resp or (default if default is not None else "")


def _prompt_dir(text: str) -> str:
    while True:
        p = _prompt(text)
        if p and os.path.isdir(p):
            return p
        print(f"  Not a valid directory: {p!r}")


def _prompt_file(text: str) -> str:
    while True:
        p = _prompt(text)
        if p and os.path.isfile(p):
            return p
        print(f"  Not a valid file: {p!r}")


def _embed_levels(text: str) -> set:
    if text.strip().upper() == "ALL":
        return {"EXACT", "HIGH", "MEDIUM", "LOW"}
    return {x.strip().upper() for x in text.split(",") if x.strip()}


# --------------------------------------------------------------------------- #
# Mode runners (thin wrappers over the libraries)
# --------------------------------------------------------------------------- #
def do_build(root: str, output: str, threshold: int, only_duplicates: bool):
    """Run the build pipeline; returns the actual CSV path written (or None)."""
    dedup.ensure_dependencies_or_exit()
    return dedup.run_pipeline(root, output, threshold, only_duplicates)


def do_review(csv_path: str, root: str, out_dir: str, thumb_size: int,
              embed: str, per_page: int, thresholds: dict) -> None:
    dedup.ensure_dependencies_or_exit()
    review.run(csv_path, root, out_dir, thumb_size, _embed_levels(embed),
               per_page, thresholds)


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #
def _gather_build_params():
    root = _prompt_dir("Path to the Takeout root folder")
    output = _prompt("Output CSV path", DEF_OUTPUT)
    threshold = int(_prompt("Hamming distance threshold", str(DEF_THRESHOLD)))
    only = _prompt("Output only duplicate groups (y/n)", "n").lower().startswith("y")
    return root, output, threshold, only


def _gather_review_params(csv_path=None, root=None):
    if csv_path is None:
        csv_path = _prompt_file("Path to duplicates.csv")
    if root is None:
        root = _prompt_dir("Path to the Takeout root folder (to locate images)")
    out_dir = _prompt("Output directory", DEF_REVIEW_OUT)
    embed = _prompt("Embed thumbnails for levels (comma list or ALL)", DEF_EMBED)
    per_page = int(_prompt("Clusters per gallery page", str(DEF_PER_PAGE)))
    thresholds = {
        "high_ssim": float(_prompt("HIGH: SSIM >=", str(review.DEFAULT_HIGH_SSIM))),
        "high_hamming": int(_prompt("HIGH: Hamming <=", str(review.DEFAULT_HIGH_HAMMING))),
        "medium_ssim": float(_prompt("MEDIUM: SSIM >=", str(review.DEFAULT_MEDIUM_SSIM))),
        "medium_hamming": int(_prompt("MEDIUM: Hamming <=", str(review.DEFAULT_MEDIUM_HAMMING))),
    }
    return csv_path, root, out_dir, embed, per_page, thresholds


def interactive() -> None:
    print("What do you want to do?")
    print("  1) Build a duplicates CSV from a photo folder (Google Takeout)")
    print("  2) Review an existing duplicates CSV")
    print("  3) Build a duplicates CSV and then review it")
    choice = _prompt("Enter 1, 2 or 3")
    if choice not in ("1", "2", "3"):
        print(f"Invalid choice: {choice!r}. Please enter 1, 2 or 3.")
        sys.exit(2)

    if choice == "1":
        root, output, threshold, only = _gather_build_params()
        do_build(root, output, threshold, only)
        return

    if choice == "2":
        csv_path, root, out_dir, embed, per_page, thresholds = _gather_review_params()
        do_review(csv_path, root, out_dir, DEF_THUMB, embed, per_page, thresholds)
        return

    # choice == "3": build, then review the freshly built CSV.
    root, output, threshold, only = _gather_build_params()
    built_csv = do_build(root, output, threshold, only)
    if not built_csv or not os.path.isfile(built_csv):
        print("\nNo CSV was produced (no images found), so there is nothing to review.")
        return
    print("\nBuild complete. Now configuring the review step...\n")
    csv_path, root, out_dir, embed, per_page, thresholds = _gather_review_params(
        csv_path=built_csv, root=root)
    do_review(csv_path, root, out_dir, DEF_THUMB, embed, per_page, thresholds)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpic.py",
        description="Google Photos duplicate toolkit: build a duplicates CSV, or review one.")

    # Mode triggers (choose one).
    p.add_argument("--build", action="store_true",
                   help="Build a duplicates CSV from a photo folder.")
    p.add_argument("--review-csv-path", dest="review_csv_path",
                   help="Review this existing duplicates CSV.")

    # Shared.
    p.add_argument("--root", help="Takeout root folder containing the photos.")

    # Build parameters.
    p.add_argument("--output", help=f"Build: output CSV path (default {DEF_OUTPUT}).")
    p.add_argument("--threshold", type=int,
                   help=f"Build: Hamming distance threshold (default {DEF_THRESHOLD}).")
    p.add_argument("--only-duplicates", action="store_true",
                   help="Build: only write clusters that contain 2+ images.")

    # Review parameters.
    p.add_argument("--output-dir", help=f"Review: output directory (default {DEF_REVIEW_OUT}).")
    p.add_argument("--thumb-size", type=int, default=DEF_THUMB,
                   help=f"Review: max thumbnail edge in px (default {DEF_THUMB}).")
    p.add_argument("--embed", default=DEF_EMBED,
                   help=f"Review: confidence levels to embed thumbnails for "
                        f"(default {DEF_EMBED}; use ALL for every level).")
    p.add_argument("--per-page", type=int, default=DEF_PER_PAGE,
                   help=f"Review: clusters per gallery page (default {DEF_PER_PAGE}).")
    p.add_argument("--high-ssim", type=float, default=review.DEFAULT_HIGH_SSIM)
    p.add_argument("--high-hamming", type=int, default=review.DEFAULT_HIGH_HAMMING)
    p.add_argument("--medium-ssim", type=float, default=review.DEFAULT_MEDIUM_SSIM)
    p.add_argument("--medium-hamming", type=int, default=review.DEFAULT_MEDIUM_HAMMING)

    # Utilities.
    p.add_argument("--check", action="store_true",
                   help="Run the OS/dependency check and exit.")
    p.add_argument("--selftest-build", action="store_true",
                   help="Dry-run the build pipeline on synthetic data.")
    p.add_argument("--selftest-review", action="store_true",
                   help="Dry-run the review pipeline on synthetic data.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.selftest_build:
        dedup._selftest()
        return
    if args.selftest_review:
        review._selftest()
        return
    if args.check:
        missing = dedup.check_dependencies(verbose=True)
        if missing:
            print("\nInstall the missing libraries with:")
            print(f"    {dedup.pip_install_command(missing)}")
            sys.exit(1)
        print("\nAll required libraries are installed.")
        return

    review_requested = args.review_csv_path is not None
    build_requested = args.build or (args.root is not None and not review_requested)

    if build_requested and review_requested:
        print("Choose one mode: either --build or --review-csv-path, not both.")
        sys.exit(2)

    if review_requested:
        csv_path = args.review_csv_path
        if not os.path.isfile(csv_path):
            print(f"CSV not found: {csv_path!r}")
            csv_path = _prompt_file("Path to duplicates.csv")
        root = args.root or _prompt_dir("Path to the Takeout root folder (to locate images)")
        out_dir = args.output_dir or DEF_REVIEW_OUT
        thresholds = {
            "high_ssim": args.high_ssim, "high_hamming": args.high_hamming,
            "medium_ssim": args.medium_ssim, "medium_hamming": args.medium_hamming,
        }
        do_review(csv_path, root, out_dir, args.thumb_size, args.embed,
                  args.per_page, thresholds)
    elif build_requested:
        root = args.root or _prompt_dir("Path to the Takeout root folder")
        output = args.output or DEF_OUTPUT
        threshold = args.threshold if args.threshold is not None else DEF_THRESHOLD
        do_build(root, output, threshold, args.only_duplicates)
    else:
        interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)
