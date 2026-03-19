"""Clean and deduplicate all downloaded Thai datasets.

Usage:
    # Process all files
    python scripts/clean_and_dedup_all.py

    # Process only specific files (for splitting work across machines)
    python scripts/clean_and_dedup_all.py --only mangosteen_curated,mangosteen_web,mc4

    # Adjust parallelism
    python scripts/clean_and_dedup_all.py --workers 8
"""

import os
import sys
import time
from pathlib import Path

from nirasa.data.clean import clean_file
from nirasa.data.dedup import dedup_file

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
DEDUP_DIR = Path("data/dedup")


def main(only: str = "", workers: int = 4):
    """Run clean and dedup pipeline.

    Args:
        only: Comma-separated list of dataset name substrings to process.
              E.g. "mangosteen_curated,mangosteen_web,mc4"
        workers: Number of parallel workers for cleaning and dedup.
    """
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        print("No JSONL files found in data/raw/")
        return

    # Filter files if --only is specified
    if only:
        filters = [f.strip() for f in only.split(",")]
        files = [f for f in files if any(filt in f.stem for filt in filters)]
        if not files:
            print(f"No files matched filter: {only}")
            return

    print(f"Found {len(files)} datasets to process (workers={workers})\n")

    for f in files:
        name = f.stem
        clean_out = CLEAN_DIR / f.name
        dedup_out = DEDUP_DIR / f.name

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Skip if dedup output already exists
        if dedup_out.exists() and dedup_out.stat().st_size > 0:
            size_mb = dedup_out.stat().st_size / (1024 * 1024)
            print(f"  SKIP — already processed ({size_mb:.1f} MB)")
            continue

        t0 = time.time()

        # Step 1: Clean (skip if clean file already exists)
        if clean_out.exists() and clean_out.stat().st_size > 0:
            size_mb = clean_out.stat().st_size / (1024 * 1024)
            print(f"  [1/2] Clean already done ({size_mb:.1f} MB)")
        else:
            print(f"  [1/2] Cleaning...")
            try:
                clean_file(str(f), str(clean_out), num_workers=workers)
            except Exception as e:
                print(f"  CLEAN FAILED: {e}")
                continue

        # Step 2: Dedup
        print(f"  [2/2] Deduplicating...")
        try:
            dedup_file(str(clean_out), str(dedup_out), num_workers=workers)
        except Exception as e:
            print(f"  DEDUP FAILED: {e}")
            continue

        elapsed = time.time() - t0
        dedup_size = dedup_out.stat().st_size / (1024 * 1024)
        print(f"  Done in {elapsed:.0f}s → {dedup_size:.1f} MB")

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    total_size = 0
    total_docs = 0
    for f in sorted(DEDUP_DIR.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f))
        total_size += size_mb
        total_docs += lines
        print(f"  {f.name:<35s} {size_mb:>8.1f} MB  {lines:>10,} docs")
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<35s} {total_size:>8.1f} MB  {total_docs:>10,} docs")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
