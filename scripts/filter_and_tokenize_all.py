"""Filter all deduped datasets, merge, and tokenize into training binary.

Usage:
    python scripts/filter_and_tokenize_all.py
    python scripts/filter_and_tokenize_all.py --workers 8
"""

import os
import time
from pathlib import Path

from nirasa.data.filter import filter_file
from nirasa.data.prepare import tokenize_and_pack

DEDUP_DIR = Path("data/dedup")
FILTER_DIR = Path("data/filtered")
MERGED_FILE = Path("data/processed/th_all_filtered.jsonl")
OUTPUT_PREFIX = "data/processed/th_all_qwen"


def main(workers: int = 4):
    """Run filter + merge + tokenize pipeline."""
    files = sorted(DEDUP_DIR.glob("*.jsonl"))
    if not files:
        print("No JSONL files found in data/dedup/")
        return

    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Filter each dataset
    print("=" * 60)
    print("STEP 1: Quality Filtering")
    print("=" * 60)

    for f in files:
        name = f.stem
        filter_out = FILTER_DIR / f.name

        if filter_out.exists() and filter_out.stat().st_size > 0:
            size_mb = filter_out.stat().st_size / (1024 * 1024)
            print(f"\n  SKIP {name} — already filtered ({size_mb:.1f} MB)")
            continue

        print(f"\n  Filtering {name}...")
        t0 = time.time()
        try:
            filter_file(
                str(f), str(filter_out),
                min_len=100,
                min_thai_ratio=0.1,
                max_repetition_ratio=0.3,
            )
        except Exception as e:
            print(f"  FILTER FAILED: {e}")
            continue
        print(f"  Done in {time.time() - t0:.0f}s")

    # Step 2: Merge all filtered files
    print(f"\n{'=' * 60}")
    print("STEP 2: Merging filtered files")
    print("=" * 60)

    filtered_files = sorted(FILTER_DIR.glob("*.jsonl"))
    total_docs = 0

    with open(MERGED_FILE, "w", encoding="utf-8") as fout:
        for f in filtered_files:
            count = 0
            with open(f, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    count += 1
            total_docs += count
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<35s} {size_mb:>8.1f} MB  {count:>10,} docs")

    merged_size = MERGED_FILE.stat().st_size / (1024 * 1024)
    print(f"  {'─' * 55}")
    print(f"  {'MERGED':<35s} {merged_size:>8.1f} MB  {total_docs:>10,} docs")

    # Step 3: Tokenize
    print(f"\n{'=' * 60}")
    print("STEP 3: Tokenizing with Qwen tokenizer")
    print("=" * 60)

    t0 = time.time()
    stats = tokenize_and_pack(
        input_path=str(MERGED_FILE),
        output_prefix=OUTPUT_PREFIX,
        model_name="Qwen/Qwen2.5-7B",
    )
    print(f"  Tokenization done in {time.time() - t0:.0f}s")

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Filtered docs:  {total_docs:,}")
    print(f"  Total tokens:   {stats['total_tokens']:,}")
    print(f"  Binary file:    {OUTPUT_PREFIX}.bin ({stats['bin_size_mb']:.1f} MB)")
    print(f"  Index file:     {OUTPUT_PREFIX}.idx ({stats['idx_size_kb']:.1f} KB)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
