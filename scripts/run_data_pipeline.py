"""Run the full data pipeline: raw text -> JSONL -> clean -> dedup -> filter."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Change to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from nirasa.data.clean import clean_file
from nirasa.data.dedup import dedup_file
from nirasa.data.filter import filter_file


def convert_raw_to_jsonl(input_path: str, output_path: str, min_chars: int = 50) -> int:
    """Convert raw text to JSONL format.

    Splits text on blank lines and creates one JSON record per document.

    Args:
        input_path: Path to raw text file.
        output_path: Path to output JSONL file.
        min_chars: Minimum character count per document.

    Returns:
        Number of documents written.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    current_doc: list[str] = []

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                if current_doc:
                    text = "\n".join(current_doc)
                    if len(text) >= min_chars:
                        record = {"text": text}
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                    current_doc = []
            else:
                current_doc.append(stripped)

        # Handle last document
        if current_doc:
            text = "\n".join(current_doc)
            if len(text) >= min_chars:
                record = {"text": text}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def file_size_str(path: str) -> str:
    """Get human-readable file size."""
    if not Path(path).exists():
        return "N/A"
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def main():
    """Run the complete data pipeline."""
    raw_file = "data/raw/th_wiki.txt"
    jsonl_file = "data/intermediate/th_wiki.jsonl"
    clean_jsonl = "data/intermediate/th_wiki_clean.jsonl"
    dedup_jsonl = "data/intermediate/th_wiki_dedup.jsonl"
    filter_jsonl = "data/processed/th_wiki_filtered.jsonl"

    print("=" * 60)
    print("Nirasa Data Pipeline")
    print("=" * 60)

    # Step 1: Convert raw text to JSONL
    print("\n[1/4] Converting raw text to JSONL...")
    if not Path(raw_file).exists():
        print(f"  ERROR: Raw file not found: {raw_file}")
        print(f"  Run: python scripts/download_wiki_th.py")
        return

    count = convert_raw_to_jsonl(raw_file, jsonl_file, min_chars=50)
    print(f"  Documents: {count:,}")

    # Step 2: Clean
    print("\n[2/4] Cleaning text...")
    clean_stats = clean_file(
        jsonl_file, clean_jsonl,
        min_thai_ratio=0.1,
        num_workers=4,
    )

    # Step 3: Dedup
    print("\n[3/4] Deduplicating...")
    dedup_stats = dedup_file(
        clean_jsonl, dedup_jsonl,
        threshold=0.8,
        num_perm=128,
    )

    # Step 4: Filter
    print("\n[4/4] Quality filtering...")
    filter_stats = filter_file(
        dedup_jsonl, filter_jsonl,
        min_len=100,
        min_thai_ratio=0.1,
        max_repetition_ratio=0.3,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"{'Stage':<25s} {'File':<40s} {'Size':>10s}")
    print("-" * 75)
    print(f"{'Raw text':<25s} {raw_file:<40s} {file_size_str(raw_file):>10s}")
    print(f"{'JSONL':<25s} {jsonl_file:<40s} {file_size_str(jsonl_file):>10s}")
    print(f"{'Cleaned':<25s} {clean_jsonl:<40s} {file_size_str(clean_jsonl):>10s}")
    print(f"{'Deduplicated':<25s} {dedup_jsonl:<40s} {file_size_str(dedup_jsonl):>10s}")
    print(f"{'Filtered':<25s} {filter_jsonl:<40s} {file_size_str(filter_jsonl):>10s}")
    print()


if __name__ == "__main__":
    main()
