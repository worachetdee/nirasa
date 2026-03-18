"""Download Thai Wikipedia and save as plain text."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    """Download Thai Wikipedia articles and save to data/raw/th_wiki.txt."""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "th_wiki.txt"

    print("Downloading Thai Wikipedia (wikimedia/wikipedia 20231101.th)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.th",
        split="train",
        trust_remote_code=True,
    )

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for article in tqdm(ds, desc="Writing articles"):
            text = article.get("text", "").strip()
            if text:
                f.write(text + "\n\n")
                count += 1

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Articles: {count:,}")
    print(f"  File:     {output_file}")
    print(f"  Size:     {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
