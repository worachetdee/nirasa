"""Prepare deduped data for MLX training.

Reads all deduped JSONL files, shuffles, and splits into
train.jsonl and valid.jsonl in the format MLX expects: {"text": "..."}
"""

import json
import os
import random
from pathlib import Path

DEDUP_DIR = Path("data/dedup")
OUTPUT_DIR = Path("data/mlx")
VALID_RATIO = 0.01  # 1% for validation
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all texts from deduped files
    all_texts = []
    for f in sorted(DEDUP_DIR.glob("*.jsonl")):
        count = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if text and len(text.strip()) > 50:
                        all_texts.append(text.strip())
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  {f.name}: {count:,} docs")

    print(f"\nTotal: {len(all_texts):,} docs")

    # Shuffle
    random.seed(SEED)
    random.shuffle(all_texts)

    # Split
    n_valid = max(1, int(len(all_texts) * VALID_RATIO))
    valid_texts = all_texts[:n_valid]
    train_texts = all_texts[n_valid:]

    # Write
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for text in valid_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    train_mb = os.path.getsize(train_path) / (1024 * 1024)
    valid_mb = os.path.getsize(valid_path) / (1024 * 1024)

    print(f"\nOutput:")
    print(f"  train: {len(train_texts):,} docs ({train_mb:.1f} MB)")
    print(f"  valid: {len(valid_texts):,} docs ({valid_mb:.1f} MB)")
    print(f"  path:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
