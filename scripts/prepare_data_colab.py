"""Prepare tokenized data for training (Colab version).

Tokenizes cleaned JSONL data with the Qwen2.5-7B tokenizer
and packs into uint32 numpy memmap files.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

os.chdir("/content/nirasa")

MODEL_NAME = "Qwen/Qwen2.5-7B"
INPUT_FILE = "data/processed/th_wiki_filtered.jsonl"
OUTPUT_PREFIX = "data/processed/th_wiki_qwen"


def main():
    """Tokenize and pack data into memmap."""
    bin_path = Path(f"{OUTPUT_PREFIX}.bin")
    idx_path = Path(f"{OUTPUT_PREFIX}.idx")
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    # Tokenize all documents
    print(f"Tokenizing: {INPUT_FILE}")
    all_tokens: list[int] = []
    doc_boundaries: list[int] = [0]
    num_docs = 0

    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "")
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(token_ids)

            if eos_id is not None:
                all_tokens.append(eos_id)

            doc_boundaries.append(len(all_tokens))
            num_docs += 1

    total_tokens = len(all_tokens)

    # Write memmap
    print(f"Writing binary: {bin_path}")
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    mm = np.memmap(str(bin_path), dtype=np.uint32, mode="w+", shape=(total_tokens,))
    mm[:] = tokens_array
    mm.flush()
    del mm

    # Write index
    print(f"Writing index: {idx_path}")
    with open(idx_path, "wb") as f:
        f.write(struct.pack("<III", num_docs, total_tokens, 4))
        for boundary in doc_boundaries:
            f.write(struct.pack("<Q", boundary))

    # Stats
    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Documents:  {num_docs:,}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Bin size:   {bin_size_mb:.1f} MB")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Max token:  {max(all_tokens):,}")


if __name__ == "__main__":
    main()
