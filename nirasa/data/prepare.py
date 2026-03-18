"""Tokenize documents and pack into numpy memmap for training."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_and_pack(
    input_path: str,
    output_prefix: str,
    model_name: str = "Qwen/Qwen2.5-7B",
    max_seq_len: int = 512,
    eos_at_doc_boundary: bool = True,
) -> dict:
    """Tokenize JSONL documents and pack into uint32 memmap.

    Qwen2.5 has a vocabulary larger than 65535, so we use uint32
    to store token IDs.

    Produces two files:
        - {output_prefix}.bin: uint32 memmap of packed token IDs
        - {output_prefix}.idx: index file with document boundaries

    Args:
        input_path: Path to input JSONL file.
        output_prefix: Prefix for output .bin and .idx files.
        model_name: HuggingFace model name for tokenizer.
        max_seq_len: Maximum sequence length for chunking.
        eos_at_doc_boundary: Whether to insert EOS token between documents.

    Returns:
        Dict with tokenization statistics.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    # First pass: count total tokens
    print("First pass: counting tokens...")
    all_tokens: list[int] = []
    doc_boundaries: list[int] = [0]
    num_docs = 0

    with open(input_path, encoding="utf-8") as f:
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

            if eos_at_doc_boundary and eos_id is not None:
                all_tokens.append(eos_id)

            doc_boundaries.append(len(all_tokens))
            num_docs += 1

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total documents: {num_docs:,}")

    # Write binary file
    print(f"Writing binary: {bin_path}")
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))
    mm[:] = tokens_array
    mm.flush()
    del mm

    # Write index file
    print(f"Writing index: {idx_path}")
    with open(idx_path, "wb") as f:
        # Header: num_docs, total_tokens, dtype_size
        f.write(struct.pack("<III", num_docs, total_tokens, 4))
        # Document boundaries
        for boundary in doc_boundaries:
            f.write(struct.pack("<Q", boundary))

    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)
    idx_size_kb = idx_path.stat().st_size / 1024

    stats = {
        "num_docs": num_docs,
        "total_tokens": total_tokens,
        "bin_size_mb": bin_size_mb,
        "idx_size_kb": idx_size_kb,
    }

    print(f"Done!")
    print(f"  Documents: {num_docs:,}")
    print(f"  Tokens:    {total_tokens:,}")
    print(f"  Bin size:  {bin_size_mb:.1f} MB")
    print(f"  Idx size:  {idx_size_kb:.1f} KB")

    return stats


def main(
    input_path: str,
    output_prefix: str = "data/processed/th_wiki_qwen",
    model_name: str = "Qwen/Qwen2.5-7B",
    max_seq_len: int = 512,
) -> None:
    """CLI entry point."""
    tokenize_and_pack(
        input_path=input_path,
        output_prefix=output_prefix,
        model_name=model_name,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
