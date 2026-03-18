"""MinHash-based near-deduplication for Thai text."""

from __future__ import annotations

import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def char_ngrams(text: str, n: int = 5) -> list[str]:
    """Generate character n-grams from text.

    Uses character-level shingling instead of word-level since Thai
    does not use spaces between words.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        List of character n-grams.
    """
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def create_minhash(text: str, num_perm: int = 128, ngram_size: int = 5) -> MinHash:
    """Create a MinHash for a text document.

    Args:
        text: Input text.
        num_perm: Number of permutations for MinHash.
        ngram_size: Character n-gram size.

    Returns:
        MinHash object.
    """
    mh = MinHash(num_perm=num_perm)
    for gram in char_ngrams(text, ngram_size):
        mh.update(gram.encode("utf-8"))
    return mh


def dedup_file(
    input_path: str,
    output_path: str,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
) -> dict:
    """Deduplicate a JSONL file using MinHash LSH.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        threshold: Jaccard similarity threshold for deduplication.
        num_perm: Number of permutations for MinHash.
        ngram_size: Character n-gram size.

    Returns:
        Dict with deduplication statistics.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # First pass: build LSH index and identify duplicates
    keep_indices: set[int] = set()
    lines: list[str] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())

    total = len(lines)
    duplicates = 0

    for idx, line in enumerate(tqdm(lines, desc="Dedup (indexing)")):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = record.get("text", "")
        if not text or len(text) < 20:
            continue

        mh = create_minhash(text, num_perm=num_perm, ngram_size=ngram_size)

        # Check for near-duplicates
        result = lsh.query(mh)
        if result:
            duplicates += 1
            continue

        # Not a duplicate — keep it
        try:
            lsh.insert(f"doc_{idx}", mh)
            keep_indices.add(idx)
        except ValueError:
            # Key already exists (shouldn't happen, but be safe)
            keep_indices.add(idx)

    # Second pass: write kept documents
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in sorted(keep_indices):
            f.write(lines[idx] + "\n")

    stats = {
        "input_docs": total,
        "output_docs": len(keep_indices),
        "duplicates_removed": duplicates,
        "dedup_ratio": duplicates / total if total > 0 else 0.0,
    }

    print(f"Dedup done: {stats['input_docs']} -> {stats['output_docs']} "
          f"(removed {stats['duplicates_removed']}, "
          f"ratio: {stats['dedup_ratio']:.2%})")
    return stats


def main(
    input_path: str,
    output_path: str,
    threshold: float = 0.8,
    num_perm: int = 128,
) -> None:
    """CLI entry point."""
    dedup_file(input_path, output_path, threshold, num_perm)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
