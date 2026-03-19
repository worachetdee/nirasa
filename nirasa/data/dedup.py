"""MinHash-based near-deduplication for Thai text."""

from __future__ import annotations

import json
from multiprocessing import Pool
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


def _compute_minhash_worker(args: tuple) -> tuple[int, str, MinHash | None]:
    """Worker function for parallel MinHash computation."""
    idx, line, num_perm, ngram_size = args

    if not line:
        return idx, line, None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return idx, line, None

    text = record.get("text", "")
    if not text or len(text) < 20:
        return idx, line, None

    mh = create_minhash(text, num_perm=num_perm, ngram_size=ngram_size)
    return idx, line, mh


def dedup_file(
    input_path: str,
    output_path: str,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
    num_workers: int = 4,
) -> dict:
    """Deduplicate a JSONL file using MinHash LSH.

    Two-phase approach for speed:
      1. Compute all MinHashes in parallel (CPU-bound, benefits from multiprocessing)
      2. Build LSH index sequentially (index must be built incrementally)

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        threshold: Jaccard similarity threshold for deduplication.
        num_perm: Number of permutations for MinHash.
        ngram_size: Character n-gram size.
        num_workers: Number of parallel workers for MinHash computation.

    Returns:
        Dict with deduplication statistics.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all lines
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    total = len(lines)

    # Phase 1: Compute MinHashes in parallel
    print(f"  Phase 1: Computing MinHashes ({num_workers} workers)...")
    args_list = [(idx, line, num_perm, ngram_size) for idx, line in enumerate(lines)]

    minhashes: dict[int, MinHash] = {}

    if num_workers <= 1:
        for args in tqdm(args_list, desc="Dedup (hashing)"):
            idx, line, mh = _compute_minhash_worker(args)
            if mh is not None:
                minhashes[idx] = mh
    else:
        with Pool(num_workers) as pool:
            for idx, line, mh in tqdm(
                pool.imap(_compute_minhash_worker, args_list, chunksize=500),
                total=len(args_list),
                desc="Dedup (hashing)",
            ):
                if mh is not None:
                    minhashes[idx] = mh

    print(f"  Phase 1 done: {len(minhashes)} hashes computed")

    # Phase 2: Build LSH index sequentially
    print(f"  Phase 2: LSH dedup...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices: set[int] = set()
    duplicates = 0

    for idx in tqdm(sorted(minhashes.keys()), desc="Dedup (indexing)"):
        mh = minhashes[idx]

        result = lsh.query(mh)
        if result:
            duplicates += 1
            continue

        try:
            lsh.insert(f"doc_{idx}", mh)
            keep_indices.add(idx)
        except ValueError:
            keep_indices.add(idx)

    # Write kept documents
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
    num_workers: int = 4,
) -> None:
    """CLI entry point."""
    dedup_file(input_path, output_path, threshold, num_perm, num_workers=num_workers)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
