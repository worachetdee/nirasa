"""Quality filtering for Thai text documents."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from nirasa.data.clean import thai_ratio


def _repetition_ratio(text: str, n: int = 5) -> float:
    """Calculate the ratio of repeated n-grams in text.

    Args:
        text: Input text.
        n: N-gram size for repetition detection.

    Returns:
        Ratio of repeated n-gram tokens to total n-gram tokens.
    """
    if len(text) < n:
        return 0.0

    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated_count = sum(count for count in counts.values() if count > 1)
    return repeated_count / len(ngrams)


def _mean_line_length(text: str) -> float:
    """Calculate the mean line length of text.

    Args:
        text: Input text.

    Returns:
        Mean line length in characters.
    """
    lines = text.split("\n")
    lines = [line for line in lines if line.strip()]
    if not lines:
        return 0.0
    return sum(len(line) for line in lines) / len(lines)


def filter_document(
    text: str,
    min_len: int = 100,
    max_len: int = 500000,
    min_thai_ratio: float = 0.1,
    max_repetition_ratio: float = 0.3,
    min_mean_line_len: float = 10.0,
) -> bool:
    """Check if a document passes quality filters.

    Args:
        text: Document text.
        min_len: Minimum character length.
        max_len: Maximum character length.
        min_thai_ratio: Minimum Thai character ratio.
        max_repetition_ratio: Maximum repetition ratio.
        min_mean_line_len: Minimum mean line length.

    Returns:
        True if document passes all filters.
    """
    # Length filter
    if len(text) < min_len or len(text) > max_len:
        return False

    # Thai ratio filter
    if thai_ratio(text) < min_thai_ratio:
        return False

    # Repetition filter
    if _repetition_ratio(text) > max_repetition_ratio:
        return False

    # Mean line length filter
    if _mean_line_length(text) < min_mean_line_len:
        return False

    return True


def filter_file(
    input_path: str,
    output_path: str,
    min_len: int = 100,
    max_len: int = 500000,
    min_thai_ratio: float = 0.1,
    max_repetition_ratio: float = 0.3,
    min_mean_line_len: float = 10.0,
) -> dict:
    """Filter a JSONL file based on quality criteria.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        min_len: Minimum character length.
        max_len: Maximum character length.
        min_thai_ratio: Minimum Thai character ratio.
        max_repetition_ratio: Maximum repetition ratio.
        min_mean_line_len: Minimum mean line length.

    Returns:
        Dict with filtering statistics.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    filter_reasons: Counter = Counter()

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Filtering"):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                filter_reasons["json_error"] += 1
                continue

            text = record.get("text", "")
            if not text:
                filter_reasons["empty"] += 1
                continue

            # Check individual filters for stats
            if len(text) < min_len:
                filter_reasons["too_short"] += 1
                continue
            if len(text) > max_len:
                filter_reasons["too_long"] += 1
                continue
            if thai_ratio(text) < min_thai_ratio:
                filter_reasons["low_thai_ratio"] += 1
                continue
            if _repetition_ratio(text) > max_repetition_ratio:
                filter_reasons["high_repetition"] += 1
                continue
            if _mean_line_length(text) < min_mean_line_len:
                filter_reasons["short_lines"] += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "input_docs": total,
        "output_docs": kept,
        "filtered": total - kept,
        "filter_reasons": dict(filter_reasons),
    }

    print(f"Filtering done: {stats['input_docs']} -> {stats['output_docs']} "
          f"(filtered {stats['filtered']})")
    for reason, count in filter_reasons.items():
        print(f"  {reason}: {count}")

    return stats


def main(
    input_path: str,
    output_path: str,
    min_len: int = 100,
    max_len: int = 500000,
    min_thai_ratio: float = 0.1,
    max_repetition_ratio: float = 0.3,
) -> None:
    """CLI entry point."""
    filter_file(
        input_path, output_path,
        min_len=min_len, max_len=max_len,
        min_thai_ratio=min_thai_ratio,
        max_repetition_ratio=max_repetition_ratio,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
