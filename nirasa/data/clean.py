"""Thai text cleaning utilities."""

from __future__ import annotations

import json
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

# Regex patterns
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MULTIPLE_SPACES_RE = re.compile(r" {2,}")
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def thai_ratio(text: str) -> float:
    """Calculate the ratio of Thai characters in text.

    Thai Unicode range: U+0E00 to U+0E7F.

    Args:
        text: Input text.

    Returns:
        Ratio of Thai characters to total characters (0.0 to 1.0).
    """
    if not text:
        return 0.0
    total = len(text)
    thai_count = sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")
    return thai_count / total


def clean_text(text: str) -> str:
    """Clean a single text string.

    Steps:
        1. NFKC normalization
        2. HTML tag removal
        3. URL removal
        4. Control character removal
        5. Collapse whitespace

    Args:
        text: Raw input text.

    Returns:
        Cleaned text.
    """
    # NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove BOM (not stripped by NFKC or control char regex)
    text = text.replace("\ufeff", "")

    # Remove HTML tags
    text = HTML_TAG_RE.sub("", text)

    # Remove URLs
    text = URL_RE.sub("", text)

    # Remove control characters
    text = CONTROL_CHAR_RE.sub("", text)

    # Collapse multiple spaces and newlines
    text = MULTIPLE_SPACES_RE.sub(" ", text)
    text = MULTIPLE_NEWLINES_RE.sub("\n\n", text)

    return text.strip()


def process_line(line: str, min_thai_ratio: float = 0.3) -> str | None:
    """Process a single JSONL line.

    Args:
        line: JSONL line with a "text" field.
        min_thai_ratio: Minimum Thai character ratio to keep.

    Returns:
        Cleaned JSONL line or None if filtered out.
    """
    line = line.strip()
    if not line:
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    text = record.get("text", "")
    if not text:
        return None

    cleaned = clean_text(text)
    if not cleaned:
        return None

    if thai_ratio(cleaned) < min_thai_ratio:
        return None

    record["text"] = cleaned
    return json.dumps(record, ensure_ascii=False)


def _process_line_wrapper(args: tuple) -> str | None:
    """Wrapper for multiprocessing."""
    return process_line(*args)


def clean_file(
    input_path: str,
    output_path: str,
    min_thai_ratio: float = 0.3,
    num_workers: int = 4,
) -> dict:
    """Clean an entire JSONL file using multiprocessing.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        min_thai_ratio: Minimum Thai character ratio to keep.
        num_workers: Number of worker processes.

    Returns:
        Dict with statistics (input_lines, output_lines, filtered).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    total_input = len(lines)
    args_list = [(line, min_thai_ratio) for line in lines]

    results = []
    if num_workers <= 1:
        for args in tqdm(args_list, desc="Cleaning"):
            results.append(_process_line_wrapper(args))
    else:
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_line_wrapper, args_list, chunksize=1000),
                    total=len(args_list),
                    desc="Cleaning",
                )
            )

    cleaned = [r for r in results if r is not None]

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")

    stats = {
        "input_lines": total_input,
        "output_lines": len(cleaned),
        "filtered": total_input - len(cleaned),
    }
    print(f"Cleaning done: {stats['input_lines']} -> {stats['output_lines']} "
          f"(filtered {stats['filtered']})")
    return stats


def main(
    input_path: str,
    output_path: str,
    min_thai_ratio: float = 0.3,
    num_workers: int = 4,
) -> None:
    """CLI entry point."""
    clean_file(input_path, output_path, min_thai_ratio, num_workers)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
