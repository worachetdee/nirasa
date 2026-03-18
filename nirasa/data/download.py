"""Download Thai corpora from HuggingFace datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

CORPUS_CONFIGS = {
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.th",
        "split": "train",
        "text_field": "text",
    },
    "oscar": {
        "path": "oscar-corpus/OSCAR-2301",
        "name": "th",
        "split": "train",
        "text_field": "text",
    },
    "mc4": {
        "path": "mc4",
        "name": "th",
        "split": "train",
        "text_field": "text",
    },
}


def download_corpus(
    corpus: str = "wikipedia",
    output_dir: str = "data/raw",
    max_samples: int | None = None,
) -> str:
    """Download a Thai corpus and save as JSONL.

    Args:
        corpus: Corpus name (wikipedia, oscar, mc4).
        output_dir: Output directory for JSONL files.
        max_samples: Maximum number of samples to download (None for all).

    Returns:
        Path to the output JSONL file.
    """
    if corpus not in CORPUS_CONFIGS:
        raise ValueError(f"Unknown corpus: {corpus}. Choose from {list(CORPUS_CONFIGS.keys())}")

    config = CORPUS_CONFIGS[corpus]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{corpus}_th.jsonl"

    print(f"Downloading {corpus} (Thai)...")
    print(f"  Dataset: {config['path']}")
    print(f"  Config:  {config.get('name', 'default')}")

    ds = load_dataset(
        config["path"],
        name=config.get("name"),
        split=config["split"],
        trust_remote_code=True,
    )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    text_field = config["text_field"]
    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc=f"Writing {corpus}"):
            text = example.get(text_field, "")
            if text and text.strip():
                record = {"text": text.strip()}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Done: {count:,} documents, {file_size_mb:.1f} MB")
    print(f"Output: {output_file}")

    return str(output_file)


def main(
    corpus: str = "wikipedia",
    output_dir: str = "data/raw",
    max_samples: int | None = None,
) -> None:
    """CLI entry point for downloading corpora."""
    download_corpus(corpus=corpus, output_dir=output_dir, max_samples=max_samples)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
