"""Train a Thai SentencePiece BPE tokenizer."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import sentencepiece as spm


def train(
    input_dir: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    byte_fallback: bool = True,
    split_digits: bool = True,
    num_threads: int = 4,
) -> str:
    """Train a SentencePiece BPE tokenizer on Thai text.

    Args:
        input_dir: Directory containing .txt files for training.
        model_prefix: Output prefix for the model (e.g., data/tokenizer/nirasa_th_sp).
        vocab_size: Vocabulary size.
        character_coverage: Character coverage ratio.
        model_type: Model type (bpe or unigram).
        byte_fallback: Enable byte fallback for unknown characters.
        split_digits: Split digits into individual tokens.
        num_threads: Number of training threads.

    Returns:
        Path to the trained model file.
    """
    # Collect input files
    txt_files = sorted(glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    print(f"Found {len(txt_files)} text files:")
    for f in txt_files[:10]:
        print(f"  {f}")
    if len(txt_files) > 10:
        print(f"  ... and {len(txt_files) - 10} more")

    # Create output directory
    output_dir = Path(model_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = ",".join(txt_files)

    print(f"\nTraining SentencePiece {model_type.upper()} tokenizer:")
    print(f"  Vocab size:          {vocab_size}")
    print(f"  Character coverage:  {character_coverage}")
    print(f"  Byte fallback:       {byte_fallback}")
    print(f"  Model prefix:        {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        split_digits=split_digits,
        normalization_rule_name="nfkc",
        num_threads=num_threads,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
    )

    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"

    print(f"\nDone!")
    print(f"  Model: {model_path}")
    print(f"  Vocab: {vocab_path}")

    # Quick test
    sp = spm.SentencePieceProcessor(model_file=model_path)
    test_text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"
    tokens = sp.encode(test_text, out_type=str)
    print(f"\nTest: '{test_text}'")
    print(f"  Tokens ({len(tokens)}): {tokens}")

    return model_path


def main(
    input_dir: str,
    model_prefix: str,
    vocab_size: int = 32000,
) -> None:
    """CLI entry point."""
    train(input_dir=input_dir, model_prefix=model_prefix, vocab_size=vocab_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
