"""Evaluate tokenizer quality on Thai text samples."""

from __future__ import annotations

from transformers import AutoTokenizer

# Sample Thai texts for evaluation
THAI_SAMPLES = [
    # Wikipedia excerpts
    "กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย",
    "ประเทศไทยมีพื้นที่ 513,120 ตารางกิโลเมตร ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้",
    "พระบาทสมเด็จพระเจ้าอยู่หัวทรงเป็นประมุขแห่งรัฐ",
    "ภาษาไทยเป็นภาษาราชการของประเทศไทย เป็นภาษาที่มีวรรณยุกต์",
    # News style
    "รัฐบาลประกาศมาตรการกระตุ้นเศรษฐกิจครั้งใหม่เพื่อสนับสนุนการฟื้นตัวของภาคธุรกิจ",
    "สภาพอากาศวันนี้ทั่วประเทศมีฝนตกเป็นบริเวณกว้าง โดยเฉพาะภาคใต้",
    "ตลาดหุ้นไทยปิดตลาดวันนี้ปรับตัวเพิ่มขึ้น 10 จุด",
    # Literature
    "น้ำใสไหลเย็นเห็นตัวปลา ว่ายไปมาในลำธาร",
    "ดอกไม้บานสะพรั่งกลิ่นหอมหวาน ลมพัดผ่านพาความสดชื่น",
    # Technical
    "ปัญญาประดิษฐ์คือสาขาวิชาของวิทยาการคอมพิวเตอร์ที่เกี่ยวข้องกับการสร้างเครื่องจักรอัจฉริยะ",
    "การเรียนรู้ของเครื่องเป็นส่วนหนึ่งของปัญญาประดิษฐ์ที่ช่วยให้คอมพิวเตอร์เรียนรู้จากข้อมูล",
    # Conversational
    "สวัสดีครับ วันนี้คุณสบายดีไหมครับ",
    "ขอบคุณมากค่ะ ช่วยเหลือได้ดีมากเลย",
    # Food
    "ต้มยำกุ้งเป็นอาหารไทยที่มีชื่อเสียงระดับโลก รสชาติเผ็ดร้อนเปรี้ยว",
    "ผัดไทยเป็นอาหารจานเดียวที่ได้รับความนิยมทั้งในประเทศและต่างประเทศ",
]


def compute_fertility(tokenizer, texts: list[str]) -> float:
    """Compute average fertility (tokens per character) over texts.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of text samples.

    Returns:
        Average fertility ratio.
    """
    total_tokens = 0
    total_chars = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        total_chars += len(text)
    return total_tokens / total_chars if total_chars > 0 else 0.0


def compute_unk_rate(tokenizer, texts: list[str]) -> float:
    """Compute the rate of UNK tokens.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of text samples.

    Returns:
        UNK token rate (0.0 to 1.0).
    """
    total_tokens = 0
    unk_tokens = 0
    unk_id = tokenizer.unk_token_id

    if unk_id is None:
        return 0.0

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        unk_tokens += sum(1 for tid in token_ids if tid == unk_id)

    return unk_tokens / total_tokens if total_tokens > 0 else 0.0


def compute_roundtrip_accuracy(tokenizer, texts: list[str]) -> float:
    """Compute roundtrip encode-decode accuracy.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of text samples.

    Returns:
        Fraction of texts that survive roundtrip perfectly.
    """
    correct = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        if decoded.strip() == text.strip():
            correct += 1
    return correct / len(texts) if texts else 0.0


def evaluate_tokenizer(
    model_name: str = "Qwen/Qwen2.5-7B",
    custom_tokenizer_path: str | None = None,
) -> dict:
    """Evaluate tokenizer quality on Thai samples.

    Args:
        model_name: HuggingFace model name for base tokenizer.
        custom_tokenizer_path: Optional path to a custom tokenizer for comparison.

    Returns:
        Dict with evaluation results.
    """
    results = {}

    # Evaluate base Qwen tokenizer
    print(f"Loading base tokenizer: {model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    base_fertility = compute_fertility(base_tokenizer, THAI_SAMPLES)
    base_unk_rate = compute_unk_rate(base_tokenizer, THAI_SAMPLES)
    base_roundtrip = compute_roundtrip_accuracy(base_tokenizer, THAI_SAMPLES)

    print(f"\n{'='*60}")
    print(f"Base Tokenizer: {model_name}")
    print(f"{'='*60}")
    print(f"  Fertility (tokens/char): {base_fertility:.4f}")
    print(f"  UNK rate:                {base_unk_rate:.4f}")
    print(f"  Roundtrip accuracy:      {base_roundtrip:.4f}")

    results["base"] = {
        "model": model_name,
        "fertility": base_fertility,
        "unk_rate": base_unk_rate,
        "roundtrip_accuracy": base_roundtrip,
    }

    # Show per-sample tokenization
    print(f"\nPer-sample tokenization:")
    for i, text in enumerate(THAI_SAMPLES[:5]):
        tokens = base_tokenizer.encode(text, add_special_tokens=False)
        token_strs = base_tokenizer.convert_ids_to_tokens(tokens)
        print(f"  [{i}] {text[:50]}...")
        print(f"      Tokens ({len(tokens)}): {token_strs[:10]}{'...' if len(token_strs) > 10 else ''}")

    # Evaluate custom tokenizer if provided
    if custom_tokenizer_path:
        import sentencepiece as spm

        print(f"\n{'='*60}")
        print(f"Custom Tokenizer: {custom_tokenizer_path}")
        print(f"{'='*60}")

        sp = spm.SentencePieceProcessor(model_file=custom_tokenizer_path)

        total_tokens = 0
        total_chars = 0
        for text in THAI_SAMPLES:
            tokens = sp.encode(text, out_type=str)
            total_tokens += len(tokens)
            total_chars += len(text)

        custom_fertility = total_tokens / total_chars if total_chars > 0 else 0.0
        print(f"  Fertility (tokens/char): {custom_fertility:.4f}")

        results["custom"] = {
            "model": custom_tokenizer_path,
            "fertility": custom_fertility,
        }

    return results


def main(
    model_name: str = "Qwen/Qwen2.5-7B",
    custom_tokenizer_path: str | None = None,
) -> None:
    """CLI entry point."""
    evaluate_tokenizer(model_name=model_name, custom_tokenizer_path=custom_tokenizer_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
