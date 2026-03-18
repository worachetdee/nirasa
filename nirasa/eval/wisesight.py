"""Wisesight sentiment analysis evaluation."""

from __future__ import annotations

import torch
from datasets import load_dataset
from tqdm import tqdm

LABEL_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral",
    3: "question",
}

LABEL_THAI = {
    0: "เชิงบวก",
    1: "เชิงลบ",
    2: "เป็นกลาง",
    3: "คำถาม",
}

LABEL_TO_ID = {
    "positive": 0, "เชิงบวก": 0, "pos": 0,
    "negative": 1, "เชิงลบ": 1, "neg": 1,
    "neutral": 2, "เป็นกลาง": 2,
    "question": 3, "คำถาม": 3, "q": 3,
}


def _build_prompt(text: str, examples: list[dict] | None = None) -> str:
    """Build a sentiment classification prompt.

    Args:
        text: Text to classify.
        examples: Optional few-shot examples.

    Returns:
        Formatted prompt string.
    """
    instruction = (
        "จำแนกอารมณ์ของข้อความต่อไปนี้เป็น "
        "\"เชิงบวก\" (positive), \"เชิงลบ\" (negative), "
        "\"เป็นกลาง\" (neutral), หรือ \"คำถาม\" (question)\n\n"
    )

    prompt_parts = [instruction]

    if examples:
        for ex in examples:
            prompt_parts.append(
                f"ข้อความ: {ex['text']}\n"
                f"อารมณ์: {LABEL_THAI[ex['label']]}\n\n"
            )

    prompt_parts.append(
        f"ข้อความ: {text}\n"
        f"อารมณ์:"
    )

    return "".join(prompt_parts)


def _parse_prediction(text: str) -> int:
    """Parse model output to a label ID.

    Args:
        text: Generated text.

    Returns:
        Label ID (0-3), defaulting to 2 (neutral) if unparseable.
    """
    text = text.strip().lower()

    for label_text, label_id in LABEL_TO_ID.items():
        if label_text in text:
            return label_id

    return 2  # default to neutral


def evaluate(
    model,
    tokenizer,
    num_shots: int = 0,
    max_samples: int = 500,
    max_new_tokens: int = 32,
) -> dict:
    """Run Wisesight sentiment evaluation.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        num_shots: Number of few-shot examples.
        max_samples: Maximum number of samples to evaluate.
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        Dict with accuracy and per-class metrics.
    """
    print("Loading Wisesight sentiment dataset...")
    ds = load_dataset("wisesight_sentiment", split="test")

    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Gather few-shot examples
    few_shot_examples = []
    eval_start = 0
    if num_shots > 0:
        for i in range(min(num_shots * 4, len(ds))):  # try to get balanced examples
            item = ds[i]
            label = item["category"]
            if len([e for e in few_shot_examples if e["label"] == label]) < num_shots:
                few_shot_examples.append({
                    "text": item["texts"],
                    "label": label,
                })
            if len(few_shot_examples) >= num_shots * 4:
                break
        eval_start = len(few_shot_examples)

    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(4)}
    per_class_total = {i: 0 for i in range(4)}

    for i in tqdm(range(eval_start, len(ds)), desc="Wisesight"):
        item = ds[i]
        text = item["texts"]
        gold_label = item["category"]

        prompt = _build_prompt(
            text,
            few_shot_examples if num_shots > 0 else None,
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction_text = tokenizer.decode(generated, skip_special_tokens=True)
        predicted_label = _parse_prediction(prediction_text)

        per_class_total[gold_label] += 1
        if predicted_label == gold_label:
            correct += 1
            per_class_correct[gold_label] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {}
    for label_id, label_name in LABEL_MAP.items():
        cls_total = per_class_total[label_id]
        cls_correct = per_class_correct[label_id]
        per_class_acc[label_name] = cls_correct / cls_total if cls_total > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "num_samples": total,
    }

    print(f"Wisesight Results: Accuracy={accuracy:.4f} ({total} samples)")
    for name, acc in per_class_acc.items():
        print(f"  {name}: {acc:.4f}")

    return results
