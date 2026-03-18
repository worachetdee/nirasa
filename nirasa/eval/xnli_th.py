"""XNLI Thai evaluation for natural language inference."""

from __future__ import annotations

import torch
from datasets import load_dataset
from tqdm import tqdm

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

LABEL_THAI = {
    0: "เกี่ยวข้อง",
    1: "เป็นกลาง",
    2: "ขัดแย้ง",
}

LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "เกี่ยวข้อง": 0,
    "เป็นกลาง": 1,
    "ขัดแย้ง": 2,
}


def _build_prompt(premise: str, hypothesis: str, examples: list[dict] | None = None) -> str:
    """Build an NLI prompt.

    Args:
        premise: Premise text.
        hypothesis: Hypothesis text.
        examples: Optional few-shot examples.

    Returns:
        Formatted prompt string.
    """
    instruction = (
        "จากข้อความต่อไปนี้ ให้ระบุความสัมพันธ์ระหว่างข้อความที่ 1 และข้อความที่ 2 "
        "ว่าเป็น \"เกี่ยวข้อง\" (entailment), \"เป็นกลาง\" (neutral), "
        "หรือ \"ขัดแย้ง\" (contradiction)\n\n"
    )

    prompt_parts = [instruction]

    if examples:
        for ex in examples:
            prompt_parts.append(
                f"ข้อความที่ 1: {ex['premise']}\n"
                f"ข้อความที่ 2: {ex['hypothesis']}\n"
                f"คำตอบ: {LABEL_THAI[ex['label']]}\n\n"
            )

    prompt_parts.append(
        f"ข้อความที่ 1: {premise}\n"
        f"ข้อความที่ 2: {hypothesis}\n"
        f"คำตอบ:"
    )

    return "".join(prompt_parts)


def _parse_prediction(text: str) -> int:
    """Parse model output to a label ID.

    Args:
        text: Generated text.

    Returns:
        Label ID (0, 1, or 2), defaulting to 1 (neutral) if unparseable.
    """
    text = text.strip().lower()

    for label_text, label_id in LABEL_TO_ID.items():
        if label_text in text:
            return label_id

    return 1  # default to neutral


def evaluate(
    model,
    tokenizer,
    num_shots: int = 0,
    max_samples: int = 500,
    max_new_tokens: int = 32,
) -> dict:
    """Run XNLI Thai evaluation.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        num_shots: Number of few-shot examples.
        max_samples: Maximum number of samples to evaluate.
        max_new_tokens: Maximum new tokens to generate.

    Returns:
        Dict with accuracy and per-class metrics.
    """
    print("Loading XNLI dataset (Thai)...")
    ds = load_dataset("xnli", "th", split="test")

    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Gather few-shot examples
    few_shot_examples = []
    eval_start = 0
    if num_shots > 0:
        for i in range(min(num_shots, len(ds))):
            item = ds[i]
            few_shot_examples.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": item["label"],
            })
        eval_start = num_shots

    correct = 0
    total = 0
    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total = {0: 0, 1: 0, 2: 0}

    for i in tqdm(range(eval_start, len(ds)), desc="XNLI-th"):
        item = ds[i]
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        gold_label = item["label"]

        prompt = _build_prompt(
            premise, hypothesis,
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

    print(f"XNLI-th Results: Accuracy={accuracy:.4f} ({total} samples)")
    for name, acc in per_class_acc.items():
        print(f"  {name}: {acc:.4f}")

    return results
