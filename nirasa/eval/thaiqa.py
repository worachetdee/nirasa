"""Thai Question Answering evaluation."""

from __future__ import annotations

import re
import unicodedata

import torch
from datasets import load_dataset
from tqdm import tqdm


def normalize_thai_text(text: str) -> str:
    """Normalize Thai text for comparison.

    Args:
        text: Raw Thai text.

    Returns:
        Normalized text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def char_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute character-level F1 score.

    Args:
        prediction: Predicted answer text.
        ground_truth: Ground truth answer text.

    Returns:
        Character-level F1 score (0.0 to 1.0).
    """
    prediction = normalize_thai_text(prediction)
    ground_truth = normalize_thai_text(ground_truth)

    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0

    pred_chars = list(prediction)
    gt_chars = list(ground_truth)

    common_chars = set(pred_chars) & set(gt_chars)
    if not common_chars:
        return 0.0

    # Count character overlaps
    pred_counter: dict[str, int] = {}
    for c in pred_chars:
        pred_counter[c] = pred_counter.get(c, 0) + 1

    gt_counter: dict[str, int] = {}
    for c in gt_chars:
        gt_counter[c] = gt_counter.get(c, 0) + 1

    overlap = 0
    for c in common_chars:
        overlap += min(pred_counter.get(c, 0), gt_counter.get(c, 0))

    precision = overlap / len(pred_chars) if pred_chars else 0.0
    recall = overlap / len(gt_chars) if gt_chars else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.

    Args:
        prediction: Predicted answer text.
        ground_truth: Ground truth answer text.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return 1.0 if normalize_thai_text(prediction) == normalize_thai_text(ground_truth) else 0.0


def _build_prompt(question: str, context: str, examples: list[dict] | None = None) -> str:
    """Build a QA prompt.

    Args:
        question: Question text.
        context: Context passage.
        examples: Optional few-shot examples.

    Returns:
        Formatted prompt string.
    """
    prompt_parts = []

    if examples:
        for ex in examples:
            prompt_parts.append(
                f"บริบท: {ex['context']}\n"
                f"คำถาม: {ex['question']}\n"
                f"คำตอบ: {ex['answer']}\n"
            )

    prompt_parts.append(
        f"บริบท: {context}\n"
        f"คำถาม: {question}\n"
        f"คำตอบ:"
    )

    return "\n".join(prompt_parts)


def evaluate(
    model,
    tokenizer,
    num_shots: int = 0,
    max_samples: int = 500,
    max_new_tokens: int = 64,
    dataset_name: str = "thaiqa_squad",
) -> dict:
    """Run Thai QA evaluation.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        num_shots: Number of few-shot examples.
        max_samples: Maximum number of samples to evaluate.
        max_new_tokens: Maximum new tokens to generate.
        dataset_name: HuggingFace dataset name.

    Returns:
        Dict with F1 and exact match scores.
    """
    print(f"Loading ThaiQA dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_name, split="test")
    except Exception:
        ds = load_dataset(dataset_name, split="validation")

    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Gather few-shot examples from the beginning if needed
    few_shot_examples = []
    eval_start = 0
    if num_shots > 0:
        for i in range(min(num_shots, len(ds))):
            item = ds[i]
            few_shot_examples.append({
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "answer": item.get("answers", {}).get("text", [""])[0] if isinstance(item.get("answers"), dict) else "",
            })
        eval_start = num_shots

    total_f1 = 0.0
    total_em = 0.0
    count = 0

    for i in tqdm(range(eval_start, len(ds)), desc="ThaiQA"):
        item = ds[i]
        question = item.get("question", "")
        context = item.get("context", "")

        answers = item.get("answers", {})
        if isinstance(answers, dict):
            ground_truths = answers.get("text", [])
        else:
            ground_truths = [str(answers)]

        if not ground_truths:
            continue

        prompt = _build_prompt(question, context, few_shot_examples if num_shots > 0 else None)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Take first line as answer
        prediction = prediction.split("\n")[0].strip()

        # Compute best score against all ground truths
        best_f1 = max(char_f1_score(prediction, gt) for gt in ground_truths)
        best_em = max(exact_match_score(prediction, gt) for gt in ground_truths)

        total_f1 += best_f1
        total_em += best_em
        count += 1

    avg_f1 = total_f1 / count if count > 0 else 0.0
    avg_em = total_em / count if count > 0 else 0.0

    results = {
        "f1": avg_f1,
        "exact_match": avg_em,
        "num_samples": count,
    }

    print(f"ThaiQA Results: F1={avg_f1:.4f}, EM={avg_em:.4f} ({count} samples)")
    return results
