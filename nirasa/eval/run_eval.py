"""Unified evaluation runner for Nirasa benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nirasa.eval import thaiqa, xnli_th, wisesight, perplexity


BENCHMARK_REGISTRY = {
    "thaiqa": thaiqa.evaluate,
    "xnli": xnli_th.evaluate,
    "wisesight": wisesight.evaluate,
    "perplexity": perplexity.evaluate,
}


def load_model_and_tokenizer(
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-7B",
    device: str | None = None,
) -> tuple:
    """Load a model and tokenizer, handling LoRA adapters.

    Args:
        model_path: Path to model or LoRA adapter.
        base_model: Base model name for LoRA.
        device: Device to load on (auto if None).

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model_path = Path(model_path)

    # Check if this is a LoRA adapter (has adapter_config.json)
    if (model_path / "adapter_config.json").exists():
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device is None else device,
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))
        model = model.merge_and_unload()
    else:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto" if device is None else device,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def evaluate(
    model_path: str,
    benchmarks: str = "thaiqa,xnli,wisesight,perplexity",
    base_model: str = "Qwen/Qwen2.5-7B",
    output_file: str | None = None,
    num_shots: int = 0,
    max_samples: int = 500,
) -> dict:
    """Run selected benchmarks on a model.

    Args:
        model_path: Path to model or LoRA adapter.
        benchmarks: Comma-separated list of benchmark names.
        base_model: Base model name for LoRA.
        output_file: Optional path to save results JSON.
        num_shots: Number of few-shot examples.
        max_samples: Maximum samples per benchmark.

    Returns:
        Dict with benchmark results.
    """
    model, tokenizer = load_model_and_tokenizer(model_path, base_model)

    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    results = {
        "model_path": str(model_path),
        "base_model": base_model,
        "benchmarks": {},
    }

    for bench_name in benchmark_list:
        if bench_name not in BENCHMARK_REGISTRY:
            print(f"Unknown benchmark: {bench_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {bench_name}")
        print(f"{'='*60}")

        start_time = time.time()
        eval_fn = BENCHMARK_REGISTRY[bench_name]

        if bench_name == "perplexity":
            bench_results = eval_fn(model, tokenizer)
        else:
            bench_results = eval_fn(
                model, tokenizer,
                num_shots=num_shots,
                max_samples=max_samples,
            )

        elapsed = time.time() - start_time
        bench_results["elapsed_seconds"] = elapsed
        results["benchmarks"][bench_name] = bench_results

        print(f"  Results: {bench_results}")
        print(f"  Time: {elapsed:.1f}s")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for bench_name, bench_results in results["benchmarks"].items():
        metric = bench_results.get("accuracy", bench_results.get("f1", bench_results.get("perplexity", "N/A")))
        print(f"  {bench_name:>15s}: {metric}")

    # Save results
    if output_file is None:
        output_file = Path(model_path) / "eval_results.json"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return results


def main() -> None:
    """CLI entry point."""
    import fire

    fire.Fire(evaluate)


if __name__ == "__main__":
    main()
