"""Text generation utilities for Nirasa."""

from __future__ import annotations

from typing import Generator

import torch
import torch.nn.functional as F


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Apply repetition penalty to logits for previously generated tokens.

    Args:
        logits: Raw logits tensor of shape (vocab_size,).
        token_ids: List of previously generated token IDs.
        penalty: Repetition penalty factor (>1.0 penalizes repetition).

    Returns:
        Modified logits tensor.
    """
    if not token_ids or penalty == 1.0:
        return logits

    unique_ids = set(token_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> int:
    """Sample a token from logits with temperature, top-p, and top-k.

    Args:
        logits: Raw logits tensor of shape (vocab_size,).
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.

    Returns:
        Sampled token ID.
    """
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_strings: list[str] | None = None,
) -> str:
    """Generate text from a prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        repetition_penalty: Penalty for repeated tokens.
        stop_strings: Optional list of strings that stop generation.

    Returns:
        Generated text (without the prompt).
    """
    tokens = []
    for token in generate_stream(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
    ):
        tokens.append(token)
    return "".join(tokens)


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_strings: list[str] | None = None,
) -> Generator[str, None, None]:
    """Generate text from a prompt, yielding tokens as they are produced.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        repetition_penalty: Penalty for repeated tokens.
        stop_strings: Optional list of strings that stop generation.

    Yields:
        Generated text tokens one at a time.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids: list[int] = []
    generated_text = ""

    eos_token_id = tokenizer.eos_token_id

    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                last_token = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device=model.device
                )
                outputs = model(last_token, past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :].clone()

        # Apply repetition penalty
        all_ids = input_ids[0].tolist() + generated_ids
        logits = _apply_repetition_penalty(logits, all_ids, repetition_penalty)

        # Sample
        token_id = _sample_token(logits, temperature, top_p, top_k)

        # Check EOS
        if token_id == eos_token_id:
            break

        generated_ids.append(token_id)
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        generated_text += token_text

        yield token_text

        # Check stop strings
        if stop_strings:
            for stop in stop_strings:
                if stop in generated_text:
                    return


def main(
    prompt: str = "กรุงเทพมหานครเป็น",
    model_path: str = "Qwen/Qwen2.5-7B",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> None:
    """CLI entry point for standalone generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nPrompt: {prompt}")
    print(f"Output: ", end="", flush=True)

    for token in generate_stream(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    ):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
