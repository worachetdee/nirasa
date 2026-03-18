"""Test text generation with base Qwen and LoRA-adapted model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.chdir("/content/nirasa")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B"
CHECKPOINT_DIR = "/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th"

# Thai test prompts
TEST_PROMPTS = [
    "ประเทศไทยมี",
    "กรุงเทพมหานครเป็น",
    "ภาษาไทยมีลักษณะ",
    "อาหารไทยที่",
    "ปัญญาประดิษฐ์คือ",
]


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint by step number."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None

    checkpoints = []
    for d in ckpt_path.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                checkpoints.append((step, str(d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    """Test generation with base and LoRA models."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load base model
    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # Find and load LoRA checkpoint
    ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)

    lora_model = None
    if ckpt_path:
        print(f"Loading LoRA from: {ckpt_path}")
        lora_model = PeftModel.from_pretrained(base_model, ckpt_path)
        lora_model = lora_model.merge_and_unload()
        lora_model.eval()
    else:
        print(f"No checkpoint found in {CHECKPOINT_DIR}, skipping LoRA comparison")

    # Generate with both models
    print("\n" + "=" * 70)
    print("Generation Comparison")
    print("=" * 70)

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)

        # Base model
        base_output = generate_text(base_model, tokenizer, prompt)
        print(f"  Base:  {base_output[:200]}")

        # LoRA model
        if lora_model is not None:
            lora_output = generate_text(lora_model, tokenizer, prompt)
            print(f"  LoRA:  {lora_output[:200]}")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
