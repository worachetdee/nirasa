"""Test text generation comparing base Qwen vs Nirasa (LoRA-adapted)."""
import os
import torch
from pathlib import Path

os.chdir("/content/nirasa")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B"
CHECKPOINT_DIR = "/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th-v2"

TEST_PROMPTS = [
    "ประเทศไทยมี",
    "กรุงเทพมหานครเป็น",
    "ภาษาไทยมีลักษณะ",
    "อาหารไทยที่",
    "ปัญญาประดิษฐ์คือ",
]


def find_latest_checkpoint(checkpoint_dir):
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
        # Check if adapter files are directly in the dir
        if (ckpt_path / "adapter_config.json").exists():
            return str(ckpt_path)
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def generate(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# Step 1: Test base model
# ============================================================
print("=" * 60)
print("Nirasa-7B Thai Generation Test")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading base model: {MODEL_NAME}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()

print("\n--- Base Qwen2.5-7B (no Thai training) ---")
for prompt in TEST_PROMPTS:
    output = generate(base_model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output[:300]}")
    print("-" * 50)

# Free memory
del base_model
torch.cuda.empty_cache()

# ============================================================
# Step 2: Test LoRA model
# ============================================================
ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
if ckpt_path:
    print(f"\n\nLoading base model again for LoRA...")
    lora_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {ckpt_path}")
    lora_model = PeftModel.from_pretrained(lora_base, ckpt_path)
    lora_model.eval()

    print("\n--- Nirasa-7B (LoRA-adapted for Thai) ---")
    for prompt in TEST_PROMPTS:
        output = generate(lora_model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output[:300]}")
        print("-" * 50)
else:
    print(f"\nNo checkpoint found in {CHECKPOINT_DIR}")

print("\nDone!")
