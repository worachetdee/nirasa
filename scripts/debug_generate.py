"""Debug generation — test step by step."""
import os
import json
import torch

os.chdir("/content/nirasa")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B"
CHECKPOINT = "/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th/step_1000"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Test base model first
prompt = "กรุงเทพมหานครเป็น"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"\nInput token IDs: {inputs['input_ids'][0].tolist()}")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f"Base output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

# Check adapter config
print(f"\nAdapter config:")
with open(os.path.join(CHECKPOINT, "adapter_config.json")) as f:
    config = json.load(f)
    print(json.dumps(config, indent=2))

# Load LoRA
print(f"\nLoading LoRA from {CHECKPOINT}...")
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()

# Test LoRA model
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f"LoRA output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

# Check if the issue is with merge
print("\nMerging LoRA weights...")
model = model.merge_and_unload()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f"Merged output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
