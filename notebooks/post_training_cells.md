# Post-Training Cells for Colab

Paste these into new cells in order after the base training completes.

---

## Cell A: Backup Base Model

```python
import shutil
backup_dir = '/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th-v3-base-backup'
source_dir = '/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th-v3/step_5000'
shutil.copytree(source_dir, backup_dir)
print(f'Backed up to: {backup_dir}')
```

---

## Cell B: Chat Fine-Tuning

```python
# === Chat Fine-Tuning on WangchanThaiInstruct ===
import json, os, random
from datasets import load_dataset

# Step 1: Prepare chat data
os.makedirs('/content/nirasa/data/chat', exist_ok=True)
ds = load_dataset('airesearch/WangchanThaiInstruct', split='train')
samples = []
for row in ds:
    instruction = row.get('Instruction', '')
    inp = row.get('Input', '') or ''
    output = row.get('Output', '')
    if not instruction or not output:
        continue
    prompt = f'{instruction}\n{inp}' if inp else instruction
    text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>'
    samples.append({'text': text})

random.seed(42)
random.shuffle(samples)
n_valid = max(1, int(len(samples) * 0.02))

chat_jsonl = '/content/nirasa/data/chat/wangchan_instruct.jsonl'
with open(chat_jsonl, 'w') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'Chat data ready: {len(samples)} samples')

# Step 2: Tokenize
import struct
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
all_tokens = []
for s in samples[n_valid:]:
    ids = tokenizer.encode(s['text'], add_special_tokens=False)
    all_tokens.extend(ids)
    all_tokens.append(tokenizer.eos_token_id)

total_tokens = len(all_tokens)
tokens_array = np.array(all_tokens, dtype=np.uint32)
bin_path = '/content/nirasa/data/chat/chat.bin'
idx_path = '/content/nirasa/data/chat/chat.idx'
mm = np.memmap(bin_path, dtype=np.uint32, mode='w+', shape=(total_tokens,))
mm[:] = tokens_array
mm.flush()
with open(idx_path, 'wb') as f:
    f.write(struct.pack('<III', len(samples) - n_valid, total_tokens, 4))
print(f'Tokenized: {total_tokens:,} tokens')

# Step 3: Train chat LoRA
CHAT_OUTPUT = '/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th-v3-chat'
CHAT_STEPS = 1000
CHAT_LR = 5e-5

dataset_chat = MemmapDataset(bin_path, idx_path, seq_len=MAX_SEQ_LEN)
print(f'Chat dataset: {len(dataset_chat):,} samples')

dataloader_chat = DataLoader(
    dataset_chat, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
)

optimizer_chat = torch.optim.AdamW(
    model.parameters(), lr=CHAT_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
)
scheduler_chat = get_cosine_schedule_with_warmup(optimizer_chat, 50, CHAT_STEPS)

model.train()
global_step_chat = 0
total_loss_chat = 0.0
data_iter_chat = iter(dataloader_chat)
start_time_chat = time.time()
log_start_chat = time.time()
tokens_chat = 0

print(f'\nChat training: 0 -> {CHAT_STEPS}')
while global_step_chat < CHAT_STEPS:
    optimizer_chat.zero_grad()
    accum_loss = 0.0
    for _ in range(GRAD_ACCUM):
        try:
            batch = next(data_iter_chat)
        except StopIteration:
            data_iter_chat = iter(dataloader_chat)
            batch = next(data_iter_chat)
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()
        tokens_chat += (labels != -100).sum().item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer_chat.step()
    scheduler_chat.step()
    global_step_chat += 1
    total_loss_chat += accum_loss
    if global_step_chat % 10 == 0:
        elapsed = time.time() - log_start_chat
        avg = total_loss_chat / 10
        lr = scheduler_chat.get_last_lr()[0]
        tps = tokens_chat / max(elapsed, 1e-6)
        eta = (CHAT_STEPS - global_step_chat) / global_step_chat * (time.time() - start_time_chat)
        print(f'chat {global_step_chat:>5d}/{CHAT_STEPS} | loss {avg:.4f} | lr {lr:.2e} | tok/s {tps:.0f} | ETA {eta/3600:.1f}h')
        total_loss_chat = 0.0
        tokens_chat = 0
        log_start_chat = time.time()
    if global_step_chat % 500 == 0:
        ckpt = Path(CHAT_OUTPUT) / f'step_{global_step_chat}'
        print(f'Saving: {ckpt}')
        model.save_pretrained(str(ckpt))

final = Path(CHAT_OUTPUT) / f'step_{global_step_chat}'
model.save_pretrained(str(final))
print(f'\nChat training done! {(time.time()-start_time_chat)/3600:.1f}h')
```

---

## Cell D0: Test Raw Qwen (no LoRA — diagnostic)

```python
del model
torch.cuda.empty_cache()

raw_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
raw_base.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

prompt = 'ประเทศไทยมี'
inputs = tokenizer(prompt, return_tensors='pt').to(raw_base.device)
with torch.no_grad():
    output = raw_base.generate(
        **inputs, max_new_tokens=100, temperature=0.7,
        top_p=0.9, repetition_penalty=1.2, do_sample=True,
    )
print('Raw Qwen2.5-7B (no LoRA):')
print(tokenizer.decode(output[0], skip_special_tokens=True))

del raw_base
torch.cuda.empty_cache()
```

---

## Cell C0: Test Base Model (without chat fine-tuning)

```python
from peft import PeftModel

del model
torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
base_lora_path = '/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th-v3-base-backup'
model = PeftModel.from_pretrained(base, base_lora_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

prompts = [
    'ประเทศไทยมี',
    'กรุงเทพมหานครเป็น',
    'อาหารไทยที่มีชื่อเสียงที่สุดคือ',
    'ภาษาไทยเป็นภาษาที่',
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'\n{"="*60}')
    print(f'Prompt: {prompt}')
    print(f'Output: {text}')
```

---

## Cell C: Test Generation (FIXED)

```python
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

prompts = [
    'สวัสดีครับ',
    'กรุงเทพมหานครเป็นเมืองอะไร',
    'อาหารไทยที่มีชื่อเสียงที่สุดคืออะไร',
    'ช่วยอธิบายเกี่ยวกับภาษาไทยให้หน่อย',
]

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
    result = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'\n{"="*60}')
    print(f'Q: {prompt}')
    print(f'A: {result}')
```
