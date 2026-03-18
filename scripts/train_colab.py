"""Training script for Google Colab with LoRA on Qwen2.5-7B."""

from __future__ import annotations

import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir("/content/nirasa")

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATA_BIN = "data/processed/th_wiki_qwen.bin"
DATA_IDX = "data/processed/th_wiki_qwen.idx"
OUTPUT_DIR = "/content/drive/MyDrive/nirasa_checkpoints/nirasa-7b-th"

MAX_SEQ_LEN = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
MAX_STEPS = 1000
SAVE_STEPS = 100
LOG_STEPS = 10
WARMUP = 50
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
SEED = 42

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class MemmapDataset(Dataset):
    """Dataset backed by a uint32 memmap file."""

    def __init__(self, bin_path: str, idx_path: str, seq_len: int = 512):
        self.seq_len = seq_len

        with open(idx_path, "rb") as f:
            header = f.read(12)
            num_docs, total_tokens, dtype_size = struct.unpack("<III", header)

        self.total_tokens = total_tokens
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(total_tokens,))
        self.num_samples = max(0, (total_tokens - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, self.total_tokens)

        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))
        input_ids = tokens[:-1]
        labels = tokens[1:]

        if len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "labels": labels}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Linear warmup + cosine decay scheduler."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    """Find latest checkpoint and return (path, step)."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None, 0

    checkpoints = []
    for d in output_path.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                checkpoints.append((step, str(d)))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1], checkpoints[-1][0]


def main():
    """Run training."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for resume
    resume_path, start_step = find_latest_checkpoint(OUTPUT_DIR)
    if resume_path:
        print(f"Resuming from: {resume_path} (step {start_step})")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load adapter weights if resuming
    if resume_path:
        print(f"Loading LoRA weights from {resume_path}")
        model.load_adapter(resume_path, adapter_name="default")

    # Dataset
    print(f"Loading dataset: {DATA_BIN}")
    dataset = MemmapDataset(DATA_BIN, DATA_IDX, seq_len=MAX_SEQ_LEN)
    print(f"Dataset: {len(dataset):,} samples, {dataset.total_tokens:,} tokens")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, MAX_STEPS)

    for _ in range(start_step):
        scheduler.step()

    # Training loop
    model.train()
    global_step = start_step
    total_loss = 0.0
    total_tokens_processed = 0
    start_time = time.time()
    log_start_time = time.time()
    data_iter = iter(dataloader)

    print(f"\nTraining: step {start_step} -> {MAX_STEPS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  LR: {LR}, LoRA r={LORA_R}, alpha={LORA_ALPHA}")

    while global_step < MAX_STEPS:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()
            total_tokens_processed += (labels != -100).sum().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        global_step += 1
        total_loss += accum_loss

        if global_step % LOG_STEPS == 0:
            elapsed = time.time() - log_start_time
            avg_loss = total_loss / LOG_STEPS
            current_lr = scheduler.get_last_lr()[0]
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-6)
            total_elapsed = time.time() - start_time

            print(
                f"step {global_step:>6d}/{MAX_STEPS} | "
                f"loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"tok/s {tok_per_sec:.0f} | "
                f"elapsed {total_elapsed:.0f}s"
            )

            total_loss = 0.0
            total_tokens_processed = 0
            log_start_time = time.time()

        if global_step % SAVE_STEPS == 0:
            ckpt_dir = output_path / f"step_{global_step}"
            print(f"Saving checkpoint: {ckpt_dir}")
            model.save_pretrained(str(ckpt_dir))
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": global_step},
                str(ckpt_dir / "training_state.pt"),
            )

    # Final save
    final_dir = output_path / f"step_{global_step}"
    print(f"Saving final: {final_dir}")
    model.save_pretrained(str(final_dir))

    total_time = time.time() - start_time
    print(f"\nTraining complete! {global_step} steps in {total_time:.0f}s ({total_time/3600:.1f}h)")


if __name__ == "__main__":
    main()
