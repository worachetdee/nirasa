"""Main training script for Nirasa Thai language model with LoRA."""

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


class MemmapDataset(Dataset):
    """Dataset backed by a uint32 memmap file.

    Reads token IDs from a .bin file and returns fixed-length sequences.
    An accompanying .idx file provides document boundaries but is optional
    for simple sequential reading.
    """

    def __init__(self, bin_path: str, idx_path: str, seq_len: int = 512):
        self.seq_len = seq_len

        # Read index to get total tokens
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
        end = start + self.seq_len + 1  # +1 for labels shift
        end = min(end, self.total_tokens)

        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))

        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Pad if necessary
        if len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "labels": labels}


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory by step number.

    Args:
        output_dir: Directory containing step_* checkpoint directories.

    Returns:
        Path to the latest checkpoint, or None if none found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for d in output_path.iterdir():
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


def train(
    data_bin: str,
    data_idx: str,
    output_dir: str = "checkpoints/nirasa-7b-th",
    model_name: str = "Qwen/Qwen2.5-7B",
    max_steps: int = 1000,
    max_seq_len: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    save_steps: int = 100,
    log_steps: int = 10,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    resume: bool = True,
    seed: int = 42,
) -> None:
    """Train Nirasa with LoRA on Thai data.

    Args:
        data_bin: Path to tokenized data .bin file.
        data_idx: Path to tokenized data .idx file.
        output_dir: Directory for checkpoints.
        model_name: Base model name.
        max_steps: Maximum training steps.
        max_seq_len: Maximum sequence length.
        batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        warmup_steps: Number of warmup steps.
        weight_decay: Weight decay for AdamW.
        max_grad_norm: Maximum gradient norm for clipping.
        save_steps: Save checkpoint every N steps.
        log_steps: Log metrics every N steps.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        resume: Whether to resume from latest checkpoint.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for resume
    start_step = 0
    resume_path = None
    if resume:
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path:
            step_name = Path(resume_path).name
            start_step = int(step_name.split("_")[1])
            print(f"Resuming from checkpoint: {resume_path} (step {start_step})")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    if resume_path:
        from peft import PeftModel
        print(f"Loading LoRA weights from {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset and dataloader
    print(f"Loading dataset: {data_bin}")
    dataset = MemmapDataset(data_bin, data_idx, seq_len=max_seq_len)
    print(f"Dataset: {len(dataset):,} samples, {dataset.total_tokens:,} tokens")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    # Restore optimizer/scheduler state if resuming
    if resume_path:
        state_path = Path(resume_path) / "training_state.pt"
        if state_path.exists():
            print(f"Restoring training state from {state_path}")
            training_state = torch.load(state_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
        else:
            print("No training_state.pt found, fast-forwarding scheduler")
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

    print(f"\nStarting training from step {start_step} to {max_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}")

    while global_step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for accum_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            total_tokens_processed += (labels != -100).sum().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()
        global_step += 1
        total_loss += accum_loss

        # Logging
        if global_step % log_steps == 0:
            elapsed = time.time() - log_start_time
            avg_loss = total_loss / log_steps
            current_lr = scheduler.get_last_lr()[0]
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-6)
            total_elapsed = time.time() - start_time

            print(
                f"step {global_step:>6d}/{max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"tok/s {tok_per_sec:.0f} | "
                f"elapsed {total_elapsed:.0f}s"
            )

            total_loss = 0.0
            total_tokens_processed = 0
            log_start_time = time.time()

        # Save checkpoint
        if global_step % save_steps == 0:
            ckpt_dir = output_path / f"step_{global_step}"
            print(f"Saving checkpoint: {ckpt_dir}")
            model.save_pretrained(str(ckpt_dir))
            # Save optimizer state
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                },
                str(ckpt_dir / "training_state.pt"),
            )

    # Save final checkpoint
    final_dir = output_path / f"step_{global_step}"
    print(f"Saving final checkpoint: {final_dir}")
    model.save_pretrained(str(final_dir))

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Total time:  {total_time:.0f}s ({total_time/3600:.1f}h)")


def main() -> None:
    """CLI entry point."""
    import fire

    fire.Fire(train)


if __name__ == "__main__":
    main()
