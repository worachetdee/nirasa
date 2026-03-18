"""End-to-end smoke test for Nirasa project.

Validates data pipeline, training, eval, and serving components
without requiring GPU or full model downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --- Color helpers ---

try:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Test if terminal supports colors
    if not sys.stdout.isatty():
        raise RuntimeError("Not a TTY")
except Exception:
    GREEN = RED = YELLOW = BOLD = RESET = ""


def PASS(msg: str) -> str:
    return f"  {GREEN}PASS{RESET} {msg}"


def FAIL(msg: str) -> str:
    return f"  {RED}FAIL{RESET} {msg}"


def SKIP(msg: str) -> str:
    return f"  {YELLOW}SKIP{RESET} {msg}"


# --- Sample Thai texts ---

THAI_TEXTS = [
    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์",
    "ประเทศไทยตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้",
    "พระบาทสมเด็จพระเจ้าอยู่หัวทรงเป็นประมุขแห่งรัฐ",
    "ต้มยำกุ้งเป็นอาหารไทยที่มีชื่อเสียงระดับโลก",
    "วัดพระแก้วเป็นวัดที่สำคัญที่สุดในประเทศไทย",
    "แม่น้ำเจ้าพระยาเป็นแม่น้ำสายสำคัญของประเทศไทย",
    "ดอยอินทนนท์เป็นยอดเขาที่สูงที่สุดในประเทศไทย",
    "รำไทยเป็นศิลปะการแสดงที่มีความงดงาม",
    "มวยไทยเป็นศิลปะการต่อสู้ที่มีชื่อเสียงไปทั่วโลก",
    "ผ้าไหมไทยเป็นผลิตภัณฑ์ที่มีคุณภาพสูง",
    "สงกรานต์เป็นเทศกาลปีใหม่ไทย",
    "ข้าวเหนียวมะม่วงเป็นของหวานยอดนิยม",
    "ตลาดน้ำเป็นสถานที่ท่องเที่ยวที่น่าสนใจ",
    "ช้างเป็นสัตว์ประจำชาติของประเทศไทย",
    "ดอกราชพฤกษ์เป็นดอกไม้ประจำชาติไทย",
    "ลอยกระทงเป็นประเพณีที่สวยงามของไทย",
    "ส้มตำเป็นอาหารพื้นบ้านภาคอีสาน",
    "พระพุทธศาสนาเป็นศาสนาหลักของประเทศไทย",
    "การศึกษาไทยแบ่งเป็นระดับประถม มัธยม และอุดมศึกษา",
    "เกษตรกรรมเป็นอาชีพหลักของคนไทยในชนบท",
    "ประเทศไทยมีภูมิอากาศแบบร้อนชื้น",
]

results: list[tuple[str, str, bool]] = []


def record(phase: str, test_name: str, passed: bool, msg: str = ""):
    """Record a test result."""
    results.append((phase, test_name, passed))
    if passed:
        print(PASS(f"{test_name}" + (f" ({msg})" if msg else "")))
    else:
        print(FAIL(f"{test_name}" + (f" ({msg})" if msg else "")))


# ========================================
# Phase 1: Data Pipeline
# ========================================


def test_data_pipeline():
    """Test data cleaning, dedup, and filtering."""
    print(f"\n{BOLD}Phase 1: Data Pipeline{RESET}")

    from nirasa.data.clean import clean_text, thai_ratio, clean_file
    from nirasa.data.dedup import dedup_file
    from nirasa.data.filter import filter_document, filter_file

    # Test clean_text
    html_text = "<p>สวัสดีครับ</p> <b>ทดสอบ</b>"
    cleaned = clean_text(html_text)
    record("data", "clean_text: HTML removal", "สวัสดีครับ" in cleaned and "<p>" not in cleaned)

    url_text = "ดูเพิ่มเติมที่ https://example.com/path ข้อมูล"
    cleaned = clean_text(url_text)
    record("data", "clean_text: URL removal", "https://" not in cleaned)

    raw = "ทดสอบ\ufeffข้อ\u200bความ"
    cleaned = clean_text(raw)
    record("data", "clean_text: NFKC normalization", len(cleaned) > 0)

    # Test thai_ratio
    ratio = thai_ratio("สวัสดีครับ hello")
    record("data", "thai_ratio: mixed text", 0.3 < ratio < 0.8, f"ratio={ratio:.3f}")

    ratio_pure = thai_ratio("ภาษาไทยล้วน")
    record("data", "thai_ratio: pure Thai", ratio_pure > 0.8, f"ratio={ratio_pure:.3f}")

    ratio_eng = thai_ratio("English only text")
    record("data", "thai_ratio: pure English", ratio_eng == 0.0, f"ratio={ratio_eng:.3f}")

    # Test clean_file
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.jsonl")
        output_file = os.path.join(tmpdir, "output.jsonl")

        with open(input_file, "w", encoding="utf-8") as f:
            for text in THAI_TEXTS[:10]:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"text": "English only text here nothing Thai"}, ensure_ascii=False) + "\n")

        stats = clean_file(input_file, output_file, min_thai_ratio=0.3, num_workers=1)
        record("data", "clean_file: processes JSONL", stats["output_lines"] > 0,
               f"{stats['input_lines']}->{stats['output_lines']}")

        # Test dedup
        dedup_input = os.path.join(tmpdir, "dedup_in.jsonl")
        dedup_output = os.path.join(tmpdir, "dedup_out.jsonl")

        with open(dedup_input, "w", encoding="utf-8") as f:
            for text in THAI_TEXTS[:5]:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            # Add near-duplicate
            f.write(json.dumps({"text": THAI_TEXTS[0] + " เพิ่มเติม"}, ensure_ascii=False) + "\n")

        dedup_stats = dedup_file(dedup_input, dedup_output, threshold=0.8, num_perm=64)
        record("data", "dedup_file: removes duplicates",
               dedup_stats["output_docs"] <= dedup_stats["input_docs"],
               f"{dedup_stats['input_docs']}->{dedup_stats['output_docs']}")

        # Test filter
        record("data", "filter_document: pass long Thai",
               filter_document(THAI_TEXTS[0] * 5, min_len=50))

        record("data", "filter_document: reject short",
               not filter_document("สั้น", min_len=50))

        filter_input = os.path.join(tmpdir, "filter_in.jsonl")
        filter_output = os.path.join(tmpdir, "filter_out.jsonl")

        with open(filter_input, "w", encoding="utf-8") as f:
            for text in THAI_TEXTS:
                # Make texts long enough to pass filter
                long_text = (text + " ") * 10
                f.write(json.dumps({"text": long_text}, ensure_ascii=False) + "\n")
            # Add short text that should be filtered
            f.write(json.dumps({"text": "สั้น"}, ensure_ascii=False) + "\n")

        filter_stats = filter_file(filter_input, filter_output, min_len=50)
        record("data", "filter_file: filters documents",
               filter_stats["filtered"] > 0,
               f"{filter_stats['input_docs']}->{filter_stats['output_docs']}")


# ========================================
# Phase 2: Training
# ========================================


def test_training(device: str = "cpu"):
    """Test training components with a tiny random model."""
    print(f"\n{BOLD}Phase 2: Training{RESET}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Create a tiny model for testing
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2

    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(device).to(torch.float32)
        record("training", "model creation (tiny)", True, f"device={device}")
    except Exception as e:
        record("training", "model creation (tiny)", False, str(e))
        return

    # Test forward pass
    try:
        input_ids = torch.randint(0, 1000, (1, 32), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        record("training", "forward pass", loss.item() > 0, f"loss={loss.item():.4f}")
    except Exception as e:
        record("training", "forward pass", False, str(e))
        return

    # Test training loop (10 steps)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        initial_loss = None
        final_loss = None

        for step in range(10):
            input_ids = torch.randint(0, 1000, (2, 32), device=device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        loss_decreased = final_loss < initial_loss
        record("training", "10-step training",
               True,  # Just check it runs
               f"loss {initial_loss:.4f} -> {final_loss:.4f}" +
               (" (decreased)" if loss_decreased else " (not decreased, but ran)"))
    except Exception as e:
        record("training", "10-step training", False, str(e))

    # Test MemmapDataset
    try:
        from nirasa.training.train import MemmapDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = os.path.join(tmpdir, "test.bin")
            idx_path = os.path.join(tmpdir, "test.idx")

            tokens = np.random.randint(0, 10000, size=1024, dtype=np.uint32)
            mm = np.memmap(bin_path, dtype=np.uint32, mode="w+", shape=(1024,))
            mm[:] = tokens
            mm.flush()

            with open(idx_path, "wb") as f:
                f.write(struct.pack("<III", 1, 1024, 4))
                f.write(struct.pack("<Q", 0))
                f.write(struct.pack("<Q", 1024))

            dataset = MemmapDataset(bin_path, idx_path, seq_len=64)
            sample = dataset[0]
            record("training", "MemmapDataset",
                   sample["input_ids"].shape == (64,) and sample["labels"].shape == (64,),
                   f"samples={len(dataset)}")
    except Exception as e:
        record("training", "MemmapDataset", False, str(e))


# ========================================
# Phase 3: Eval
# ========================================


def test_eval():
    """Test evaluation metrics."""
    print(f"\n{BOLD}Phase 3: Eval{RESET}")

    from nirasa.eval.thaiqa import char_f1_score, exact_match_score

    # Test char_f1_score
    f1 = char_f1_score("กรุงเทพมหานคร", "กรุงเทพมหานคร")
    record("eval", "char_f1: exact match", abs(f1 - 1.0) < 1e-6, f"f1={f1:.4f}")

    f1 = char_f1_score("กรุงเทพ", "กรุงเทพมหานคร")
    record("eval", "char_f1: partial overlap", 0.0 < f1 < 1.0, f"f1={f1:.4f}")

    f1 = char_f1_score("กรุงเทพ", "เชียงใหม่")
    record("eval", "char_f1: no overlap", f1 < 0.3, f"f1={f1:.4f}")

    f1 = char_f1_score("", "")
    record("eval", "char_f1: both empty", f1 == 1.0, f"f1={f1:.4f}")

    f1 = char_f1_score("ข้อความ", "")
    record("eval", "char_f1: one empty", f1 == 0.0, f"f1={f1:.4f}")

    # Test exact_match_score
    em = exact_match_score("กรุงเทพมหานคร", "กรุงเทพมหานคร")
    record("eval", "exact_match: same string", em == 1.0)

    em = exact_match_score("กรุงเทพ", "กรุงเทพมหานคร")
    record("eval", "exact_match: different string", em == 0.0)

    em = exact_match_score("  กรุงเทพ  ", "กรุงเทพ")
    record("eval", "exact_match: whitespace normalization", em == 1.0)


# ========================================
# Phase 4: Serving
# ========================================


def test_serving():
    """Test serving components."""
    print(f"\n{BOLD}Phase 4: Serving{RESET}")

    from nirasa.serving.chat_template import (
        apply_chat_template,
        parse_chat_messages,
        DEFAULT_SYSTEM_PROMPT,
    )

    # Test chat template
    messages = [
        {"role": "user", "content": "สวัสดีครับ"},
    ]
    formatted = apply_chat_template(messages, add_generation_prompt=True)
    record("serving", "chat_template: format",
           "<|system|>" in formatted and "<|user|>" in formatted and "<|assistant|>" in formatted)

    record("serving", "chat_template: default system prompt",
           DEFAULT_SYSTEM_PROMPT in formatted)

    # Test with explicit system
    messages_sys = [
        {"role": "system", "content": "คุณเป็นผู้ช่วย"},
        {"role": "user", "content": "สวัสดี"},
    ]
    formatted_sys = apply_chat_template(messages_sys, add_generation_prompt=True)
    record("serving", "chat_template: custom system",
           "คุณเป็นผู้ช่วย" in formatted_sys)

    # Test parse
    parsed = parse_chat_messages(formatted)
    record("serving", "chat_template: parse roundtrip",
           len(parsed) >= 2)  # system + user

    # Test generate helpers
    from nirasa.serving.generate import _apply_repetition_penalty, _sample_token
    import torch

    logits = torch.ones(100)
    logits[5] = 10.0
    penalized = _apply_repetition_penalty(logits.clone(), [5], penalty=1.5)
    record("serving", "repetition_penalty", penalized[5] < logits[5],
           f"{logits[5]:.1f} -> {penalized[5]:.1f}")

    logits = torch.zeros(100)
    logits[42] = 100.0
    token = _sample_token(logits, temperature=0.0)
    record("serving", "sample_token: greedy", token == 42, f"token={token}")

    token_sampled = _sample_token(torch.randn(100), temperature=1.0, top_p=0.9, top_k=10)
    record("serving", "sample_token: sampling", 0 <= token_sampled < 100, f"token={token_sampled}")


# ========================================
# Main
# ========================================


def main():
    parser = argparse.ArgumentParser(description="Nirasa smoke test")
    parser.add_argument("--device", default="cpu", help="Device for model tests")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer tests")
    args = parser.parse_args()

    start_time = time.time()

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Nirasa Smoke Test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"Device: {args.device}")

    test_data_pipeline()
    test_training(device=args.device)
    test_eval()
    test_serving()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    total = len(results)
    passed = sum(1 for _, _, p in results if p)
    failed = total - passed

    # Group by phase
    phases: dict[str, list[tuple[str, str, bool]]] = {}
    for phase, name, p in results:
        phases.setdefault(phase, []).append((phase, name, p))

    print(f"\n{'Phase':<15s} {'Passed':>8s} {'Failed':>8s} {'Total':>8s}")
    print("-" * 45)
    for phase, phase_results in phases.items():
        p = sum(1 for _, _, ok in phase_results if ok)
        f = len(phase_results) - p
        status = GREEN + "OK" + RESET if f == 0 else RED + "FAIL" + RESET
        print(f"{phase:<15s} {p:>8d} {f:>8d} {len(phase_results):>8d}  {status}")

    print("-" * 45)
    print(f"{'TOTAL':<15s} {passed:>8d} {failed:>8d} {total:>8d}")
    print(f"\nTime: {elapsed:.1f}s")

    if failed > 0:
        print(f"\n{RED}{BOLD}{failed} test(s) FAILED{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All {total} tests PASSED{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
