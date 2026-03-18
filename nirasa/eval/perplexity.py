"""Perplexity evaluation on held-out Thai text."""

from __future__ import annotations

import math

import torch

# Held-out Thai text for perplexity evaluation
THAI_EVAL_TEXT = """กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย
เป็นศูนย์กลางการปกครอง การศึกษา การคมนาคมขนส่ง การเงินการธนาคาร การพาณิชย์
การสื่อสาร และความเจริญของประเทศ เป็นเมืองที่มีชื่อยาวที่สุดในโลก
ตั้งอยู่บนสามเหลี่ยมปากแม่น้ำเจ้าพระยา มีพื้นที่ 1,568.737 ตารางกิโลเมตร
ประเทศไทยมีชื่อเป็นทางการว่า ราชอาณาจักรไทย มีระบอบการปกครองแบบประชาธิปไตย
อันมีพระมหากษัตริย์ทรงเป็นประมุข ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้
มีอาณาเขตทิศเหนือติดกับประเทศเมียนมาและสาธารณรัฐประชาธิปไตยประชาชนลาว
ทิศตะวันออกติดกับสาธารณรัฐประชาธิปไตยประชาชนลาวและราชอาณาจักรกัมพูชา
ทิศใต้ติดกับอ่าวไทยและประเทศมาเลเซีย ทิศตะวันตกติดกับทะเลอันดามันและประเทศเมียนมา
ภาษาไทยเป็นภาษาที่มีระดับเสียงของคำ แต่ละคำจะมีระดับเสียงสูงต่ำที่แน่นอน
ซึ่งเรียกว่าวรรณยุกต์ การเปลี่ยนวรรณยุกต์จะทำให้ความหมายของคำเปลี่ยนไป
ภาษาไทยมีพยัญชนะ 44 ตัว สระ 32 รูป วรรณยุกต์ 4 รูป 5 เสียง
อาหารไทยเป็นที่รู้จักไปทั่วโลก ด้วยรสชาติที่เป็นเอกลักษณ์
ทั้งเผ็ด เปรี้ยว หวาน เค็ม ที่ผสมผสานกันอย่างลงตัว
ต้มยำกุ้งเป็นอาหารที่มีชื่อเสียงที่สุด มีรสชาติเผ็ดร้อนและเปรี้ยว
ส้มตำเป็นอาหารพื้นบ้านภาคอีสาน ที่ได้รับความนิยมทั่วประเทศ
แกงเขียวหวานเป็นแกงที่มีสีเขียวจากพริกเขียว มีรสชาติหวานมัน"""


def compute_perplexity(
    model,
    tokenizer,
    text: str | None = None,
    window_size: int = 512,
    stride: int = 256,
) -> dict:
    """Compute perplexity using sliding window.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        text: Text to evaluate (uses default Thai text if None).
        window_size: Sliding window size in tokens.
        stride: Stride for sliding window.

    Returns:
        Dict with token perplexity and character perplexity.
    """
    if text is None:
        text = THAI_EVAL_TEXT

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + window_size, seq_len)
        target_len = end_loc - begin_loc

        input_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            nll = outputs.loss.item()

        # The loss is averaged over the sequence, so multiply by length
        # Subtract 1 because labels are shifted
        num_tokens = target_len - 1
        if num_tokens > 0:
            total_nll += nll * num_tokens
            total_tokens += num_tokens

        if end_loc >= seq_len:
            break

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    token_ppl = math.exp(avg_nll) if avg_nll < 100 else float("inf")

    # Character perplexity: adjust by average characters per token
    num_chars = len(text)
    chars_per_token = num_chars / total_tokens if total_tokens > 0 else 1.0
    char_ppl = token_ppl ** (1.0 / chars_per_token) if chars_per_token > 0 else float("inf")

    results = {
        "perplexity": token_ppl,
        "char_perplexity": char_ppl,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "total_chars": num_chars,
        "chars_per_token": chars_per_token,
    }

    print(f"Perplexity Results:")
    print(f"  Token PPL:     {token_ppl:.2f}")
    print(f"  Char PPL:      {char_ppl:.2f}")
    print(f"  Avg NLL:       {avg_nll:.4f}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Chars/token:   {chars_per_token:.2f}")

    return results


def evaluate(
    model,
    tokenizer,
    **kwargs,
) -> dict:
    """Evaluation entry point compatible with run_eval.

    Args:
        model: The language model.
        tokenizer: The tokenizer.

    Returns:
        Dict with perplexity results.
    """
    return compute_perplexity(model, tokenizer)
