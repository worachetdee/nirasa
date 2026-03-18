"""Download additional Thai corpora: Wisesight, Prachatai, ThaiGov."""
import os
import json
import time

os.makedirs("data/raw", exist_ok=True)

from datasets import load_dataset

# ============================================================
# 1. Wisesight Sentiment (social media)
# ============================================================
print("=" * 50)
print("[1/3] Wisesight Sentiment (social media)")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("wisesight_sentiment", split="train")
    count = 0
    with open("data/raw/th_wisesight.jsonl", "w") as f:
        for row in ds:
            text = row.get("texts", "")
            if text and len(text.strip()) > 20:
                f.write(json.dumps({"id": f"ws_{count}", "text": text}, ensure_ascii=False) + "\n")
                count += 1
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 2. Prachatai News (Creative Commons)
# ============================================================
print()
print("=" * 50)
print("[2/3] Prachatai-67k (Thai news)")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("prachatai67k", split="train")
    count = 0
    with open("data/raw/th_prachatai.jsonl", "w") as f:
        for row in ds:
            title = row.get("title", "")
            body = row.get("body_text", "")
            text = f"{title}\n{body}" if title else body
            if text and len(text.strip()) > 50:
                f.write(json.dumps({"id": f"pt_{count}", "text": text}, ensure_ascii=False) + "\n")
                count += 1
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 3. ThaiGov (government corpus)
# ============================================================
print()
print("=" * 50)
print("[3/3] Thai Government Gazette")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("kobkrit/thai-government-gazette", split="train", streaming=True)
    count = 0
    with open("data/raw/th_thaigov.jsonl", "w") as f:
        for row in ds:
            text = row.get("text", "")
            if text and len(text.strip()) > 50:
                f.write(json.dumps({"id": f"gov_{count}", "text": text}, ensure_ascii=False) + "\n")
                count += 1
                if count >= 100000:
                    break
                if count % 20000 == 0:
                    print(f"    {count:,} docs...")
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")
    # Fallback: try another Thai government dataset
    print("  Trying alternative...")
    try:
        ds = load_dataset("Thaweewat/thai-text-corpus", split="train", streaming=True)
        count = 0
        with open("data/raw/th_thaitext.jsonl", "w") as f:
            for row in ds:
                text = row.get("text", "")
                if text and len(text.strip()) > 50:
                    f.write(json.dumps({"id": f"tt_{count}", "text": text}, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= 100000:
                        break
                    if count % 20000 == 0:
                        print(f"    {count:,} docs...")
        print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
    except Exception as e2:
        print(f"  ALSO FAILED: {e2}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 50)
print("Download Summary (all sources)")
print("=" * 50)
total_size = 0
total_docs = 0
for fname in sorted(os.listdir("data/raw")):
    if fname.endswith(".jsonl"):
        path = os.path.join("data/raw", fname)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = sum(1 for _ in open(path))
        total_size += size_mb
        total_docs += lines
        print(f"  {fname}: {size_mb:.1f} MB, {lines:,} docs")
print(f"  TOTAL: {total_size:.1f} MB, {total_docs:,} docs")
