"""Download multiple Thai text corpora for training."""
import os
import json
import time

os.makedirs("data/raw", exist_ok=True)

def save_jsonl(dataset, output_path, text_field="text", max_docs=None):
    """Save dataset to JSONL format."""
    count = 0
    with open(output_path, "w") as f:
        for row in dataset:
            text = row.get(text_field, "")
            if text and len(text.strip()) > 50:
                f.write(json.dumps({"id": f"doc_{count}", "text": text}, ensure_ascii=False) + "\n")
                count += 1
                if max_docs and count >= max_docs:
                    break
                if count % 50000 == 0:
                    print(f"    {count:,} docs...")
    return count


from datasets import load_dataset

# ============================================================
# 1. Thai Wikipedia (~150K articles)
# ============================================================
print("=" * 50)
print("[1/5] Thai Wikipedia")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
    count = save_jsonl(ds, "data/raw/th_wiki.jsonl")
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 2. CulturaX Thai (cleaned Common Crawl + OSCAR)
# ============================================================
print()
print("=" * 50)
print("[2/5] CulturaX Thai (500K sample)")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("uonlp/CulturaX", "th", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_culturax.jsonl", max_docs=500000)
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 3. mC4 Thai
# ============================================================
print()
print("=" * 50)
print("[3/5] mC4 Thai (500K sample)")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("allenai/c4", "th", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_mc4.jsonl", max_docs=500000)
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 4. Thai National Corpus (Wisesight - social media)
# ============================================================
print()
print("=" * 50)
print("[4/5] Wisesight-1000 Thai social media")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("wisesight/wisesight-1000", split="train")
    count = save_jsonl(ds, "data/raw/th_wisesight.jsonl", text_field="texts")
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# 5. Thai Textbook QA (iApp)
# ============================================================
print()
print("=" * 50)
print("[5/5] iApp Thai Wikipedia QA")
print("=" * 50)
t0 = time.time()
try:
    ds = load_dataset("iapp_wiki_qa_squad", split="train", trust_remote_code=True)
    count = 0
    with open("data/raw/th_iapp_qa.jsonl", "w") as f:
        for row in ds:
            context = row.get("context", "")
            if context and len(context.strip()) > 50:
                f.write(json.dumps({"id": f"doc_{count}", "text": context}, ensure_ascii=False) + "\n")
                count += 1
    print(f"  Saved {count:,} docs ({time.time()-t0:.0f}s)")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 50)
print("Download Summary")
print("=" * 50)
total_size = 0
for fname in sorted(os.listdir("data/raw")):
    if fname.endswith(".jsonl") or fname.endswith(".txt"):
        path = os.path.join("data/raw", fname)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = sum(1 for _ in open(path))
        total_size += size_mb
        print(f"  {fname}: {size_mb:.1f} MB, {lines:,} docs")
print(f"  TOTAL: {total_size:.1f} MB")
