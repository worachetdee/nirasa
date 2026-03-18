"""Download all Thai training data from legally clear sources.

Data sources organized by tier:
  Tier 1 (Core): Large-scale, high-quality
  Tier 2 (Curated): Smaller but legally pristine (CC/public domain)
  Tier 3 (Domain): Specialized domains for diversity
"""
import json
import os
import time
import sys

os.makedirs("data/raw", exist_ok=True)

from datasets import load_dataset


def save_jsonl(dataset, output_path, text_field="text", max_docs=None, min_len=50):
    """Save dataset to JSONL format."""
    count = 0
    with open(output_path, "w") as f:
        for row in dataset:
            text = row.get(text_field, "")
            if text and len(text.strip()) > min_len:
                f.write(json.dumps({"id": f"doc_{count}", "text": text.strip()}, ensure_ascii=False) + "\n")
                count += 1
                if max_docs and count >= max_docs:
                    break
                if count % 50000 == 0:
                    print(f"    {count:,} docs...")
    return count


def report_size(path):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = sum(1 for _ in open(path))
        print(f"  -> {lines:,} docs, {size_mb:.1f} MB")
        return lines, size_mb
    return 0, 0


def download_source(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        print(f"  Time: {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  FAILED: {e}")


# =============================================================
# TIER 1: Core large-scale datasets
# =============================================================
print("\n" + "#"*60)
print("# TIER 1: Core datasets")
print("#"*60)

def dl_wikipedia():
    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
    count = save_jsonl(ds, "data/raw/th_wiki.jsonl")
    report_size("data/raw/th_wiki.jsonl")

def dl_mc4():
    ds = load_dataset("allenai/c4", "th", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_mc4.jsonl", max_docs=500000)
    report_size("data/raw/th_mc4.jsonl")

def dl_mangosteen_web():
    """WangchanLION web corpus — Mangosteen's cleaned Common Crawl for Thai."""
    ds = load_dataset("aisingapore/WangchanLION-Web", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_mangosteen_web.jsonl", max_docs=500000)
    report_size("data/raw/th_mangosteen_web.jsonl")

def dl_mangosteen_curated():
    """WangchanLION curated — Mangosteen's non-web Thai sources (CC/public domain)."""
    ds = load_dataset("aisingapore/WangchanLION-Curated", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_mangosteen_curated.jsonl", max_docs=500000)
    report_size("data/raw/th_mangosteen_curated.jsonl")

download_source("[1/14] Thai Wikipedia (CC BY-SA)", dl_wikipedia)
download_source("[2/14] mC4 Thai (ODC-BY)", dl_mc4)
download_source("[3/14] Mangosteen Web — cleaned Common Crawl (permissive)", dl_mangosteen_web)
download_source("[4/14] Mangosteen Curated — CC/public domain (permissive)", dl_mangosteen_curated)


# =============================================================
# TIER 2: Curated CC / Public Domain datasets
# =============================================================
print("\n" + "#"*60)
print("# TIER 2: Curated (CC / Public Domain)")
print("#"*60)

def dl_thai_law():
    ds = load_dataset("pythainlp/thailaw-v1.0", split="train")
    count = save_jsonl(ds, "data/raw/th_law.jsonl")
    report_size("data/raw/th_law.jsonl")

def dl_thai_gov():
    ds = load_dataset("pythainlp/thaigov-v2-corpus-31032024", split="train")
    count = save_jsonl(ds, "data/raw/th_gov.jsonl")
    report_size("data/raw/th_gov.jsonl")

def dl_thai_constitution():
    ds = load_dataset("pythainlp/thai-constitution-corpus", split="train")
    count = save_jsonl(ds, "data/raw/th_constitution.jsonl")
    report_size("data/raw/th_constitution.jsonl")

def dl_thai_open_data():
    ds = load_dataset("pythainlp/thai-open-data-text-v1", split="train")
    count = save_jsonl(ds, "data/raw/th_opendata.jsonl")
    report_size("data/raw/th_opendata.jsonl")

def dl_thai_oldbooks():
    ds = load_dataset("pythainlp/thai-oldbooks", split="train")
    count = save_jsonl(ds, "data/raw/th_oldbooks.jsonl")
    report_size("data/raw/th_oldbooks.jsonl")

download_source("[5/14] Thai Law (public domain)", dl_thai_law)
download_source("[6/14] Thai Government corpus (public domain)", dl_thai_gov)
download_source("[7/14] Thai Constitution (public domain)", dl_thai_constitution)
download_source("[8/14] Thai Open Data (public domain)", dl_thai_open_data)
download_source("[9/14] Thai Old Books (public domain/CC)", dl_thai_oldbooks)


# =============================================================
# TIER 3: Additional large-scale datasets
# =============================================================
print("\n" + "#"*60)
print("# TIER 3: Additional large-scale (web crawl)")
print("#"*60)

def dl_oscar():
    """OSCAR Thai — large web crawl, gated (needs HF login)."""
    ds = load_dataset("oscar-corpus/OSCAR-2301", "th", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_oscar.jsonl", max_docs=500000)
    report_size("data/raw/th_oscar.jsonl")

def dl_cc100():
    """CC-100 Thai — deduplicated web crawl used to train XLM-R."""
    ds = load_dataset("cc100", lang="th", split="train", streaming=True)
    count = save_jsonl(ds, "data/raw/th_cc100.jsonl", max_docs=500000)
    report_size("data/raw/th_cc100.jsonl")

download_source("[10/14] OSCAR Thai — web crawl (CC0 metadata, gated)", dl_oscar)
download_source("[11/14] CC-100 Thai — deduplicated web (no IP claims)", dl_cc100)


# =============================================================
# TIER 4: Domain-specific datasets
# =============================================================
print("\n" + "#"*60)
print("# TIER 4: Domain-specific")
print("#"*60)

def dl_wisesight():
    ds = load_dataset("wisesight_sentiment", split="train")
    count = 0
    with open("data/raw/th_wisesight.jsonl", "w") as f:
        for row in ds:
            text = row.get("texts", "")
            if text and len(text.strip()) > 20:
                f.write(json.dumps({"id": f"ws_{count}", "text": text.strip()}, ensure_ascii=False) + "\n")
                count += 1
    report_size("data/raw/th_wisesight.jsonl")

def dl_prachatai():
    """Prachatai-67K — Thai news articles (Apache 2.0)."""
    ds = load_dataset("PyThaiNLP/prachathai67k", split="train")
    count = 0
    with open("data/raw/th_prachatai.jsonl", "w") as f:
        for row in ds:
            title = row.get("title", "")
            body = row.get("body_text", row.get("text", ""))
            text = f"{title}\n{body}" if title and body else (body or title or "")
            if text and len(text.strip()) > 50:
                f.write(json.dumps({"id": f"pt_{count}", "text": text.strip()}, ensure_ascii=False) + "\n")
                count += 1
    report_size("data/raw/th_prachatai.jsonl")

def dl_wongnai():
    """Wongnai reviews — colloquial Thai (LGPL-3.0)."""
    ds = load_dataset("Wongnai/wongnai_reviews", split="train")
    count = 0
    with open("data/raw/th_wongnai.jsonl", "w") as f:
        for row in ds:
            text = row.get("review_body", row.get("text", ""))
            if text and len(text.strip()) > 30:
                f.write(json.dumps({"id": f"wn_{count}", "text": text.strip()}, ensure_ascii=False) + "\n")
                count += 1
    report_size("data/raw/th_wongnai.jsonl")

download_source("[12/14] Wisesight Sentiment — social media (CC0)", dl_wisesight)
download_source("[13/14] Prachatai-67K — news journalism (Apache 2.0)", dl_prachatai)
download_source("[14/14] Wongnai Reviews — colloquial/opinion (LGPL-3.0)", dl_wongnai)


# =============================================================
# Summary
# =============================================================
print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)
total_size = 0
total_docs = 0
for fname in sorted(os.listdir("data/raw")):
    if fname.endswith(".jsonl"):
        path = os.path.join("data/raw", fname)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = sum(1 for _ in open(path))
        total_size += size_mb
        total_docs += lines
        print(f"  {fname:<35s} {size_mb:>8.1f} MB  {lines:>10,} docs")
print(f"  {'─'*55}")
print(f"  {'TOTAL':<35s} {total_size:>8.1f} MB  {total_docs:>10,} docs")
