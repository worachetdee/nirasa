# Nirasa (นิราศ) — Thai Language Model

> *"นิราศ" — a traditional Thai literary genre of journey poems, reflecting the model's journey from a multilingual foundation to Thai specialization.*

Continued pretraining of **Qwen2.5-7B** on large-scale Thai corpora using LoRA, following the [Swallow methodology](https://arxiv.org/abs/2404.17790) for language adaptation.

## Highlights

- **Base model**: Qwen2.5-7B (7.7B parameters, bfloat16)
- **Training**: LoRA (r=64, 2% trainable params) — trainable on a single A100
- **Data**: 3.7 GB+ Thai text from Wikipedia, mC4, Wisesight, and more
- **Eval**: ThaiQA, XNLI-th, Wisesight sentiment, perplexity
- **Cost**: Under $10 on Google Colab Pro
- **Runs locally**: Inference on M4 Max MacBook or RTX 4090

## Why Nirasa?

Thai is underserved in the open-source LLM space. While Japanese has Swallow, PLaMo, and Japanese-StableLM, Thai has very few dedicated models. Nirasa aims to change that by providing:

1. A high-quality Thai LLM built on a strong multilingual foundation
2. A fully open-source training pipeline anyone can reproduce
3. Research into Thai tokenization, data efficiency, and cross-lingual transfer

## Training Data

| Source | Type | Docs | Size |
|--------|------|------|------|
| Thai Wikipedia | Encyclopedia | 159K | 939 MB |
| mC4 Thai | Web text | 500K | 2.8 GB |
| Wisesight | Social media (CC0) | 16K | 5 MB |

**14 total sources organized in 4 tiers:**

| Tier | Source | License | Type |
|------|--------|---------|------|
| **1. Core** | Thai Wikipedia | CC BY-SA | Encyclopedia |
| | mC4 Thai | ODC-BY | Web text |
| | Mangosteen Web | Permissive | Cleaned Common Crawl |
| | Mangosteen Curated | CC / Public Domain | Non-web Thai sources |
| **2. Curated** | Thai Law | Public Domain | Legal statutes |
| | Thai Gov corpus | Public Domain | Government documents |
| | Thai Constitution | Public Domain | Constitutional texts |
| | Thai Open Data | Public Domain | Government open data |
| | Thai Old Books | Public Domain / CC | Classical literature |
| **3. Web crawl** | OSCAR Thai | CC0 (gated) | Large web crawl (~12 GB) |
| | CC-100 Thai | No IP claims | Deduplicated web |
| **4. Domain** | Wisesight | CC0 | Social media |
| | Prachatai-67K | Apache 2.0 | News journalism |
| | Wongnai | LGPL-3.0 | Restaurant reviews |

All data sources are either public domain, Creative Commons licensed, or standard web crawl corpora used by major LLMs (GPT, LLaMA, BLOOM, XLM-R). See [Mangosteen](https://github.com/vistec-AI/Mangosteen) by VISTEC and the [PyThaiNLP pretrained datasets collection](https://huggingface.co/collections/pythainlp/datasets-for-pretrained-thai-llm-65db96ab730386b492889a98) for curated sources.

### Data Pipeline

```
Raw text → NFKC normalize → BOM removal → HTML/URL removal → Control char removal
        → Collapse whitespace → Thai ratio filter → MinHash dedup (char 5-grams)
        → Quality filter (length, repetition, line length) → Tokenize (Qwen) → Binary
```

## Architecture

| Component | Details |
|-----------|---------|
| Base model | Qwen/Qwen2.5-7B |
| Adaptation | LoRA (r=64, alpha=128) |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable params | 161M / 7.7B (2.08%) |
| Precision | bfloat16 |
| Optimizer | AdamW (lr=2e-4, cosine schedule) |
| Sequence length | 512 tokens |

## Quick Start

### On Google Colab (recommended)

```bash
# Clone and install
!git clone https://github.com/worachetdee/nirasa.git /content/nirasa
!cd /content/nirasa && pip install torch transformers peft datasets accelerate -q

# Download Thai data
!cd /content/nirasa && python scripts/download_wiki_th.py

# Run data pipeline
!cd /content/nirasa && PYTHONPATH=/content/nirasa python scripts/run_data_pipeline.py

# Tokenize with Qwen tokenizer
!cd /content/nirasa && PYTHONPATH=/content/nirasa python scripts/prepare_data_colab.py

# Train (saves checkpoints to Google Drive)
!cd /content/nirasa && PYTHONPATH=/content/nirasa python scripts/train_colab.py

# Test generation
!cd /content/nirasa && PYTHONPATH=/content/nirasa python scripts/test_generate.py
```

### Local Development

```bash
git clone https://github.com/worachetdee/nirasa.git
cd nirasa
pip install -e ".[dev]"

# Run smoke test
make smoke-test

# Full pipeline
make data-pipeline
make pretrain
make eval
make serve
```

## Evaluation Benchmarks

| Benchmark | Type | Metric | Description |
|-----------|------|--------|-------------|
| ThaiQA | Question Answering | Char F1, EM | Reading comprehension |
| XNLI-th | NLI | Accuracy | Entailment / contradiction / neutral |
| Wisesight | Sentiment | Accuracy | Positive / negative / neutral / question |
| Perplexity | Language Modeling | PPL | Held-out Thai text |

## Project Structure

```
nirasa/
├── nirasa/
│   ├── data/            # Download, clean, dedup, filter, tokenize
│   ├── tokenizer/       # SentencePiece training & evaluation
│   ├── training/        # LoRA training with Qwen2.5-7B
│   ├── eval/            # ThaiQA, XNLI-th, Wisesight, perplexity
│   └── serving/         # FastAPI server, generation, chat template
├── scripts/             # Colab-friendly run scripts
├── configs/             # Training & tokenizer YAML configs
├── tests/               # Unit tests
├── docker/              # Train & serve Dockerfiles
├── ROADMAP.md           # Project roadmap & research directions
├── pyproject.toml
└── Makefile
```

## Scaling Path

```
Nirasa-7B   →  Current. Colab Pro ($10/mo), single A100.
Nirasa-14B  →  Qwen2.5-14B base, 2x A100 (~$500).
Nirasa-72B  →  Qwen2.5-72B base, 8x H100 (~$5K-10K).
```

Same codebase, same pipeline — just change the model config and add GPUs.

## Research Questions

### RQ1: Does a Chinese-trained base model transfer better to Thai?

Thai is linguistically closer to Chinese than to English — both are tonal, analytic (no inflection), SVO word order, no spaces between words, and share thousands of loanwords. We hypothesize that a Chinese-heavy base model (Qwen) transfers better to Thai than English-heavy models (LLaMA, Mistral).

| Feature | Thai | Chinese | Japanese | English |
|---------|------|---------|----------|---------|
| Tonal | 5 tones | 4 tones | No | No |
| Word order | SVO | SVO | SOV | SVO |
| Word boundaries | No spaces | No spaces | No spaces | Spaces |
| Grammar | Analytic | Analytic | Agglutinative | Fusional |
| Shared loanwords | — | Many | Some | Few |

**Experiment:** Run identical continued pretraining on Qwen2.5-7B vs LLaMA 3.1-8B vs Mistral-7B. Compare loss curves and downstream benchmark scores. No one has done this systematic comparison for Thai.

### RQ2: Thai tokenizer design

SentencePiece BPE vs morpheme-aware tokenization (PyThaiNLP) vs character-level. Thai's lack of word boundaries makes tokenizer design critical. Compare fertility, downstream task performance, and training efficiency.

### RQ3: Data efficiency for low-resource adaptation

How much Thai data is actually needed to meaningfully improve a multilingual model? Plot loss and benchmark scores at 100M, 500M, 1B, 5B, and 10B tokens. This has direct implications for all low-resource languages.

### RQ4: Thai-English bilingual training

Does continued pretraining on Thai degrade English performance? Can mixed Thai-English training maintain both? Measure cross-lingual transfer effects.

## Sister Project

**[Zensei (禅精)](https://github.com/worachetdee/zenzei)** — Same methodology applied to Japanese with DeepSeek-V3 / Qwen2.5-7B. First training complete, generates fluent Japanese.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba — base model
- [Swallow](https://arxiv.org/abs/2404.17790) by TokyoTech — methodology for language adaptation
- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) — Thai NLP tools
