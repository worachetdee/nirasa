# Nirasa (นิราศ) — Thai Language Model

Continued pretraining of Qwen2.5-7B on Thai data, following the Swallow methodology for language adaptation.

## Overview

Nirasa applies the Swallow approach — originally developed for adapting LLMs to Japanese — to the Thai language. We perform continued pretraining of Qwen2.5-7B on large-scale Thai corpora with LoRA, preserving the base model's multilingual capabilities while significantly improving Thai language understanding and generation.

The name "นิราศ" (Nirasa) refers to a traditional Thai literary genre of travel poetry, reflecting the model's journey from a multilingual foundation to Thai specialization.

## Architecture

- **Base model**: Qwen/Qwen2.5-7B
- **Adaptation**: LoRA (Low-Rank Adaptation)
  - Rank: 64
  - Alpha: 128
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision**: bfloat16
- **Max sequence length**: 512 (configurable)

## Data

| Corpus | Description | Size |
|--------|-------------|------|
| Thai Wikipedia | `wikimedia/wikipedia` (20231101.th) | ~200K articles |
| OSCAR-th | Thai subset of OSCAR | Large-scale web crawl |

### Data Pipeline

1. **Download** — Fetch corpora via HuggingFace `datasets`
2. **Clean** — NFKC normalization, HTML/URL removal, Thai ratio filtering
3. **Dedup** — MinHash LSH near-deduplication (character 5-grams)
4. **Filter** — Length, repetition, and quality filtering
5. **Tokenize** — Pack into uint32 memmap with Qwen tokenizer

## Evaluation

| Benchmark | Type | Metric |
|-----------|------|--------|
| ThaiQA | Question Answering | Character F1 |
| XNLI-th | Natural Language Inference | Accuracy |
| Wisesight | Sentiment Analysis | Accuracy |
| Perplexity | Language Modeling | PPL (token & char) |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run data pipeline
make data-pipeline

# Train with LoRA
make pretrain

# Evaluate
make eval

# Serve
make serve
```

## Project Structure

```
nirasa/
├── nirasa/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py      # Corpus download
│   │   ├── clean.py         # Text cleaning
│   │   ├── dedup.py         # MinHash dedup
│   │   ├── filter.py        # Quality filtering
│   │   └── prepare.py       # Tokenization & packing
│   ├── tokenizer/
│   │   ├── train_tokenizer.py
│   │   └── eval_tokenizer.py
│   ├── training/
│   │   └── train.py         # LoRA training
│   ├── eval/
│   │   ├── run_eval.py      # Unified eval runner
│   │   ├── thaiqa.py        # Thai QA eval
│   │   ├── xnli_th.py       # XNLI Thai eval
│   │   ├── wisesight.py     # Sentiment eval
│   │   └── perplexity.py    # Perplexity eval
│   └── serving/
│       ├── api_server.py     # FastAPI server
│       ├── generate.py       # Text generation
│       └── chat_template.py  # Chat formatting
├── scripts/
│   ├── download_wiki_th.py
│   ├── run_data_pipeline.py
│   ├── prepare_data_colab.py
│   ├── train_colab.py
│   ├── test_generate.py
│   └── smoke_test.py
├── configs/
│   ├── training/lora_finetune.yaml
│   └── tokenizer/tokenizer_train.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── tests/
│   ├── test_data_pipeline.py
│   └── test_eval.py
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
