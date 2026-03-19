# Nirasa Local — Thai AI That Runs on Your Device

> No cloud. No API key. No cost.

Build the best **small Thai LLM** that runs locally on phones, laptops, and edge devices. Same data, same pipeline — just smaller and optimized for inference.

---

## Target Models

| Model | Params | Target Device | RAM Needed (Q4) | Use Case |
|-------|--------|---------------|-----------------|----------|
| **Nirasa-0.5B** | 500M | Phones, Raspberry Pi | ~400 MB | Autocomplete, simple QA |
| **Nirasa-1.5B** | 1.5B | Phones, cheap laptops | ~1 GB | Chat, translation, summarization |
| **Nirasa-3B** | 3B | Any laptop, tablets | ~2 GB | General assistant, coding help |
| **Nirasa-7B** | 7B | MacBook, gaming PC | ~4.5 GB | Full capability, research baseline |

Base models: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B

---

## Distribution Channels

| Channel | Format | Platform |
|---------|--------|----------|
| **Ollama** | GGUF | `ollama run nirasa` — Mac, Linux, Windows |
| **LM Studio** | GGUF | One-click download — Mac, Linux, Windows |
| **Apple MLX** | MLX | Native M-series performance |
| **HuggingFace** | Safetensors + GGUF | Direct download, API |
| **nirasa.org** | All formats | Download page + web demo |

---

## Phase 1: Data Pipeline (In Progress)

- [x] Download 11 Thai sources (17.5 GB, 1.7M docs)
- [x] Clean (NFKC, HTML/URL removal, Thai ratio filter)
- [x] Deduplicate (MinHash char 5-grams)
- [ ] Quality filter (length, repetition, Thai ratio)
- [ ] Tokenize with Qwen tokenizer → binary memmap
- [ ] Create train/validation split (99/1)

**Output:** Single processed dataset used for all model sizes.

---

## Phase 2: Train Nirasa-1.5B (Start Here)

Why 1.5B first:
- Small enough to train fast on Colab A100 (~2h for 5000 steps)
- Large enough to produce useful Thai output
- Runs on any laptop — broadest audience
- Fast iteration for validating the pipeline before scaling up/down

### 2a: Continued Pretraining
- [ ] Base: `Qwen/Qwen2.5-1.5B`
- [ ] Full fine-tuning (NOT LoRA — model is small enough)
- [ ] Seq length: 2048
- [ ] Train 5000-10000 steps on full dataset
- [ ] Monitor loss curve, compare vs base Qwen2.5-1.5B
- [ ] Estimated cost: ~$3-5 on Colab Pro

### 2b: Instruction Tuning
- [ ] Collect Thai instruction data:
  - PyThaiNLP Thai instruction dataset collection (HuggingFace)
  - Translate Alpaca/OpenHermes subset to Thai
  - Thai-specific: cooking, Buddhism, government services, travel
- [ ] Fine-tune on chat/instruction format
- [ ] Add Qwen chat template support
- [ ] Test conversational quality

### 2c: Quantize & Package
- [ ] Convert to GGUF (Q4_K_M, Q5_K_M, Q8_0)
- [ ] Convert to MLX format
- [ ] Benchmark: tokens/sec on M4 Max, RTX 5090, M1 Air, iPhone 15
- [ ] Measure quality degradation per quantization level

### 2d: Validate
- [ ] Run ThaiQA, XNLI-th, Wisesight, perplexity
- [ ] Compare: base Qwen2.5-1.5B vs Nirasa-1.5B vs Typhoon 1B
- [ ] Test real-world prompts: Thai chat, QA, summarization, translation
- [ ] If quality is good → proceed to Phase 3
- [ ] If quality is bad → diagnose before training other sizes

---

## Phase 3: Train Full Family

Only after 1.5B is validated:

### Nirasa-3B
- [ ] Full fine-tune `Qwen/Qwen2.5-3B`
- [ ] Same data, same pipeline
- [ ] ~4h on Colab A100
- [ ] Quantize + benchmark

### Nirasa-0.5B
- [ ] Full fine-tune `Qwen/Qwen2.5-0.5B`
- [ ] Focus on fast inference, basic Thai capability
- [ ] Target: phones and edge devices
- [ ] Quantize + benchmark

### Nirasa-7B
- [ ] Full fine-tune `Qwen/Qwen2.5-7B` (or LoRA if memory-constrained)
- [ ] This is the "flagship" quality model
- [ ] Quantize + benchmark

---

## Phase 4: Ollama & LM Studio Release

- [ ] Create Ollama Modelfile for each size
- [ ] Test `ollama run nirasa:1.5b`, `nirasa:3b`, `nirasa:7b`
- [ ] Upload GGUF to HuggingFace (all quant levels)
- [ ] Submit to Ollama library (official registry)
- [ ] Submit to LM Studio catalog
- [ ] Create MLX model for Apple Silicon users
- [ ] Write model cards with benchmarks, recommended quant per device

---

## Phase 5: nirasa.org

### Landing Page
- [ ] Hero: "Thai AI that runs on your device"
- [ ] One-click install instructions (Ollama, LM Studio)
- [ ] Model picker: choose size based on your device
- [ ] Live demo (chat UI powered by Nirasa-7B on backend)
- [ ] Benchmarks comparison table
- [ ] Link to GitHub, HuggingFace, paper

### Chat Demo
- [ ] ChatGPT-style web UI
- [ ] Streaming responses
- [ ] Powered by Nirasa-7B on a GPU server
- [ ] Disclaimer: "For best privacy, download and run locally"

### Download Page
- [ ] GGUF files for each model size and quant level
- [ ] MLX files for Apple Silicon
- [ ] Direct download + HuggingFace links
- [ ] Size and device compatibility table

---

## Phase 6: Community & Launch

- [ ] Blog post: "Why we built a local Thai LLM"
- [ ] Post on Thai AI communities, Reddit r/LocalLLaMA, Hacker News
- [ ] Thai developer meetup demo
- [ ] GitHub Discussions for feedback
- [ ] Accept contributions (data, evals, translations)

---

## Phase 7: Research Paper

Findings worth publishing (ACL/EMNLP workshop):

1. **Base model selection for Thai** — Qwen vs LLaMA vs Mistral at 1.5B/3B scale
2. **Data efficiency** — loss curves at 100M, 500M, 1B, 3B tokens across model sizes
3. **Full fine-tuning vs LoRA for small models** — when does LoRA make sense?
4. **Quantization impact on Thai** — does Thai degrade more than English at Q4?
5. **Scaling laws for Thai** — how do 0.5B/1.5B/3B/7B compare on Thai benchmarks?

---

## Timeline Estimate (With Claude Code Helping)

| Phase | Work Sessions | GPU Time | Wall Clock |
|-------|-------------|----------|------------|
| 1. Data pipeline (finish) | 1-2h | 2-3h local | 1 day |
| 2. Train Nirasa-1.5B | 3-4h | 4-6h Colab | 3-5 days |
| 3. Train full family | 3-4h | 10-15h Colab | 1 week |
| 4. Ollama/LM Studio release | 2-3h | — | 2 days |
| 5. nirasa.org | 4-6h | — | 3-5 days |
| 6. Launch | 1-2h | — | 1 day |
| 7. Paper (optional) | 5-10h | — | 2-4 weeks |
| **Total** | **~20-30h** | **~20h Colab** | **~3-5 weeks** |

Estimated Colab cost: ~$15-25 total for all sizes.

---

## Key Decisions

### Why full fine-tuning instead of LoRA (for small models)?
- At 0.5B-3B, the entire model fits in A100 memory for full fine-tuning
- LoRA on the v2 7B run showed quality degradation — small models need every parameter updated
- Full fine-tuning is simpler and more effective at this scale

### Why 1.5B first?
- Sweet spot: useful quality + runs anywhere
- Fastest to validate the pipeline
- If 1.5B fails, we diagnose before wasting GPU on larger models
- If 1.5B succeeds, we have confidence the pipeline works

### Why Qwen2.5 base?
- Validated by OpenThaiGPT 1.5 (same base, good Thai results)
- Chinese-Thai linguistic overlap (tonal, analytic, no word boundaries)
- Full size range (0.5B to 72B) from a single family
- Strong multilingual baseline already includes some Thai
