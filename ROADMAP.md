# Nirasa (นิราศ) — Roadmap

## Vision
Build the best open-source Thai language model by continued pretraining of Qwen2.5-7B on diverse Thai data, with a clear path to scale up.

---

## Completed

### Phase 1: Infrastructure
- [x] Project repo created (40 files, full pipeline)
- [x] GitHub: github.com/worachetdee/nirasa
- [x] Data pipeline: download → clean → dedup → filter → tokenize
- [x] Training script: Qwen2.5-7B + LoRA on Colab A100
- [x] Eval suite: ThaiQA, XNLI-th, Wisesight, perplexity
- [x] Serving: FastAPI OpenAI-compatible server
- [x] Smoke test for validation

### Phase 2: Data Collection (In Progress)
- [x] Thai Wikipedia — 159K docs, 939 MB
- [x] mC4 Thai — 500K docs, 2.8 GB
- [x] Wisesight sentiment — 16K docs, 5 MB
- [ ] OSCAR Thai — ~12 GB (needs HF login, gated)
- [ ] CC-100 Thai — ~5 GB
- [ ] Prachatai-67K — ~300 MB (journalism)
- [ ] Wongnai — ~40 MB (reviews, colloquial)

### Phase 3: Data Pipeline (In Progress)
- [x] Clean (NFKC, HTML/URL removal, Thai ratio filter)
- [x] Dedup (MinHash char 5-grams)
- [x] Filter (length, quality, repetition)
- [ ] Tokenize with Qwen tokenizer → binary format
- [ ] Back up processed data to Google Drive

---

## Next Steps

### Phase 4: First Training Run (Colab A100)
- [ ] Tokenize filtered data with Qwen2.5-7B tokenizer
- [ ] Train LoRA (r=64) for 1000 steps as validation
- [ ] Test generation with Thai prompts
- [ ] Compare output vs base Qwen2.5-7B
- **Estimated cost:** ~$1 (47 min on Colab Pro)

### Phase 5: Add More Data + Retrain
- [ ] Set up HuggingFace CLI authentication
- [ ] Download OSCAR Thai (~12 GB)
- [ ] Download CC-100 Thai (~5 GB)
- [ ] Download Prachatai-67K, Wongnai
- [ ] Re-run pipeline on combined ~20 GB corpus
- [ ] Train for 5000-10000 steps
- **Target:** loss < 1.8 on Thai held-out text

### Phase 6: Evaluation & Benchmarking
- [ ] Run ThaiQA (reading comprehension, char F1)
- [ ] Run XNLI-th (natural language inference, accuracy)
- [ ] Run Wisesight (sentiment classification, accuracy)
- [ ] Run perplexity on held-out Thai text
- [ ] Compare: base Qwen vs Nirasa on all benchmarks
- [ ] Publish results

### Phase 7: Model Release
- [ ] Merge LoRA weights into base model
- [ ] Upload to HuggingFace Hub as `worachetdee/nirasa-7b-th`
- [ ] Write model card with benchmarks
- [ ] Demo via API server

---

## Future (When Budget Allows)

### Scale Up Path
```
Nirasa-7B   → current, Colab Pro ($10/mo)
Nirasa-14B  → Qwen2.5-14B base, 2x A100 (~$500)
Nirasa-72B  → Qwen2.5-72B base, 8x H100 (~$5K-10K)
```

### Research Directions
1. **Thai tokenizer study** — SentencePiece vs morpheme-aware (PyThaiNLP) vs character-level. Compare downstream impact. Publishable at ACL/EMNLP.
2. **Cross-lingual transfer** — Does Chinese knowledge in Qwen help Thai? Systematic comparison of base models (Qwen vs LLaMA vs Mistral) for Thai adaptation.
3. **Low-resource data efficiency** — How much Thai data is actually needed? Plot loss curves at 1B, 5B, 10B, 50B tokens.
4. **Thai-English bilingual** — Mixed training on Thai + English to maintain English capability.
5. **Full fine-tuning vs LoRA** — Compare quality at 7B scale when budget allows.

### Additional Data Sources to Explore
- Thai National Corpus (TNC) — academic, may need permission
- ThaiGov gazette — government documents (public domain)
- Chula Thai Archives — university research texts
- Thai social media (Twitter/X, Pantip) — needs scraping

---

## Hardware & Cost Summary

| Setup | What it can do | Cost |
|-------|---------------|------|
| M4 Max 36GB (current) | Dev, testing, small inference | Already owned |
| Colab Pro | LoRA training on A100 | $10/mo |
| RTX 5090 32GB (gaming PC) | Local training, inference | Already owned |
| Cloud 8x H100 | 72B model training | ~$5K-10K per run |

**Current approach:** M4 Max for dev + Colab Pro for training = most cost-effective.

---

## Sister Project
- **Zensei (禅精)** — Same methodology applied to Japanese
- Repo: github.com/worachetdee/zenzei
- Status: First training complete, generates fluent Japanese
- Shared codebase architecture
