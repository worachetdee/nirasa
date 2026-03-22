# Nirasa (นิราศ) — Roadmap

## Vision
Build the best open-source Thai language model, ship it as a **ChatGPT-style web UI at nirasa.org**, and open-source everything so others can reproduce, improve, and build on it.

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

### Phase 2: Data Collection
- [x] Thai Wikipedia — 159K docs, 939 MB (CC BY-SA)
- [x] mC4 Thai — 500K docs, 2.8 GB (ODC-BY)
- [x] Mangosteen Web — 500K docs, 4.3 GB (permissive)
- [x] Mangosteen Curated — 393K docs, 8.1 GB (CC/public domain)
- [x] Thai Law — 52K docs, 871 MB (public domain)
- [x] Thai Government — 44K docs, 343 MB (public domain)
- [x] Thai Constitution — 20 docs, 3.8 MB (public domain)
- [x] Thai Open Data — 21 docs, 2 MB (public domain)
- [x] Thai Old Books — 75 docs, 89 MB (public domain/CC)
- [x] Wisesight — 16K docs, 5 MB (CC0)
- [x] Wongnai Reviews — 40K docs, 59 MB (LGPL-3.0)
- **Total: 11 sources, 1.7M docs, 17.5 GB**

**Not available (upstream issues):**
- [ ] OSCAR Thai — gated, needs access approval at huggingface.co/datasets/oscar-corpus/OSCAR-2301
- [ ] CC-100 Thai — HF deprecated dataset script, broken upstream
- [ ] Prachatai-67K — same, broken upstream

### Phase 3: v2 Training (Wiki-Only Baseline)
- [x] Tokenize Thai Wikipedia with Qwen tokenizer (134M tokens)
- [x] Train LoRA for 1000 steps on Colab A100 (0.8h, ~$1)
- [x] Loss: 1.60 → 1.42 (plateaued ~step 700)
- [x] Test generation vs base Qwen2.5-7B
- **Result: LoRA outputs worse than base model — wiki-only data insufficient**

### Phase 4: Full Data Pipeline (In Progress)
- [x] Clean all 11 datasets (HTML/URL removal, NFKC, BOM removal, Thai ratio filter)
- [ ] Deduplicate (MinHash char 5-grams, threshold 0.8) — bug fixed, needs re-run
- [ ] Quality filter (length, repetition, Thai ratio)
- [ ] Tokenize combined corpus with Qwen tokenizer → binary memmap
- [ ] Back up processed data to Google Drive

### Bug Fixes & Infrastructure (Completed)
- [x] Fix BOM character not stripped by clean_text (NFKC doesn't remove U+FEFF)
- [x] Fix MinHash dedup unreliable with low num_perm (tests now use 128)
- [x] Fix filter tests using repetitive text that triggered repetition filter
- [x] Fix API server returning HTTP 200 on errors (now returns 503)
- [x] Fix health endpoint always returning "ok" even when model not loaded
- [x] Add serving endpoint tests (health, models, chat completions)
- [x] Fix training resume: use PeftModel.from_pretrained() instead of get_peft_model() + load_adapter()
- [x] Fix optimizer/scheduler state not restored on checkpoint resume
- [x] Align train_colab.py dataset with train.py (manual label shift, +1 token read)
- [x] Increase max_seq_len from 512 to 2048 (512 caused truncated outputs)
- [x] Add setuptools package discovery config to pyproject.toml

---

## Next Steps

### Phase 5: v3 Training — Full Dataset
- [ ] Train LoRA (r=64) for 5000 steps on full corpus
- [x] ~~Increase seq_len from 512 → 2048~~ (done — config and code updated)
- [ ] Monitor loss curve — expect lower floor than v2 with diverse data
- [ ] Test generation at step 1000, 2500, 5000
- [ ] Compare vs base Qwen and vs v2 wiki-only
- **Estimated cost:** ~$5-8 (4-6h on Colab Pro A100)

**Key changes from v2:**
- ~30x more training data (17.5 GB vs 605 MB filtered)
- Diverse domains (legal, government, social media, reviews, literature)
- seq_len 2048 (was 512, now fixed)
- Fixed checkpoint resume (LoRA weights + optimizer/scheduler state)

### Phase 6: Evaluation & Benchmarking
- [ ] Run ThaiQA (reading comprehension, char F1)
- [ ] Run XNLI-th (natural language inference, accuracy)
- [ ] Run Wisesight (sentiment classification, accuracy)
- [ ] Run perplexity on held-out Thai text
- [ ] Compare all three: base Qwen vs v2 (wiki) vs v3 (full)
- [ ] If v3 still underperforms base → diagnose (see Contingency below)

### Phase 7: Model Release
- [ ] Merge LoRA weights into base model
- [ ] Upload to HuggingFace Hub as `worachetdee/nirasa-7b-th`
- [ ] Write model card with benchmarks and training details
- [ ] Demo via API server
- [ ] Announce on Thai AI community channels

### Phase 8: Chat Fine-Tuning
- [ ] Collect Thai instruction/chat data (translate Alpaca, use existing Thai QA pairs)
- [ ] Fine-tune LoRA checkpoint on instruction-following format
- [ ] Add system prompt and chat template support
- [ ] Evaluate conversational quality (fluency, helpfulness, safety)

### Phase 9: Web UI (nirasa.org)
- [ ] Build chat frontend — React/Next.js, ChatGPT-style interface
  - Chat bubbles with streaming token display
  - Conversation history (local storage or DB)
  - Mobile responsive
  - Thai/English language toggle
- [ ] Production backend — harden existing FastAPI server
  - Streaming SSE (already partially implemented)
  - Rate limiting and API key auth
  - Proper HTTP error codes (fix current 200-on-error bug)
  - Health check that actually reflects model state
  - CORS configuration
- [ ] Deploy
  - GPU server (single A100/L4 or quantized on RTX 4090/5090)
  - Frontend on Vercel/Cloudflare Pages → nirasa.org
  - Backend API on GPU instance → api.nirasa.org
  - SSL, CDN, monitoring
- [ ] Landing page at nirasa.org
  - Project description and motivation
  - Live demo (chat UI)
  - Link to GitHub repo, HuggingFace model, benchmarks
  - "Try Nirasa" CTA

### Phase 10: Community & Launch
- [ ] HuggingFace model card with benchmarks
- [ ] Blog post: methodology, results, what we learned
- [ ] Share on Thai AI/ML communities, Hacker News, Reddit r/LocalLLaMA
- [ ] Set up GitHub Discussions for community feedback
- [ ] Accept contributions (data sources, eval benchmarks, translations)

---

## Contingency: If v3 Still Underperforms Base Qwen

If the full-dataset LoRA still doesn't beat base Qwen, investigate in this order:

1. **Learning rate too high** — try 5e-5 instead of 2e-4. LoRA on a model that already knows some Thai may need gentler updates.
2. **LoRA rank too high** — r=64 with alpha=128 is aggressive. Try r=16, alpha=32 for less catastrophic forgetting.
3. **Seq length** — confirm 2048 is being used. 512 alone could explain poor outputs.
4. **Data mixing** — ensure mangosteen_curated (8 GB) doesn't dominate. Consider upsampling smaller high-quality sources (law, wikipedia).
5. **Validate on smaller model first** — train Qwen2.5-1.5B with full fine-tuning (not LoRA) to validate the pipeline works. If 1.5B improves, the issue is LoRA capacity at 7B.
6. **Full fine-tuning** — if LoRA can't bridge the gap, consider QLoRA with 4-bit quantization to fit full fine-tuning on A100.

---

## Future (When Budget Allows)

### Scale Up Path
```
Nirasa-7B   → current, Colab Pro ($10/mo)
Nirasa-14B  → Qwen2.5-14B base, 2x A100 (~$500)
Nirasa-72B  → Qwen2.5-72B base, 8x H100 (~$5K-10K)
```

### Research Directions
1. **RQ1: Base model selection** — Does Chinese knowledge in Qwen help Thai? Systematic comparison of Qwen vs LLaMA vs Mistral for Thai adaptation. No one has published this.
2. **RQ2: Thai tokenizer study** — SentencePiece vs morpheme-aware (PyThaiNLP) vs character-level. Compare fertility, downstream impact. Publishable at ACL/EMNLP.
3. **RQ3: Data efficiency** — How much Thai data is needed? Plot loss and benchmarks at 100M, 500M, 1B, 5B tokens. Direct implications for all low-resource languages.
4. **RQ4: Bilingual training** — Does continued pretraining on Thai degrade English? Can mixed training maintain both?
5. **RQ5: LoRA vs full fine-tuning** — Compare quality at 7B scale. Quantify the LoRA tax for language adaptation.

---

## Hardware & Cost Summary

| Setup | What it can do | Cost |
|-------|---------------|------|
| M4 Max 36GB (current) | Dev, data pipeline, small inference | Already owned |
| Colab Pro A100 | LoRA training 7B | $10/mo |
| RTX 5090 32GB (gaming PC) | Local training, inference | Already owned |
| Cloud 8x H100 | 72B model training | ~$5K-10K per run |

**Current approach:** M4 Max for dev/data + Colab Pro for training = most cost-effective.

---

## Sister Project
- **Zensei (禅精)** — Same methodology applied to Japanese
- Repo: github.com/worachetdee/zenzei
- Status: First training complete, generates fluent Japanese
- Shared codebase architecture
