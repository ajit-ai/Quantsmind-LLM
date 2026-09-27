# Quantsmind Coder — Execution Status Tracker

> **Direction update (2026-09-27):** project pivoted to a **coding LLM** for corporate use.
> Python + JavaScript/TypeScript specialist, 15–100M params, all use cases (completion, chat, review).
> See [CODE_LLM_STRATEGY](CODE_LLM_STRATEGY.md) for full strategy.

Living document. Update at every phase end; phase completion requires develop→main merge.
Legend: [x] done · [~] partially done / verify pending · [ ] not started

---

## CODE LLM PIVOT — Implementation Status

### Step 1: Code Tokenizer — ✅ COMPLETE

- [x] Code tokenizer config (`configs/tokenizer/code_bpe.yaml`)
- [x] Code-specific special tokens (FIM, repo, file, language)
- [x] Tests (`tests/test_code_tokenizer.py`)

### Step 2: Code Model Config — ✅ COMPLETE

- [x] `configs/model/code_30m.yaml` — d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, seq_len=2048
- [x] ~30M params (within 15–100M target)

### Step 3: Code Corpus Pipeline — ✅ COMPLETE

- [x] `data/build_code_corpus.py` — GitHub API download, license filter, language detect, quality filter, dedup, chunk
- [x] Supports Python, JavaScript, TypeScript
- [x] Permissive license filter (MIT, Apache-2.0, BSD, ISC)
- [x] Syntax validation for Python

### Step 4: Code Pretraining Config — ✅ COMPLETE

- [x] `configs/train/code_local.yaml` — seq_len=2048, max_steps=10000, 65k tokens/step

### Step 5: SFT Training Loop — ✅ COMPLETE

- [x] `training/sft.py` — instruction-response training with loss masking
- [x] `data/build_code_sft.py` — SFT pair generator (code gen, explanation, review, refactor)
- [x] Tests (`tests/test_sft.py`)

### Step 6: Code Evaluation — ✅ COMPLETE

- [x] `evaluation/eval_code.py` — HumanEval + MBPP + code perplexity
- [x] Updated `evaluation/promotion.yaml` — code benchmarks + security gates
- [x] Tests (`tests/test_eval_code.py`)

### Step 7: Code Serving — ✅ COMPLETE

- [x] `serving/code_chat.py` — Gradio UI with syntax highlighting, language selector
- [x] `serving/completion_server.py` — REST API for IDE completion (FastAPI)
- [x] `serving/review_server.py` — REST API for code review (FastAPI)
- [x] Tests (`tests/test_code_serving.py`)

### Step 8: Corporate Fine-Tune Pipeline — ✅ COMPLETE

- [x] `data/ingest_internal.py` — internal repo ingestion with secret scanning
- [x] `configs/train/corporate_ft.yaml` — continued pretrain + SFT config
- [x] Tests (`tests/test_internal_ingest.py`)

### Step 9: Documentation — ✅ COMPLETE

- [x] `README.md` — updated for coding LLM
- [x] `docs/CODE_LLM_STRATEGY.md` — full corporate strategy
- [x] `docs/STATUS.md` — this file

### Step 10: Dependencies — ✅ COMPLETE

- [x] `requirements.txt` — added fastapi, uvicorn, pydantic

---

## Parked: Multimodal (No Active Development)

| Component | Status |
|---|---|
| `training/vision.py` | Parked — maintain as-is |
| `training/multimodal.py` | Parked — maintain as-is |
| `training/train_mm.py` | Parked — maintain as-is |
| `evaluation/eval_caption.py` | Parked — maintain as-is |
| `evaluation/eval_vqa.py` | Parked — maintain as-is |
| `evaluation/india_finance/` | Parked — maintain as-is |

---

## Milestone Board

| Milestone | Target | Status |
|---|---|---|
| Code tokenizer + corpus pipeline | Week 1 | ✅ COMPLETE |
| Code model config + pretrain config | Week 1 | ✅ COMPLETE |
| SFT training loop + data builder | Week 2 | ✅ COMPLETE |
| Code evaluation harness | Week 2 | ✅ COMPLETE |
| Code serving (chat + completion + review) | Week 3 | ✅ COMPLETE |
| Corporate fine-tune pipeline | Week 3 | ✅ COMPLETE |
| Documentation + tests | Week 3 | ✅ COMPLETE |
| **First training run** | Week 4 | 🔲 PENDING |
| **Corporate deployment** | Week 6 | 🔲 PENDING |

---

## Next Actions

1. **Run tests:** `python -m pytest tests -q`
2. **Build code corpus:** `python -m data.build_code_corpus --max-repos 10`
3. **Train tokenizer:** `python -m tokenizer.train_tokenizer --input data/code_corpus_v0.jsonl --out tokenizer/qm_code_bpe_v0.json --vocab-size 32768`
4. **Pretrain:** `python -m training.pretrain --config configs/train/code_local.yaml --out-dir runs/code_30m`
5. **Generate SFT data:** `python -m data.build_code_sft --corpus data/code_corpus_v0.jsonl`
6. **SFT:** `python -m training.sft --config configs/train/code_local.yaml --sft-data data/code_sft_v0.jsonl --init-ckpt runs/code_30m/checkpoint_last.pt`
7. **Evaluate:** `python -m evaluation.eval_code --ckpt runs/code_sft/checkpoint_last.pt`
8. **Deploy:** `python -m serving.code_chat --ckpt runs/code_sft/checkpoint_last.pt`
