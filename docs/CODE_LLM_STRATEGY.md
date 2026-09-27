# Quantsmind Coder — Corporate Coding LLM Strategy

> **Pivot from finance/multimodal to coding LLM for corporate use.**
> Target: Python + JavaScript/TypeScript specialist, 15–100M params, all use cases.

---

## 1. Product Definition

| Item | Decision |
|---|---|
| Name | Quantsmind Coder |
| Use cases | Code completion, code chat, code review/QA |
| Languages | Python, JavaScript, TypeScript |
| Scale | 15–100M params (local/single-GPU deployment) |
| Data | Public code (permissive licenses) + internal corporate codebase |
| Multimodal | Parked — maintain existing code, no new investment |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                         │
│  Public GitHub (MIT/Apache) → filter → dedup → encode   │
│  Internal repos → scrub secrets → dedup → encode        │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    TOKENIZER                             │
│  qm_code_bpe_v0.json (32,768 vocab, code-optimized)     │
│  Special tokens: <|fim_prefix|>, <|fim_suffix|>, etc.   │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    MODEL                                 │
│  QMModel: RMSNorm/RoPE/GQA/SwiGLU, ~30M params          │
│  d_model=256, n_layers=8, n_heads=8, seq_len=2048      │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    TRAINING                              │
│  Phase 1: Code pretrain (public code, 10k steps)        │
│  Phase 2: SFT (code tasks: generation, review, debug)   │
│  Phase 3: Corporate fine-tune (internal codebase)       │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    SERVING                               │
│  Code Chat UI (Gradio) — conversational coding help     │
│  Completion API (FastAPI) — IDE inline suggestions      │
│  Review API (FastAPI) — automated code review           │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Training Phases

### Phase 1: Code Pretraining (Weeks 1–3)

| Item | Detail |
|---|---|
| Corpus | Public GitHub Python + JS/TS repos (MIT/Apache-2.0) |
| Tokenizer | `qm_code_bpe_v0.json` (32,768 vocab) |
| Model | `code_30m.yaml` (~30M params) |
| Config | `code_local.yaml` (10k steps, 65k tokens/step) |
| Compute | Local CPU/GPU, ₹0 |
| Gate | Loss curve sane, held-out PPL < 15 |

### Phase 2: SFT — Code Tasks (Weeks 4–6)

| Task | Source | Volume |
|---|---|---|
| Code generation | Docstring → function body | 20k pairs |
| Code explanation | Function → NL explanation | 15k pairs |
| Bug fixing | Buggy code + error → fix | 10k pairs |
| Code review | Code → structured review | 10k pairs |
| Refactoring | Code → improved version | 5k pairs |

- **Teacher model:** DeepSeek-Coder / CodeLlama (synthetic data generation)
- **Training:** `training/sft.py` with loss masking on assistant tokens
- **LR:** 5e-5 (much lower than pretrain)

### Phase 3: Corporate Fine-Tune (Weeks 7–9)

| Step | Detail |
|---|---|
| Internal corpus | Ingest internal repos via `data/ingest_internal.py` |
| Continued pretrain | 1-2B tokens internal code, LR 5e-4 |
| Internal SFT | Code review/QA pairs from internal patterns |
| Privacy | All processing local, secret scrubbing, no external APIs |

---

## 4. Serving Interfaces

### 4.1 Code Chat UI
- **Entry point:** `python -m serving.code_chat --ckpt runs/code_30m/checkpoint_last.pt`
- **URL:** http://localhost:7860
- **Features:** Syntax highlighting, language selector, code context upload

### 4.2 Completion API
- **Entry point:** `python -m serving.completion_server --port 8000`
- **Endpoint:** `POST /complete`
- **Input:** `{prefix, suffix, language, max_tokens, temperature, top_k}`
- **Output:** `{completions[], scores[], model}`
- **Use:** IDE plugin, CLI tool

### 4.3 Review API
- **Entry point:** `python -m serving.review_server --port 8001`
- **Endpoint:** `POST /review`
- **Input:** `{code, language, context}`
- **Output:** `{issues[], summary, model}`
- **Use:** CI/CD pipeline, PR review automation

---

## 5. Evaluation

| Benchmark | Target | Measurement |
|---|---|---|
| HumanEval | Pass@1 ≥ 20% | Code execution correctness |
| MBPP | Pass@1 ≥ 25% | Basic Python programming |
| Code PPL | < 15 | Held-out perplexity |
| Internal completion | Top-5 ≥ 40% | Corporate-specific accuracy |
| Security scan | 100% | No secrets in generated code |

---

## 6. Corporate Deployment Checklist

- [ ] Legal review: internal code usage rights for training
- [ ] Security: secret scanning on all ingested code
- [ ] Privacy: no external API calls, all inference local
- [ ] Evaluation: internal benchmark suite defined
- [ ] Serving: GPU provisioned (single A100/consumer GPU sufficient)
- [ ] Monitoring: logging, feedback collection
- [ ] Update cadence: monthly fine-tune with new internal code

---

## 7. What Gets Parked

| Component | Status | Reason |
|---|---|---|
| `training/vision.py` | Parked | Not needed for coding LLM |
| `training/multimodal.py` | Parked | Not needed for coding LLM |
| `training/train_mm.py` | Parked | Not needed for coding LLM |
| `evaluation/eval_caption.py` | Parked | Not needed for coding LLM |
| `evaluation/eval_vqa.py` | Parked | Not needed for coding LLM |
| `evaluation/india_finance/` | Parked | Not needed for coding LLM |

All parked code remains in repo for future multimodal revival.
