# Taksh Coder — Corporate Coding LLM Strategy

> **This document defines the product direction.** The single authoritative
> *execution* roadmap is [ROADMAP.md](ROADMAP.md); M2 acceptance governance lives in
> `docs/source/roadmap/M2_EXECUTION.rst`. Where milestone numbering differs between
> documents, ROADMAP.md governs.
>
> **Pivot from finance/multimodal to coding LLM for corporate use.**
> Target: Python + JavaScript/TypeScript specialist, 15–100M params, all use cases.
> Multimodal is **deferred** (see ROADMAP.md → *Deferred / Future Multimodal Track*),
> not abandoned.
>
> **Reality check (2026-10-14).** This document is a *plan*. Where it previously read as
> a record, the numbers were wrong and have been corrected: the model is **16,257,280**
> params (not ~30M), the code tokenizer realises **5,822** vocab (not 32,768), the shipped
> corpus is **100,244 tokens of this repository's own source** (not GitHub, and contains
> **0 TypeScript**), the SFT set is **quarantined**, and **no HumanEval or MBPP score
> exists**. See [VERSIONING.md](VERSIONING.md) and [OPEN_ISSUES.md](OPEN_ISSUES.md).

---

## 1. Product Definition

| Item | Decision |
|---|---|
| Name | Taksh Coder |
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
│  qm_code_bpe_v0.json — REALISED vocab 5,822                 │
│  Special tokens present: <pad> <bos> <eos> <unk> ONLY.      │
│  FIM/system/assistant markers are NOT in the vocabulary.    │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL                                  │
│  QMModel: RMSNorm/RoPE/GQA/SwiGLU — 16,257,280 params       │
│  d_model=256, n_layers=8, n_heads=8, seq_len=2048           │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING                               │
│  Phase 1: Code pretrain — TARGET 10k steps, RUN 200 steps  │
│  Phase 2: SFT — BLOCKED, corpus quarantined                 │
│  Phase 3: Corporate fine-tune — 2,126-token demo corpus    │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVING                                │
│  Code Chat UI (Gradio) — NOT DEPLOYABLE, see A1           │
│  Completion API (FastAPI) — NOT DEPLOYABLE, see A1         │
│  Review API (FastAPI) — NOT DEPLOYABLE, see A1             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Phases

### Phase 1: Code Pretraining (Weeks 1–3) — PARTIALLY EXECUTED

| Item | Target | Actual |
|---|---|---|
| Corpus | Public GitHub Python + JS/TS repos (MIT/Apache-2.0) | **77 chunks, 100,244 tokens, 100% `source: local:*` (this repo). 0 TypeScript** |
| Tokenizer | 32,768 vocab target | **5,822 realised** |
| Model | ~30M params | **16,257,280** |
| Config | `code_local.yaml` (10k steps, 65k tokens/step) | configured; **only 200 steps run at seq_len 512** |
| Compute | Local CPU/GPU, ₹0 | ✅ free, as designed |
| Gate | Loss curve sane, held-out PPL < 15 | **PPL NOT MEASURABLE.** Held-out tail is 1,003 tokens, one window needs 2,049 |

### Phase 2: SFT — Code Tasks (Weeks 4–6) — BLOCKED

Targets below were aspirational. Actual: **442 pairs, no teacher model, 423 unusable.**

| Task | Target volume | Actual | Status |
|---|---|---|---|
| Code generation | 20k pairs | **19**, all malformed by a faulty regex extractor | ❌ |
| Code explanation | 15k pairs | 96, identical boilerplate | ❌ quarantined |
| Bug fixing | 10k pairs | 0 — generator not implemented | ❌ missing |
| Code review | 10k pairs | 231, one byte-identical template | ❌ quarantined |
| Refactoring | 5k pairs | 96, original code returned unchanged while claiming edits were made | ❌ quarantined |

- **Teacher model (DeepSeek-Coder / CodeLlama):** never used. No distillation step exists in the repo.
- **Training:** `training/sft.py` with loss masking on assistant tokens. Blocked by the corpus quarantine and by Open Issues B1/B3.
- **LR:** 5e-5 as configured.

### Phase 3: Corporate Fine-Tune (Weeks 7–9) — CODE ONLY

| Step | Detail | Actual |
|---|---|---|
| Internal corpus | Ingest internal repos via `data/ingest_internal.py` | **2,126 tokens from 5 synthetic demo files** |
| Continued pretrain | 1–2B tokens internal code, LR 5e-4 | target only; shipped checkpoint is 495,808 params trained 100 steps |
| Internal SFT | Code review/QA pairs from internal patterns | blocked by the SFT quarantine |
| Privacy | All processing local, secret scrubbing, no external APIs | partially: local ✅, scrubbing is 4 regexes (A7) |

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

## 5. Evaluation — TARGETS, NOT RESULTS

Every row below is a **gate to be met**, not a measurement. No value in this table has
been achieved. `evaluation/promotion.yaml` is declarative; no code currently enforces
it (Open Issue D4).

| Benchmark | Target | Measurement | Current state |
|---|---|---|---|
| HumanEval | Pass@1 ≥ 20% | Code execution correctness | **UNMEASURED.** Harness holds a 5-problem subset of 164; reports stamp `promotion_eligible: false` |
| MBPP | Pass@1 ≥ 25% | Basic Python programming | **UNMEASURED.** 5-problem subset of 500 |
| Code PPL | < 15 | Held-out perplexity | **NOT MEASURABLE.** Corpus too small to yield one held-out window |
| Internal completion | Top-5 ≥ 40% | Corporate-specific accuracy | no internal eval suite defined |
| Security scan | 100% | No secrets in generated code | not implemented |

Two previously committed reports recorded a **false perfect perplexity of 1.0** produced
by an empty held-out split. Both JSON files are preserved unchanged and retracted in
`runs/code_pretrain/eval_report.RETRACTED.md` and `runs/code_smoke/eval_report.RETRACTED.md`.

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
