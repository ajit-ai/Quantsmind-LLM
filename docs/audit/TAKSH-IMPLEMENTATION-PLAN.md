# TAKSH — Implementation Plan (Coding-Only Genius Programme)

**Document ID:** `TAKSH-IMPLEMENTATION-PLAN-v1`
**Date:** 2026-11-19
**Owner:** ajit-ai
**Status of this document:** approved direction by owner; implementation gated per unit below.
**Honesty constraint:** Taksh has no demonstrated coding capability today (frozen Python benchmark 0/30; Native v0.1 0/20). This plan never claims otherwise.

---

## 1. Objective

Make Taksh genuinely usable as a **coding-only AI tool**:

1. usable locally as its own coding assistant;
2. attachable to clients (OpenCode, Cline, Continue, OpenAI-compatible endpoints);
3. distributable worldwide through Hugging Face and Ollama/llama.cpp once qualified, with OpenRouter only as a later hosted-provider step;
4. sovereign by construction: own weights, own tokenizer, own evaluation, own serving — no external LLM required for the core intelligence.

Non-negotiables:

- native TAKSH is the default core; external providers are optional, clearly labelled adapters;
- never silently substitute an external provider for native inference;
- never claim capability from tests/docs alone; frozen executable benchmarks decide;
- no publication, release, merge to protected branches, model redistribution or training without explicit owner authorization;
- every milestone is reversible and evidence-gated.

---

## 2. Current measured state (do not contradict)

| Item | State |
|---|---|
| Frozen Python benchmark | 0/30 |
| Native v0.1 benchmark | 0/20 syntax, 0/20 executable, 0/20 correct; 16/20 empty; 20/20 degenerate |
| Canonical 111M checkpoint | missing |
| Available checkpoint `runs/v180_sft/checkpoint_last.pt` | ~28.8M params, step 200, undertrained |
| Tokenizer | `taksh_code_bpe_v1.json`, 5,822 vocab, BOS 1, EOS 2, no role/FIM markers |
| Gradio UI | works technically; streaming is a character-at-a-time replay; default tokenizer stale |
| Serving skeleton | OpenAI-compatible API, completion API, review API; loopback-only; tool calling unsupported; streaming not implemented |
| Native v0/v0.1/v0.2 artifacts | recovered in closure worktree, not committed on any branch |
| Corpus | Python-only, frozen; four protected corpus directories must not enter Git without legal review |
| Checkpoint/tokenizer hashes | verified 2026-11-14 (see Step 0.2 report) |

---

## 3. What already exists (reuse map)

| Area | Implementation | Reuse |
|---|---|---|
| Transformer (RMSNorm/RoPE/GQA/SwiGLU, KV cache) | `training/model.py`, `taksh/kv_cache.py` | KEEP |
| Training loops, checkpoint/resume, manifests | `training/pretrain.py`, `sft.py`, `train.py`, `manifest.py`, `checkpoint*.py` | KEEP |
| Prompt identity | `training/chat_template.py` (v1/v2 contracts, single renderer) | KEEP |
| Safe checkpoint loading | `serving/local_inference.py` (`weights_only=True`, narrow NumPy allowlist) | KEEP |
| Data contracts | `data/schema.py`, provenance/licensing gates | KEEP |
| Evaluation harness | `evaluation/native_v01/*`, sandbox runner; frozen benchmarks | KEEP immutable |
| Serving skeleton | `openai_server.py`, `completion_server.py`, `review_server.py`, `server_security.py` | KEEP, extend |
| Provider abstraction | `serving/providers.py` (ModelProvider, capabilities, external/S2/Seed providers) | KEEP, extend |
| DLCAE runtime | `taksh/dlcae/*` (six engines, coding orchestrator, evidence) | KEEP |
| Gradio UI shell | `serving/code_chat.py` | KEEP, fix defects |
| Governance | roadmap registry, validator, release governance, audits | KEEP |

---

## 4. Defects found in code scan (implementation backlog)

| ID | Defect | File | Fix | Unit |
|---|---|---|---|---|
| C1 | Gradio "streaming" iterates a finished string one character at a time; `generate_code` returns `str` but is consumed as a generator | `serving/code_chat.py` | Render once or implement real streaming callback | I1 |
| C2 | Gradio default tokenizer `qm_code_bpe_v0.json`; native work used `taksh_code_bpe_v1.json` | `serving/code_chat.py` | Derive from checkpoint/default v1 | I1 |
| C3 | `generate_ids(use_kv_cache=False)` contradicts docstring "(default)" | `serving/local_inference.py` | Flip default to True after equivalence check | I1 |
| A6 | `completion_server.top_k` used as completion count → N serial generations; trivial DoS | `serving/completion_server.py` | Reinterpret as sampling cutoff (or explicit `n_completions` with cap) | I1 |
| N8 | `stream: true` rejected with structured 400 | `serving/openai_server.py` | Add SSE streaming | I1/I5 |
| N7 | Tool/function calling unsupported | `openai_server.py`, `agent_profile.py` | Minimal validated tool-call contract; never fake | I5/I6 |
| P1 | `NativeTakshProvider` absent on current branch; closure carries one | `serving/providers.py`, closure worktree | Integrate closure version after owner approval | I2 |
| P2 | `serving/foundation_inference.py` primary copy is older (baseline metadata missing) | primary vs closure worktree | Integrate closure version | I2 |
| T1 | Tokenizer has no role/FIM/chat markers | `taksh_code_bpe_v1.json` | Tokenizer v2 (32k, control tokens) in scale-up | I4/V7 |
| M1 | 28.8M dense, undertrained; no MoE | `training/model.py`, configs | Capability recovery first; MoE only after dense proof | I4/V9 |

---

## 5. Implementation units

### I0 — Owner baseline acceptance
**Goal:** pick the integration baseline.
**Input:** `docs/audit/TAKSH-STEP0.3-CONSOLIDATION-PLAN.md`
**Output:** owner decision, recorded in `docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md`
**Gate:** explicit owner approval. No code changes.

### I1 — Quick correctness fixes (CPU-only, no training)
**Files:** `serving/code_chat.py`, `serving/local_inference.py`, `serving/completion_server.py`
**Tasks:** fix C1, C2, C3, A6; preserve owner's loopback/NumPy edits.
**Tests:** `pytest tests/test_m7_serving.py tests/test_openai_server.py tests/test_legacy_serving_security.py tests/test_kv_cache.py -q`
**Gate:** serving suite green; no model-capability claim.

### I2 — Native artifact integration
**Source:** closure worktree `C:/Users/Lenovo/.local/share/opencode/worktree/545eb7/integration/taksh-l060-1-closure`
**Files:**
- Native v0: `serving/foundation_inference.py`, `serving/native_provider.py`, `tests/test_native_v160_inference.py`
- Native v0.1 (immutable): `evaluation/native_v01/*`, `tests/test_native_v01_evaluation.py`, `docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md`
- Native v0.2 (deferred): `evaluation/native_v02/*`, `tests/test_native_v02_diagnosis.py`, `docs/audit/TAKSH-NATIVE-V0.2-DIAGNOSIS.md`

**Tasks:** integrate v0 + v0.1; keep v0.2 as diagnostic evidence; hash-verify before/after; add provider identity to native responses.
**Tests:** `pytest tests/test_native_v160_inference.py tests/test_native_v01_evaluation.py tests/test_native_provider.py -q`
**Gate:** Native v0.1 benchmark definition byte-identical; no capability claim.
**Owner decision required before staging/committing anything.**

### I3 — Recovery/CI selective integration
**Sources:** `61ae12a`, `30facf5`, `90eba0a`, `61e2dba` (merge commits `8f1fbda`, `0215c6d` excluded)
**Review:** file-level diff vs current L0.7.4 CI qualification; extract non-conflicting safeguards only.
**Gate:** no regression in `tests/test_ci_workflows.py` and current CI qualification.

### I4 — Capability recovery (the critical path)
**Files/configs:** `configs/train/sft_v030_aligned.yaml` (new), `training/sft.py`, `training/chat_template.py`, `evaluation/native_v01/*` (immutable)
**Tasks:** aligned prompt contract from V1 diagnosis; small clean full-function SFT set; bounded CPU run with fixed seeds; evaluate on immutable Native v0.1; compare syntax/execution/correctness/repetition/instruction-adherence.
**Gate:** pass@1 > 0, else document capacity recommendation. **No scale-up before this gate.**

### I5 — Serving qualification
**Tasks:** deterministic generation tests; invalid checkpoint/tokenizer rejection; context-overflow handling; timeout/cancellation; memory + latency measurements; SSE streaming; tool-call contract if authorized.
**Deliverable:** `docs/audit/TAKSH-STEP0.6-NATIVE-SERVING-QUALIFICATION.md`
**Gate:** safe loads verified; resource envelope measured; no unsafe paths.

### I6 — Product integration (Gradio-first)
**Files:** `serving/code_chat.py` (extend, not rebuild), `serving/providers.py`, `taksh/dlcae/coding/*`, selective reviewed candidates from five-version branch (`taksh/agent.py`, `taksh/engineering.py`, `serving/engineering_api.py`, `ide/vscode/`)
**Tasks:** native model selector; per-response provider identity; code explain/debug/repair; safe patch proposals with explicit approval; controlled test execution; cancellation.
**Gate:** UI works end-to-end with native model alone; external adapters labelled.

### I7 — Release and distribution
**Order:** Hugging Face model repo → Ollama/GGUF → OpenAI-compatible endpoint (auth + loopback default) → OpenCode/Cline integration docs → OpenRouter only as hosted provider (later).
**Prerequisites:** capability gates, safety gates, licence decisions, owner publication authorization.
**Deliverables:** model card, tokenizer/config, licence, provenance, benchmark evidence, limitations, reproducibility script; `TAKSH-STEP0.9-RELEASE-QUALIFICATION.md`.

---

## 6. Vertical critical path

```text
I0 owner baseline acceptance
   ↓
I1 quick correctness fixes ──────────────┐
   ↓                                     │ (I3 CI review, parallel)
I2 native artifact integration           │
   ↓                                     │
I4 capability recovery  ←────────────────┘
   ↓
I5 serving qualification
   ↓
I6 Gradio product integration
   ↓
I7 release & distribution
   ↓
V7 dense scale-up (100M→300M→1B→3B; tokenizer v2)
   ↓
V8 repository intelligence
   ↓
V9 MoE (only after dense proof)
   ↓
V10 world distribution (HF, Ollama, OpenAI-compatible, OpenRouter)
```

---

## 8. Implementation discipline

Branch naming, commit format, testing gates G1–G5, CI reality, the merge-to-main approval procedure, rollback and unit-to-branch mapping are governed by `docs/audit/TAKSH-REPO-WORKFLOW.md`.

---

## 7. Definition of done (whole programme)

1. Native model demonstrates frozen-benchmark capability (pass@1 > 0, then competitive thresholds per release gate).
2. Serving is safe: `weights_only=True`, no unauthenticated model loading, no external calls.
3. Gradio product works with native model unaided; external providers are labelled adapters.
4. Hugging Face release with honest model card and provenance — owner authorized.
5. Ollama/GGUF for local worldwide use.
6. OpenCode/Cline attach via OpenAI-compatible endpoint with streaming.
7. No capability claim anywhere without benchmark evidence and reproducible commands.

---

## 8. Standing prohibitions

- No training/fine-tuning without owner compute + data authorization.
- No merge/tag/release/push/publication without explicit owner authorization.
- Protected corpora (`data/l02_overfit/`, `l02_sft/`, `l03_corpus/`, `l04_code_corpus/`) never enter Git/releases without legal review.
- Native v0.1 benchmark never modified.
- Karkain remains DISCOVERED until `kcc.exe` works; no language qualification without evidence.
- No capability claims from unexecuted benchmarks.
