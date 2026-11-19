# Taksh-LLM — Design Architecture

> **Correction notice (2026-10-14).** This document is a forward-looking design. The
> ladder figures below do not match the configs they name:
> `pilot_300m.yaml` = **113,145,408** params, `prod_1p5b.yaml` = **1,315,014,656**,
> `scale_7b.yaml` = **6,073,618,432**. "300M pilot" throughout means the 113M-named
> config. **Nothing in this ladder has been trained**: the only text pretrain ran **10
> steps** (`local_15m`, 7,767,680 params) and the only code pretrain ran **200 steps**
> (`code_30m.yaml`, 16,257,280 params).
>
> | Ladder rung | Config | Actual params | Trained? |
> |---|---|---|---|
> | Local text proto | `local_15m.yaml` | 7,767,680 | 10 steps |
> | Code proto | `code_30m.yaml` | 16,257,280 | 200 steps |
> | Pilot | `pilot_300m.yaml` | **113,145,408** | no |
> | Production | `prod_1p5b.yaml` | **1,315,014,656** | no |
> | Flagship | `scale_7b.yaml` | **6,073,618,432** | no |
>
> The "49k BPE" tokenizer claim is **correct** for the general tokenizer
> (`tokenizer/qm_bpe_v0.json`). The code tokenizer realises **5,822**.

> **Decision record (ADR-0003, FINAL — supersedes the interim open-base plan; extended by
> [SCOPE_AND_PRINCIPLES](SCOPE_AND_PRINCIPLES.md) to multimodal)**:
> Taksh is a **general multimodal (image ↔ text) foundation** built as a
> **from-scratch, sovereign-weights Indian model family**, targeting **B2B2C** distribution,
> with finance/trading as the first application vertical:
> **300M pilot → 1.5B "Taksh-Bharat" production → 7B when funded.**
> Founder decision: weight ownership and the "born-in-India, owned-weights" story are the
> mission. Open models (Qwen/Llama/Mistral/Qwen-VL/SD-class) may appear in evaluation
> **as benchmark rivals to beat — never as the product core.** Accepted trade: raw capability
> below GPT-4-class on open-ended knowledge, compensated by RAG + tool-grounding + domain focus.
> Roadmap: [ROADMAP](ROADMAP.md) · Scope/rules: [SCOPE_AND_PRINCIPLES](SCOPE_AND_PRINCIPLES.md).

---

## 1. Product Scope

| Item | Decision |
|---|---|
| Vertical | Multi-domain **general foundation**; finance & trading = first application vertical (other verticals later) |
| Customer model | **B2B2C**: partners = brokers / wealth platforms / fintechs; their retail end-users consume via partner apps |
| Product shape | White-label multimodal copilot: OpenAI-compatible API + embed SDK, partner-scoped RAG |
| Base strategy | **From-scratch pretrain**: 300M pilot → 1.5B production → 7B when funded |
| Modalities | **Image → Text** (v1) · Text → Text (v0) · **Text → Image** (v2) — one shared AR decoder |
| Model ownership | 100% sovereign weights — architecture, tokenizer, data, training all in-repo |
| Open models | Benchmark rivals in eval harness only; never the product core |
| Market | India-first (₹, NSE/BSE/SEBI conventions), world-ready corpus mix (~70/30) |
| Context | 2k (pilot) → 8k (1.5B) → extend toward 32k via RoPE scaling post-launch |
| Serving economics | AWQ-int4 1.5B on single consumer/A100-class GPU — very low B2B2C cost per message |
| License posture | All weights, data ledger, and serving stack are project-owned from birth |

**Non-goals (v1):** general-purpose chatbot market (finance-first vertical), real-time alpha signals, unlicensed personalized advice, video/audio/3D generation, other industry verticals (see ROADMAP Wave-2+). Multimodal image↔text is now **a core goal** (ROADMAP Phase A–B).

## 1.1 B2B2C value proposition

> "Finance-literate, arithmetic-grounded, deployable-in-partner-environment copilot at commodity-GPU economics."

| Partner pain | Taksh answer |
|---|---|
| Support/advisor teams flooded with repetitive investor questions | Grounded Q&A over partner holdings + market snapshot |
| Analyst hours burned on reports/transcripts/circulars | Doc intelligence: summarize, extract guidance/risk, quarter-compare |
| Cannot send customer data to US clouds; OpenAI bills unpredictable | Self-hostable, per-token predictable pricing, tenant isolation |
| Compliance exposure from free-form chat answers | Guardrail layer: advisory refusals, mandatory disclaimers, audit logs |

---

## 2. System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│ DATA PLATFORM                                                          │
│ filings/news/transcripts acquire → license filter → dedup → quality    │
│   → PII scrub → decontaminate vs evals → CPT/SFT/DPO corpora (DVC)     │
└───────────┬───────────────────────────────────────────────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│ PRE-TRAINING PLATFORM                                                  │
│ tokenizer (49k BPE) → 300M pilot (12B tok) → 1.5B Taksh-Bharat    │
│   (60B tok, India-finance-weighted) → mid-train tail → long-ctx ext    │
│ open models run through the same eval harness as benchmark rivals only │
└───────────┬───────────────────────────────────────────────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│ POST-TRAINING PLATFORM                                                 │
│ SFT (chat + finance tasks) → DPO → safety/advisory-refusal tuning      │
│   → calculator/tool-call curriculum                                    │
└───────────┬───────────────────────────────────────────────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│ EVALUATION GATE                                                        │
│ standard suites + FinQA/TAT-QA/ConvFinQA + private India-finance set   │
│ + LLM-judge chat quality + red-team/advisory-refusal regression        │
│ promotion.yaml gates every release                                     │
└───────────┬───────────────────────────────────────────────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│ PRODUCT / SERVING PLATFORM                                             │
│ partner app → API gateway (tenant router, auth, quota, SSE)            │
│   → guardrails (injection filter, disclaimer injector, PII redaction)  │
│   → RAG service (partner docs + market snapshots, vector store)        │
│   → vLLM engine pool (champion adapter set, AWQ-int4, spec-decode)     │
│ observability OTel/Prometheus · billing hooks · feedback flywheel      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Strategy

### 3.1 Pre-training ladder (sovereign weights)

| Run | Params | Tokens | Corpus | Compute (A100-class) | Est. cost* |
|---|---|---|---|---|---|
| Pilot | 300M | 12B | India-weighted general v0 | 8 GPUs × 1–2 days | < ₹10K |
| **Production** | **1.5B "Taksh-Bharat"** | **60B** | general + India-finance tail | 32 GPUs × ~1.5 days | **₹3–5 lakh** |
| Flagship (funded) | 7B | 200B+ | full mix + long-ctx | 64+ GPUs × ~2 wks | ₹30–60 lakh |

*\*cloud rental estimates; configs already in `configs/model/`.*

Scaling-law discipline: LR probes at 100M-token scale before each ladder run; 1.5B loss
trajectory must match pilot-extrapolated prediction within tolerance before mid-train.

### 3.2 Post-training ladder

| Step | Method | Scale |
|---|---|---|
| P1 SFT | Chat template + finance task mixture (QA, extraction, summarization, calc-tool calls) | 50–150k pairs |
| P2 DPO | Preference pairs incl. refusal-of-personalized-advice style targets | 10–40k pairs |
| P3 Safety pass | Adversarial jailbreak + advisory-boundary SFT/DPO mixture | red-team suite |
| P4 Tool curriculum | Calculator/unit-converter/API-call traces for numeric grounding | synthetic + audited |

### 3.3 Benchmark rivals (not product core)
Qwen2.5/3, Llama-3.x, Mistral run through the identical eval battery each release cycle.
Purpose: (a) honest public scorecard "sovereign 1.5B vs open 7B-class on India-finance",
(b) regression alarms, (c) they never appear in the serving path.

### 3.4 Numeric grounding rules (non-negotiable product requirement)
- All arithmetic presented to end-users must come from the calculator/tool path, never free-token math
- Digit-group tokenization behavior trained in via tool-trace data
- Eval gate includes exact-match arithmetic suite (percentages, CAGR, ratios, unit conversions)

### 3.5 Draft model (speculative decoding)
The same from-scratch stack trains a ~150M draft model on domain text, cutting serving
latency ~2× at Phase-4 time.

### 4.1 Corpora (pre-training + post-training)
| Corpus | Sources | Use |
|---|---|---|
| General pretrain mix | Web (clean), wiki, code (licensed), books/academic (open) | 300M pilot + 1.5B base |
| India-finance pretrain layer | NSE/BSE-style filings (per license), earnings transcripts, arXiv q-fin, open financial textbooks, market explainers, SEBI circular summaries (public), ₹/crore conventions | 1.5B mid-train tail |
| SFT mixture | Teacher-distilled finance instruction data + human-audited samples | P1 |
| DPO pairs | Sampled from our own SFT model, ranked by judge + human spot-check | P2 |
| Red-team set | Jailbreaks, advice-boundary probes, injection attacks | P3 |

### 4.2 Pipeline (unchanged engineering)
acquire → license/compliance filter → exact+fuzzy dedup → quality classifier → PII scrub → decontamination vs all benchmarks → version-tagged shards (DVC) → provenance ledger.

### 4.3 Budgets
A1: 5–20B tokens · A2: 50–150k pairs · A3: 10–40k pairs. Quality > quantity at this stage;
every corpus ships a data card.

---

## 5. Evaluation & Promotion Gate

- General: MMLU-Pro subset, GSM8K, HumanEval (guard against catastrophic forgetting)
- Finance: FinQA, TAT-QA, ConvFinQA + **private India-finance held-out set** (never trained on)
- Arithmetic tool-suite: exact-match, hard-fail
- Chat: MT-Bench/Arena-Hard style LLM-judge, fixed judge across runs
- Safety: advisory-boundary refusals ≥95% precision, jailbreak hard-fail set 100%, injection resistance
- Forgetting check: no general benchmark −2% relative vs previous release without sign-off
- Benchmark rivals (open 7B-class) scored on the same battery for the public India-finance scorecard
- Rules encoded in `evaluation/promotion.yaml`; every release auto-generates eval report + model card

---

## 6. B2B2C Serving Architecture

```
partner app/webhook ──► Gateway
                         ├─ tenant router (partner_id → config, quotas, model alias)
                         ├─ authn/z (partner keys→JWT) · rate limit · billing metering
                         ├─ input guardrails: prompt-injection filter, PII redaction
                         ├─ RAG service: partner docs ∪ market snapshot → context pack
                         ├─ vLLM engine pool (AWQ-int4 champion, continuous batching)
                         ├─ tool bus: calculator, quote-snapshot, doc-retrieval
                         └─ output guardrails: disclaimer injector (SEBI-style),
                            advisory-refusal policy, response policy checks
audit log (immutable, per-message) · OTel metrics · feedback capture → flywheel
```

SLOs: TTFT p95 <800ms @1k-token prompt · ≥2,500 tok/s/worker (1.5B int4, single GPU) · 99.5% uptime.
Deployment: K8s HPA; per-partner canary aliases; nightly eval smoke in prod shadow mode.

---

## 7. Safety & Compliance (B2B2C specifics)

- Mandatory contextual disclaimers on market-information responses; configurable per partner jurisdiction
- Hard boundary: no personalized buy/sell recommendations unless partner contract + feature flag + logged consent flow
- Immutable per-message audit trail (prompt hash, retrieval sources, tool outputs, guardrail verdicts)
- PII redaction inbound; leakage scans outbound; abuse monitoring with per-tenant kill switch
- Model card + eval report per release; incident runbooks; monthly red-team cadence

## 8. Repository Layout

```
quantsmind-llm/
├── configs/          # model/train/data YAMLs (+ toy CI configs)
├── tokenizer/        # byte-BPE trainer (draft-model + data tooling)
├── data/             # sample gen, encode pipelines (real ingestion lands Phase 2)
├── training/         # model.py, pretrain.py (draft-model/from-scratch path), CPT/SFT/DPO land later
├── evaluation/       # harness configs + promotion.yaml gate
├── safety/           # red-team suites, guardrail policies
├── serving/          # gateway, RAG service, engine worker, docker/k8s
├── infra/            # cluster provisioning
├── experiments/      # run registry notes
├── scripts/          # smoke pipeline (CI + local)
├── tests/            # unit tests (pytest)
└── docs/             # this file, ROADMAP.md, adr/
```

## 9. Key Technology Choices (ADR summary)

| Area | Choice | Alternative considered |
|---|---|---|
| Base vs scratch | **From-scratch sovereign pretrain (ADR-0003, FINAL)** | Open-base extension (interim ADR-000, superseded) |
| Pretrain ladder | 300M pilot → 1.5B Taksh-Bharat → 7B when funded | Single big run (no pipeline proof) |
| Open models | Benchmark rivals in eval only | Product core (rejected: ownership mission) |
| Alignment | SFT + DPO | PPO-RLHF (ops-heavy) |
| Numerics | Tool-bus calculator mandate | Free-token arithmetic |
| Serving | vLLM + FastAPI gateway, AWQ-int4 | TensorRT-LLM (vendor lock) |
| Retrieval | Hybrid BM25+dense, pluggable vector store | Raw long-context stuffing |
| Tracking/versioning | W&B + DVC | MLflow, git-lfs |
| Draft decoding | Self-trained ~150M draft | n-gram/prompt-lookup only |

Each finalized decision gets `docs/adr/NNNN-*.md`.
