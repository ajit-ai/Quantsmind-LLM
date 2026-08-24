# Quantsmind-LLM — Design Architecture

> **Decision record (ADR-000)**: Quantsmind-LLM is a **finance/trading vertical LLM platform**
> targeting **B2B2C** distribution, built by **extending an Apache-2.0 open base model**
> (recommended: Qwen3-8B class) rather than pretraining from scratch. From-scratch
> pretraining remains a documented future option (Phase X) once revenue supports it.
> Rationale: a self-trained 7B cannot match industrial open bases (~36T training tokens);
> adaptation costs ~2–5% of pretraining while adapted weights remain fully ours to serve.

---

## 1. Product Scope

| Item | Decision |
|---|---|
| Vertical | Finance & Trading (single-vertical focus; other industries deferred) |
| Customer model | **B2B2C**: partners = brokers / wealth platforms / fintechs; their retail end-users consume via partner apps |
| Product shape | White-label "Investor Copilot": OpenAI-compatible API + embed SDK, partner-scoped RAG |
| Base strategy | Extend Apache-2.0 open base → finance CPT → SFT → DPO → safety |
| Recommended base | Qwen3-8B / Qwen2.5-7B (Apache 2.0; strong reasoning/math, Hinglish-capable) |
| Market | India-first (₹, NSE/BSE/SEBI conventions), world-ready corpus mix (~70/30) |
| Context | 32k native-class; partner docs handled via RAG, not raw context stuffing |
| Serving economics | AWQ-int4 8B on single A100/4090-class GPU — commodity B2B2C cost per message |
| License posture | Adapted weights + data ledger + adapters + serving stack are project-owned |

**Non-goals (v1):** general-purpose chatbot market, real-time alpha signals, unlicensed personalized advice, multimodal, other industry verticals (see ROADMAP Wave-2+).

## 1.1 B2B2C value proposition

> "Finance-literate, arithmetic-grounded, deployable-in-partner-environment copilot at commodity-GPU economics."

| Partner pain | Quantsmind answer |
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
│ ADAPTATION PLATFORM                                                    │
│ base bake-off (eval harness) → champion ADR                            │
│   → finance continued-pretrain (5–20B tok, LoRA/full FT ladder)        │
│   → SFT (chat + finance tasks) → DPO → safety/advisory-refusal tuning  │
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

### 3.1 Base bake-off (Phase 1)
Candidates: Qwen3-8B, Qwen2.5-7B-Instruct, Mistral-7B-v0.3, Gemma-2-9B (license backup).
Each runs the full eval battery (general + finance + Hindi/Hinglish probes).
Champion selected by weighted score in `evaluation/promotion.yaml`; decision recorded as ADR.

### 3.2 Adaptation ladder

| Step | Method | Scale |
|---|---|---|
| A1 Domain CPT | Full-parameter or LoRA-r128 continued pretrain on finance mix | 5–20B tokens |
| A2 SFT | Chat template + finance task mixture (QA, extraction, summarization, calc-tool calls) | 50–150k pairs |
| A3 DPO | Preference pairs incl. refusal-of-personalized-advice style targets | 10–40k pairs |
| A4 Safety pass | Adversarial jailbreak + advisory-boundary SFT/DPO mixture | red-team suite |
| A5 Tool curriculum | Calculator/unit-converter/API-call traces for numeric grounding | synthetic + audited |

### 3.3 Numeric grounding rules (non-negotiable product requirement)
- All arithmetic presented to end-users must come from the calculator/tool path, never free-token math
- Digit-group tokenization preserved from base tokenizer behavior; tool-trace data reinforces it
- Eval gate includes exact-match arithmetic suite (percentages, CAGR, ratios, unit conversions)

### 3.4 Draft model (speculative decoding)
Our `training/` from-scratch stack is repurposed to train a ~150M draft model on domain text,
cutting serving latency ~2×. This keeps the from-scratch codepath alive and commercially useful.

### 3.5 Future from-scratch option (Phase X)
If revenue funds it: Llama-style dense 7B per configs already in repo. Kept as ADR trail only.

---

## 4. Data Platform

### 4.1 Corpora
| Corpus | Sources | Use |
|---|---|---|
| Finance CPT mix | EDGAR/NSE-BSE-style filings (per license), earnings transcripts, arXiv q-fin, open financial textbooks, market explainers | A1 |
| India-weighted layer | SEBI circulars summaries (public), exchange education content (licensed), ₹/crore conventions, Indian market terminology | A1 tail |
| SFT mixture | Teacher-distilled finance instruction data + human-audited samples | A2 |
| DPO pairs | Sampled from our own SFT model, ranked by judge + human spot-check | A3 |
| Red-team set | Jailbreaks, advice-boundary probes, injection attacks | A4 |

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
- Forgetting check: no general benchmark −2% relative vs champion base without sign-off
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

SLOs: TTFT p95 <800ms @1k-token prompt · ≥1,500 tok/s/worker (8B int4, A100-class) · 99.5% uptime.
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
| Base vs scratch | Open-base extension, Apache-2.0 (ADR-000) | From-scratch (deferred, Phase X) |
| Primary base candidate | Qwen3-8B / Qwen2.5-7B | Mistral-7B, Gemma-2-9B |
| Alignment | SFT + DPO | PPO-RLHF (ops-heavy) |
| Numerics | Tool-bus calculator mandate | Free-token arithmetic |
| Serving | vLLM + FastAPI gateway, AWQ-int4 | TensorRT-LLM (vendor lock) |
| Retrieval | Hybrid BM25+dense, pluggable vector store | Raw long-context stuffing |
| Tracking/versioning | W&B + DVC | MLflow, git-lfs |
| Draft decoding | Self-trained ~150M draft | n-gram/prompt-lookup only |

Each finalized decision gets `docs/adr/NNNN-*.md`.
