# Quantsmind-LLM — Design Architecture

> **Decision record**: Quantsmind-LLM is trained **from scratch** (full control over weights,
> license, data provenance, and architecture). This is the expensive path — it is chosen
> deliberately for sovereignty and long-term product control.

---

## 1. Product Scope

| Item | Decision |
|---|---|
| Name | Quantsmind-LLM |
| Domain | Quantitative finance + general reasoning |
| Training | From scratch, decoder-only Transformer |
| Size tier | Pilot 300M → Production 1.5B → Scale 7B |
| Context length | 8k at launch → extend to 128k via RoPE scaling |
| License of output model | Fully owned by project (data licenses permitting) |
| Serving target | Self-hosted GPU inference, OpenAI-compatible API |

**Non-goals (v1):** multimodality (vision/audio), MoE, on-device quantized mobile models.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PLATFORM                                  │
│  crawl/acquire ─► license filter ─► dedup ─► quality filter ─► PII scrub │
│        ─► decontaminate ─► mix & shard ─► tokenized shards (parquet/arrow)│
└──────────────┬──────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PLATFORM                                 │
│  Tokenizer ─► Pre-train (300M→1.5B→7B) ─► Mid-train (domain)            │
│        ─► SFT ─► DPO/KTO ─► Safety tuning                                │
│  Infra: PyTorch + FSDP/tensor parallel, W&B tracking, checkpoint store   │
└──────────────┬──────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION GATE                                   │
│  lm-eval-harness (standard suites) + FinQA/FinBen + LLM-judge + red team │
│  Promotion requires passing thresholds defined in evaluation/promotion.yaml│
└──────────────┬──────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SERVING PLATFORM                                  │
│  vLLM engine ─► FastAPI gateway (auth, rate-limit, SSE) ─► clients       │
│  K8s autoscaling, FP8/AWQ quantization, observability (OTel + Grafana)   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture

Decoder-only Transformer, Llama-class design with modern choices:

| Component | Choice | Rationale |
|---|---|---|
| Normalization | RMSNorm, pre-norm | Stable training, cheaper than LayerNorm |
| Attention | Multi-head w/ **GQA** | KV-cache ÷4, big serving win |
| Positional encoding | RoPE | Proven; enables context extension later |
| FFN activation | SwiGLU | Standard best practice |
| Precision | bf16 mixed, fp32 master weights | Stability on A100/H100 |
| Attention kernel | FlashAttention-2 | Memory + speed |
| Bias terms | None (except none anywhere) | Cleaner scaling behavior |

### Reference configs (`configs/`)

| Param | pilot-300m | prod-1.5b | scale-7b |
|---|---|---|---|
| d_model | 768 | 2048 | 4096 |
| n_layers | 12 | 24 | 32 |
| n_heads | 12 | 16 | 32 |
| n_kv_heads (GQA) | 4 | 8 | 8 |
| ffn dim | 2048 | 5504 | 11008 |
| vocab | 49,152 | 49,152 | 49,152 |
| context | 2,048 | 8,192 | 8,192 →128k |
| max LR | 3e-3 | 3e-4? no — 2e-4? see train cfg | 1.5e-4? see train cfg |

*(Exact LRs live in `configs/train/*.yaml`; tune per-run via scaling-law probes in Phase 0.)*

### Long-context plan
Phase 2+: continue-pretrain 20B tokens with YaRN-scaled RoPE to 32k, then 128k.
Validate with needle-in-a-haystack + long-context FinQA.

### Arithmetic robustness (finance-critical)
- Digit-level token splitting for numbers
- Calculator/tool-use head start in mid-training data
- Curriculum: synthetic arithmetic → financial tables → FinQA-style QA

---

## 4. Tokenizer Design

- **Algorithm**: Byte-level BPE (HF `tokenizers`), trained on ≥50GB sample of final corpus
- **Vocab**: 49,152 — balances embedding cost vs compression for finance text
- **Domain merges**: tickers (`AAPL`, `NIFTY50`), currency amounts, ISO dates, percent strings
- **Numbers**: split into 1–3 digit groups for arithmetic generalization
- **Tests** (CI-enforced): round-trip fidelity, no unk on financial corpora sample, fertility < 1.15 tokens/word on domain text, multilingual spot checks

---

## 5. Data Platform

### 5.1 Sources (general)
Web crawls (FineWeb-C / CommonCrawl-derived), code (The Stack-v2-licensed subset),
books/academic (licensed or open), math (OpenWebMath/GSM-style synth), Wikipedia.

### 5.2 Domain corpus (Quantsmind edge)
Market news filings (EDGAR, exchange disclosures per license), earnings-call transcripts,
financial textbooks (open-licensed), research papers (arXiv q-fin), synthetic financial
tables/QA generated by a teacher model with human audit sampling.

### 5.3 Pipeline stages
1. **Acquire** — resumable downloaders, checksums, license tags attached per-doc
2. **License/compliance filter** — drop disallowed licenses/domains; keep provenance ledger
3. **Dedup** — exact (hash) → fuzzy MinHashLSH (Jaccard ≥ 0.8) across the whole corpus
4. **Quality** — fastText classifier trained on labeled good/bad docs; perplexity filter vs clean reference LM
5. **PII scrub** — regex + NER pass (names, accounts, PAN/Aadhaar/SSN patterns)
6. **Decontamination** — n-gram overlap removal against every eval benchmark
7. **Mixing** — configurable weights, curriculum ordering (general → general+code → domain-heavy)
8. **Tokenize & shard** — packed sequences (no cross-doc attention masking optional; we mask), Arrow shards of ~2k documents each, content-addressed by corpus version tag

### 5.4 Data governance
- Every shard carries: source, license, filter scores, pipeline version (reproducible builds)
- DVC/LakeFS tracks dataset versions; W&B logs mix proportions per run

### Data budget (Chinchilla-optimal ≈ 20 tokens/param)

| Model | Minimum tokens | Target tokens | Notes |
|---|---|---|---|
| 300M pilot | 6B | 10–15B | Pipeline validation |
| 1.5B prod | 30B | 60–100B | First production candidate |
| 7B scale | 140B | 200–400B | Requires serious cluster |

---

## 6. Training Platform

### 6.1 Phases

| Phase | Objective | Method |
|---|---|---|
| P0 Pre-train | Language + world knowledge | Next-token CE, cosine schedule, warmup 2k steps |
| P1 Mid-train | Finance domain + long ctx + math tools | Continued pretrain on domain-heavy mix, LR 25% of peak |
| P2 SFT | Instruction/chat format | Chat template, 100–500k curated pairs, pack length 4k |
| P3 DPO | Preference alignment | On-policy-ish pairs from our own samples, β≈0.1 |
| P4 Safety | Refusals + jailbreak resistance | Adversarial SFT + DPO mixture, red-team regression suite |

### 6.2 Engineering stack

- **Framework**: PyTorch + native FSDP2 (start) — migrate paths documented for Megatron-LM/torchtitan when multi-node
- **Parallelism ladder**: single-node FSDP → TP=2/SP for 7B → PP only if >64 GPUs
- **Precision**: bf16 autocast, fp32 optimizer states; grad clip 1.0
- **Optimizer**: AdamW (β1 0.9, β2 0.95, wd 0.1) — consider Lion/Muon after ablation
- **Checkpointing**: async distributed checkpoint every 1–2k steps; keep ring buffer (last 3) + milestone set; store optimizer state for exact resume
- **Failure playbook**: loss spike (>3σ) → rollback to last good ckpt, skip batch, halve LR; hang → elastic restart; silent NaN watchdog every 100 steps

### 6.3 Compute estimates (A100-80G class)

| Run | GPUs | Wall-clock (approx) |
|---|---|---|
| 300M × 12B tok | 8 | ~1–2 days |
| 1.5B × 60B tok | 32 | ~2 weeks |
| 7B × 200B tok | 256 | ~6–8 weeks |

(MFU target ≥ 40%; measure before scaling.)

---

## 7. Evaluation & Promotion Gate

### 7.1 Suites
- **General**: MMLU-Pro, HellaSwag, ARC-C, WinoGrande, GSM8K, HumanEval
- **Finance**: FinQA, ConvFinQA, TAT-QA, FinBen suite, numeric-reasoning custom set (kept private)
- **Chat quality**: MT-Bench, Arena-Hard (LLM-judge, judge fixed across runs)
- **Long ctx**: needle-in-haystack @ multiple depths, long-doc summarization rubric
- **Safety**: jailbreak success rate, refusal precision/recall on benign prompts, PII leak scan

### 7.2 Promotion rules (`evaluation/promotion.yaml`)
- No benchmark regresses > 2% relative vs current champion without sign-off
- Safety suite must pass 100% hard-fail cases
- Loss curve sanity (no spikes in last 20% of training)
- Full eval report archived as artifact; model card regenerated automatically

---

## 8. Serving Architecture

```
client ──► API Gateway (FastAPI)
            ├── authn/z (API keys → JWT), quota, rate limit
            ├── guardrails: prompt-injection filter, output policy check
            ├── router: model version selection (canary %)
            └── engine pool: vLLM workers (K8s HPA on queue depth)
                    ├── paged KV cache, continuous batching
                    ├── FP8 or AWQ-int4 weights
                    └── speculative decoding (draft = our 300M)
metrics/logs/traces ──► Prometheus + Grafana + OTel
feedback capture ──► thumbs up/down + transcripts (consented) ──► data flywheel
```

SLOs (v1): TTFT p95 < 800ms @ 1k-token prompt; throughput ≥ 1,500 tok/s/worker (7B int4, A100); availability 99.5%.

---

## 9. Safety, Legal, Compliance

- Model card + datasheets per release; eval reports public-facing summary
- Financial-advice disclaimer enforced at gateway layer for advisory intents
- Red-team cadence: pre-release full pass; monthly regression
- Abuse monitoring: anomaly detection on usage patterns; kill-switch per API key
- Data provenance ledger retained for every training shard (audit trail)

---

## 10. Repository Layout

```
quantsmind-llm/
├── configs/
│   ├── model/          # pilot_300m.yaml, prod_1p5b.yaml, scale_7b.yaml
│   ├── train/          # pretrain/midtrain/sft/dpo stage configs
│   └── data/           # corpus mix definitions per stage
├── tokenizer/          # BPE training scripts + CI tests
├── data/               # acquire/dedup/filter/pii/mix/tokenize pipelines
├── training/           # pretrain.py, sft.py, dpo.py, utils, schedulers
├── evaluation/         # harness configs, finance bench, promotion gate
├── safety/             # red-team suites, guardrail policies, classifiers
├── serving/            # vllm worker, api gateway, docker, k8s manifests
├── infra/              # terraform / slurm / cluster bootstrap
├── experiments/        # run registry notes (W&B links, findings)
├── docs/               # this doc, ROADMAP.md, ADRs, model cards
└── .github/workflows/  # CI: unit tests, smoke-train, lint
```

## 11. Key Technology Choices (ADR summary)

| Area | Choice | Alternative considered |
|---|---|---|
| From-scratch vs base-adapt | From scratch (decision) | Qwen/Llama continued pretrain |
| Framework | FSDP2/torchtitan path | Megatron-LM, DeepSpeed |
| Tokenizer | byte-BPE 49k | SentencePiece unigram 32k |
| Alignment | DPO/KTO | PPO-RLHF (ops-heavy) |
| Serving | vLLM | TensorRT-LLM (locked-in vendor) |
| Experiment tracking | Weights & Biases | MLflow |
| Data versioning | DVC (+LakeFS later) | git-lfs |

Each decision gets an `docs/adr/NNNN-*.md` file as it is finalized.
