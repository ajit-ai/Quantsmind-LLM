# Quantsmind-LLM — Development Roadmap

**Scope**: Finance/Trading vertical · Tier-1 use cases (RAG + tools + 7B-class) · B2B2C.
**Strategy**: extend an Apache-2.0 open base (Qwen3-8B class) → finance adaptation → partner pilot.
Durations assume 1–2 engineers + rented GPU. Each phase has hard exit criteria; do not advance without passing them.

---

## Phase 0 — Foundations (Weeks 1–2)  ← current

| Workstream | Deliverables |
|---|---|
| Repo & CI | Scaffold, lint, pytest suite, CPU smoke-train of tiny model in CI |
| Training infra | `training/model.py` + `pretrain.py` (LR schedule, checkpoint/resume, watchdog) — powers draft-model now, from-scratch option later |
| Eval infra | Harness config skeleton, `promotion.yaml` gate rules |
| Tokenizer tooling | byte-BPE trainer + tests (draft-model/data tooling) |

**Exit**: CI green on GitHub Actions — unit tests pass and 30-step toy run completes with checkpoint+resume.

## Phase 1 — Base bake-off (Weeks 2–4)

> **Competitive clock** (see COMPETITIVE_ANALYSIS.md): Sarvam is circling BFSI with
> sovereign positioning; Stoxo proved retail demand. Window to demonstrate
> "finance-tuned sovereign engine" is ~2 quarters — M2 is the proof point.

- Download candidates: Qwen3-8B, Qwen2.5-7B-Instruct, Mistral-7B-v0.3, Gemma-2-9B
- Wire lm-eval-harness + finance suites (FinQA/TAT-QA/ConvFinQA) + Hindi/Hinglish probes
- Run full battery on each candidate on 1×A100; log to W&B
- Build **private India-finance held-out eval set** (v0, ~500 items, never trained on)

**Exit**: champion selected by weighted score → ADR-001; eval harness reproducible via one command.

## Phase 2 — Finance adaptation (Weeks 4–8)

- Domain corpus build: filings/transcripts/q-fin per license ledger → CPT mix v1 (~10B tokens)
- A1 CPT run (full-FT if ≥4×A100 else LoRA-r256 ladder); LR probes first at 100M-token scale
- A2 SFT mixture build (50–150k pairs: QA, extraction, summarization, calc-tool traces) + train
- A3 DPO pairs (10–40k) + train
- A4 safety pass: advisory-boundary + jailbreak SFT/DPO mix; red-team round 1
- A5 calculator/tool-call curriculum baked into SFT data; exact-match arithmetic eval added

**Exit**: candidate beats base by clear margin on finance suite; arithmetic suite 100% via-tool;
refusal precision ≥95%; no general-benchmark regression >2%; all data cards published.

## Phase 3 — B2B2C product layer (Weeks 8–10)

- Gateway: tenant router, partner keys, quotas, billing meter hooks, SSE streaming
- RAG service: hybrid retrieval over filings/news/partner docs; market-snapshot context packer
- Guardrails: injection filter, PII redaction, SEBI-style disclaimer injector, advisory-refusal policy
- Serving: vLLM AWQ-int4 engine pool; speculative decoding w/ our draft model; K8s manifests
- Immutable audit log per message; OTel dashboards (TTFT/TPOT/tool-error rates)

**Exit**: end-to-end demo — partner-key request → grounded, disclaimed answer with tool-verified numbers; load test 2× projected peak.

## Phase 4 — Pilot launch (Weeks 10–12)

> **Positioning rule** (COMPETITIVE_ANALYSIS.md §5): sell the compliant engine, not the
> app. Pilot target = a broker/wealth platform that needs its own branded AI answer to
> Stoxo-class products and cannot risk foreign-API wrappers under SEBI's forthcoming
> tiered AI rules.

- One named B2B pilot partner (broker/wealth app) — success criteria contracted up front
- SEBI-RA partner firm signed for advisory-adjacent features (RA owns accountability; we power)
- Compliance pack per partner: audit-log export, residency attestation, kill-switch demo,
  model-inventory row (RBI MD / SEBI CSCRF aligned)
- Shadow-mode week → canary 5% → full ramp; weekly eval reports; feedback flywheel capture
- Model card + eval report published internally; go/no-go review board

**Exit**: pilot SLOs met in production; partner sign-off; compliance pack accepted by
partner's compliance officer; rollback rehearsed.

## Phase 5 — Compound & decide Wave-2 (Quarter 2)

- Data flywheel → monthly SFT/DPO refreshes; Hindi/Hinglish strengthening
- Draft-model scale-up; context extension experiments
- Wave-2 vertical (agri/commodity adjacent) only if: corpus identified + eval suite built + anchor customer signed

---

## Milestones

| M | Target | Gate |
|---|---|---|
| M0 | wk 2 | CI green incl. smoke-train |
| M1 | wk 4 | Champion base chosen (ADR-001) |
| M2 | wk 8 | Finance-adapted model passes promotion gate |
| M3 | wk 10 | Product layer demo end-to-end |
| M4 | wk 12 | Production pilot live |
| M5 | Q2 | Refresh cadence + Wave-2 decision |

## Risk Register

| Risk | Mitigation |
|---|---|
| Base license change / model withdrawal | Pin weights + hashes in DVC; bake-off keeps 2 Apache alternates ready |
| Finance benchmark contamination | Strict decontamination; private held-out set is the deciding metric |
| Arithmetic hallucination reaching users | Tool-mandate architecture + exact-match hard-fail gate |
| Advisory-compliance breach | Refusal training + disclaimer injector + audit trail + partner contract flags |
| Competitor verticalizes first (Sarvam→BFSI, Stoxo→white-label) | 2-quarter window tracked in COMPETITIVE_ANALYSIS.md; M2 demo is the counter-move; quarterly market re-scan |
| SEBI AI guidelines finalize with stricter controls | Map final text to our controls (audit/residency/kill-switch already built); update safety/ policies |
| Single-GPU serving cost creep | int4 + spec-decode budgets enforced in load tests |
| Data licensing gaps | Provenance ledger + legal review before CPT starts |
| Key-person dependency | Runbooks + this repo as single source of truth |

## Branching & Release Flow

- `feat/*` → PR into `develop` (integration)
- `develop` → release PR into protected `main`, attaching the eval report
- Tags: model releases tagged `model-vN.M` with git SHA + dataset tag + config hash
- Every phase ends with develop→main merge per project rule
