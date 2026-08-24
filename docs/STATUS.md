# Quantsmind-LLM — Execution Status Tracker

Living document. Update at every phase end; phase completion requires develop→main merge.
Legend: [x] done · [~] partially done / verify pending · [ ] not started

---

## PHASE 0 — Foundations (Weeks 1–2) — ~90%

- [x] 0.1 Repo scaffold, configs, .gitignore, branch flow
- [x] 0.2 Strategy locked: B2B2C finance vertical, open-base extension (ADR-000)
- [x] 0.3 Transformer implementation (RMSNorm/RoPE/GQA/SwiGLU)
- [x] 0.4 Pretrain loop: cosine LR, checkpoint/resume, NaN watchdog
- [x] 0.5 Tokenizer trainer + round-trip sanity tests
- [x] 0.6 Data tools: corpus generator, encode-to-tokens pipeline
- [x] 0.7 Unit tests + CPU smoke pipeline scripts
- [~] 0.8 GitHub Actions CI pushed — verify first run green on GitHub
- [x] 0.9 Promotion gate rules (evaluation/promotion.yaml)
- [ ] 0.10 Lint config (ruff) in CI
- [ ] 0.11 Formalize docs/adr/0001 files

### Preview track (pulled forward from Phase 3)

- [x] P0.a Finance chat UI (Gradio) — prompt → financial knowledge, disclaimer enforced
- [x] P0.b Pluggable OpenAI-compatible backend (works with Ollama/vLLM/OpenRouter today)
- [x] P0.c UI E2E verified on user machine (Gradio 6.25; demo mode confirmed 2026-08-24)
- [ ] P0.d Optional: connect real backend (Ollama qwen2.5) and verify grounded answers

Exit when: CI green on GitHub + 0.10/0.11 closed.

## PHASE 1 — Base Bake-off (Weeks 2–4) — 0%

- [ ] 1.1 Provision GPU box (1×A100/4090-class rented)
- [ ] 1.2 Download candidates: Qwen3-8B, Qwen2.5-7B-Instruct, Mistral-7B-v0.3, Gemma-2-9B
- [ ] 1.3 Wire lm-eval-harness: general suites + FinQA/TAT-QA/ConvFinQA
- [ ] 1.4 Hindi/Hinglish probe set (~200 prompts)
- [ ] 1.5 Build private India-finance held-out eval v0 (~500 items)
- [ ] 1.6 Run full battery on all candidates → weighted scorecard
- [ ] 1.7 Select champion → ADR-001

Exit when: one-command reproducible evals; champion named.

## PHASE 2 — Finance Adaptation (Weeks 4–8) — 0%

- [ ] 2.1 Real data ingestion + license/provenance ledger
- [ ] 2.2 CPT mix v1 (~10B tokens) + decontamination audit
- [ ] 2.3 LR probes → CPT run (full-FT or LoRA ladder)
- [ ] 2.4 SFT mixture build (50–150k pairs incl. calculator traces) + train
- [ ] 2.5 DPO pair generation + train
- [ ] 2.6 Safety pass: advisory-boundary refusals + red-team round 1
- [ ] 2.7 Exact-match arithmetic eval suite (hard-fail gate)
- [ ] 2.8 Pass promotion.yaml gate vs base

Exit when: finance scores up, arithmetic 100% via-tool, refusal >=95%, forgetting <=2%.

## PHASE 3 — B2B2C Product Layer (Weeks 8–10) — 0%

- [ ] 3.1 API gateway: tenant router, partner keys, quotas, billing hooks, SSE
- [ ] 3.2 RAG service: hybrid retrieval + market-snapshot context packer
- [ ] 3.3 Guardrails: injection filter, PII redaction, disclaimer injector
- [ ] 3.4 vLLM serving pool (AWQ-int4) + K8s manifests
- [ ] 3.5 Speculative decoding with self-trained ~150M draft model
- [ ] 3.6 Immutable per-message audit log + OTel dashboards

Exit when: partner-key request -> grounded disclaimed tool-verified answer E2E; load-tested 2x peak.

## PHASE 4 — Pilot Launch (Weeks 10–12) — 0%

- [ ] 4.1 Sign pilot B2B partner + contracted success criteria
- [ ] 4.2 Shadow-mode week -> canary 5% -> ramp
- [ ] 4.3 Weekly eval reports; feedback capture loop live
- [ ] 4.4 Model card + eval report; go/no-go review

Exit when: production SLOs met; partner sign-off.

## PHASE 5 — Compound & Wave-2 Decision (Q2) — 0%

- [ ] 5.1 Data flywheel -> monthly SFT/DPO refresh cadence
- [ ] 5.2 Hindi/Hinglish strengthening
- [ ] 5.3 Draft-model scale-up; context extension experiments
- [ ] 5.4 Wave-2 gate: corpus + eval suite + anchor customer all present

---

## Milestone Board

| Milestone | Target | Status |
|---|---|---|
| M0 Pipeline proven | wk 2 | IN PROGRESS (CI verification left) |
| M1 Champion base chosen | wk 4 | PENDING |
| M2 Finance-adapted model passes gate | wk 8 | PENDING |
| M3 Product demo E2E | wk 10 | PENDING |
| M4 Production pilot live | wk 12 | PENDING |
| M5 Refresh cadence + Wave-2 gate | Q2 | PENDING |
