# Quantsmind-LLM — Development Roadmap

**Scope**: Finance/Trading vertical · Tier-1 use cases (RAG + tools) · B2B2C.
**Core (ADR-0003, FINAL)**: **from-scratch sovereign pretrain** — 300M pilot → 1.5B
"Quantsmind-Bharat" → 7B when funded. Open models = benchmark rivals only.
Durations assume 1–2 engineers + rented GPU. Hard exit criteria per phase; every phase
ends with develop→main merge.

---

## Phase 0 — Foundations (Weeks 1–2) — ~95% complete

- [x] Repo, CI, training stack (transformer + pretrain loop + checkpoint/resume + watchdogs)
- [x] Tokenizer trainer + tests; data encode pipeline; smoke pipeline
- [x] Product UI shell (Gradio) with India residency guard + SEBI-aware prompt
- [x] Strategy docs: ADR-0003 from-scratch sovereign; competitive analysis
- [~] Verify first GitHub Actions run green
- [ ] ruff lint in CI; ADR-0001 formalization

## Phase 1 — Data v0 + 300M Pilot (Weeks 2–6)

- [ ] 1.1 Corpus acquisition v0 → 12B tokens India-weighted general mix (web/wiki/code/finance)
- [ ] 1.2 Train 49k BPE tokenizer on corpus sample; pass round-trip + fertility tests
- [ ] 1.3 Dedup + quality filter + PII scrub + decontamination; provenance ledger live
- [ ] 1.4 Private India-finance held-out eval set v0 (~500 items, never trained on)
- [ ] 1.5 LR probes at 100M-token scale (2–3 short runs)
- [ ] 1.6 **300M pilot pretrain** — 8×A100, 1–2 days, <₹10K
- [ ] 1.7 Eval battery on pilot + benchmark rivals (open 300M-class models)

**Exit**: smooth loss (no unrecovered spikes), MFU ≥40%, checkpoint/resume drill passed,
pilot scores recorded vs rivals; data card v0 published.

## Phase 2 — 1.5B Quantsmind-Bharat Pretrain (Weeks 6–12)

- [ ] 2.1 Corpus scale-up to 60B tokens; India-finance layer as mid-train tail (~20%)
- [ ] 2.2 Provision 32×A100 rental; storage + checkpoint redundancy drill
- [ ] 2.3 **1.5B pretrain** (₹3–5 lakh) with daily eval snapshots + spike playbook armed
- [ ] 2.4 Mid-train finance tail; long-context extension (16k) if stable
- [ ] 2.5 Loss trajectory vs scaling-law prediction from pilot (tolerance gate)

**Exit**: sovereign 1.5B base weights exist; beats 300M pilot across suite;
India-finance set floor met; milestone tag `qm-base-1.5b-v1`.

## Phase 3 — Post-Training: SFT + DPO + Safety (Weeks 12–16)

- [ ] 3.1 SFT mixture build (50–150k pairs: chat, finance QA, extraction, summarization,
      calculator-tool traces) + train
- [ ] 3.2 DPO pair generation + train (10–40k pairs)
- [ ] 3.3 Safety pass: advisory-boundary refusals + jailbreak mixture; red-team round 1
- [ ] 3.4 Exact-match arithmetic suite (hard-fail, via-tool only)
- [ ] 3.5 Promotion gate vs base + benchmark rivals scorecard published

**Exit**: finance scores ↑ vs base; arithmetic 100% via-tool; refusal precision ≥95%;
no general-benchmark regression >2%; tag `qm-chat-1.5b-v1`.

## Phase 4 — B2B2C Product Layer (Weeks 16–18)

- [ ] 4.1 Gateway: tenant router, partner keys, quotas, billing hooks, SSE
- [ ] 4.2 RAG service: hybrid retrieval + market-snapshot context packer
- [ ] 4.3 Guardrails: injection filter, PII redaction, SEBI disclaimer injector
- [ ] 4.4 vLLM serving pool (1.5B AWQ-int4) + speculative decoding w/ self-trained draft
- [ ] 4.5 Immutable per-message audit log + OTel dashboards
- [ ] 4.6 India-region deployment target (Mumbai region / Indian DC)

**Exit**: partner-key request → grounded, disclaimed, tool-verified answer E2E; load test 2× peak.

## Phase 5 — Pilot Launch (Weeks 18–20)

> Positioning (COMPETITIVE_ANALYSIS.md): sell the compliant engine, not the app.

- [ ] 5.1 Sign pilot B2B partner (broker/wealth platform) + success criteria
- [ ] 5.2 SEBI-RA partner firm for advisory-adjacent features
- [ ] 5.3 Compliance pack: audit-log export, residency attestation, kill-switch demo,
      model-inventory row (RBI MD / SEBI CSCRF aligned)
- [ ] 5.4 Shadow week → canary 5% → ramp; weekly eval reports; feedback flywheel
- [ ] 5.5 Model card + eval report; go/no-go review

**Exit**: production SLOs met; partner + compliance sign-off; rollback rehearsed.

## Phase 6 — Compound & 7B Decision (Q3+)

- Data flywheel → monthly SFT/DPO refreshes; Hindi/Hinglish strengthening
- Context extension toward 32k; tool-use expansion
- 7B run (₹30–60 lakh) only when: pilot revenue/commitments + cluster plan exist

---

## Milestones

| M | Target | Gate |
|---|---|---|
| M0 Pipeline proven | wk 2 | CI green incl. smoke-train |
| M1 300M pilot trained + evals proven | wk 6 | exit Phase 1 |
| M2 Sovereign 1.5B base weights | wk 12 | exit Phase 2 |
| M3 Aligned chat model passes gate | wk 16 | exit Phase 3 |
| M4 Product layer E2E demo | wk 18 | exit Phase 4 |
| M5 Production pilot live | wk 20 | exit Phase 5 |
| M6 7B decision | Q3 | funded + cluster plan |

## Budget Summary (cloud rental, A100-class)

| Item | Estimate |
|---|---|
| Phase 1 pilot + probes | < ₹15K |
| Phase 2 1.5B pretrain | ₹3–5 lakh |
| Phase 3 post-training | ₹20–50K |
| Phase 4 serving (pilot scale) | ₹30–60K/mo |
| 7B (optional, Phase 6) | ₹30–60 lakh |

## Risk Register

| Risk | Mitigation |
|---|---|
| Loss spikes at scale | Spike playbook, async checkpoints, LR margin from probes |
| 1.5B quality floor too low for Tier-1 | Pilot gate before big spend; RAG+tools compensate; 7B is the escalation path |
| Data licensing gaps | Provenance ledger; legal review gate before pretrain |
| Benchmark contamination | Decontamination + private held-out set decides |
| Arithmetic hallucination to users | Tool mandate + exact-match hard-fail gate |
| Advisory-compliance breach | Refusal training + disclaimers + audit trail + RA partner |
| Competitor verticalizes first | 2-quarter window (COMPETITIVE_ANALYSIS.md); M2 demo is the counter-move |
| Compute cost overrun | Phase-gated spend; kill criteria per phase |

## Branching & Release Flow

- `feat/*` → PR into `develop` → release PR into protected `main` with eval report attached
- Model tags: `qm-pilot-300m-vN`, `qm-base-1.5b-vN`, `qm-chat-1.5b-vN` with git SHA + dataset tag + config hash
