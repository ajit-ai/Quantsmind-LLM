# Quantsmind-LLM — Development Roadmap

From-scratch training program. Durations assume a small dedicated team (2–4 ML engineers)
and rented GPU capacity; adjust to your cluster reality. Each phase has explicit
**exit criteria** — do not scale up until the previous phase passes.

---

## Phase 0 — Foundations (Weeks 1–3)

| Workstream | Deliverables |
|---|---|
| Repo & CI | Scaffold (this commit), lint, unit tests, smoke-train of a 10M-param model in CI |
| Tokenizer | Train byte-BPE 49k on corpus sample; pass round-trip/fertility tests |
| Data v0 | Acquire + dedup + quality-filter first 100B raw tokens; provenance ledger live |
| Eval harness | lm-eval-harness wired for pilot sizes; finance bench stub; promotion.yaml drafted |
| Infra | GPU access verified end-to-end: launch, log to W&B, checkpoint to storage, resume |

**Exit criteria:** a 10M toy model trains to sane loss (< 4.0 on 1B tokens) with full
logging/checkpoint/resume working; tokenizer tests green.

## Phase 1 — Pilot Run 300M (Weeks 3–6)

- Train **300M on ~12B tokens** general mix (~8×A100 × 1–2 days per attempt)
- Purpose: validate *everything* cheaply — data quality, LR schedule, throughput, evals
- Ablate: 2–3 LRs via short scaling-law probes before the real run
- Evaluate: perplexity curves vs reference open models at same size, basic benchmarks

**Exit criteria:** loss curve smooth (no unrecovered spikes); MFU ≥ 40%; eval harness
produces comparable numbers to published 300M-class models (±10%); pipeline restarts
from checkpoint cleanly after induced kill.

## Phase 2 — Data Scale-Up + Mid-Training Design (Weeks 5–9, overlaps)

- Grow corpus toward 60–100B tokens; add domain corpus ingestion (filings/news/papers/synthetic QA)
- PII + decontamination audits signed off by second engineer
- Build SFT dataset (50–150k pairs) and DPO pair set via teacher-model distillation + human audit
- Long-context plan: YaRN experiment scripted and tested on the 300M pilot

**Exit criteria:** dataset v1 frozen, version-tagged, fully documented data card; decontam
report shows 0 benchmark contamination above threshold; SFT/DPO datasets reviewed.

## Phase 3 — Production Pre-Train 1.5B (Weeks 9–14)

- Train **1.5B on ~60B tokens**: general → +code/math curriculum, domain-heavy final 20%
- Ops: daily eval snapshots; spike playbook armed; budget checkpoints every 2k steps
- Mid-train continuation on finance corpus (+ long-context extension to 16–32k if stable)

**Exit criteria:** beats 300M baseline across suite; finance bench ≥ defined floor;
loss trajectory matches scaling-law prediction from pilot within tolerance.

## Phase 4 — Post-Training: SFT + DPO (Weeks 13–17)

- Chat template finalized; SFT on curated pairs; DPO on preference pairs
- Safety pass: adversarial SFT mixture + red-team round 1
- Iterate: 2–3 cycles of (eval → fix data → retrain) — budgeted explicitly

**Exit criteria:** MT-Bench ≥ 7.0 target or documented gap analysis; jailbreak hard-fail
set 100% blocked; refusal precision ≥ 95% on benign prompts.

## Phase 5 — Serving & Launch (Weeks 16–20)

- vLLM deployment, quantization bake-off (FP8 vs AWQ-int4), speculative decoding with 300M draft
- Gateway: auth, quotas, streaming, guardrails, canary routing, feedback capture
- Load test to 2× projected peak; chaos drill (kill worker mid-stream)
- Model card + eval report published internally; go/no-go review

**Exit criteria:** all SLOs met under load; rollback path rehearsed; sign-off from safety review.

## Phase 6 — Scale & Compound (Quarter 2+)

- 7B run on 200B+ tokens (requires multi-node plan from Phase 3 learnings)
- RL-with-verifiable-rewards for math/finance reasoning (GRPO-style)
- Context extension to 128k; tool-use/calculator integration for arithmetic guarantees
- Data flywheel: consented user feedback → monthly SFT/DPO refreshes
- Consider MoE variant for inference economics once dense 7B is stable

---

## Milestone Summary

| Milestone | Target date | Gate |
|---|---|---|
| M0 Pipeline proven (toy train) | wk 3 | exit P0 |
| M1 Pilot 300M done | wk 6 | exit P1 |
| M2 Dataset v1 frozen | wk 9 | exit P2 |
| M3 Base 1.5B pretrained | wk 14 | exit P3 |
| M4 Aligned chat model | wk 17 | exit P4 |
| M5 Production serving | wk 20 | exit P5 |
| M6 7B scale run | Q2 | roadmap |

## Risk Register

| Risk | Mitigation |
|---|---|
| Loss spikes at scale | Spike playbook, frequent async checkpoints, LR margin from probes |
| Data licensing gaps | Provenance ledger, license filter early in pipeline, legal review gate |
| Finance benchmarks contaminated | Strict decontamination + private held-out finance set |
| Compute cost overrun | Scaling-law probes before big runs; kill criteria per phase |
| Single-point infra failure | Multi-region checkpoint copies; elastic restart drills |
| Key-person dependency | Runbooks + paired rotations on data/training/serving |

## Branching & Release Flow

- `main` = protected release branch (tagged releases only)
- `develop` = integration branch (this work happens here)
- Feature branches `feat/*` → PR into `develop`; develop → main via release PR with eval report attached
- Every model artifact tagged with git SHA + dataset version + config hash
