# Taksh - Development Roadmap (M0–M7 record and history)

> **Forward planning.** From `v1.1.0` onward, the canonical development roadmap is
> [roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md](roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md).
> It defines the approved version responsibilities (`v1.1.0`–`v3.0.0`), the
> milestone groups M-A–M-E and the per-version release gates. Where this file
> states a future intent that conflicts with it, the master roadmap governs.
>
> **Authority for history.** This document remains the authoritative record of
> the completed M0–M7 programme and of the decisions taken up to `v1.0.0`.
> Where another historical document defines a completed milestone differently,
> this record governs.
>
> **Direction (2026-10-14 pivot):** QuantsMind-LLM is a **Sovereign Coding LLM** —
> Python, JavaScript/TypeScript, software engineering. See
> [CODE_LLM_STRATEGY.md](CODE_LLM_STRATEGY.md) for the product definition.
>
> **Multimodal is deferred, not deleted.** The historical multimodal plan is
> retained verbatim in [Deferred / Future Multimodal Track](#deferred--future-multimodal-track)
> at the end of this file. It does **not** define the current execution path unless
> explicitly reactivated. Supporting detail:
> [SCOPE_AND_PRINCIPLES.md](SCOPE_AND_PRINCIPLES.md) · [VERSIONING.md](VERSIONING.md) ·
> [STATUS.md](STATUS.md) · [OPEN_ISSUES.md](OPEN_ISSUES.md).

**Scope**: a sovereign, from-scratch coding model family · permissive-licensed public
code + internal corporate code · B2B2C product path.
**Strategy (ADR-0003, extended)**: sovereign weights only; **₹0 local prototype before any
rented-GPU spend**.
**Governance**: a milestone slice is not accepted because tests pass. Four validation
layers and six acceptance states are defined in the M2 execution record
(`docs/source/roadmap/M2_EXECUTION.rst`).

---

## Milestones (authoritative)

| M | Milestone | Gate |
|---|---|---|
| M0 | **Foundation** — architecture, tokenizer, model implementation, initial data pipeline, serving foundation, repo infrastructure | Complete (infrastructure only; no capability claim) |
| M1 | **Data, Token, and Training Correctness** — dataset integrity, token contracts, tokenizer correctness, deterministic encoding, smoke correctness, checkpoint correctness, provenance, licensing/contamination controls, artifact correctness | Substantially implemented; limitations open — see M1 record below |
| M2 | **Dataset Expansion and Training Readiness** — everything required before serious base-model training | **NOT COMPLETE** — 12 slices below, plus the 15-condition completion gate |
| M3 | **Base Model Training** — real pretraining after M2 gates pass | **IMPLEMENTED, AUTOMATED TESTED, END-TO-END DEMONSTRATED** — 2,853,280-param model, 400 steps, 1,638,400 tokens, train loss 8.67→3.06, val 5.34→4.06 (ppl 58), 4.59 nats better than random init; checkpoint reloaded in a fresh process and generated locally. Generated code does **not** parse. Awaiting user acceptance and technical review |
| M4 | **Instruction / Conversational Tuning** - SFT and instruction following | **IMPLEMENTED, NEGATIVE CAPABILITY RESULT** - see the M4 execution record; no functional code evidence |
| M5 | **Evaluation and Capability Validation** - human + automated evaluation, coding benchmarks, robustness, safety, regression | **INSUFFICIENT_EVIDENCE** - see the M5 execution record |
| M6 | **Local Capability Validation** - local candidate acquisition and evaluation under the sovereign boundary | **INSUFFICIENT_EVIDENCE** - published M6 record; serving code predates it and is not an M6 acceptance |
| M7 | **Final packaging and public release** - package and close the completed M7 sovereign capability experiment at v1.0.0 | **RELEASE PUBLISHED** - capability is negative (0/40 locked syntax, 0/8 functional, evidence level 0); v1.0.0 Git tag and GitHub release published 2026-10-21; PyPI upload blocked on credentials; packaging, clean-install and serving evidence are in `docs/source/releases/M7_RELEASE_READINESS.rst` |

> **Milestone scope reconciliation (2026-10-20).** The published M6 record is
> *Local Capability Validation*, not serving acceptance. Serving code existed from
> earlier foundation work and was repaired for M7 compatibility. M7 is the
> **final packaging and public-release milestone** for the completed M7
> sovereign capability experiment; there is no M8 milestone. The experiment was
> previously documented under the interim target `v0.7.0`; that label is
> historical and was never released. Current package/release metadata is
> `v1.0.0` (Apache-2.0 source release), without asserting model capability. Historical
> artifacts retain their original milestone labels.

### M1 — Data, Token, and Training Correctness (substantially implemented)

Implemented: unified corpus pipeline with mandatory licence gating, token artifact with
verified provenance, token-aware PII scrubbing, document-separator encoding, strict
checkpoint loading, append-only loss logging, SFT dataset-coverage accounting,
end-to-end training smoke coverage.

**Preserved open limitations** (M1 is *not* production-ready):
- Corpus is **~2.24M tokens against a 100M target**.
- Two licensing decisions unresolved (`psf/black` GPL-2.0 subtree, `pypa/setuptools`
  AGPL-3.0 text) — excluded, awaiting human review.
- No role tokens in the tokenizer vocabulary.
- The pilot token artifact is general prose, not the code corpus.

### M2 — Dataset Expansion and Training Readiness

Authoritative status, acceptance state and evidence:
**`docs/source/roadmap/M2_EXECUTION.rst`**. Summary:

| Slice | Status | Evidence basis |
|---|---|---|
| M2.1 Architecture and configuration | Complete | 7 train configs validate; `validate_train_config()`; models defined |
| M2.2 Resumable data acquisition | Complete | `--resume`, interrupted-run recovery tests |
| M2.3 Corpus expansion | **Blocked** | 2,239,771 of 100,000,000 tokens (2.24%); needs allowlist expansion (human) + 2 licence decisions |
| M2.4 Unified dataset schema | Implemented / automated-tested / manual-tested | conformance measured on real shards; 4 manual-found diagnostic defects fixed |
| M2.5 Deterministic dataset processing | Implemented / automated-tested | 14 determinism/reproducibility tests |
| M2.6 Deduplication and corpus composition | Implemented / automated-tested | 11 dedup/leakage tests |
| M2.7 Tokenization and token budget | Implemented / automated-tested / manual-tested | separator/budget/artifact tests; code token artifact built from the released corpus |
| M2.8 Dataset versioning and release artifacts | Implemented / automated-tested / manual-tested | `data/release.py`: `RELEASE.json` + `CHECKSUMS.sha256`, 9 verify checks, token-to-shard provenance chain |
| M2.9 Training preparation | Implemented / automated-tested / manual-tested | config validation, strict checkpoints, loss log, e2e smoke; `pretrain_code_v2.yaml` targets the released artifact |
| M2.10 Evaluation preparation | Implemented / automated-tested / manual-tested | held-out set with verified separation; reproducibility records; human-eval rubric + reliability gate. **No human judgements collected** |
| M2.11 Documentation | Updated | `M2_EXECUTION.rst`, `M2_RELEASE_AND_GATE.rst`, acceptance checklist; Sphinx builds clean |
| M2.12 M2 release / training-readiness gate | **Implemented; verdict BLOCKED** | `data/m2_gate.py`: 1 technical failure, 3 blocked on human decisions |

**M2 is not complete.** The gate verdict is `M2 BLOCKED`: `python -m data.m2_gate`.
Three items require human decisions (two licence judgements, source
authorisation, annotator recruitment); no code change resolves them. Layers C
(user acceptance) and D (technical review) also remain outstanding for every
slice.

### M3 — Base Model Training (implemented, demonstrated, not accepted)

M3 trained a real model from random initialisation and proved the full loop:
train → checkpoint → reload in a fresh process → generate locally.

.. list-table:: M3 outcome
   :header-rows: 1
   :widths: 34 66

   * - Item
     - Value
   * - Model
     - 2,853,280 parameters, vocab 5822 (matched to `qm_code_bpe_v0`)
   * - Data
     - 1,165,217 train tokens from 7 licence-verified sources; 162 records
       withheld pending licence review; M2 holdout untouched
   * - Run
     - 400 steps, 1,638,400 tokens, 44.2 min on CPU
   * - Loss
     - train 8.6730 → 3.0633; validation 5.3431 → 4.0614 (ppl 209 → 58)
   * - Control
     - untrained model scores 8.6557 on the same windows: training gained
       **4.5943 nats**
   * - Checkpoints
     - `runs/m3_base/checkpoint_last.pt`, `checkpoint_best.pt` (33.5 MB each),
       with `run_manifest.json` carrying full provenance
   * - Generation
     - local, fresh process, own weights; **output does not parse as Python**
   * - Verdict
     - COMPLETE — AWAITING USER ACCEPTANCE and TECHNICAL REVIEW

Full record: `reports/M3_TRAINING_REPORT.rst` and
`docs/source/roadmap/M3_EXECUTION.rst`.

The model has **not** demonstrated useful coding capability. The binding
constraints are data volume (1.2% of the M2 corpus target) and CPU-only compute,
not code. No comparison against any other coding model is claimed.

---

## Deferred / Future Multimodal Track

> **Status: DEFERRED (2026-10-14 pivot).** The plan below is the project's earlier
> multimodal direction. It is retained **verbatim** for history and possible future
> reactivation. It does **not** define the current execution path, does not gate M2, and
> its M-numbering is superseded by the authoritative milestone table above.
> Current multimodal work is limited to maintaining existing code.
> Reactivating this track requires an explicit decision, recorded in
> [OPEN_ISSUES.md](OPEN_ISSUES.md) and reflected in this file.

### Historical plan (verbatim, superseded)

| M | Milestone | Target | Gate |
|---|---|---|---|
| ~~M1~~ | **MM-understanding local prototype** (image→text + text→text, ≤30M params, CPU/1-GPU) | end Phase A | ViT+decoder end-to-end loss sane; zero-shot caption works; VQA smoke passes; no text-task regression |
| ~~M2~~ | **Text→image local prototype** (own VQ-VAE tokens + AR decode, CPU/1-GPU) | end Phase B | image-token reconstruction ≥ floor; prompt→image qualitatively sane; style/checkpoint documented |
| ~~M3~~ | 300M multimodal pilot | end Phase C | smooth loss, MFU ≥40%, beats M1 prototype on MM eval; rivals scored |
| ~~M4~~ | Sovereign **1.5B multimodal base** | end Phase D | beats 300M across suite; India-finance floor met; tag `qm-mm-1.5b-v1` |
| ~~M5~~ | Aligned chat + first vertical passes gate | end Phase E | finance scores ↑; arithmetic 100% via-tool; refusal ≥95%; MM tasks ≥ floor; tag `qm-chat-1.5b-v1` |
| ~~M6~~ | Product layer E2E + pilot live | end Phase F | partner-key request → grounded disclaimed verified answer; load-tested 2× peak |
| ~~M7~~ | 7B decision | end Phase G | funded + cluster plan exist |

Phases A–G below are unchanged historical content.

---

## Phase A — Multimodal Understanding v0 (local, ₹0) — DEFERRED

- [ ] A1 Image data pipeline v0: downloaders, license/provenance ledger, dedup, data card
- [ ] A2 `VisionEncoder` from scratch: patchify → layer-norm → pos-enc/RoPE, ~10–20M params, unit tests
- [ ] A3 MLP projector + interleaved image-token decoding in the existing QM Decoder (special `<|img|>` tokens)
- [ ] A4 Image–text pairs v0 (~30–50k, licensed; India-aware captions where available)
- [ ] A5 Local prototype train: 7,767,680-param LLM + ~10M ViT → captioning + visual-QA on CPU/1-GPU
- [ ] A6 Eval: zero-shot captions, VQA seed, forgetting check on text-only tasks
- [ ] A7 CI: MM smoke-train job green

**Exit (M1):** image→text works end-to-end locally; no text-only regression; artifacts + model card.

## Phase B — Text→Image Generation v0 (local, ₹0) — DEFERRED

- [ ] B1 Image tokenizer from scratch (small VQ-VAE/VQGAN, ~5–15M params) trained on image set
- [ ] B2 Discrete image-token vocab merged into the shared decoder; autoregressive text→image sampling
- [ ] B3 Generation eval: reconstruction fidelity, sampling sanity, prompt adherence
- [ ] B4 Joint training on interleaved corpora (caption → image, image → caption objectives)
- [ ] B5 CI: text→image smoke job green

**Exit (M2):** text→image works locally; quality floor documented for scale-up.

## Phase C — 300M Multimodal Pilot (rented GPU, < ₹15K) — DEFERRED

- [ ] C1 Corpus v1 scale-out: text tier (12B tokens) + image-text tier
- [ ] C2 LR probes at 100M-token scale for the joint MM objective
- [ ] C3 300M MM pretrain (8×A100, 1–2 days) with NaN watchdog + async checkpoints
- [ ] C4 Eval battery on pilot + open MM rivals (Qwen-VL/LLaVA-class; SD-class for generation)
- [ ] C5 MFU report, checkpoint/resume drill

**Exit (M3):** smooth loss, MFU ≥40%, M1-prototype beaten on MM eval; honest scorecard published.

## Phase D — 1.5B Sovereign Multimodal Base (₹3–5 lakh) — DEFERRED

- [ ] D1 Corpus: 60B token text tier + MM tier scale-out; India vertical tail
- [ ] D2 32×A100 rental; storage + checkpoint-redundancy drill
- [ ] D3 1.5B MM pretrain with daily eval snapshots + spike playbook
- [ ] D4 Mid-train vertical tail (India finance + chart/doc), long-ctx extension (16k)

**Exit (M4):** sovereign 1.5B multimodal weights; beats pilot suite; tag `qm-mm-1.5b-v1`.

## Phase E — Post-Training: SFT + DPO + Safety — DEFERRED (multimodal scope)

- [ ] E1 SFT mixture (50–150k pairs: chat, chart inference, doc extraction, image-text QA, calculator-tool traces) + train
- [ ] E2 DPO pair generation + train (10–40k)
- [ ] E3 Safety pass: advisory-boundary refusals (finance vertical), jailbreak set, red-team round 1
- [ ] E4 Arithmetic exact-match suite (hard-fail, via-tool only)
- [ ] E5 Promotion gate vs base + rivals scorecard (`evaluation/promotion.yaml`)

**Exit (M5):** vertical scores ↑; arithmetic 100% via-tool; refusal ≥95%; MM task floor met.

## Phase F — Product Layer + Pilot Launch — DEFERRED

- [ ] F1 API gateway: tenant router, partner keys, quotas, billing hooks, SSE
- [ ] F2 RAG service: hybrid retrieval + market-snapshot context packer
- [ ] F3 Guardrails: injection filter, PII redaction, disclaimer injector
- [ ] F4 vLLM serving pool (AWQ-int4) + speculative decoding (self-trained draft)
- [ ] F5 Immutable per-message audit log + OTel dashboards
- [ ] F6 India-region deployment (Mumbai / Indian DC)
- [ ] F7 Pioneer B2B partner signed; shadow → canary → ramp
- [ ] F8 Weekly eval reports + feedback flywheel; go/no-go review

**Exit (M6):** E2E partner-key request → grounded, disclaimed, verified answer; load-tested 2× peak.

## Phase G — Compound & 7B Decision — DEFERRED

- [ ] G1 Data flywheel → monthly SFT/DPO refresh
- [ ] G2 Hindi/Hinglish + image-gen scale-up; context → 32k
- [ ] G3 7B multimodal run (₹30–60 lakh) only when revenue commitments + cluster plan exist

**Exit (M7):** evidence-backed 7B go/no-go.

---

## Parallel workstreams

- **Vertical layer (finance):** the existing 26/500 India-finance eval seed grows → held-out set v1;
  SEBI-aware guardrails and tool-mandated numerics remain (ADR-0002, scope doc).
- **Corpus/provenance:** every text and image corpus carries a data card; license ledger precedes use.

## Budget Summary (rental, A100-class)

| Item | Estimate |
|---|---|
| Phase C pilot + probes | < ₹15K |
| Phase D 1.5B pretrain | ₹3–5 lakh |
| Phase E post-training | ₹20–50K |
| Phase F serving (pilot scale) | ₹30–60K/mo |
| Phase G 7B (optional) | ₹30–60 lakh |

Local paths (Phases A–B) stay at ₹0.

## Risk Register

| Risk | Mitigation |
|---|---|
| MM dataset scarcity / licensing gaps | provenance ledger + legal gate before use; data cards |
| ViT-pretrain quality floor too low | M1 prototype gate; CLIP-score/zero-shot baselines first |
| MM joint training destabilizes text (forgetting / spikes) | co-training schedule from C2 probes; forgetting eval gate; spike playbook |
| Text→image dev cost or drift | image-tokenizer fidelity gate before scale-up (B3) |
| Both modalities regress vs uni-modal | per-modality gates in `promotion.yaml`; rivals as regression alarm |
| Compute overrun | phase-gated spend; hard kill criteria per phase |
| Advisory/compliance breach (vertical) | refusal training + disclaimers + audit trail + RA partner |

## Branching & Release Flow

- `feat/*` → PR into `develop` → release PR into protected `main` with eval report attached
- Model tags: `qm-mm-pilot-300m-vN`, `qm-mm-1.5b-vN`, `qm-chat-1.5b-vN` (git SHA + dataset tag + config hash)
- Rule: **every phase ends with a develop → main merge**