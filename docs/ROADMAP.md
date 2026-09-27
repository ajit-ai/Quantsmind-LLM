# Quantsmind — Development Roadmap

> **One sovereign model family, two modalities.** Image → Text · Text → Text · Text → Image.
> Scope & hard rules: [SCOPE_AND_PRINCIPLES](SCOPE_AND_PRINCIPLES.md).

**Scope**: General multimodal foundation (image ↔ text) · one from-scratch sovereign
model family · B2B2C product path · finance/trading = first application vertical.
**Strategy (ADR-0003, extended)**: sovereign weights only; **₹0 local prototype before any
rented-GPU spend**; **understanding before generation** (image→text ships before text→image).
**Hard exit criteria per phase**; every phase ends with a develop→main merge.

| Phase | Focus | Device | Cost |
|---|---|---|---|
| A | Multimodal **understanding** v0 (local proof) | CPU / 1-GPU | ₹0 |
| B | Multimodal **generation** v0 (local proof) | CPU / 1-GPU | ₹0 |
| C | 300M multimodal pilot | 8×A100 | < ₹15K |
| D | 1.5B sovereign multimodal base | 32×A100 | ₹3–5 lakh |
| E | Post-training: SFT + DPO + safety | 4×A40/A100 | ₹20–50K |
| F | Product layer + pilot launch | serve pool | ₹30–60K/mo |
| G | Compound & 7B decision | 64+ GPUs | ₹30–60 lakh (if funded) |

---

## Milestones (north star)

| M | Milestone | Target | Gate |
|---|---|---|---|
| M0 | Text pipeline proven | done | 46 tests green; smoke train + CI |
| M1 | **MM-understanding local prototype** (image→text + text→text, ≤30M params, CPU/1-GPU) | end Phase A | ViT+decoder end-to-end loss sane; zero-shot caption works; VQA smoke passes; no text-task regression |
| M2 | **Text→image local prototype** (own VQ-VAE tokens + AR decode, CPU/1-GPU) | end Phase B | image-token reconstruction ≥ floor; prompt→image qualitatively sane; style/checkpoint documented |
| M3 | 300M multimodal pilot | end Phase C | smooth loss, MFU ≥40%, beats M1 prototype on MM eval; rivals scored |
| M4 | Sovereign **1.5B multimodal base** | end Phase D | beats 300M across suite; India-finance floor met; tag `qm-mm-1.5b-v1` |
| M5 | Aligned chat + first vertical passes gate | end Phase E | finance scores ↑; arithmetic 100% via-tool; refusal ≥95%; MM tasks ≥ floor; tag `qm-chat-1.5b-v1` |
| M6 | Product layer E2E + pilot live | end Phase F | partner-key request → grounded disclaimed verified answer; load-tested 2× peak |
| M7 | 7B decision | end Phase G | funded + cluster plan exist |

---

## Phase A — Multimodal Understanding v0 (local, ₹0)

- [ ] A1 Image data pipeline v0: downloaders, license/provenance ledger, dedup, data card
- [ ] A2 `VisionEncoder` from scratch: patchify → layer-norm → pos-enc/RoPE, ~10–20M params, unit tests
- [ ] A3 MLP projector + interleaved image-token decoding in the existing QM Decoder (special `<|img|>` tokens)
- [ ] A4 Image–text pairs v0 (~30–50k, licensed; India-aware captions where available)
- [ ] A5 Local prototype train: 11M LLM + ~10M ViT → captioning + visual-QA on CPU/1-GPU
- [ ] A6 Eval: zero-shot captions, VQA seed, forgetting check on text-only tasks
- [ ] A7 CI: MM smoke-train job green

**Exit (M1):** image→text works end-to-end locally; no text-only regression; artifacts + model card.

## Phase B — Text→Image Generation v0 (local, ₹0)

- [ ] B1 Image tokenizer from scratch (small VQ-VAE/VQGAN, ~5–15M params) trained on image set
- [ ] B2 Discrete image-token vocab merged into the shared decoder; autoregressive text→image sampling
- [ ] B3 Generation eval: reconstruction fidelity, sampling sanity, prompt adherence
- [ ] B4 Joint training on interleaved corpora (caption → image, image → caption objectives)
- [ ] B5 CI: text→image smoke job green

**Exit (M2):** text→image works locally; quality floor documented for scale-up.

## Phase C — 300M Multimodal Pilot (rented GPU, < ₹15K)

- [ ] C1 Corpus v1 scale-out: text tier (12B tokens) + image-text tier
- [ ] C2 LR probes at 100M-token scale for the joint MM objective
- [ ] C3 300M MM pretrain (8×A100, 1–2 days) with NaN watchdog + async checkpoints
- [ ] C4 Eval battery on pilot + open MM rivals (Qwen-VL/LLaVA-class; SD-class for generation)
- [ ] C5 MFU report, checkpoint/resume drill

**Exit (M3):** smooth loss, MFU ≥40%, M1-prototype beaten on MM eval; honest scorecard published.

## Phase D — 1.5B Sovereign Multimodal Base (₹3–5 lakh)

- [ ] D1 Corpus: 60B token text tier + MM tier scale-out; India vertical tail
- [ ] D2 32×A100 rental; storage + checkpoint-redundancy drill
- [ ] D3 1.5B MM pretrain with daily eval snapshots + spike playbook
- [ ] D4 Mid-train vertical tail (India finance + chart/doc), long-ctx extension (16k)

**Exit (M4):** sovereign 1.5B multimodal weights; beats pilot suite; tag `qm-mm-1.5b-v1`.

## Phase E — Post-Training: SFT + DPO + Safety

- [ ] E1 SFT mixture (50–150k pairs: chat, chart inference, doc extraction, image-text QA, calculator-tool traces) + train
- [ ] E2 DPO pair generation + train (10–40k)
- [ ] E3 Safety pass: advisory-boundary refusals (finance vertical), jailbreak set, red-team round 1
- [ ] E4 Arithmetic exact-match suite (hard-fail, via-tool only)
- [ ] E5 Promotion gate vs base + rivals scorecard (`evaluation/promotion.yaml`)

**Exit (M5):** vertical scores ↑; arithmetic 100% via-tool; refusal ≥95%; MM task floor met.

## Phase F — Product Layer + Pilot Launch

- [ ] F1 API gateway: tenant router, partner keys, quotas, billing hooks, SSE
- [ ] F2 RAG service: hybrid retrieval + market-snapshot context packer
- [ ] F3 Guardrails: injection filter, PII redaction, disclaimer injector
- [ ] F4 vLLM serving pool (AWQ-int4) + speculative decoding (self-trained draft)
- [ ] F5 Immutable per-message audit log + OTel dashboards
- [ ] F6 India-region deployment (Mumbai / Indian DC)
- [ ] F7 Pioneer B2B partner signed; shadow → canary → ramp
- [ ] F8 Weekly eval reports + feedback flywheel; go/no-go review

**Exit (M6):** E2E partner-key request → grounded, disclaimed, verified answer; load-tested 2× peak.

## Phase G — Compound & 7B Decision

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