# Quantsmind — Scope & Principles

> **What we are building:** one sovereign, from-scratch, multimodal model family —
> **Image → Text · Text → Text · Text → Image** — trained on your machine, owned by you.
> General-domain foundation; finance/trading is the first application vertical.

---

## 1. Mission

*India-owned multimodal intelligence: images and text, one model family, zero foreign base,
₹0 forever on the local path.*

## 2. What Quantsmind is

- A **from-scratch sovereign model family** (ADR-0003): architecture, tokenizer, data,
  training, and weights are all project-owned and in-repo.
- **Multimodal by core design**, not by bolt-on: one shared autoregressive transformer
  backbone (RMSNorm / RoPE / GQA / SwiGLU) jointly consumes and produces text and image tokens.
- **General domain foundation** with a **finance/trading vertical** as the first specialized
  application (SEBI/RBI-aware guardrails, tool-grounded numerics, India conventions).
- **₹0-first engineering**: every capability is first proven as a small local prototype on
  CPU / one consumer GPU (like today's ~11M text prototype) *before* any rented-GPU spend.
- **Understanding before generation**: Image→Text v1 → Text→Image v2.
- **B2B2C product path**: partners distribute a compliant engine via an OpenAI-compatible API.

| Direction | Status | Mechanism |
|---|---|---|
| Text → Text | v0 (done, ~11M proto, 46 tests) | QM Transformer decoder |
| Image → Text | Phase A (v1) | `VisionEncoder` (ViT) → MLP projector → decoder |
| Text → Image | Phase B (v2) | own VQ-VAE image tokens → shared AR decoder |

## 3. What Quantsmind is NOT (v1–v2)

- **Not an app business** — we build and license the engine; partners own the customer surface.
- **Not built on open-model cores** — open models are benchmark rivals in the eval harness only.
- **Not a black box** — model cards, eval reports, provenance ledgers, and repos are public.
- **Not a finance-only text chatbot** — finance is one vertical on a general multimodal model.
- **Not video/audio/3D** in v1–v2 — static multi-image interleaving is the v1 ceiling
  (multi-image scene reasoning over multiple charts/documents is supported).

## 4. Principles

### P1 — Sovereign from scratch, always
All weights learned from our own architecture and data. No open-base backbone in the product
core, now or later. *Enforced:* ADR-0003; open models appear in `evaluation/` as rivals only.

### P2 — One shared backbone, not adapters on a pile of models
A single decoder that handles image, text, and image-generation tokens. Vision encoder and
image tokenizer are trained from scratch and co-trained with the decoder. *Enforced:*
`training/model.py` is the one home for the stack; no frozen third-party feature extractor
in the product path.

### P3 — ₹0 before ₹₹₹
Every stage ships as a small local prototype first. Rented GPU is approved only when the
local gate is passed. *Enforced:* Phases A–B are CPU/1-GPU; Phase C starts only at M1/M2 exit.

### P4 — Understanding before generation
Image→text (comprehension) lands first; text→image (generation) is a later, gated phase.
*Enforced:* roadmap Phase A precedes Phase B.

### P5 — Prove it, then scale it
Evals, model cards, and gates precede every promotion. Private held-out sets are never
touched by training. *Enforced:* `evaluation/promotion.yaml`, per-modality gates,
decontamination checks.

### P6 — Data provenance and licensing ledger
Every text corpus and image set has a license holder, a data card, and a versioned record
before it trains anything. *Enforced:* `data/` pipeline writes ledger entries; legal gate
before any Phase C+ corpus use.

### P7 — India-first data residency (ADR-0002)
Default-deny egress (`QM_STRICT_RESIDENCY=1`): prompts stay on the controlled network.
Conscious override is a documented, auditable operator choice. Production targets Indian DCs.

### P8 — Hard numeric grounding
Arithmetic shown to users comes from the tool path (calculator), never free-token math.
*Enforced:* exact-match arithmetic suite is a hard-fail gate (Phase E).

### P9 — Honest public scorecard
We report sovereign-vs-open comparisons truthfully on every release (text, MM, and
generation), including when we lose. *Enforced:* release eval report + model card.

### P10 — Phase-gated spend and kill criteria
Each phase has a defined exit gate and a stop condition. Spending never runs past a failed
gate. *Enforced:* milestone gates M1–M7; budget table in ROADMAP.

### P11 — Reproducible on a laptop
Any engineer can run the local prototypes from a README command, pytest suite green.
*Enforced:* CI smoke jobs for text, MM-understanding, and generation.

## 5. Non-negotiables (violation blocks a release)

- [ ] Sovereign from-scratch weights (no open-backbone product core)
- [ ] Provenance/ledger current for every training corpus (text + image)
- [ ] Private held-out eval untouched by training (decontamination verified)
- [ ] `evaluation/promotion.yaml` gate fully green before any promotion
- [ ] Numeric claims via the tool path only
- [ ] Residency guard default-deny

## 6. Deferred / out of scope

| Item | Note |
|---|---|
| Video, audio, 3D, real-time generation | post-v2 exploration |
| Personalized investment advice | hard SEBI boundary; RA-partner feature later (ADR-0002) |
| Multi-tenant production serving | Phase F, gated |
| 7B flagship | funded + cluster plan only (M7) |