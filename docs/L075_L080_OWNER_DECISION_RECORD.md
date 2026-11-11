# TAKSH — L0.8.0 Owner Decision Record

**Document:** L0.8.0 Pretraining Authorization  
**Baseline:** L0.7.5  
**Current readiness:** BLOCKED  
**Purpose:** Record explicit owner decisions before any foundation-model training begins.

## 1. Current verified baseline

| Item | Current status |
|---|---|
| Canonical foundation model | `taksh-foundation-111m`, 110,658,816 parameters |
| Canonical trained checkpoint | MISSING |
| Frozen research corpus | `l072-l04-fdb1fee2beaae9c2` |
| Technically eligible corpus | 1,050 samples / 1,380,611 tokens |
| Training-authorized tokens | ZERO |
| Language corpus coverage | Python only |
| Other 52 languages | Zero qualified corpus tokens |
| Karkain | DISCOVERED; zero qualified corpus tokens |
| Frozen Python benchmark | 0/30 |
| Current compute | Not qualified for training |
| L0.7.5 verdict | BLOCKED |

This document does not change any of these facts.

## 2. Decision A1.1 — Foundation-model training authorization

**Question:** Do you explicitly authorize TAKSH to begin L0.8.0 foundation-model pretraining using only the existing internally curated research corpus?

Select exactly one:

- [ ] **A1.1-A — Authorize a bounded research training run.**
- [ ] **A1.1-B — Do not authorize training yet.**
- [x] **A1.1-C — Authorize preparation and preflight only; no training.**

If A1.1-A is selected, the authorization is limited to:

- Existing frozen snapshot `l072-l04-fdb1fee2beaae9c2`.
- Existing internally curated Python corpus, subject to its documented conditions.
- Research-only experimentation.
- Reproducible training, checkpointing, recovery and evaluation.
- The canonical 111M architecture and its pinned configuration.
- Explicit reporting of data exposure, compute consumption, model changes and benchmark results.

**This authorization does NOT include:**

- External dataset acquisition or downloading.
- Commercial training or commercial distribution.
- Public model publication or Hugging Face upload.
- Paid APIs, GPU rental, cloud provisioning or purchases.
- Expanding the dataset or changing its licensing conditions.
- Replacing the canonical model with `v180_sft`.
- Claiming coding capability without benchmark evidence.

## 3. Decision A2 — Compute environment authorization

**Question:** Which compute approach may TAKSH engineering investigate or use?

Select one:

- [ ] **A2-A — Existing local hardware only.** No purchases or provisioning.
- [ ] **A2-B — Owner-provided GPU hardware.** Use only hardware explicitly identified and approved below.
- [ ] **A2-C — External compute investigation only.** Research options and report estimates; no account creation, provisioning or spending.
- [ ] **A2-D — External compute execution authorized**, subject to the explicit limits below.
- [x] **A2-E — No compute authorization yet.**

If A2-D is selected, complete all fields:

- Approved provider:
- Approved instance/GPU:
- Maximum total expenditure:
- Maximum runtime:
- Maximum storage expenditure:
- Approved region:
- Data-storage and deletion requirements:
- Authorization expiry date:

No blank field is to be interpreted as permission.

**Technical qualification requirement:** The selected environment must be measured against the actual model, optimizer, activation, checkpoint and recovery requirements. The preliminary 16 GiB VRAM recommendation is not a guarantee of feasibility.

## 4. Separate restrictions — unchanged unless separately approved

| Activity | Authorization |
|---|---|
| Existing internal research corpus | Conditional permission only |
| External corpus acquisition | Not authorized |
| External corpus legal clearance | Human legal review required |
| Commercial distribution | Not authorized |
| Public model publication | Not authorized |
| Paid compute | Not authorized unless explicitly bounded under A2-D |
| Karkain repository/compiler changes | Not authorized |
| Merge to `develop` or `main` | Not authorized |
| Git tags/releases | Not authorized |

## 5. Mandatory L0.8.0 entry conditions

Even after A1.1 and A2 decisions, the engineering team must independently verify:

1. Training authorization is explicit and recorded.
2. Every training record is legally and technically eligible for the authorized use.
3. The canonical model configuration and tokenizer fingerprints match.
4. A valid training initialization path exists; no incompatible checkpoint substitution.
5. Compute preflight passes on the actual selected environment.
6. Checkpoint integrity, recovery and reproducibility gates pass.
7. The frozen evaluation benchmark and acceptance thresholds remain unchanged.
8. The seven inherited CI regression fixes remain covered by tests, and CI limitations are disclosed.
9. Resource limits, stop conditions and failure handling are enforced.
10. No prohibited activity is triggered by the implementation.

**Any unchecked mandatory condition means NO-GO.**

## 6. Owner decision record

Owner decision date: 2026-11-11

A1.1 decision: **A1.1-C — Authorize preparation and preflight only; no training.**

A2 decision: **A2-E — No compute authorization yet.**

Additional restrictions or conditions (explicitly supplied by the owner, 2026-11-11):

- No model training, pretraining, fine-tuning or weight updates.
- No checkpoint creation or replacement through training.
- No external dataset acquisition, downloads, mirroring or ingestion.
- No GPU rental, cloud provisioning, purchases or other compute spending.
- No paid APIs, paid services or external model execution.
- No publication or distribution.
- No changes to `develop` or `main`.
- No tags, releases or publication.

No other decisions are modified, inferred or implied by this record. Every other entry in this record remains exactly as previously filed: A1.1-A and A1.1-B remain unchecked; A2-A/A2-B/A2-C/A2-D remain unchecked; the A2-D bounding fields remain unspecified and unauthorized; no paid purchase is authorized; no external data authority exists.

Owner authorization reference / evidence: `TAKSH-OWNER-DECISION-2026-11-11`

**A1.1-C authorizes preparation only. It does NOT authorize training, checkpoint generation, external data acquisition, compute spending, publication, or any change to `develop` or `main`. A preparation authorization must never be interpreted as a training authorization.**

## 7. Final authorization rule

A completed decision record is not itself proof that training is ready.

L0.8.0 may begin only after the selected authorizations are recorded, their evidence is validated, and every mandatory technical and legal entry condition passes.

Until then:

**TAKSH remains at L0.7.5 — BLOCKED.**

The complete L0–L3 roadmap, 53 original languages, Karkain as the 54th first-class language, DLCAE, independent foundation model and adaptive coding intelligence objectives remain unchanged.
