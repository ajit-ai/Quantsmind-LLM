# TAKSH Five-Version Implementation Baseline Report

**Program:** Five-Version Consolidated Implementation Program (1.1.0 → 2.0.0)
**Baseline:** `feature/taksh-five-version-ga-program` @ `0215c6d039fceffd334f8931aa96a3287010a580` (from `develop`)
**Date:** 2026-11-12
**Authorization:** A1.1-C preparation-only / A2-E no compute / TAKSH-OWNER-DECISION-2026-11-11

---

## 1. Repository Inventory (current state)

| Area | Files | Notes |
|---|---|---|
| `training/` | 86 | Complete training pipeline: pretrain, SFT, data, model, checkpoint, tokenizer contracts, validation, replay, geometry, scaling, precision |
| `serving/` | 20 | Local inference, OpenAI-compatible API, providers (TakshS2Provider, Seed-Coder, external adapters), security |
| `taksh/` | 2 | Package metadata (`__init__.py`, `__main__.py`) |
| `evaluation/` | 83 | L0.6.x–L1.0.0-ENG3 validators, DLCAE inference qualification, benchmark harnesses, release tooling |
| `configs/` | 34 | Foundation, model, tokenizer, data, train, feasibility, roadmap registries |
| `tests/` | 102 | ~600 focused tests across M3–ENG3 milestones |
| `docs/` | ~180 | Milestone reports, roadmap, architecture docs, audit registers |
| `data/` | 15+ | Corpus files, manifests, contamination audits (frozen research corpus) |
| `artifacts/` | 11 | Frozen corpus snapshot `l072-l04-fdb1fee2beaae9c2` |
| `taksh-hf-publication/` | 453 | Publication staging (owner-only, not integrated) |

## 2. Existing Capability Assessment

### 2.1 Native model (M3/v1.0.0 lineages)
- `training/model.py` — RMSNorm, RoPE, GQA, SwiGLU, tied embeddings
- `configs/model/code_m3_9m.yaml` — 9.36M-param researched target
- `configs/model/toy_ci.yaml` — tiny test configuration
- `configs/foundation/l060_foundation.yaml` — canonical 111M foundation config (L0.6.0)
- `configs/model/foundation_111m.yaml` — **missing; created in `feature/l110-native-model-foundation @ c7443f9`**

### 2.2 Inference and serving
- `serving/inference_engine.py` — ModelLoader, TokenizerAdapter, ContextManager, SamplingController, CancellationController, InferenceTrace
- `serving/local_inference.py` — `generate_ids()` with safe deserialization, EOS stopping, binding repair
- `serving/openai_server.py` — OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`, `/health`)
- `serving/providers.py` — TakshS2Provider, Seed-Coder preflight (fail-closed), provider registry
- `serving/generation_contract.py` — **new; created in `feature/l111-native-inference-mechanics @ 8c896b2`**

### 2.3 Training infrastructure
- `training/pretrain.py` — resumable pretraining, AdamW, cosine LR, distributed-safe
- `training/sft.py` — SFT loop, quarantine enforcement, loss masking
- `training/checkpoint.py` — atomic writes, fingerprint verification, safe load
- `training/geometry.py` — checkpoint/tokenizer/geometry compatibility inspector
- `training/tokenizer_contract.py` — control-token contract (v1/v2)
- `training/tokenizer_fingerprint.py` — **new; created in `feature/l111-native-inference-mechanics @ 8c896b2`**

### 2.4 Data and corpus
- Frozen research corpus: `artifacts/l072_l04_freeze/snapshot.json` — 1,050 technical-eligible records, 1,380,611 tokens (Python-only, research-only)
- 53-language registry (untracked owner artifact `l051_languages.yaml`)
- Karkain: 54th language, `DISCOVERED`, KCC binary fails (`0xC0000135`), KARK-BENCH 0 tasks

### 2.5 DLCAE and coding intelligence
- `taksh/dlcae/` — engine package (comprehension, sequences, knowledge, registry, adaptation, validation, learning, persistence, Karkain adapter)
- `taksh/dlcae/semantic/` — symbol/type/dependency analysis, toolchain adapters, failure correlation
- `taksh/dlcae/coding/` — end-to-end coding intelligence pipeline (model gate, orchestrator, dev tasks)

### 2.6 Evaluation and benchmarks
- Frozen Python benchmark: `854c7ca833…` (0/30, preserved)
- Multilingual benchmark: `7a839060…` (64 tasks, unmeasured)
- `evaluation/eval_code.py` — HumanEval/MBPP harness (never executed against a model)

## 3. Integration State (as of 2026-11-12)

| Branch | HEAD | Contents |
|---|---|---|
| `develop` | `0215c6d` | CI recovery (PRs #2/#3), SFT tokenizer fix, docs reconciliation |
| `main` | `d50de3e` | develop merged by owner (PR #1) |
| `feature/l110-native-model-foundation` | `c7443f9` | `configs/model/foundation_111m.yaml`, model architecture tests |
| `feature/l111-native-inference-mechanics` | `8c896b2` | Generation contract, tokenizer fingerprint, inference validation |
| `feature/l060-foundation-model-qualification` | `78b6cdf` | L0.6.x–L1.0.0-ENG3 milestones, DLCAE runtime, semantic validation |

## 4. What Is Missing

### 4.1 Model artifact
- Canonical `taksh-foundation-111m` checkpoint does not exist.
- `v180_sft` is GEOMETRY_INCOMPATIBLE (d_model 384 vs 768).

### 4.2 Training execution
- Training source code exists but is not authorized to run (A1.1-C).

### 4.3 Coding capability evaluation
- Frozen Python benchmark 0/30; no model has been measured against it.
- Coding intelligence orchestrator operates on random-weight fixtures only.

### 4.4 Karkain
- KCC `0xC0000135`; no corpus, no benchmark tasks, no qualification.

### 4.5 Production readiness
- No security audit, no deployment automation, no release artifacts, no GA decision.

## 5. Five-Version Program Boundaries

> **Version-label note.** This program uses `1.1.0`, `1.2.0`, `1.3.0`, `1.4.0`
> and `2.0.0` as consolidation labels for the five requested slices
> (Native Foundation → Native Model Capability → Coding Intelligence →
> Autonomous Engineering → GA). The long-range
> [`TAKSH_MASTER_DEVELOPMENT_ROADMAP.md`](TAKSH_MASTER_DEVELOPMENT_ROADMAP.md)
> uses a longer `v1.0.0`–`v3.0.0` index with different responsibilities. The two
> must not be conflated. The master roadmap remains the long-range plan until
> the owner reconciles the labels; this branch does not silently redefine it.
> The control matrix is
> [`TAKSH-PROGRAM-VERSION-GATES.md`](TAKSH-PROGRAM-VERSION-GATES.md), backed by
> `configs/roadmap/five_version_program.yaml` and
> `reports/taksh_version_gates.json`. Program labels are not package versions;
> the package remains `1.0.0` until an authorized release.

| Version | Authorization required | Status |
|---|---|---|
| 1.1.0 Native Foundation | None (source/config only) | Implementing now |
| 1.2.0 Model Capability | A1.1-A + A2 compute + corpus authorization | Blocked until owner approval |
| 1.3.0 Coding Intelligence | Depends on 1.2.0 artifact (or synthetic fixtures for source) | Source work possible |
| 1.4.0 Autonomous Platform | Depends on 1.3.0 capability evaluation | Design and CLI/SDK work possible |
| 2.0.0 Production GA | Owner release decision after 1.4.0 | Not started |

## 6. Authorization and Limitations

- **A1.1-C:** Preparation and preflight only; no training.
- **A2-E:** No compute authorization.
- **`TAKSH_ENABLE_TRAINING_CI`:** Must remain unset.
- **Frozen benchmark `854c7ca8…`:** 0/30, never to be modified or re-scored without owner authorization.
- **v180_sft:** GEOMETRY_INCOMPATIBLE; never substituted for canonical 111M.
- **Karkain:** DISCOVERED; KCC `0xC0000135`; KARK-BENCH 0 tasks.
- **Publication:** No tags, releases, PyPI/Hugging Face uploads without explicit owner decision.
- **develop/main merges:** Require explicit owner authorization per PR.

## 7. Next Actions (executed in this session)

1. Port 1.1.0 native foundation (config, model, inference contract) from `feature/l111-native-inference-mechanics` into this program branch.
2. Implement 1.2.0 source components: training readiness report (machine-readable), checkpoint qualification helpers, evaluation framework core.
3. Implement 1.3.0 source components: coding task orchestration engine, Python specialist interfaces, repair loop.
4. Implement 1.4.0 source components: project understanding (file index, symbol index), multi-file engineering primitives, CLI extensions, Python SDK.
5. Implement 2.0.0 source components: security checklist/probes, deployment documentation, compatibility matrix, qualification report generator.

All work is source-code implementation with focused non-training tests. **No training, checkpoint creation, data acquisition, compute spend, publication, or merge to develop/main is authorized.**

## 8. Cross-cutting distribution planning

Channel readiness is assessed in
`docs/roadmap/TAKSH-DISTRIBUTION-AND-PUBLICATION-PLAN.md`, with machine-readable
evidence in `reports/taksh_publication_readiness.json` generated by
`python -m evaluation.publication_readiness`. As of 2026-11-12, only the source
package and a chat-only local OpenAI-compatible surface are partially prepared;
Hugging Face model/dataset, Cline, OpenCode and OpenRouter are blocked or not
started.

---

**Product status:** MODEL_UNAVAILABLE · ACTUAL_CODING_CAPABILITY NOT_EVALUATED · Karkain DISCOVERED · L1.0.0 GA NOT_ESTABLISHED · autonomous engineering NOT_ESTABLISHED
