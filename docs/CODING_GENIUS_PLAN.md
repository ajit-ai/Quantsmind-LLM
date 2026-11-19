# Taksh — Coding-Only Genius Plan

**Date:** 2026-11-14
**Direction:** Taksh is a coding-only AI tool. It is not a general assistant, a finance product, a multimodal system or a non-code autonomous platform.
**Aspiration:** build a world-class coding model that is sovereign, locally deployable and evidence-gated.
**Honesty constraint:** Taksh has **no demonstrated coding capability today**. Every target in this document is a release gate, not a result.

---

## 1. Identity

> **Taksh is a sovereign coding genius: the only coding intelligence that can run inside a controlled, auditable, locally deployed environment — with its own weights, tokenizer, data lineage, evaluation and release governance.**

The product is optimized for software engineering:

1. generating correct, executable code,
2. reasoning about code and algorithms,
3. finding and repairing defects,
4. generating and validating tests,
5. reviewing and refactoring under behavioral constraints,
6. understanding repositories and multi-file changes,
7. serving those workflows through an IDE, CI/CD and engineering-agent interface.

Useful adjacent capabilities are allowed only when they materially improve coding:
system-design assistance, API contract generation, migration planning, architecture review and documentation grounded in a repository.

No general chat product, non-code assistant, finance product, vision product or unrestricted autonomous agent is in scope.

---

## 2. Current measured state

| Layer | Current state | Evidence |
|---|---|---|
| Package | `taksh 1.0.0`, Apache-2.0 source release | `taksh/__init__.py`, `README.md` |
| Canonical model | `taksh-foundation-111m` is missing | `docs/L076_MASTER_ROADMAP_REPORT.md` |
| Largest relevant artifact | `v180_sft`, approximately 28.8M parameters | `docs/L073_TRAINING_READINESS_REPORT.md` |
| Frozen Python benchmark | **0/30** | Frozen Python benchmark and reports |
| Locked syntax-valid probes | **0/40**; **0/8** functional probes | M7 experiment record |
| Instruction following | Not demonstrated | No instruction-tuned product checkpoint |
| HumanEval / MBPP | Unmeasured; only 5-case smoke subsets exist | `docs/STATUS.md`, `docs/OPEN_ISSUES.md` |
| Capability claim | **NOT_EVALUATED / NOT DEMONSTRATED** | This repository |
| Corpus | 1,050 samples / 1,380,611 frozen eligible Python tokens | `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md` |
| Tokenizer | 5,822-vocab code BPE; no chat/control/FIM tokens | `tokenizer/taksh_code_bpe_v1.json` |
| Context | Canonical architecture 2,048 tokens | `configs/foundation/l060_foundation.yaml` |
| Karkain | DISCOVERED; `kcc.exe` fails with `0xC0000135`; zero Karkain tasks | `docs/TAKSH-MASTER-ROADMAP-v3.0.0.md` |
| Training + compute | Not authorized | `docs/L075_L080_OWNER_DECISION_RECORD.md` |

**Decision:** do not publish, tag, merge to protected branches or claim capability until the owner authorizes the relevant gate and the gate passes.

---

## 3. The product ladder

| Version | Working title | Scope | Gate before claim |
|---|---|---|---|
| v1.0.0 | Coding Genius Beta | Python completion, generation, repair, explanation and review | Frozen Python benchmark passes an owner-approved absolute bar |
| v1.1.x | Python Coding Genius | Robust instruction following, reliable code output, constrained repair | Measured instruction + execution gates; no regression |
| v1.2.x | Functional Correctness | Test-grounded function generation | HumanEval/MBPP-style execution pass rate above a frozen threshold |
| v1.3.x | Debugging and Tests | Fault localization, repair, generated tests | Repair/no-regression/mutation gates pass |
| v1.4.x | Developer Workflows | Review, documentation, IDE completion, latency/serving | Workflow gates and serving SLO pass on named profile |
| v1.5.x | Python Coding Genius Alpha | Frozen qualification and reproducible release decision | Reproduce release from manifests; independent review |
| v2.0.0 | Multilingual Coding Genius | 16 initial languages + Karkain when evidence exists | Per-language ≥16 executed tasks and ≥50% pass ratio |
| v2.1–v2.8.x | Repository Coding Genius | Multi-file generation, repository debugging, TDD, builds, review, quality | Frozen multi-file and repository gates pass |
| v2.9–v2.10.x | Repository Engineering GA | Repository-aware assistant and qualification | Repository qualification and owner GA decision |
| v3.0.0 | Agentic Coding Genius | Permissioned, auditable, coding-specialized agent | SWE-bench-class evidence, reproducibility, safety and owner GA decision |

These are gated versions, not dates. A version can be held or refused without invalidating prior work.

---

## 4. Capability bar

The bar measures code, not prose.

### 4.1 Foundation gates

- syntax validity: generated code parses in the target language;
- exact or structural AST equivalence where reference answers exist;
- parameter binding and undefined-name checks;
- sandboxed execution with timeout and no network;
- negative controls must not be credited;
- frozen benchmarks must be hash-verified before and after runs;
- repeated runs report variance; no single lucky run qualifies a release.

### 4.2 Product gates

| Gate | Proposed release target | Current state |
|---|---:|---:|
| Locked syntax-valid | ≥ 50% for the foundation release; ≥ 90% for Coding Genius | 0/40 |
| HumanEval pass@1 | > 0 first, then ≥ 20% before beta; ≥ 60% only after evidence | unmeasured |
| MBPP pass@1 | ≥ 25% beta; ≥ 65% aspirational | unmeasured |
| BigCodeBench pass@1 | ≥ 45% before broad claims | unmeasured |
| Repair success | ≥ 30% on frozen fault set | unmeasured |
| Generated-tests mutation score | ≥ 30% | unmeasured |
| Repository localization | ≥ 50% | unmeasured |
| Sandbox escapes | 0 | 0 observed, but no model capability |
| Secret leakage | 0 | not tested at model level |
| SWE-bench-class | ≥ 15% at v3.0.0 candidate | unmeasured |

No global performance claim may exceed the weakest claim that has evidence.

### 4.3 System gates

| Gate | Proposed threshold |
|---|---:|
| TTFT | ≤ 400 ms p95 on a published profile |
| TPOT | ≤ 50 ms p95 on a published profile |
| Memory bound | ≤ declared budget at 8 concurrent requests |
| Reproducibility | manifest rebuild matches hashes; documented nondeterminism only |
| Recovery | interruption resume without silent divergence |
| Security | no unauthenticated remote load; loopback by default; explicit `--allow-remote` only |

---

## 5. Execution order

### Step 0 — Stabilize the baseline (CPU-only)

1. Branch and artifact audit.
2. Close known defects in the current working tree without touching protected artifacts.
3. Canonical prompt renderer, checkpoint integrity and manifests.
4. Full regression suite.
5. Document evidence and known limitations.

### Step 1 — Coding-genius evaluation and foundation (CPU-only)

1. Harden evaluation runner and negative controls.
2. Add code-specific control tokens to the next tokenizer version; do not mutate frozen v1 tokenizer.
3. Add serving streaming and KV-cache qualification.
4. Build the coding-genius Design Architect subsystem as a source/config and engineering feature.

### Step 2 — Training authorization packet (owner decision)

1. Corpus contract and license memo.
2. Compute request and budget.
3. Training config and checkpoint plan.
4. Evaluation and rollback plan.
5. Legal review for external data.

### Step 3 — First real model (GPU, owner-gated)

1. Build governed 100M+ token corpus.
2. Train 32,768-token vocabulary tokenizer v2 with reserved control tokens.
3. Train canonical 111M model.
4. Binary pass@1 evaluation.

### Step 4 — v1.0.0 Coding Genius Beta

1. SFT on execution-verified code tasks.
2. Repair and tests.
3. Serving, IDE, safety gates.
4. Frozen qualification.
5. Honest release or refusal.

### Step 5 — v2.0.0 multilingual coding genius

Acquire and qualify languages one gate at a time. Python remains the anchor; Karkain remains a first-class target only after the toolchain is repaired.

### Step 6 — v3.0.0 agentic coding genius

Only after repository gates pass. Local-first, permissioned, auditable and coding-specialized.

---

## 11. Implementation plan and programme status

The approved plan for executing this direction, including reuse map, defect list, implementation units I0–I7, critical path, standing prohibitions and required owner decisions, is recorded in:

- `docs/audit/TAKSH-IMPLEMENTATION-PLAN.md` — full implementation plan;
- `docs/audit/TAKSH-FOUNDATION-PROGRAMME-STATUS.md` — live status register (statuses only change when their evidence gates pass).

Execution order: `I0 owner baseline acceptance → I1 quick correctness fixes → I2 native artifact integration → I3 CI review (parallel) → I4 capability recovery → I5 serving qualification → I6 Gradio product → I7 release & distribution`.

---

## 6. Design Architect — coding-adjacent feature

The next feature is an **agentic architecture and software-design specialist**, not a general chat mode. It converts requirements and repository context into reviewable artifacts:

- assumptions and open questions;
- ADRs;
- C4/Mermaid diagrams;
- API/data contracts;
- migration and rollout plans;
- risks and trade-offs;
- design review findings.

Implementation remains the coding engine's responsibility. P0 design is recorded in `docs/architecture/DESIGN-ARCHITECT.md`; the feature will be qualified through the same frozen-benchmark and evidence-gate process as the rest of Taksh and will not be treated as a proven capability until measured.

---

## 7. Governance boundaries

- No external data acquisition, model training, compute spend, publication, tag, release or protected-branch merge is authorized by this document.
- Training and compute require explicit owner decision records.
- Frozen benchmarks, tokenizer v1, frozen corpus, canonical config and historical checkpoints are protected artifacts.
- Karkain remains DISCOVERED until `kcc.exe` works and at least the declared KARK-BENCH subset can run.
- Capability claims are tied only to measured evidence.
- The owner can hold or refuse any version; holding a version never weakens its gates.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| fluent but wrong output | sandboxed execution + negative controls + human review |
| benchmark contamination | pinned suites, hashes, decontamination scan |
| small corpus overfitting | held-out splits, early stop, repeated runs |
| loss improves but code does not | token loss is never sole gate; execution is mandatory |
| multilingual ambition without data | per-language fail-closed gates; Python remains the core until proven |
| safety or secret leakage | secret scanning, sandbox, no network, explicit permission boundaries |
| model cannot run locally | quantization and serving measurements on declared hardware |
| unsupported market claims | statements are target-only; independent review before release |

---

## 9. Success definition

Taksh is not "the world's best coding AI" because a document says so. It becomes competitive only when:

1. a canonical model exists and is reproducible,
2. frozen coding benchmarks pass at owner-approved thresholds,
3. repository-level tasks pass with audit logs,
4. serving meets local latency and safety SLOs,
5. the evidence and limitations are published honestly.

The coding-genius focus is the product strategy; training and measured capability are what make it real.
