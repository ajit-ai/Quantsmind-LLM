# TAKSH — Master Development Roadmap to v3.0.0

**Document ID:** `TAKSH-V3-MASTER-ROADMAP`
**Version:** 1.0 — established 2026-11-11 by milestone L0.7.6
**Authority:** planning and specification only. Authorizes no purchase, training, dataset, compute, release or publication.
**Target:** **TAKSH v3.0.0 — Advanced Autonomous Software Engineering**

---

## 0. Status snapshot (verified 2026-11-11; never auto-upgraded)

| Item | Current status |
|---|---|
| Current milestone | **L0.7.6 — Master roadmap and release governance (IN_PROGRESS)** |
| L0.7.5 verdict | **BLOCKED** |
| L0.7.2 snapshot `l072-l04-fdb1fee2beaae9c2` | FROZEN_RESEARCH_ONLY, byte-identical |
| Training-authorized tokens | **ZERO** |
| Canonical `taksh-foundation-111m` checkpoint | **MISSING** (110,658,816 params, config qualified) |
| Python corpus | 1,050 samples / 1,380,611 eligible tokens (frozen) |
| Other 52 original languages | Zero qualified corpus tokens |
| Karkain (54th first-class) | DISCOVERED; zero corpus; kcc.exe `0xC0000135` on host; compiler pinned 1.1.0; TAKSH-KARK-BENCH 0 tasks |
| Frozen Python benchmark | **0/30** |
| Owner authorization | **MISSING** — `docs/L075_L080_OWNER_DECISION_RECORD.md` is BLANK by owner instruction |
| Compute | **NOT QUALIFIED** — L0.7.4: MX150 4 GiB CUDA 11.1, CPU-only torch; host ~95% memory-saturated |
| CI | production trigger policy; manual qualification workflow exists on branch, **invisible to dispatch API from non-default branch** |
| Python 3.11 runtime | CONDITIONAL (static analysis only; never executed) |
| External data acquisition | **NOT AUTHORIZED** |
| Paid compute | **NOT AUTHORIZED** |
| Publication | **NOT AUTHORIZED** |
| Package version | `1.0.0` |

**No capability is claimed for TAKSH at any point in this roadmap until a frozen executable benchmark proves it. No version number, milestone or document grants authority.**

---

## 1. Identity (non-negotiable)

**TAKSH — Independent Adaptive Coding Intelligence**

TAKSH is an independently trained, locally deployable coding intelligence platform specializing in:

- programming across the original **53 programming languages** plus **Karkain** (54th first-class);
- **independent foundation-model** development (own tokenizer, own corpus lineage, own architecture);
- the **Dynamic Language Comprehension and Adaptation Engine (DLCAE)**;
- code generation, explanation, debugging and repair;
- testing, validation and software quality engineering;
- repository understanding and multi-file engineering;
- controlled autonomous software development;
- continual learning with internal knowledge formation (and optional retrieval — never a RAG-only substitute);
- reproducible, benchmark-proven capability.

TAKSH is **not** a general chatbot, a retrieval-only assistant, a wrapper around external models, or a conventional IDE extension.

---

## 2. Authority and precedence

| Document | Role | Authority order |
|---|---|---|
| This file (`docs/TAKSH-MASTER-ROADMAP-v3.0.0.md`) | v3 roadmap, v1–v3 release composition | forward roadmap authority (planning only) |
| `configs/roadmap/taksh-v3-roadmap.yaml` | machine-readable registry | operative registry for validation/audit |
| `docs/TAKSH-RELEASE-GOVERNANCE.md` | versioning + release qualification + authorization boundaries | release-qualification authority |
| `docs/TAKSH-ROADMAP-CHANGELOG.md` | roadmap revision history | audit trail; never rewritten |
| `docs/ROADMAP.md` | M0–M7 historical record (v1.0.0 baseline, published 2026-10-21) | historical record |
| `docs/roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md` | previous forward roadmap (v1.1.0+ planning) | **superseded for milestone planning by this document** (v3.0.0 milestone program); history retained |
| `docs/L0600_R1_ARCHITECTURE_ALIGNMENT_REPORT.md` | L0.6.0R1 alignment baseline | architectural baseline |
| `docs/L075_L080_OWNER_DECISION_RECORD.md` | blank owner authorization record | owner-only document |
| `docs/L075_L080_ENTRY_CONTRACT.md` | L0.8.0 preconditions | authorization-preconditions authority |
| `docs/L075_FINAL_PRETRAINING_READINESS_REPORT.md` | L0.7.5 final readiness verdict (BLOCKED) | readiness authority |

Rules:

1. Historical documents govern what happened; this roadmap governs what happens next.
2. Where this roadmap conflicts with an earlier milestone *intent*, the earlier milestone's *record* still stands as a record, but the forward path is defined here.
3. No document under this roadmap authorizes spend, training, acquisition, publication, or default-branch changes.

---

## 3. Release architecture

| Release | Name | Objective | Gate(s) |
|---|---|---|---|
| L0 | Independent foundation and learning | tokenizer → pretraining → coding specialization → benchmark qualification | L0.10.0 + L0.10.3 |
| v1.0.0 | Independent Adaptive Coding Assistant | local, independent coding intelligence with benchmark-proven capability | L1.0.0 → L1.9.0 maturity |
| v2.0.0 | Repository-Aware Software Engineering | from snippets to complete repository engineering | L2.9 → L2.10 |
| v3.0.0 | Advanced Autonomous Software Engineering | controlled, evidence-driven autonomous software engineering platform | L3.6 → L3.7 |

**Initial L1 language target:** Python, JavaScript, TypeScript, C, C++, Java, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Dart, SQL, Bash (16) + Karkain = **17**.

---

## 4. Milestone program (status recorded, never invented)

Authority: `configs/roadmap/taksh-v3-roadmap.yaml` (machine-readable).

### L0 — Foundation (13 qualified milestones, 1 active, 1 blocked-decision, rest pending)

| Milestone | Title | Status |
|---|---|---|
| L0.5 | Model feasibility | **CONDITIONAL_PASS** (owner-approval pending) |
| L0.5.1 | Multilingual feasibility | **CONDITIONAL_PASS** (owner-approval pending) |
| L0.6.0 | Foundation model qualification | **QUALIFIED** |
| L0.6.0-R1 | Architecture alignment (DLCAE, Karkain) | **QUALIFIED** |
| L0.6.1 | Training pipeline (DLCAE-aware) | **QUALIFIED** |
| L0.6.2 | Local inference qualification | **CONDITIONAL_PASS** (host blocked) |
| L0.7.0 | Data acquisition readiness | **QUALIFIED** |
| L0.7.1 | Multilingual corpus engineering | **QUALIFIED** |
| L0.7.2 | Corpus freeze & eligibility | **BLOCKED** (FROZEN_RESEARCH_ONLY; owner sign-off needed) |
| L0.7.3 | Training readiness & CI regression closure | **QUALIFIED** |
| L0.7.4 | CI execution, runtime & resource strategy | **QUALIFIED** (execution blocked; documented) |
| L0.7.5 | Final authorization readiness | **BLOCKED** (owner decisions pending) |
| L0.7.6 | Master roadmap & release governance | **IN_PROGRESS** |
| L0.8.0 | Foundation code pretraining | **BLOCKED** (owner gate; not started) |
| L0.8.1 | Training stability | NOT_STARTED |
| L0.8.2 | Language balance | NOT_STARTED |
| L0.9.0 | Coding instruction training | NOT_STARTED |
| L0.9.1 | Code generation | NOT_STARTED |
| L0.9.2 | Debugging and repair | NOT_STARTED |
| L0.9.3 | Testing and reasoning | NOT_STARTED |
| L0.9.4 | Multilingual alignment | NOT_STARTED |
| L0.10.0 | Frozen benchmark qualification | NOT_STARTED |
| L0.10.1 | Multilingual qualification | NOT_STARTED |
| L0.10.2 | Safety and reproducibility | NOT_STARTED |
| L0.10.3 | Release candidate | NOT_STARTED |

### v1.0.0 — L1 (all NOT_STARTED)

| Release | Milestone | Scope |
|---|---|---|
| v1.0 | L1.0.0 | Coding assistant GA (17 languages initial target; benchmark-proven) |
| v1.1 | L1.1.0 | Accuracy and reliability |
| v1.2 | L1.2.0 | 12 enterprise/legacy languages |
| v1.3 | L1.3.0 | 10 scientific/functional languages |
| v1.4 | L1.4.0 | Full 53-language expansion + Karkain qualification as evidence exists |
| v1.5 | L1.5.0 | Advanced programming |
| v1.6 | L1.6.0 | Cross-language programming |
| v1.7 | L1.7.0 | Developer productivity |
| v1.8 | L1.8.0 | Local inference optimization |
| v1.9 | L1.9.0 | L1 maturity qualification |

### v2.0.0 — L2 (all NOT_STARTED)

| Release | Milestone | Scope |
|---|---|---|
| v2.0 | L2.0 | Repository understanding |
| v2.0 | L2.1 | Multi-file code generation |
| v2.0 | L2.2 | Repository debugging |
| v2.0 | L2.3 | Test-driven development |
| v2.0 | L2.4 | Build and dependency management |
| v2.0 | L2.5 | Code review and refactoring |
| v2.0 | L2.6 | Controlled Git-aware development |
| v2.0 | L2.7 | Project documentation |
| v2.0 | L2.8 | Software quality engineering |
| v2.0 | L2.9 | Repository-engineering qualification |
| v2.0 | L2.10 | Repository-aware coding assistant |

### v3.0.0 — L3 (all NOT_STARTED)

| Release | Milestone | Scope |
|---|---|---|
| v3.0 | L3.0 | Task planning |
| v3.0 | L3.1 | Controlled task execution |
| v3.0 | L3.2 | Autonomous implementation |
| v3.0 | L3.3 | Autonomous debugging |
| v3.0 | L3.4 | Multi-stage engineering |
| v3.0 | L3.5 | Advanced multilingual engineering |
| v3.0 | L3.6 | Engineering agent qualification |
| v3.0 | L3.7 | Advanced local coding platform |

---

## 5. Dependency graph and critical path

```
L0.5 → L0.5.1 → L0.6.0 → L0.6.0-R1 → L0.6.1 → L0.6.2 → L0.7.0 → L0.7.1 → L0.7.2 → L0.7.3 → L0.7.4 → L0.7.5 → L0.7.6
                                                                                           │ (CURRENT)
L0.8.0 (gated: owner auth + technical gates) → L0.8.1 → L0.8.2 → L0.9.x → L0.10.x → L1.0.0 → … → L3.7
   ▲ Blockers: owner decision record (A1.1/A2) · canonical 111M checkpoint · viable compute · legal eligibility for new corpora · CI execution · Python 3.11 runtime · frozen benchmark displaced honestly · Karkain toolchain repair and corpus
```

**Cross-cutting (all L0–L3):** DLCAE · evaluation/regression framework · safety and reproducibility · observability and audit · documented provenance · 53-language catalogue + Karkain.

Critical-path truth: **no version number upgrades readiness.** Every gate above is an evidence gate.

---

## 6. Architecture requirements (preserved in full)

**Eight logical subsystems** (not eight models; logical cooperation): Learning Engine · Knowledge Engine · Code Intelligence Engine · Engineering Engine · DLCAE · Memory Engine · Evaluation Engine · Agent Engine.

**DLCAE — six engines (preserved exactly):** Incremental Comprehension · Sequence Orchestration · Knowledge Formation · Language Adaptation · Validation · Learning Feedback.

**Progressive language comprehension hypothesis:** `C → C1 → C2 → C3 → C4` and independent sequences `D1 → D2 → D3`. Character/byte-level processing remains a *research hypothesis*— never a mandated strategy without comparative evidence. Retrieval vs internal knowledge formation vs hybrid: **no winner without measured experiments**.

**Karkain (54th first-class):** permanent across L0–L3. Separate benchmark categories: syntax · types/semantics · compilation · KIR · standard library · debugging/repair · algorithms · translation · repository engineering · adaptation to new Karkain features. Pinned SPEC/compiler (1.1.0). Compiler-assisted ≠ model-only measurements. `.kark` remains the extension of record. KCC remains the authoritative validation oracle. `kcc.exe 0xC0000135` on the current host — external blocker, unresolved, **never bypassed**.

---

## 7. Cross-cutting engineering requirements (every milestone)

1. Functional correctness · 2. Reliability and reproducibility · 3. Performance and resource constraints · 4. Security and safe execution · 5. Privacy and data governance · 6. Legal provenance and licensing · 7. Observability and auditability · 8. Recovery and interruption handling · 9. Backward compatibility · 10. Documentation and developer experience · 11. Regression coverage · 12. Benchmark qualification.

> No arbitrary numeric targets are introduced by this roadmap. Acceptance criteria remain frozen in place until the owner approves a change; any change requires owner sign-off and a roadmap changelog entry.

---

## 8. Release qualification policy (summary)

Full policy: `docs/TAKSH-RELEASE-GOVERNANCE.md`. Every release must ship: exact commit, branch/remote verification, change inventory, architecture review, focused + regression evidence, CI results (or explicit CI limitation), frozen benchmark results, security assessment, measurements, known issues, **explicit capability claims tied to evidence**, and a release decision. Status vocabulary: NOT_STARTED · IN_PROGRESS · BLOCKED · CONDITIONAL_PASS · QUALIFIED · RELEASE_CANDIDATE · GA · DEFERRED. **No status upgrade without the gate.**

GA additionally requires: frozen benchmarks passing at owner-accepted thresholds · reproducibility across seeds · owner sign-off · no publication without explicit authorization.

---

## 9. Blockers (current, live)

| Blocker | Status | Owner action |
|---|---|---|
| Owner training authorization (A1.1) | **MISSING** | fill `docs/L075_L080_OWNER_DECISION_RECORD.md` |
| Compute environment authorization (A2) | **MISSING** | decide + authorize or provide environment |
| L0.8.0 entry contract preconditions | NOT MET | cannot begin L0.8.0 |
| Canonical 111M checkpoint | **MISSING** | requires authorized training run |
| Python 3.11 runtime qualification | CONDITIONAL (static only) | authorize interpreter provisioning or accept static-only |
| Full CI execution | **BLOCKED** on this branch | owner action: push CI qualification workflow to default branch / authorize PR |
| Karkain toolchain (`kcc.exe 0xC0000135`) | **BLOCKED** | owner-side repair of Karkain compiler environment |
| External corpus legal clearance | HUMAN_LEGAL_REVIEW_REQUIRED | counsel review before any acquisition |

**None of these resolve by reaching a version number.**

---

## 10. Risks and deferred decisions

| Risk / decision | Disposition |
|---|---|
| Python-only corpus vs 53-language ambition | Deferred to authorized acquisition legal clearance (A3.2); nothing acquired without it |
| Frozen Python benchmark 0/30 ceiling | Stands until an authorized run produces evidence; no pre-claiming |
| Character/token-level DLCAE hypothesis | Research; comparative experiments required before any mandated strategy |
| Retrieval integration in TAKSH | Optional; never a RAG-only substitute for internal knowledge formation |
| Karkain qualification order | Deferred until KCC/spec toolchain is repaired and evidence exists |
| Publication pathway (publication to PyPI/HF) | Deferred; unauthorized until owner grants release approval |
| v2/v3 milestone GA selections | Deferred until L0 gates are discharged with evidence |
| CI on the feature branch | Deferred pending owner default-branch action |
| Python-only misread as TAKSH scope | Explicitly rejected; catalogue unchanged; corpus is Python today by legal limit, not by ambition |
