# Taksh Master Development Roadmap

**Document ID:** `TAKSH-ROADMAP-MASTER`
**Roadmap document version:** 2.0 (reconciled to the approved version and milestone structure)
**Status:** Canonical forward-development reference from `v1.1.0` onward; `v1.0.0` recorded as the shipped baseline
**Baseline release:** `v1.0.0` — Apache-2.0 source release, published 2026-10-21
**Repository branch:** `develop` (baseline commit `cb9b81e`; `v1.0.0` release commit `a156892`)
**Owner / sole approver:** `ajit-ai`
**Last updated:** 2026-10-21
**Authority of this document:** planning and specification only. It authorises no spend, no training run, no inference run, no tag and no publication.

---

## 0. How to read this roadmap

### 0.1 Authority and precedence

This document is the **canonical future-development reference** for Taksh from
`v1.1.0` onward. It carries the approved version responsibilities, the approved
milestone groups, and the engineering specification for every version.

Precedence rules:

- **Historical records stay authoritative for what happened.** The M0–M7
  records under `docs/source/roadmap/`, `docs/source/releases/`, `runs/**` and
  `reports/` remain the source of truth for completed work, including the
  negative M7 result. Nothing here rewrites them.
- **This document governs forward planning.** Where an older document states a
  future intent that conflicts with a version responsibility defined here, this
  document wins.
- **`docs/ROADMAP.md` keeps its history and gains a navigation banner.** The
  M0–M7 table, the M1/M2 slice status and the deferred multimodal track are
  preserved; the banner points here for forward planning.
- **No document may claim model capability that evaluation evidence does not
  support.** Unchanged by this reconciliation.

### 0.2 Requirements basis

Sections were written from three inputs:

1. **The approved version responsibilities** (§0.9) and **approved milestone
   groups** (§0.5). These are fixed inputs, not proposals.
2. **The mandatory ten-section schema** applied to every version (§0.4).
3. **Verified repository evidence**: `README.md`, `docs/ROADMAP.md`,
   `docs/STATUS.md`, `docs/VERSIONING.md`, `docs/RELEASING.md`,
   `docs/OPEN_ISSUES.md`, `docs/CODE_LLM_STRATEGY.md`,
   `docs/SCOPE_AND_PRINCIPLES.md`, `docs/RULEBOOK.md`, `docs/adr/0002-*`,
   `docs/adr/0003-*`, `docs/source/roadmap/M2_RELEASE_AND_GATE.rst`,
   `docs/source/roadmap/M7_EXPERIMENT_REPORT.rst`,
   `docs/source/releases/M7_RELEASE_READINESS.rst`, `CHANGELOG.md`,
   `configs/**`, `data/**` manifests, `evaluation/promotion.yaml`,
   `evaluation/m7_*` and `runs/**/run_manifest.json`.

Where a threshold is stated, its basis is named: **measured** (evidence exists
in the repository), **specified** (an accepted gate recorded in project
documentation) or **proposed** (a planning target awaiting owner approval).

### 0.3 Status vocabulary

Every requirement, threshold and claim in this document carries one of these
statuses. They are not interchangeable.

| Status | Meaning | May a release claim it? |
|---|---|---|
| **Measured** | Evidence exists in the repository, reproducible from a recorded command | Yes |
| **Specified** | A gate fixed by accepted project documentation | Not until measured |
| **Proposed** | A planning target awaiting owner approval | No |
| **Planned** | Work not started; no evidence | No |
| **Deferred** | Explicitly postponed beyond v3.0.0 | No |
| **Refused** | Evaluated and declined, with a recorded reason | No |
| **Blocked** | Cannot proceed without an owner decision (§23) | No |

**Gates, not forecasts.** All performance numbers are release gates. None is a
promise of outcome, a schedule commitment or a claim about current capability.

### 0.4 The ten-section schema

Every version section uses the same ten subsections, in this order:

1. **Objective and architecture** — what the version must achieve and the shape
   of the system that achieves it.
2. **Components and modules** — the code surfaces that must exist or change.
3. **Functional features** — user-visible behaviour delivered by the version.
4. **Dataset and training requirements** — data sources, governance, mixture,
   training configuration, provenance.
5. **Inference and API changes** — serving contract, endpoints, compatibility.
6. **Benchmarks and release gates** — what is measured, how, and what must pass.
7. **Hardware and compute considerations** — resource envelope and budget.
8. **Dependencies and risks** — prerequisites, blockers and mitigations.
9. **Deliverables and Definition of Done** — artifacts and the exit checklist.
10. **Explicit exclusions** — what the version deliberately does not do.

A version is **not** complete if any of the ten subsections is missing, even if
its work is delivered.

### 0.5 Milestone groups (approved)

| Group | Versions | Responsibility | Close condition |
|---|---|---|---|
| **M-A** | v1.1.0–v1.4.0 | **Python Learning Foundation** | A Python model that trains reproducibly from governed data, renders one prompt identity, and shows measured syntax and completion capability |
| **M-B** | v1.5.0–v1.9.0 | **Python Developer Capability** | Instruction following, functional correctness, debugging and tests, developer workflows, and a qualified Python alpha |
| **M-C** | v2.0.0–v2.6.0 | **Multilingual Expansion** | SQL and JavaScript, TypeScript/web, Shell and configuration, Java, Go, C#/.NET, C/C++, each with a measured capability level and retained Python |
| **M-D** | v2.7.0–v2.8.0 | **Software Engineering Intelligence** | Cross-language translation and repository-aware engineering, both measured |
| **M-E** | v2.9.0–v3.0.0 | **Production Qualification** | Production qualification passed, then a production-ready multilingual release |

Groups are **gates, not sprints**. A group closes only when every version in it
has passed its own gates; a version may not advance a group by being partially
delivered.

### 0.6 Programme flow

```
M-A                M-B                     M-C                         M-D              M-E
v1.1 integrity     v1.5 instruction        v2.0 SQL + JavaScript       v2.7 xlang       v2.9 qualification
v1.2 data/tok      v1.6 functional         v2.1 TypeScript + web       v2.8 repo-aware  v3.0 production
v1.3 pretraining   v1.7 debug + tests      v2.2 Shell + config
v1.4 syntax+comp   v1.8 dev workflows      v2.3 Java
                   v1.9 Python alpha       v2.4 Go
                                           v2.5 C# / .NET
                                           v2.6 C / C++
```

The v1 line is Python-only by design. Breadth begins at `v2.0.0`, after the
Python capability loop is measurable end to end. A breadth version is never
used to escape a Python gate.

### 0.7 Release independence (restated)

Each version is an independent release decision. A version may be released, held
or refused without invalidating a prior release. Holding a version does not
permit skipping its gates; refusing a version requires a published reason and an
updated plan. Promotion from `develop` to a release tag is always a separate,
explicit owner action (see §22.6).

### 0.8 Verified baseline facts (what v1.1.0 starts from)

These are measured facts about M7, not plans. Detail lives in §1.

- The M7 checkpoint has **9,359,104** parameters and was trained with
  `qm.plain_instruction/1` while the serving path rendered a different chat
  template.
- Locked evaluation: **0/40** syntax-valid, **0/8** functional, 0/29 functional
  cases; candidate validation loss 5.9473 against M3 4.9685 and M4 4.9220.
- The approved diagnostic is **`MODEL_UNDERTRAINED`**, with
  **`SERVING_PROMPT_MISMATCH`** as a contributing factor.
- The v1.0.0 release is a **source release**: no model weights were published
  and no capability claim was made.

### 0.9 Version index (approved)

| Version | Group | Approved responsibility | Key delivery |
|---|---|---|---|
| v1.0.0 | baseline | Existing foundation and source release | Shipped baseline; negative M7 result recorded |
| v1.1.0 | M-A | Training integrity and reproducibility | One prompt identity, deterministic training, provable checkpoints |
| v1.2.0 | M-A | Python data and tokenizer foundation | Governed corpus, trained tokenizer, frozen data contract |
| v1.3.0 | M-A | Python-focused pretraining | A Python base model that beats its controls |
| v1.4.0 | M-A | Python syntax and completion | Measured syntax validity and fill-in-the-middle completion |
| v1.5.0 | M-B | Natural-language instruction tuning | Instruction following on the frozen contract |
| v1.6.0 | M-B | Functional Python correctness | Executable, test-passing Python functions |
| v1.7.0 | M-B | Python debugging and test generation | Fault localisation, repair and generated tests |
| v1.8.0 | M-B | Python developer workflows | Review, docs, IDE completion, serving maturity |
| v1.9.0 | M-B | Python alpha qualification | Freeze, reproduce, qualify or publish a refusal |
| v2.0.0 | M-C | SQL and JavaScript foundation | Multilingual platform, SQL, JavaScript |
| v2.1.0 | M-C | TypeScript and web development | TypeScript correctness and web tasks |
| v2.2.0 | M-C | Shell scripting and configuration | Shell, YAML/TOML, Dockerfile, safety |
| v2.3.0 | M-C | Java programming | Compiling, test-passing Java |
| v2.4.0 | M-C | Go programming | Idiomatic, race-free Go |
| v2.5.0 | M-C | C# and .NET | Compiling C# and .NET idioms |
| v2.6.0 | M-C | C and C++ | Memory-safe, sanitizer-clean C/C++ |
| v2.7.0 | M-D | Cross-language translation | Behaviour-preserving translation |
| v2.8.0 | M-D | Repository-aware software engineering | Multi-file, tool-using repo tasks |
| v2.9.0 | M-E | Production qualification | SLOs, isolation, hardening, runbooks |
| v3.0.0 | M-E | Production-ready multilingual Taksh | Qualified production release |

### 0.10 Cross-version components

Four concerns recur in every version. They are specified once here and
referenced, never re-invented:

- **Prompt identity** — one canonical renderer shared by training, evaluation
  and serving; contract id travels with every artifact (§22.3).
- **Promotion gate** — the same declarative gate decides every promotion; a
  gate is never weakened inside a version (§22.5).
- **Evidence levels 0–5** — the unit of capability claim; a version may not
  claim a level its evidence does not reach (§22.4).
- **Release qualification** — the seven-point checklist that precedes any tag
  (§22.6).

### 0.11 What this document does not decide

It does not approve model sizes, compute budgets, benchmark thresholds as
achieved results, or deployment targets. Those are **proposed** inputs in the
relevant version and **owner decisions** in §23. Approving this document does
not approve any of them.

---

## 1. v1.0.0 — Existing foundation and source release

**Group:** baseline · **Model rung:** S2 (9,359,104 params) · **Evidence
level:** 0 (measured, negative) · **Status:** shipped 2026-10-21

### 1.1 Objective and architecture

v1.0.0 is the fixed starting point, not a version to be re-planned. It comprises
the M0–M7 experiment programme, a decoder-only transformer trained from scratch
on a Python code corpus, a FastAPI serving package, a declarative promotion
gate, a release toolchain and an Apache-2.0 source release. The objective was
to determine whether the current architecture, data and budget could produce a
useful Python coding model. The measured answer was **no**, and that answer is
recorded.

Architecture as shipped: byte-level BPE tokenizer with specials
`<pad> <bos> <eos> <unk>` (no reserved control tokens); decoder-only
transformer; `qm.plain_instruction/1` prompt dialect used in training; a
different chat template rendered at serving time; `quantsmind.checkpoint/1`
checkpoint schema (no tokenizer or contract binding).

### 1.2 Components and modules

| Module | Responsibility as shipped |
|---|---|
| `configs/model/*.yaml` | Model geometries, including the 9M used for M7 |
| `data/**` + manifests | Corpus, tokenised shards, source manifests |
| `training/**` | Preprocessing, tokenizer training, training loop |
| `serving/**` | FastAPI app, chat template, generation loop |
| `evaluation/**` | Locked suites, promotion gate, M7 diagnostics |
| `tools/**`, `scripts/**` | Release preparation and validation |
| `runs/**` | Run manifests and checkpoints (not committed) |

### 1.3 Functional features

Delivered: a trainable and servable model stack, reproducible evaluation
harnesses, a promotion gate definition, a release toolchain and an honest
public record of a negative capability result. No coding capability is claimed.

### 1.4 Dataset and training requirements

Python source from a vetted subset of repositories, 437 records / 2,239,771
tokens in `data/code_corpus_v2.jsonl`, against a 100M-token M2 target. M7 was
trained at 9,359,104 parameters. The shortfall against the data target and the
absent data-governance artifacts are inputs to v1.2.0, not retroactive defects
in this record.

### 1.5 Inference and API changes

None. v1.0.0 defines the baseline serving behaviour that v1.1.0 must correct:
`/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, with a chat
template that does not match training.

### 1.6 Benchmarks and release gates

| Measurement | Result | Status |
|---|---|---|
| Locked syntax-valid | **0/40** | measured |
| Locked functional | **0/8** | measured |
| Functional cases | **0/29** | measured |
| Candidate val loss | 5.9473 | measured |
| M3 / M4 control val loss | 4.9685 / 4.9220 | measured |
| Diagnostic | `MODEL_UNDERTRAINED` + `SERVING_PROMPT_MISMATCH` | measured |

The release gate for v1.0.0 was a **source release with an accurate README**,
which passed. Capability gates were not met and were not claimed.

### 1.7 Hardware and compute considerations

The M7 run completed within the project's existing single-machine envelope. No
new hardware class was provisioned. Exact device-hours are recorded in the run
manifest and are not restated here as a planning input.

### 1.8 Dependencies and risks

Dependencies: Python, PyTorch, FastAPI, the vetted corpus and the evaluation
harnesses. Carried risks: the corpus is too small for the data target; prompt
identity is broken; the checkpoint schema cannot prove which tokenizer or
contract produced it. All three are owned by v1.1.0 and v1.2.0.

### 1.9 Deliverables and Definition of Done

Source tree, configs, evaluation harnesses, promotion gate definition, release
notes, `CHANGELOG.md`, and the M7 experiment and release-readiness records.
Done: tag `v1.0.0` exists, the README claim matches the measured result, and the
negative finding is discoverable from `docs/ROADMAP.md` and
`docs/source/roadmap/M7_EXPERIMENT_REPORT.rst`.

### 1.10 Explicit exclusions

No weights, no capability claim, no hosted endpoint, no multilingual support, no
instruction-following claim, no re-interpretation of the M7 result. v1.0.0 is
closed and is not amended by later versions.

---

## 2. v1.1.0 — Training integrity and reproducibility

**Group:** M-A · **Model rung:** S2 (9,359,104 params, unchanged) · **Evidence
level target:** 0–1 · **Blocks:** all later versions

### 2.1 Objective and architecture

**Objective.** Make every later claim provable. After v1.1.0 the project must be
able to answer, from artifacts alone: which tokenizer, which configuration and
which prompt contract produced this checkpoint; and would the same command
produce the same run again. No capability improvement is attempted here.

**Architecture.** One canonical renderer becomes the only code path that
formats a prompt. Training, evaluation and serving all call
`training/chat_template.py`; no other module may build prompt strings. The
renderer owns a reserved control-token range (system, user, assistant, end-of-
turn, tool, fill-in-the-middle), each encoded to exactly one id. The checkpoint
schema moves to `quantsmind.checkpoint/2`, binding tokenizer hash, config hash,
contract id and code revision. Runs become deterministic by construction and
auditable by manifest.

### 2.2 Components and modules

| Module | Change |
|---|---|
| `training/chat_template.py` | Single renderer; contract `qm.chat_instruction/2`; control-token table |
| `training/tokenizer/**` | Reserved-special reservation; single-id assertion at build time |
| `training/checkpoint/**` | `quantsmind.checkpoint/2`; strict loader; tokenizer/config/contract binding |
| `training/train.py` | Seed plumbing, deterministic data order, loss-log schema |
| `training/manifest.py` | Manifest completeness: hashes, seeds, versions, command |
| `serving/app.py` | Renderer import; contract check at startup; typed `ContractMismatch` |
| `evaluation/**` | Prompt-identity and checkpoint round-trip checks |
| `tools/reproduce.py` | One-command reproduction from a manifest |

### 2.3 Functional features

- A prompt rendered for training is byte-identical to the same conversation
  rendered for serving.
- Loading a checkpoint whose tokenizer, config or contract disagrees with the
  server raises a typed error instead of silently producing garbage.
- A run manifest is sufficient to reproduce the run without oral history.
- A loss record is machine-readable and comparable across runs.

### 2.4 Dataset and training requirements

The v1.0.0 corpus is reused unchanged; this version changes the *pipeline*, not
the data. Data order is a deterministic function of seed and shard order.
Required artifacts: corpus hash, shard hashes, tokenizer hash, config hash,
code revision, command line, seed, device and library versions. Provenance
rules are those in §22.2; no new sources are admitted.

### 2.5 Inference and API changes

`/v1/chat/completions` renders through the canonical template. The server
reports its contract id at `/v1/models` and refuses a request whose declared
contract differs. Error taxonomy gains `contract_mismatch`,
`checkpoint_incompatible` and `tokenizer_missing`. No new capability endpoint.

### 2.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Prompt identity | 100% of fixtures byte-identical across train/eval/serve | specified |
| Legacy dialect leakage | 0 occurrences of `qm.plain_instruction/1` in the serving path | specified |
| Control-token encoding | Each reserved token encodes to exactly one id | specified |
| Checkpoint round-trip | Fresh process loads all 11 provenance fields or refuses | specified |
| Manifest completeness | 100% of required fields present | specified |
| Repeat-run identity | Two runs, same manifest: identical loss log, or documented nondeterminism with cause | specified |
| Reproduction drill | Rebuild from manifest on a clean environment within the documented interval | specified |
| Contract mismatch handling | Mismatch produces the typed refusal in 100% of cases | specified |

Syntax and functional numbers are recorded, not gated. v1.1.0 claims no
capability.

### 2.7 Hardware and compute considerations

No scale-up: one S2 (9M) rung. Compute is short-run and single-machine; the
reproduction drill needs a second clean environment. Budget class is **proposed**
(§23, D14) and must be approved before the drill is scheduled.

### 2.8 Dependencies and risks

Depends on the frozen v1.0.0 tree and a working tokenizer build. Risks: (a)
determinism is unattainable on some GPU kernels — mitigated by documenting the
exact nondeterministic operation and isolating it; (b) hidden second prompt path
— mitigated by a repo-wide grep gate in CI; (c) checkpoint schema churn — frozen
at `/2` for the remainder of v1.

### 2.9 Deliverables and Definition of Done

Renderer and contract spec; checkpoint `/2` schema and strict loader; manifest
schema; reproduction tool; CI gates for prompt identity and control tokens;
updated serving app; evidence bundle (repeat-run logs, reproduction transcript,
gap register). Done when every gate in §2.6 passes and the gap register lists
every remaining integrity gap with an owner.

### 2.10 Explicit exclusions

No data expansion, no tokenizer retraining (v1.2.0), no parameter-count change,
no instruction-tuning objective, no capability claim, no change to the M7
record. v1.1.0 does not attempt to fix the undertrained model.

---

## 3. v1.2.0 — Python data and tokenizer foundation

**Group:** M-A · **Model rung:** S2 (no training run required) · **Evidence
level target:** 0 · **Blocks:** v1.3.0

### 3.1 Objective and architecture

**Objective.** Replace the 2.24M-token corpus and the unbound tokenizer with a
governed, decontaminated, reproducible Python data foundation and a tokenizer
whose behaviour is measured. The 100M-token M2 target is the design point; a
shortfall is acceptable only if recorded and approved.

**Architecture.** A declarative **data contract** drives the build: sources,
allowlist, licence status, pinned commits, exclusion rules, dedup and
decontamination steps, split assignment and output hashes. The corpus is built
by one reproducible command into immutable shards. The tokenizer is trained from
the training split only, with the reserved control range of v1.1.0, and is
frozen with its hash recorded in the data contract.

### 3.2 Components and modules

| Module | Change |
|---|---|
| `data/contracts/python_v1.yaml` | New declarative contract |
| `data/build/**` | Acquire, filter, dedup, decontaminate, split, shard |
| `data/manifests/**` | Per-source manifests with commit, licence, hash |
| `training/tokenizer/**` | Train from train split; fertility report |
| `evaluation/decontamination/**` | N-gram overlap against locked suites |
| `tools/data_card.py` | Generates the data card from the contract |

### 3.3 Functional features

- One command rebuilds the corpus byte-identically from pinned sources.
- Every shard is traceable to source, licence and commit.
- A data card states size, composition, exclusions, dedup rate and
  contamination results in plain language.
- The tokenizer reports fertility on code, comments and natural language.

### 3.4 Dataset and training requirements

Sources: the existing vetted allowlist, widened only by an owner decision
(§23, D3). Licence gate: a source without a decided licence is **not**
included (§23, D1–D2). Dedup: exact and near-duplicate removal with recorded
rates. Decontamination: n-gram overlap against every locked suite, zero
tolerance. Splits: by source group, never by random line, to prevent leakage.
Targets: 100M tokens (specified); ≥ 60% Python; ≥ 8% natural-language
comments/docstrings; held-out Python ≥ 5%. Tokenizer vocabulary is **proposed**
(§23, D6) and must be fixed before training.

### 3.5 Inference and API changes

None user-visible. Serving continues on the v1.0.0 model; the new tokenizer is
not yet used for a released checkpoint. The tokenizer artifact is versioned and
loadable for v1.3.0.

### 3.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Corpus size | ≥ 100M tokens, or approved documented shortfall | specified |
| Licence gate | 100% of sources have a decided licence | specified |
| Allowlist | No source outside the approved allowlist | specified |
| Contamination | 0 n-gram overlaps with locked suites | specified |
| Dedup | Duplicate rate recorded; > 30% near-duplicate removed | proposed |
| Split integrity | 0 source groups straddling train/eval | specified |
| Rebuild identity | Two builds produce identical shard hashes | specified |
| Tokenizer fertility | Reported on code, comments and English; no regression vs v1.0.0 tokenizer on Python | proposed |

### 3.7 Hardware and compute considerations

CPU/IO bound: acquisition, dedup and tokenizer training. No GPU training run.
Storage grows with the corpus (order of a few GB at 100M tokens). A short
single-machine job is assumed; budget is **proposed** (§23, D14).

### 3.8 Dependencies and risks

Depends on v1.1.0 control-token reservation and manifest schema. Risks: (a)
licence ambiguity blocks sources — mitigated by excluding first and deciding
later; (b) dedup removes too much — mitigated by recording before/after rates
per source; (c) tokenizer trained on contaminated data — mitigated by training
on the train split only; (d) corpus shortfall — mitigated by a documented,
owner-approved deviation rather than silent padding.

### 3.9 Deliverables and Definition of Done

Data contract, build pipeline, manifests, shards, tokenizer artifact plus
fertility report, data card, decontamination report. Done when the corpus and
tokenizer rebuild identically twice, every gate in §3.6 passes, and the data
card is linked from the release notes.

### 3.10 Explicit exclusions

No model training (v1.3.0), no capability claim, no non-Python languages
(v2.0.0 onward), no instruction data, no tokenizer changes after freeze, no
widening of the allowlist without an owner decision.

---

## 4. v1.3.0 — Python-focused pretraining

**Group:** M-A · **Model rung:** S3 (16,257,280 params) · **Evidence level
target:** 1–2 · **Blocks:** v1.4.0

### 4.1 Objective and architecture

**Objective.** Train a Python base model that measurably beats the v1.0.0
controls on held-out Python, on the frozen v1.2.0 corpus and tokenizer, with no
change to prompt identity or checkpoint provability. This is the first
scale-up (9M → 16M) and the first legitimate occasion to claim learning.

**Architecture.** Decoder-only transformer from `configs/model/code_30m.yaml`
(16,257,280 parameters, verified twice), trained on the v1.2.0 shards with the
frozen tokenizer. Pretraining is raw-text next-token prediction; the canonical
renderer is not involved, but the tokenizer's reserved control range is
preserved. Checkpoints use `/2` and are written with full manifests.

### 4.2 Components and modules

| Module | Change |
|---|---|
| `configs/model/code_30m.yaml` | Verified geometry; no silent edits |
| `training/train.py` | Deterministic pretraining loop; checkpoint `/2`; loss schema |
| `training/data_loader.py` | Shard streaming in manifest order |
| `training/schedules.py` | Warmup/decay; recorded in config |
| `evaluation/perplexity.py` | Held-out Python perplexity |
| `runs/**` | Manifests and checkpoints (never committed) |

### 4.3 Functional features

- A base model that continues Python tokens better than the v1.0.0 model.
- A loss curve and perplexity record that are comparable across runs.
- A checkpoint that loads only with its own tokenizer and config.

### 4.4 Dataset and training requirements

Data: frozen v1.2.0 shards only; no new sources in this version. Mixture per the
data contract (≥ 60% Python, ≥ 8% comments/docstrings). Sequence length, batch
size, learning rate and schedule are recorded in the run manifest; values are
**proposed** (§23, D7) and fixed before the run. At least one repeat run for
variance. Controls: the v1.0.0 M7 checkpoint and the recorded M3/M4 runs.

### 4.5 Inference and API changes

No serving change. The new checkpoint is evaluated offline; it is not deployed
because it is not yet syntax- or instruction-capable.

### 4.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Held-out Python loss | Lower than both M3 and M4 controls | specified |
| Held-out perplexity | < 15 | proposed |
| Repeated-run variance | Spread reported; median used for the gate | specified |
| Tokenizer/contract binding | 100% of checkpoints load or refuse | specified |
| Corpus identity | Trains only on v1.2.0 shard hashes | specified |
| Smoke syntax | > 0.05 on the locked suite (smoke only, not a claim) | proposed |

The smoke syntax figure exists to catch a dead model. It is **not** a capability
claim and does not satisfy any v1.4.0 gate.

### 4.7 Hardware and compute considerations

A 16M-parameter model at ~100M tokens is a single-GPU job in the project's
existing class. Estimated device-hours are **proposed** (§23, D14); no new
hardware is requested. Wall-clock depends on the sequence length fixed above.

### 4.8 Dependencies and risks

Depends on v1.2.0 corpus and tokenizer and on v1.1.0 manifest/checkpoint work.
Risks: (a) loss improves without usable syntax — expected, handled by v1.4.0;
(b) overfitting to the small corpus — mitigated by held-out splits and early
stopping criteria recorded in the manifest; (c) accidental corpus drift —
mitigated by hash checks at load; (d) control comparison is unfair because of
different tokenizers — mitigated by reporting loss per token on a shared
held-out set tokenised once per tokenizer.

### 4.9 Deliverables and Definition of Done

Trained S3 checkpoint plus manifests, loss/perplexity records, repeat-run
variance report, control comparison table. Done when §4.6 passes and the run is
reproducible from its manifest.

### 4.10 Explicit exclusions

No instruction tuning (v1.5.0), no fill-in-the-middle objective (v1.4.0), no
functional-correctness claim, no scale beyond S3, no new data, no serving
deployment, no multilingual data.

---

## 5. v1.4.0 — Python syntax and completion

**Group:** M-A · **Model rung:** S3 (16M) · **Evidence level target:** 2 ·
**Closes M-A**

### 5.1 Objective and architecture

**Objective.** Turn the pretrained base into a model whose Python output
actually parses, and that fills code in the middle of a file. This is the first
version with a user-visible capability claim, and it closes M-A.

**Architecture.** Two objectives on one base: continued pretraining on
syntax-dense Python with a parse-based acceptance loop, and fill-in-the-middle
(FIM) training using reserved sentinels. Decoding gains a structured completion
path with stop conditions at the renderer's end-of-turn token. Evaluation is
AST-based, not string-based.

### 5.2 Components and modules

| Module | Change |
|---|---|
| `training/fim.py` | PSM and SAD transforms; sentinel handling |
| `training/train.py` | Syntax-weighted sampling; FIM objective |
| `evaluation/syntax.py` | `ast.parse` + delimiter balance + import sanity |
| `evaluation/completion.py` | Exact-match, top-5, in-context parse |
| `serving/app.py` | `/v1/completions` accepts `suffix`; stop-token handling |

### 5.3 Functional features

- Completes a function body from a signature and docstring.
- Fills a marked gap in the middle of a Python file.
- Produces output that parses at a measured rate.
- Stops at the end-of-turn token instead of rambling.

### 5.4 Dataset and training requirements

Data: v1.2.0 corpus plus deterministic FIM transforms of the training split
(no cross-file leakage). Syntax-weighted sampling targets definitions, calls,
comprehensions, decorators and error-handling blocks. Held-out completion tasks
are authored or curated, never generated by the model. FIM ratios and sentinel
placement are recorded per run; values are **proposed** (§23, D7).

### 5.5 Inference and API changes

`/v1/completions` accepts an optional `suffix` field and switches to FIM
rendering when present. Responses expose `finish_reason` (`stop`, `length`,
`end_of_turn`). The contract id remains `qm.chat_instruction/2`; completion
rendering is a documented mode of the same renderer.

### 5.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Syntax validity (locked) | ≥ 0.50 | proposed |
| Delimiter/indentation balance | ≥ 0.90 on passing samples | proposed |
| Signature-to-body completion, exact | ≥ 0.20 | proposed |
| Completion top-5 | ≥ 0.40 | proposed |
| In-context parse rate | ≥ 0.90 | proposed |
| FIM leakage | 0 held-out completion tasks in training data | specified |
| Prompt identity | 100% unchanged from v1.1.0 | specified |

### 5.7 Hardware and compute considerations

Same S3 class; FIM doubles the usable training signal at no parameter cost.
Additional compute is data preparation plus one training run; budget is
**proposed** (§23, D14).

### 5.8 Dependencies and risks

Depends on v1.3.0 checkpoints and v1.2.0 corpus. Risks: (a) model memorises
completion fixtures — mitigated by leakage checks and held-out authoring; (b)
parse rate rises while usefulness does not — mitigated by pairing syntax gates
with completion gates; (c) FIM sentinels leak into normal completions —
mitigated by a sentinel-hygiene gate in v1.1.0's control-token tests extended
here.

### 5.9 Deliverables and Definition of Done

S3 syntax+completion checkpoint, FIM spec, syntax and completion evaluation
suites, serving mode, M-A close report. Done when all §5.6 gates pass, the model
is deployed to the internal evaluation endpoint, and the M-A report states the
measured capability with its evidence level.

### 5.10 Explicit exclusions

No natural-language instruction following (v1.5.0), no functional correctness
(v1.6.0), no debugging or tests (v1.7.0), no multi-file reasoning, no IDE
integration (v1.8.0), no non-Python language.

---

## 6. v1.5.0 — Natural-language instruction tuning

**Group:** M-B · **Model rung:** S3 (16M) · **Evidence level target:** 1–2 ·
**Opens M-B**

### 6.1 Objective and architecture

**Objective.** Make the model answer a natural-language Python request in the
canonical conversation format, without losing the syntax and completion ability
gained in v1.4.0. This is supervised fine-tuning, not preference optimisation.

**Architecture.** The v1.4.0 base is fine-tuned on instruction/response pairs
rendered *only* through `training/chat_template.py`, so the training and serving
prompts are the same object. A small, curated instruction set (~5k–20k
examples, **proposed**) covers explanation, generation, editing, refactoring and
refusal. Loss is computed on assistant tokens only.

### 6.2 Components and modules

| Module | Change |
|---|---|
| `data/instructions/**` | Curated instruction/response sets with provenance |
| `training/sft.py` | Assistant-only loss; masking; packing |
| `training/chat_template.py` | Roles exercised for the first time in training |
| `evaluation/instruction.py` | Format compliance and adherence rubric |
| `evaluation/regression.py` | Syntax/completion regression suite |

### 6.3 Functional features

- Answers "write a function that…" and "what does this do?" in the expected
  format.
- Keeps the answer inside the assistant turn and stops cleanly.
- Refuses out-of-scope requests with a short, non-fabricated response.
- Does not regress on the v1.4.0 syntax and completion suites.

### 6.4 Dataset and training requirements

Instruction data is curated or licensed; no model-generated data in this
version. Every example records source, licence and intent. Held-out
instructions are authored separately. Chat-format coverage must include
multi-turn (at least two turns) and refusal cases. Deduplication and
contamination checks reuse v1.2.0 tooling. Epochs, learning rate and masking
policy are recorded per run; values are **proposed** (§23, D7 and D11).

### 6.5 Inference and API changes

`/v1/chat/completions` becomes the primary endpoint and now returns coherent
instruction-following output. Multi-turn history is rendered by the canonical
renderer. No new endpoint.

### 6.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Format compliance | ≥ 0.90 | specified |
| Instruction adherence (rubric) | ≥ 0.70 | proposed |
| Refusal appropriateness | ≥ 0.80 on out-of-scope set | proposed |
| Syntax retention | ≥ 0.45 (no more than 0.05 below v1.4.0) | proposed |
| Completion retention | top-5 ≥ 0.35 | proposed |
| Multi-turn coherence | ≥ 0.60 on two-turn probes | proposed |
| Contamination | 0 held-out instructions in training data | specified |

### 6.7 Hardware and compute considerations

Supervised fine-tuning of a 16M model is far cheaper than pretraining; a single
GPU run is expected. The larger cost is instruction-data curation (human time).
Budget is **proposed** (§23, D14).

### 6.8 Dependencies and risks

Depends on v1.4.0 checkpoint, v1.1.0 renderer, v1.2.0 data tooling. Risks: (a)
catastrophic forgetting of syntax — mitigated by mixing pretraining replay and
by regression gates; (b) format compliance without usefulness — mitigated by
the adherence rubric; (c) instruction data licensing — mitigated by the v1.2.0
licence gate; (d) overfitting on a small set — mitigated by held-out probes and
early stopping.

### 6.9 Deliverables and Definition of Done

Instruction dataset with provenance, SFT checkpoint, instruction and regression
suites, updated serving behaviour, data card for the instruction set. Done when
§6.6 passes and the checkpoint is registered with full provenance.

### 6.10 Explicit exclusions

No preference optimisation (RLHF/DPO) in v1.5.0, no functional-correctness claim
(v1.6.0), no tool use, no multilingual data, no model-generated training data,
no parameter-count change.

---

## 7. v1.6.0 — Functional Python correctness

**Group:** M-B · **Model rung:** S3 (16M) · **Evidence level target:** 2–3

### 7.1 Objective and architecture

**Objective.** Move from *parses* to *works*: generated Python must pass
executable tests. This is the first version whose primary gate is behavioural
rather than structural.

**Architecture.** A sandboxed execution service runs generated code against
unit tests with timeouts, memory caps and no network. Training adds executable
function-spec data (docstring + signature + tests + reference solution) and
rejection of failing samples. Evaluation vendors the public suites at pinned
commits and runs them in the same sandbox.

### 7.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/execution/**` | Sandbox runner: timeout, memory, no network, no fs writes |
| `evaluation/humaneval.py`, `mbpp.py` | Pinned vendored suites |
| `training/exec_filter.py` | Executable-sample filtering before SFT |
| `data/functional/**` | Function-spec data with tests |
| `serving/**` | Optional `execute` preview flag (off by default) |

### 7.3 Functional features

- Produces functions that pass provided unit tests at a measured rate.
- Rejects its own failing generations during training.
- Reports pass@1 with confidence intervals, not a single number.
- Never executes untrusted code outside the sandbox.

### 7.4 Dataset and training requirements

Function-spec data: docstring, signature, examples, tests, reference solution;
licence-cleared and deduplicated against every suite. Execution filtering keeps
only samples whose reference solution passes its own tests and whose generated
candidates are scored. Replay of v1.5.0 instruction data prevents regression.
Suite pinning: HumanEval and MBPP at recorded commits with recorded hashes.

### 7.5 Inference and API changes

No mandatory API change. An optional `execute` flag on `/v1/chat/completions`
is **off by default** and gated per deployment; when enabled it returns test
results, never raw process output. Error taxonomy gains `execution_timeout`,
`execution_rejected`.

### 7.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Locked functional probes | ≥ 0.50 pass@1 | proposed |
| HumanEval (pinned) | ≥ 0.10 pass@1 | proposed |
| MBPP (pinned) | ≥ 0.10 pass@1 | proposed |
| Sandbox escapes | 0 | specified |
| Timeout/error handling | 100% classified, no crash | specified |
| Instruction retention | format ≥ 0.90, adherence ≥ 0.65 | specified / proposed |
| Contamination | 0 overlaps with vendored suites | specified |

Reported as pass@1 with a 95% interval over ≥ 3 seeds; a single lucky run does
not pass a gate.

### 7.7 Hardware and compute considerations

Execution is the new cost centre: thousands of sandboxed runs per evaluation.
CPU-only workers suffice; the model run is another S3 fine-tune. Budget is
**proposed** (§23, D14 and D15).

### 7.8 Dependencies and risks

Depends on v1.5.0 checkpoint and v1.4.0 syntax behaviour. Risks: (a) sandbox
escape — mitigated by no-network, no-write, process-level isolation and a
security review; (b) benchmark contamination — mitigated by pinned suites,
hashes and n-gram checks; (c) execution noise — mitigated by fixed resource
limits and retries with recorded seeds; (d) pass@1 overfitting to public suites —
mitigated by private locked probes and by reporting both.

### 7.9 Deliverables and Definition of Done

Sandbox execution service, vendored suite pins, functional data card, S3
functional checkpoint, pass@1 report with intervals. Done when §7.6 passes, the
sandbox passes its security review, and results are reproducible from recorded
commands.

### 7.10 Explicit exclusions

No debugging or repair (v1.7.0), no test generation (v1.7.0), no multi-file or
repository tasks (v2.8.0), no non-Python languages, no public leaderboard
submission, no claim of general coding ability beyond the measured suites.

---

## 8. v1.7.0 — Python debugging and test generation

**Group:** M-B · **Model rung:** S3 (16M) · **Evidence level target:** 3

### 8.1 Objective and architecture

**Objective.** Add the two closed-loop developer skills that require execution
feedback: **repair** of failing Python and **generation** of tests that actually
catch bugs. Both are measured by behaviour, not appearance.

**Architecture.** Two task heads over one checkpoint, distinguished by the
rendered task contract, not by separate models. A repair loop consumes a
traceback plus source, proposes an edit, and re-executes against the original
test. A test-generation path produces tests evaluated by coverage gained and by
mutation score against seeded faults.

### 8.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/debug.py` | Fault localisation, repair harness, no-regression check |
| `evaluation/testgen.py` | Coverage delta, mutation score |
| `data/debug/**` | Seeded faults + traces with provenance |
| `data/tests/**` | Function-spec → test pairs |
| `training/sft.py` | Task-contract conditioning; replay mixing |
| `evaluation/execution/**` | Mutation runner (extends v1.6.0 sandbox) |

### 8.3 Functional features

- Given a failing test and traceback, proposes a patch that makes it pass.
- Points at the likely faulty function or line.
- Generates tests that pass against correct code and fail against mutants.
- Does not break passing tests when repairing.

### 8.4 Dataset and training requirements

Faults are seeded deterministically into training code with recorded mutation
operators; the model never sees the mutation label at inference. Debug data
pairs include traceback, source, test and reference patch. Test-generation data
pairs functions with human-written tests plus coverage annotations. Held-out
faults are authored separately from training faults. Dedup against the v1.6.0
vendored suites is mandatory.

### 8.5 Inference and API changes

A `task` field (`repair`, `localise`, `testgen`) selects the contract mode.
Repair responses may carry a unified diff when the deployment enables patch
output (`diff` flag, off by default). No new endpoint.

### 8.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Repair success (locked) | ≥ 0.30 | proposed |
| No-regression on passing tests | ≥ 0.95 | proposed |
| Fault localisation (file/function) | ≥ 0.50 | proposed |
| Generated-test pass rate | ≥ 0.70 | proposed |
| Mutation score of generated tests | ≥ 0.30 | proposed |
| Coverage delta | ≥ +10 points over no-test baseline | proposed |
| Syntax/functional retention | no more than 0.03 below v1.6.0 | proposed |

### 8.7 Hardware and compute considerations

Execution-heavy: every repair candidate and every generated test is run, and
mutation scoring multiplies runs. CPU workers dominate; the model run is another
S3 fine-tune. Budget is **proposed** (§23, D14).

### 8.8 Dependencies and risks

Depends on v1.6.0 sandbox and functional checkpoint. Risks: (a) repair by
deleting tests — mitigated by no-regression gates and diff inspection; (b)
generated tests that pass trivially — mitigated by mutation scoring; (c) seeded
faults unrepresentative of real bugs — mitigated by including recorded real
tracebacks where licence permits; (d) execution cost — mitigated by batching and
caching.

### 8.9 Deliverables and Definition of Done

Debug and test-generation suites, seeded-fault corpus, S3 checkpoint, repair and
testgen reports. Done when §8.6 passes and the reports separate localisation,
repair and test quality.

### 8.10 Explicit exclusions

No code review, documentation or IDE workflows (v1.8.0), no repository-scale
debugging (v2.8.0), no autonomous multi-step agents, no non-Python languages, no
claim of general debugging ability.

---

## 9. v1.8.0 — Python developer workflows

**Group:** M-B · **Model rung:** S3 (16M) · **Evidence level target:** 3–4

### 9.1 Objective and architecture

**Objective.** Turn model capability into a workflow a developer can actually
use: review comments, documentation, in-editor completion, and a serving path
that is fast and correct enough to sit behind an editor. Capability gates from
earlier versions must hold.

**Architecture.** Three task contracts (review, docs, complete) share the
checkpoint and the canonical renderer. Serving gains a KV-cache path, SSE
streaming, request batching and a completion-optimised endpoint. Promotion
enforcement moves from a document into an executable runner.

### 9.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/review.py` | Comment relevance and localisation |
| `evaluation/docs.py` | Docstring completeness and correctness |
| `serving/stream.py` | SSE with backpressure and clean termination |
| `serving/cache.py` | KV cache, prefix reuse, memory bound |
| `serving/batch.py` | Continuous batching with per-request limits |
| `tools/promote.py` | Executable promotion runner reading `evaluation/promotion.yaml` |
| `integrations/ide/**` | Minimal completion client (reference implementation) |

### 9.3 Functional features

- Reviews a diff and raises comments that point at the right lines.
- Writes docstrings that match the function and its behaviour.
- Completes code in an editor with acceptable latency and streaming.
- Serves multiple concurrent requests without exceeding memory bounds.
- Promotion is executed, not argued: the runner computes verdicts.

### 9.4 Dataset and training requirements

Review data: diffs with human comments, licence-cleared, anchored to lines.
Docs data: functions with reference docstrings. Completion data reuses v1.4.0
FIM transforms with repository-realistic prefixes. Workflow data is mixed with
replay from v1.5.0–v1.7.0 to prevent forgetting. No new languages.

### 9.5 Inference and API changes

`/v1/completions` gains streaming and a `max_latency_ms` hint. `/v1/chat/completions`
supports SSE. `/v1/models` reports context length, quantisation and cache
policy. Request-level cancellation is honoured. Contract id unchanged.

### 9.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Review comment localisation | ≥ 0.50 | proposed |
| Review precision (accepted comments) | ≥ 0.40 | proposed |
| Docstring correctness (rubric) | ≥ 0.60 | proposed |
| IDE completion acceptance (proxy) | ≥ 0.30 | proposed |
| Streaming first-token p95 | ≤ 400 ms on the published profile | proposed |
| Streaming integrity | 100% of streams terminate with a valid reason | specified |
| Memory bound | Never exceeds the declared cache budget under 8 concurrent requests | specified |
| Prior capability retention | functional no more than 0.03 below v1.6.0 | proposed |

### 9.7 Hardware and compute considerations

First version with a **serving** envelope: one GPU sized for the S3 model with
KV cache, plus CPU workers for evaluation. Latency numbers are only claimable on
a published hardware profile. Budget is **proposed** (§23, D14); no latency
claim may be made on unprovisioned hardware.

### 9.8 Dependencies and risks

Depends on v1.4.0–v1.7.0 checkpoints, v1.6.0 sandbox for evaluation. Risks: (a)
latency claims on developer laptops — mitigated by the published-profile rule;
(b) cache memory blow-up — mitigated by hard bounds and eviction tests; (c)
streaming leaks a partial turn — mitigated by end-of-turn enforcement; (d)
workflow quality judged by anecdotes — mitigated by rubric and localisation
metrics.

### 9.9 Deliverables and Definition of Done

Review/docs/completion suites, streaming and cache serving, promotion runner,
reference IDE client, latency report on a named profile. Done when §9.6 passes
and the promotion runner is the only path used to mark a promotion.

### 9.10 Explicit exclusions

No repository-aware multi-file reasoning (v2.8.0), no production SLOs, isolation
or runbooks (v2.9.0), no non-Python languages, no hosted public service, no
autonomous agent loops.

---

## 10. v1.9.0 — Python alpha qualification

**Group:** M-B · **Model rung:** S3 (16M, frozen) · **Evidence level target:**
3 (5 with independent review) · **Closes M-B**

### 10.1 Objective and architecture

**Objective.** Decide, on evidence, whether the Python capability built across
M-A and M-B is fit to be called an **alpha**: frozen, reproducible and honestly
described. A refusal is a valid outcome and must be published if the gates are
red.

**Architecture.** No new model capability. The version is a freeze and
qualification exercise: artifacts are frozen, the promotion runner is executed
against the complete gate set, a reproduction drill rebuilds the release from
manifests, and an independent reviewer checks the evidence trail.

### 10.2 Components and modules

| Module | Change |
|---|---|
| `tools/freeze.py` | Artifact freeze with hashes |
| `tools/reproduce.py` | Full rebuild from manifests |
| `tools/qualify.py` | Runs the complete gate set; emits the qualification pack |
| `tools/promote.py` | Reads verdicts; writes the promotion record |
| `docs/source/releases/**` | Alpha qualification report |
| `evaluation/alpha/**` | Alpha probe set (private, locked) |

### 10.3 Functional features

- A frozen release candidate with a complete provenance chain.
- A public qualification report that states measured capability and limits.
- A refusal statement template used when gates are red.
- A support statement describing what alpha does and does not promise.

### 10.4 Dataset and training requirements

No new training data. All evaluation data is locked and hashed before the freeze.
The private alpha probes are authored and held by the owner; they are never used
in training and are revealed only in aggregate.

### 10.5 Inference and API changes

The frozen candidate is deployed to an internal alpha endpoint with the v1.8.0
API surface. Version string reports `1.9.0-alpha`. No public endpoint.

### 10.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| All prior gates | Green at their recorded thresholds | specified |
| Reproduction | Rebuild matches frozen artifacts within the documented interval | specified |
| Seed variance | Reported; alpha claim uses the median | specified |
| Human review | At least one independent reviewer records findings | specified |
| Licence position | Every distributed artifact has a decided licence | specified |
| Open P0 issues | 0 | specified |
| Claim accuracy | Every claim in the report traceable to evidence | specified |
| Alpha probes | Reported; no gate, disclosed as advisory | proposed |

### 10.7 Hardware and compute considerations

Cost is dominated by reproduction runs (at least two) and the full gate sweep.
No new hardware class. Budget is **proposed** (§23, D14).

### 10.8 Dependencies and risks

Depends on every prior version being green. Risks: (a) pressure to call alpha
when a gate is red — mitigated by the published-refusal path and the sole
approver rule; (b) reproduction drift from unpinned dependencies — mitigated by
pinned environments and recorded versions; (c) overclaiming in the report —
mitigated by claim-to-evidence traceability; (d) private probes leaking into
training — mitigated by access control and a leakage audit.

### 10.9 Deliverables and Definition of Done

Frozen artifacts, qualification pack, reproduction transcript, alpha report or
refusal statement, support statement. Done when the owner signs the promotion
record — **or** signs the refusal and the plan is updated. Both outcomes close
the version.

### 10.10 Explicit exclusions

No new features, no parameter change, no multilingual work, no production SLOs,
no public hosting, no leaderboard claims, no reinterpretation of red gates as
"close enough".

---

## 11. v2.0.0 — SQL and JavaScript foundation

**Group:** M-C · **Model rung:** S4 (113,145,408 params) · **Evidence level
target:** 3 · **Opens M-C**

### 11.1 Objective and architecture

**Objective.** Deliver the first two non-Python programming languages — **SQL**
and **JavaScript** — on a multilingual platform, and prove that adding languages
does not destroy Python. This is the first real scale-up (16M → 113M).

**Architecture.** The tokenizer is extended for SQL and JavaScript with
recorded deltas; the renderer gains language-aware contracts but no second
format. Training uses a per-language mixture with replay of Python data. SQL is
evaluated by execution against a seeded schema; JavaScript by execution in a
sandbox with a pinned runtime. Both reuse the v1.6.0 execution discipline.

### 11.2 Components and modules

| Module | Change |
|---|---|
| `configs/model/pilot_300m.yaml` | S4 geometry (113,145,408), verified twice |
| `training/tokenizer/**` | Language extension; fertility per language |
| `data/contracts/multilingual_v1.yaml` | Mixture and replay rules |
| `evaluation/sql/**` | Seeded schemas, execution oracle, result comparison |
| `evaluation/javascript/**` | Pinned runtime sandbox, test harness |
| `training/mixture.py` | Temperature-based sampling with floors |

### 11.3 Functional features

- Writes SQL that executes and returns the expected rows.
- Writes JavaScript that runs and passes tests.
- Explains and edits both languages in the canonical chat format.
- Keeps Python capability within the retention band.

### 11.4 Dataset and training requirements

SQL: queries against seeded schemas, dialect declared (SQLite first,
**proposed**). JavaScript: functions, modules and async basics with tests.
Mixture floors (Python ≥ 40%, SQL ≥ 15%, JavaScript ≥ 15%, **proposed**) prevent
dilution. Licence gate per source; no scraped code without a decided licence.
Contamination checks per language. Tokenizer delta is recorded and frozen before
training.

### 11.5 Inference and API changes

A `language` hint is accepted and validated; it changes prompting, not model
weights. `/v1/chat/completions` can return code blocks tagged with the language.
Execution preview extends to SQL and JavaScript when explicitly enabled.

### 11.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| SQL execution | ≥ 0.50 | proposed |
| SQL result correctness | ≥ 0.40 | proposed |
| JavaScript execution | ≥ 0.40 | proposed |
| JavaScript tests | ≥ 0.30 | proposed |
| Python syntax retention | ≥ 0.45 | proposed |
| Python functional retention | no more than 0.03 below v1.6.0 | proposed |
| Contamination | 0 per language | specified |
| Tokenizer fertility | No language worse than the documented ceiling | proposed |

### 11.7 Hardware and compute considerations

First S4 run: 113M parameters at the v1.2.0-derived multilingual corpus. This is
a multi-GPU or multi-day single-GPU job depending on the approved envelope.
Budget and hardware class are **proposed** (§23, D14 and D15) and must be
approved before the run is scheduled.

### 11.8 Dependencies and risks

Depends on M-A/M-B checkpoints, v1.6.0 sandbox, v1.2.0 governance extended to
two new licence regimes. Risks: (a) Python regression from mixture dilution —
mitigated by floors and retention gates; (b) SQL dialect ambiguity — mitigated by
declaring the dialect and pinning the engine; (c) JavaScript runtime
nondeterminism — mitigated by pinned runtimes and recorded versions; (d) scale-up
eats the budget before evidence exists — mitigated by a short S4 pilot first.

### 11.9 Deliverables and Definition of Done

S4 checkpoint, multilingual data contract and data card, SQL and JavaScript
suites, retention report, updated tokenizer. Done when §11.6 passes and the
pilot report justifies any further S4 spend.

### 11.10 Explicit exclusions

No TypeScript or web frameworks (v2.1.0), no Shell (v2.2.0), no Java, Go, C# or
C/C++ (v2.3.0–v2.6.0), no repository tasks, no production SLOs, no 7B-scale
work.

---

## 12. v2.1.0 — TypeScript and web development

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3

### 12.1 Objective and architecture

**Objective.** Extend the JavaScript base into **TypeScript** and practical
**web development**, where correctness is checked by a type checker as well as
by execution.

**Architecture.** TypeScript is treated as a distinct language with a declared
compiler version. Evaluation runs `tsc --noEmit --strict` as a first oracle and
the pinned runtime as a second. Web tasks use platform APIs (DOM, fetch,
storage) against a deterministic test page, not a live network.

### 12.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/typescript/**` | `tsc` oracle, diagnostics normalisation |
| `evaluation/web/**` | DOM/fetch harness with mocked network |
| `data/contracts/typescript_v1.yaml` | TS mixture and TS-specific filtering |
| `training/mixture.py` | TS floor; JS↔TS sharing |
| `serving/**` | Language hint recognises `typescript` |

### 12.3 Functional features

- Writes TypeScript that type-checks under strict mode.
- Adds types to untyped JavaScript without changing behaviour.
- Implements small web behaviours against DOM and fetch mocks.
- Keeps JavaScript and Python capability within their bands.

### 12.4 Dataset and training requirements

TypeScript sources with licence clearance and pinned compiler version; JS→TS
migration pairs; web examples with deterministic harnesses. Mixture: TS ≥ 12%,
JS ≥ 12% with Python replay (**proposed**). Type-error filtering removes samples
that do not pass `tsc`. Contamination checks include the TS suite.

### 12.5 Inference and API changes

`language: typescript` selects TS prompting. Responses may include a
`diagnostics` block when execution/type preview is enabled. No new endpoint.

### 12.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| TypeScript type-check | ≥ 0.60 | proposed |
| TypeScript execution/tests | ≥ 0.30 | proposed |
| JS→TS migration equivalence | ≥ 0.25 | proposed |
| Web behaviour tests | ≥ 0.30 | proposed |
| JavaScript retention | no more than 0.03 below v2.0.0 | proposed |
| Python retention | ≥ 0.45 syntax | proposed |

### 12.7 Hardware and compute considerations

Continued S4 fine-tuning; `tsc` is CPU-bound and cheap relative to training. No
new hardware class. Budget is **proposed** (§23, D14).

### 12.8 Dependencies and risks

Depends on v2.0.0 JS capability and the S4 checkpoint. Risks: (a) TS type errors
treated as style — mitigated by strict mode and a pinned compiler; (b) live
network flakiness in web tests — mitigated by mocks; (c) compiler version drift
— mitigated by pinning and recording; (d) JS regression from TS mixing —
mitigated by retention gates.

### 12.9 Deliverables and Definition of Done

TS and web suites, TS data card, S4 checkpoint update, migration and retention
reports. Done when §12.6 passes.

### 12.10 Explicit exclusions

No framework-specific application generation beyond the declared harness, no
Shell (v2.2.0), no backend/runtime services, no repository tasks, no production
SLOs.

---

## 13. v2.2.0 — Shell scripting and configuration

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3

### 13.1 Objective and architecture

**Objective.** Cover the glue layer of real repositories: shell scripts and the
configuration files around them. Safety is a first-class gate, because a wrong
shell command is a destructive command.

**Architecture.** Shell is evaluated in a disposable container with a recorded
filesystem snapshot; configuration files (YAML, TOML, Dockerfile, JSON) are
evaluated by parser and linter oracles. A static safety classifier blocks
destructive patterns before any execution preview.

### 13.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/shell/**` | Container sandbox, snapshot/diff, exit-code oracle |
| `evaluation/shell_safety/**` | Destructive-pattern classifier and refusal suite |
| `evaluation/config/**` | YAML/TOML/JSON parse, Dockerfile lint |
| `data/contracts/shell_v1.yaml` | Shell and config mixture |
| `serving/**` | `language: shell`; safety refusal path |

### 13.3 Functional features

- Writes POSIX shell and bash that run and exit correctly.
- Uses pipes, redirection and error handling idiomatically.
- Produces valid YAML/TOML/JSON and lint-clean Dockerfiles.
- Refuses or flags destructive commands (`rm -rf /`, unguarded `curl | sh`).

### 13.4 Dataset and training requirements

Shell sources with licence clearance, grouped by dialect (POSIX vs bash) and
shell version. Config examples paired with their schemas. Safety examples
include both safe and unsafe patterns with the expected refusal. Mixture: Shell
≥ 10%, config ≥ 5% (**proposed**). No scraped secrets; a secret scanner runs
over all data.

### 13.5 Inference and API changes

`language: shell` and `language: config` prompt modes. Safety refusals return a
structured reason code (`destructive_command`, `unbounded_network`). Execution
preview, when enabled, runs only in the disposable container.

### 13.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Shell execution | ≥ 0.40 | proposed |
| Shell dialect correctness | ≥ 0.50 on POSIX probes | proposed |
| Safety classification | ≥ 0.95 recall on destructive suite | specified |
| False-refusal rate | ≤ 0.10 on safe suite | proposed |
| Config parse validity | ≥ 0.70 | proposed |
| Dockerfile lint clean | ≥ 0.50 | proposed |
| Category retention | Python, SQL, JS, TS within 0.03 | proposed |

### 13.7 Hardware and compute considerations

CPU-bound: container runs, parsers, linters. Training continues on S4. Budget is
**proposed** (§23, D14 and D15).

### 13.8 Dependencies and risks

Depends on v2.0.0–v2.1.0 platform. Risks: (a) destructive output in a real
environment — mitigated by sandboxing, safety gating and refusal; (b) shell
dialect drift — mitigated by declared dialect and version; (c) secrets in data —
mitigated by scanning; (d) safety classifier false positives harming usefulness —
mitigated by tracking false refusals as a gate.

### 13.9 Deliverables and Definition of Done

Shell and config suites, safety suite, container sandbox, S4 checkpoint update,
safety report. Done when §13.6 passes and the safety review is signed.

### 13.10 Explicit exclusions

No infrastructure-as-code or cloud API work, no Windows PowerShell coverage, no
package-manager operations, no network execution, no Java/Go/C#/C++ (v2.3.0+).

---

## 14. v2.3.0 — Java programming

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3

### 14.1 Objective and architecture

**Objective.** Add Java, where compilation is a hard oracle and the ecosystem
expects types, packages and tests. Compiling is a necessary but insufficient
gate; execution and tests decide.

**Architecture.** A pinned JDK and an offline build tool evaluate generated
code. Compilation diagnostics are normalised into training signal. Java data is
grouped by language level, with records/streams introduced explicitly rather
than assumed.

### 14.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/java/**` | JDK compile, JUnit execution, diagnostics |
| `data/contracts/java_v1.yaml` | Java mixture, language-level tagging |
| `training/exec_filter.py` | Java compile filter |
| `serving/**` | `language: java` |

### 14.3 Functional features

- Writes Java classes and methods that compile.
- Passes JUnit tests for provided specifications.
- Uses collections, streams and records idiomatically for the declared level.
- Explains Java code and errors in the canonical chat format.

### 14.4 Dataset and training requirements

Licence-cleared Java sources with package declarations intact; method-level and
class-level tasks; test pairs. Language level (e.g. 17, **proposed**) is
declared and enforced at compile time. Mixture: Java ≥ 10% with replay of prior
languages (**proposed**). Contamination check includes the Java suite.

### 14.5 Inference and API changes

`language: java` selects Java prompting. Execution preview compiles and runs in
the sandbox and returns compiler diagnostics, never raw JVM output.

### 14.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Compilation | ≥ 0.80 | proposed |
| Test pass rate | ≥ 0.30 | proposed |
| JUnit generation validity | ≥ 0.25 | proposed |
| Idiomatic usage (rubric) | ≥ 0.50 | proposed |
| Category retention | Python, SQL, JS, TS, Shell within 0.03 | proposed |

### 14.7 Hardware and compute considerations

Compilation is CPU- and memory-hungry relative to scripting languages; the
sandbox needs a JVM-sized memory cap. Training continues on S4. Budget is
**proposed** (§23, D14).

### 14.8 Dependencies and risks

Depends on v2.0.0 platform and v1.6.0 sandbox pattern. Risks: (a) compile-rate
theatre with runtime failures — mitigated by execution and tests; (b) JDK
version drift — mitigated by pinning; (c) Maven/Gradle network access —
mitigated by offline repositories; (d) retention regression — mitigated by
mixture floors.

### 14.9 Deliverables and Definition of Done

Java suite, JDK pin, S4 checkpoint update, compile/execution report. Done when
§14.6 passes.

### 14.10 Explicit exclusions

No Spring or enterprise frameworks, no Android, no Kotlin or Scala, no build
system authoring beyond the offline harness, no repository tasks.

---

## 15. v2.4.0 — Go programming

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3

### 15.1 Objective and architecture

**Objective.** Add Go, where concurrency correctness is part of correctness. A
program that races is wrong even when its tests pass once.

**Architecture.** A pinned Go toolchain evaluates build, `go vet`, tests and the
race detector. Concurrency tasks are explicit in the training data. Error
handling is evaluated for explicit return-value treatment rather than exception
semantics.

### 15.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/go/**` | Build, vet, test, race detector |
| `evaluation/go_concurrency/**` | Concurrency probe set |
| `data/contracts/go_v1.yaml` | Go mixture and concurrency tagging |
| `training/exec_filter.py` | Go build/test filter |
| `serving/**` | `language: go` |

### 15.3 Functional features

- Writes Go that builds and passes tests.
- Handles errors explicitly and idiomatically.
- Uses goroutines and channels correctly in bounded probes.
- Produces race-detector-clean concurrent code.

### 15.4 Dataset and training requirements

Licence-cleared Go sources; concurrency examples with synchronisation
primitives; test pairs. Module paths and versions are normalised. Go version is
pinned (**proposed**). Mixture: Go ≥ 8% with replay (**proposed**). No
generated code without a licence decision.

### 15.5 Inference and API changes

`language: go` selects Go prompting. Execution preview returns build and vet
diagnostics plus test outcomes; race-detector results are returned when enabled.

### 15.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Build | ≥ 0.80 | proposed |
| Test pass rate | ≥ 0.30 | proposed |
| Race-free (race detector) | ≥ 0.90 on concurrency probes | proposed |
| `go vet` clean | ≥ 0.60 | proposed |
| Error-handling idiom (rubric) | ≥ 0.50 | proposed |
| Category retention | Prior six categories within 0.03 | proposed |

### 15.7 Hardware and compute considerations

Race-detector runs are several times more expensive than plain tests; concurrency
probes are therefore bounded and sampled. Training continues on S4. Budget is
**proposed** (§23, D14).

### 15.8 Dependencies and risks

Depends on the sandbox and S4 checkpoint. Risks: (a) passing tests that race —
mitigated by the race-detector gate; (b) goroutine leaks — mitigated by bounded
probe timeouts; (c) version drift in modules — mitigated by pinned module
cache; (d) retention regression — mitigated by mixture floors.

### 15.9 Deliverables and Definition of Done

Go suite, concurrency suite, toolchain pin, S4 checkpoint update, race report.
Done when §15.6 passes.

### 15.10 Explicit exclusions

No distributed-systems design, no Kubernetes operators, no cgo, no generics-heavy
library work beyond idiomatic basics, no repository tasks.

---

## 16. v2.5.0 — C# and .NET

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3

### 16.1 Objective and architecture

**Objective.** Add C# and the .NET idioms developers actually use: LINQ, async/
await, nullable reference types and dependency-aware unit tests.

**Architecture.** A pinned .NET SDK with an offline NuGet cache compiles and runs
generated code. Nullable warnings and async misuse are treated as first-class
diagnostics. Tests use a recorded framework version.

### 16.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/csharp/**` | SDK compile, test run, diagnostics |
| `evaluation/csharp_async/**` | Async correctness probes |
| `data/contracts/csharp_v1.yaml` | C# mixture, framework pin |
| `training/exec_filter.py` | C# compile/test filter |
| `serving/**` | `language: csharp` |

### 16.3 Functional features

- Writes C# that compiles and passes unit tests.
- Uses LINQ and collection APIs idiomatically.
- Handles async/await without deadlocks or fire-and-forget bugs.
- Respects nullable annotations.

### 16.4 Dataset and training requirements

Licence-cleared C# sources with project structure; async and LINQ examples; test
pairs. Framework target (e.g. `net8.0`, **proposed**) is pinned. NuGet access is
offline and recorded. Mixture: C# ≥ 8% with replay (**proposed**).

### 16.5 Inference and API changes

`language: csharp` selects C# prompting. Execution preview returns compiler,
nullable and test diagnostics.

### 16.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Compilation | ≥ 0.80 | proposed |
| Test pass rate | ≥ 0.30 | proposed |
| Async correctness | ≥ 0.40 on async probes | proposed |
| Nullable-warning-free | ≥ 0.50 | proposed |
| LINQ idiom (rubric) | ≥ 0.50 | proposed |
| Category retention | Prior seven categories within 0.03 | proposed |

### 16.7 Hardware and compute considerations

The .NET SDK is memory-hungry per build; the sandbox caps memory and reuses a
warm build cache. Training continues on S4. Budget is **proposed** (§23, D14).

### 16.8 Dependencies and risks

Depends on the sandbox and S4 checkpoint. Risks: (a) NuGet network access —
mitigated by an offline cache; (b) async deadlocks passing tests — mitigated by
dedicated probes and timeouts; (c) SDK drift — mitigated by pinning; (d)
retention regression — mitigated by mixture floors.

### 16.9 Deliverables and Definition of Done

C# suite, async suite, SDK pin, S4 checkpoint update, diagnostic report. Done
when §16.6 passes.

### 16.10 Explicit exclusions

No ASP.NET services, no Unity or game engines, no F# or VB.NET, no Windows-only
APIs, no repository tasks.

---

## 17. v2.6.0 — C and C++

**Group:** M-C · **Model rung:** S4 (113M) · **Evidence level target:** 3–4 ·
**Closes M-C**

### 17.1 Objective and architecture

**Objective.** Add the two languages where memory safety is the dominant risk,
and close the multilingual milestone. Output must compile *and* survive
sanitisers.

**Architecture.** Pinned compilers for C and C++ compile with warnings as
errors; tests run under AddressSanitizer and UndefinedBehaviorSanitizer where
available. C and C++ are separate contracts, not one mode.

### 17.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/c_cpp/**` | Compile, test, sanitizer matrix |
| `data/contracts/c_v1.yaml`, `cpp_v1.yaml` | Separate mixtures and standards pins |
| `training/exec_filter.py` | Compile + sanitizer filter |
| `serving/**` | `language: c`, `language: cpp` |
| `docs/source/releases/**` | M-C close report |

### 17.3 Functional features

- Writes C and C++ that compile under the declared standard.
- Passes tests without sanitizer findings.
- Manages memory explicitly without leaks or use-after-free in probes.
- Uses standard library facilities idiomatically.

### 17.4 Dataset and training requirements

Licence-cleared C and C++ sources with standards pinned (e.g. C17, C++17,
**proposed**); header/source structure preserved; test pairs. Build flags are
recorded. Mixture: C ≥ 6%, C++ ≥ 8% with replay (**proposed**). Contamination
checks per language.

### 17.5 Inference and API changes

`language: c` and `language: cpp` select prompting. Execution preview returns
compiler and sanitizer diagnostics; sanitizer findings are summarised, not
dumped raw.

### 17.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| C compilation | ≥ 0.70 | proposed |
| C++ compilation | ≥ 0.70 | proposed |
| Test pass rate | ≥ 0.25 | proposed |
| Sanitizer-clean | ≥ 0.90 of compiled samples | proposed |
| Memory-leak-free | ≥ 0.90 on leak probes | proposed |
| Category retention | Prior eight categories within 0.03 | proposed |

### 17.7 Hardware and compute considerations

Sanitiser builds are slower and memory-hungry; the evaluation matrix samples
rather than exhausts. Training continues on S4. Budget is **proposed**
(§23, D14).

### 17.8 Dependencies and risks

Depends on the sandbox and S4 checkpoint. Risks: (a) compilable but unsafe code
— mitigated by sanitizer gates; (b) undefined behaviour that passes tests —
mitigated by UBSan and bounded probes; (c) standard/compiler drift — mitigated by
pinning; (d) retention regression — mitigated by mixture floors.

### 17.9 Deliverables and Definition of Done

C and C++ suites, sanitizer matrix, standards pins, S4 checkpoint update, M-C
close report stating each language's measured level. Done when §17.6 passes and
the M-C report is published with per-language evidence.

### 17.10 Explicit exclusions

No kernel or driver code, no embedded toolchains, no template metaprogramming
beyond idiomatic basics, no build-system authoring (CMake beyond the harness), no
repository tasks.

---

## 18. v2.7.0 — Cross-language translation

**Group:** M-D · **Model rung:** S4 (113M) · **Evidence level target:** 4 ·
**Opens M-D**

### 18.1 Objective and architecture

**Objective.** Move code between the ten supported languages **without changing
behaviour**. Translation is judged by running both sides on shared vectors, not
by resemblance.

**Architecture.** A translation contract declares source and target language,
entry point and I/O types. Evaluation compiles and runs source and translation
against the same vectors and compares outputs, including error behaviour. Shared
vectors are language-neutral and versioned.

### 18.2 Components and modules

| Module | Change |
|---|---|
| `evaluation/xlang/**` | Shared vectors, cross-runtime execution, comparison |
| `evaluation/xlang_structural.py` | Structural fidelity checks |
| `data/contracts/xlang_v1.yaml` | Translation pairs and direction matrix |
| `training/sft.py` | Direction tokens; replay of all languages |
| `serving/**` | `translate_to` field; language-pair validation |

### 18.3 Functional features

- Translates a function between supported languages and preserves output.
- Preserves error semantics where the target language allows it.
- Flags constructs that cannot be translated faithfully.
- Keeps per-language capability while adding translation.

### 18.4 Dataset and training requirements

Paired implementations with shared tests; pairs must be human-verified or
derived from a single specification, not model-generated. Direction coverage is
declared; not every pair is required at v2.7.0 (**proposed**: 12 core
directions). Contamination: vectors must not appear in training.

### 18.5 Inference and API changes

`translate_to` is added to chat requests and validated against the supported
language set. Responses include a fidelity note when a construct was approximated.

### 18.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Behavioural equivalence (shared vectors) | ≥ 0.25 | proposed |
| Compile/build success | ≥ 0.70 | proposed |
| Test parity | ≥ 0.30 | proposed |
| Structural fidelity | ≥ 0.50 | proposed |
| Unsupported-construct flagging | ≥ 0.80 precision | proposed |
| Per-language retention | Each of ten within 0.03 | proposed |

### 18.7 Hardware and compute considerations

Cross-runtime execution multiplies sandbox cost: two runtimes per vector set.
CPU-bound; training continues on S4. Budget is **proposed** (§23, D14).

### 18.8 Dependencies and risks

Depends on all ten languages being at their gates (M-C closed). Risks: (a)
translations that compile but behave differently — mitigated by shared vectors;
(b) semantic gaps between languages — mitigated by explicit approximation notes
and flags; (c) pairs generated by the model — prohibited; (d) retention loss —
mitigated by replay.

### 18.9 Deliverables and Definition of Done

Translation suite, shared vector set, direction matrix, S4 checkpoint update,
equivalence report. Done when §18.6 passes and every claimed direction has
recorded evidence.

### 18.10 Explicit exclusions

No repository-aware translation, no framework migration, no performance
optimisation during translation, no new languages, no production SLOs.

---

## 19. v2.8.0 — Repository-aware software engineering

**Group:** M-D · **Model rung:** S4 + long context · **Evidence level target:**
4 · **Closes M-D**

### 19.1 Objective and architecture

**Objective.** Work on a repository, not a snippet: find the right files, read
them, propose a patch that applies, and leave the rest of the tree alone.

**Architecture.** A bounded agent loop with a small, audited tool set
(`read_file`, `list_dir`, `search`, `apply_patch`) over a sandboxed checkout.
The context builder ranks files by relevance and fits them to the context
window. All tool calls are logged; the loop has a hard step and token budget.

### 19.2 Components and modules

| Module | Change |
|---|---|
| `repo/context.py` | File ranking, chunking, context budget |
| `repo/tools.py` | Audited tools; path confinement; read-only default |
| `repo/agent.py` | Bounded loop, step limits, abort conditions |
| `evaluation/repo_tasks/**` | Issue→patch tasks, apply and test oracle |
| `evaluation/repo_leakage.py` | Held-out repo isolation checks |
| `serving/**` | `repo_session` endpoint (internal only) |

### 19.3 Functional features

- Locates the files relevant to a described issue.
- Reads and cites repository context before editing.
- Produces a patch that applies cleanly.
- Leaves unrelated files unchanged.
- Stops when the step or token budget is exhausted, reporting why.

### 19.4 Dataset and training requirements

Repository tasks with issue text, tests and reference patches; per-repository
licence clearance. Held-out repositories are never in training. Tool-use traces
are recorded from the agent loop, not scraped. Context length is extended to at
least 8k tokens (**proposed**). Replay across all ten languages prevents
regression.

### 19.5 Inference and API changes

A `repo_session` endpoint accepts a repository reference, a task and a budget.
Tool calls are returned with `tool` role messages. Patch output is a unified
diff. The endpoint is internal only; no public repository access.

### 19.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Issue→patch tasks | ≥ 0.30 resolved | proposed |
| File localisation | ≥ 0.60 | proposed |
| Patch applies cleanly | ≥ 0.80 | proposed |
| Unrelated-file change rate | ≤ 0.05 | proposed |
| Tool-safety violations | 0 escapes from the checkout | specified |
| Budget compliance | 100% of runs terminate within budget | specified |
| Held-out repo leakage | 0 training overlap | specified |
| Per-language retention | Each of ten within 0.03 | proposed |

### 19.7 Hardware and compute considerations

Long-context inference and retrieval raise memory and latency; the agent loop
multiplies inference calls per task. This is the most compute-intensive version
before production qualification. Budget is **proposed** (§23, D14 and D15).

### 19.8 Dependencies and risks

Depends on M-C closure and the S4 checkpoint. Risks: (a) destructive tool use —
mitigated by read-only default, path confinement and audited patch application;
(b) context stuffing — mitigated by ranking and hard budgets; (c) benchmark
leakage — mitigated by held-out repositories and isolation checks; (d) latency
blow-up — mitigated by step limits and caching.

### 19.9 Deliverables and Definition of Done

Repo context builder, tool layer, bounded agent, repo-task suite, leakage report,
M-D close report. Done when §19.6 passes and the M-D report states measured
features with per-language retention.

### 19.10 Explicit exclusions

No autonomous long-running agents, no CI/CD integration, no public repository
access, no issue-tracker integration, no code hosting, no multi-repository
reasoning.

---

## 20. v2.9.0 — Production qualification

**Group:** M-E · **Model rung:** S4 (frozen) · **Evidence level target:** 4–5 ·
**Opens M-E**

### 20.1 Objective and architecture

**Objective.** Prove the system can be operated, not just evaluated: defined
service levels, isolation, observability, failure handling and reversible
rollout. No new model capability is added.

**Architecture.** A deployment profile names hardware, context limits,
concurrency and quantisation. The serving stack gains structured logging,
metrics, health/readiness separation, request timeouts and admission control.
Qualification runs load, soak and failure-injection suites and exercises
rollback.

### 20.2 Components and modules

| Module | Change |
|---|---|
| `deploy/profiles/**` | Declared hardware and capacity profiles |
| `serving/observability.py` | Metrics, structured logs, tracing hooks |
| `serving/admission.py` | Concurrency and queue limits |
| `ops/runbooks/**` | Start, drain, rollback, incident, restore |
| `evaluation/load/**` | Load, soak and failure-injection suites |
| `evaluation/slo.py` | SLO computation from recorded metrics |

### 20.3 Functional features

- Publishes service levels for the named profile and meets them under load.
- Degrades predictably under overload instead of failing arbitrarily.
- Rolls forward and back without data loss.
- Produces an incident timeline from logs alone.

### 20.4 Dataset and training requirements

No training. Evaluation uses frozen model artifacts and synthetic load
generators. Any prompt data used in load tests is non-sensitive and recorded.
Provenance of the frozen artifacts is verified before qualification begins.

### 20.5 Inference and API changes

Health and readiness are separated. Requests gain timeouts and structured error
codes (`overloaded`, `timeout`, `budget_exceeded`). Metrics endpoint is exposed
on an internal interface only. No capability change.

### 20.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| Availability (soak) | ≥ 99.5% on the published profile | proposed |
| p95 latency | Within the declared profile budget | proposed |
| Error rate under rated load | ≤ 1% | proposed |
| Overload behaviour | 100% structured refusals, no crash | specified |
| Rollback drill | Restores prior version within the declared window | specified |
| Security review | Passed, findings tracked to closure | specified |
| Observability | Every request traceable end to end | specified |
| Open P0/P1 issues | 0 / tracked with owners | specified |

### 20.7 Hardware and compute considerations

This version **defines** the production profile; it does not assume one. Load
and soak tests require provisioned hardware for the declared duration. Budget,
hardware class and duration are **proposed** (§23, D14, D15 and D16) and must be
approved before scheduling.

### 20.8 Dependencies and risks

Depends on all prior versions and on frozen artifacts. Risks: (a) SLO claims on
inadequate hardware — mitigated by the published-profile rule; (b) load tests
that miss real traffic shape — mitigated by recorded traffic assumptions; (c)
rollback untested — mitigated by a mandatory drill; (d) security findings
deferred — mitigated by tracked closure before v3.0.0.

### 20.9 Deliverables and Definition of Done

Deployment profiles, observability, admission control, runbooks, load/soak/
failure reports, rollback transcript, security review, M-E opening report. Done
when §20.6 passes on a named profile and the runbooks have been exercised.

### 20.10 Explicit exclusions

No new model capability, no GA or public launch claim (v3.0.0 decides that), no
multi-region design, no autoscaling beyond the declared profile, no new
languages, no 7B-scale work.

---

## 21. v3.0.0 — Production-ready multilingual Taksh

**Group:** M-E · **Model rung:** S5 (1,315,014,656 params) · **Evidence level
target:** 5 · **Closes M-E and the roadmap**

### 21.1 Objective and architecture

**Objective.** Ship a production-ready multilingual coding model: ten
programming languages, repository awareness, translation, and the operational
guarantees of v2.9.0, at the largest rung this roadmap permits. This is a
**release decision**, not an automatic consequence of time passing.

**Architecture.** The S5 geometry (`configs/model/prod_1p5b.yaml`, 1,315,014,656
parameters, verified twice) is trained on the consolidated multilingual corpus
with the frozen tokenizer contract. Serving uses the v2.9.0 operational stack
with a declared quantisation profile. The release binds model, tokenizer,
contract, data card and code revision into one manifest.

### 21.2 Components and modules

| Module | Change |
|---|---|
| `configs/model/prod_1p5b.yaml` | S5 geometry, verified twice |
| `data/contracts/final_v1.yaml` | Consolidated mixture and licence manifest |
| `training/**` | Final training run with full provenance |
| `deploy/profiles/prod_v1.yaml` | Quantised production profile |
| `tools/release.py` | Release manifest binding all artifacts |
| `docs/source/releases/**` | Production release notes and model card |

### 21.3 Functional features

- Ten languages at their measured levels, retained under one model.
- Repository-aware tasks and cross-language translation.
- Documented limitations, refusals and safety behaviour.
- A reproducible release with a verifiable manifest.

### 21.4 Dataset and training requirements

Consolidated corpus under the final data contract; every source licence decided;
decontamination complete for every suite; per-language mixture floors documented.
Final run recorded with seeds, versions and command. Data card and model card are
release artifacts. No source is admitted after the contract is frozen.

### 21.5 Inference and API changes

The v2.9.0 API surface is frozen as the v3 contract. `/v1/models` reports the
release manifest id. Deprecations follow a one-minor-version notice. No breaking
change inside v3.

### 21.6 Benchmarks and release gates

| Gate | Threshold | Basis |
|---|---|---|
| All ten language suites | At or above their recorded version levels | specified |
| Repository tasks | ≥ 0.30 | proposed |
| Translation equivalence | ≥ 0.25 | proposed |
| Python functional | ≥ 0.50 (no regression from v1.6.0) | proposed |
| SLOs | v2.9.0 gates met on the production profile | specified |
| Promotion gate | Green, all verdicts recorded | specified |
| Independent review | Level 5 evidence confirmed by an external reviewer | specified |
| Licence completeness | 100% of distributed artifacts | specified |
| Reproducibility | Release rebuilds from manifests within the interval | specified |
| Open P0/P1 | 0 / none blocking | specified |

### 21.7 Hardware and compute considerations

The S5 run is the largest expenditure in the roadmap and requires approved
multi-GPU capacity. Inference requires production-class GPUs at the declared
quantisation. Both are **proposed** (§23, D14, D15 and D16); no production
commitment exists until approved.

### 21.8 Dependencies and risks

Depends on v2.9.0 qualification, M-C/M-D closure and complete licensing. Risks:
(a) scale-up without a data decision — prohibited by §22.1 rule 2; (b) release
pressure overriding a red gate — mitigated by the sole-approver rule and
published refusal; (c) cost overrun — mitigated by a staged pilot and a stop
rule; (d) capability regression at scale — mitigated by per-language retention
gates; (e) distribution licence defect — mitigated by the completeness gate.

### 21.9 Deliverables and Definition of Done

S5 checkpoint, final data and model cards, production profile, release manifest,
full evaluation report, independent review, release notes. Done when §21.6 passes,
the owner signs the release, and the manifest verifies from a clean checkout.

### 21.10 Explicit exclusions

**Deferred beyond v3.0.0:** the 6B/7B rung (`scale_7b.yaml` at 6,073,618,432
parameters) is explicitly out of scope; multimodal capability remains deferred;
languages beyond the ten are not promised; no hosted public service is implied by
this document; no fine-tuning service, no on-device build, no autonomous
long-running agents.

---

## 22. Programme governance

### 22.1 Model-scaling decisions

Rungs are named, counted and gated. No version may silently borrow a larger rung.

| Rung | Parameters | Source of truth | First legitimate use |
|---|---|---|---|
| S0 | < 1M | smoke config | pipeline tests only |
| S1 | ~3M | config | pipeline tests only |
| S2 | **9,359,104** | M7 checkpoint | v1.0.0 baseline; v1.1.0–v1.2.0 |
| S3 | **16,257,280** | `configs/model/code_30m.yaml` | v1.3.0 onward |
| S4 | **113,145,408** | `configs/model/pilot_300m.yaml` | v2.0.0 onward |
| S5 | **1,315,014,656** | `configs/model/prod_1p5b.yaml` | v3.0.0 |
| S6 | **6,073,618,432** | `configs/model/scale_7b.yaml` | **deferred beyond v3.0.0** |
| S7 | larger | none | not planned |

Rules:

1. **A filename is not a claim.** "300m" is 113M; "1p5b" is 1.315B; "7b" is
   6.07B. The counted value is the only value.
2. **No scale-up without a data decision.** A larger rung requires a decided
   corpus, mixture and licence position for that run.
3. **Scale-down is legal; unrecorded change is not.** Any rung change appears in
   the version's objective and in the run manifest.
4. **Every count is verified twice**, independently, before a release note.
5. **S4, S5 and S6 are proposed, not approved.** Using any of them requires the
   owner decisions in §23 (D7, D14, D16); this table records intent, not consent.

### 22.2 Dataset governance

- **Allowlist first.** A source enters only through an explicit allowlist entry.
- **Licence gate is blocking.** A source without a decided licence is excluded
  (§23, D1–D2 for the two open judgements).
- **Pinned commits.** Every source is fetched at a recorded commit and hash.
- **Source-group splits.** No split by random line; groups stay together.
- **Decontamination is mandatory** against every locked suite, with zero
  tolerance and a recorded report.
- **Data card per version.** Size, composition, exclusions, dedup and
  contamination rates, in plain language.
- **Nine release checks**: allowlist, licence, pin, split, dedup, contamination,
  card, hash, reproducibility.
- **Human decisions only** for licensing; no model or tool may decide a licence.

### 22.3 Checkpoint and prompt provenance

Every distributed checkpoint records: model geometry, parameter count,
tokenizer hash, config hash, contract id, code revision, corpus hash, training
command, seeds, library versions and creation time — eleven fields, no
exceptions.

- Loading is strict: a missing or mismatched field is a refusal, not a warning.
- Partial loads are possible only through an explicit flag and are marked in the
  served model id.
- Verification happens in a **fresh process**, never in the process that wrote
  the file.
- `runs/**` is never committed; only manifests and hashes are.

### 22.4 Evidence levels

| Level | Meaning | Claim permitted |
|---|---|---|
| 0 | Reproducible run exists; capability not demonstrated | "trained", never "works" |
| 1 | Measured on a locked internal suite | the measured number only |
| 2 | Measured on a locked suite plus a held-out split | the number with interval |
| 3 | Measured on public plus private suites, reproducible | comparison permitted |
| 4 | Measured under operational conditions (latency, load, tools) | operational claims |
| 5 | Independently reviewed, evidence trail complete | public capability claim |

A version may not claim a level its evidence does not reach. Level 5 requires a
reviewer who is not the author and who did not run the primary evaluation.

### 22.5 Promotion gate

`evaluation/promotion.yaml` is the single declarative gate. The runner
(`tools/promote.py`) computes verdicts; no one marks a promotion by hand.

- Every rule has an id, a threshold, a basis (`measured`, `specified`,
  `proposed`) and a verdict.
- A missing measurement is **failing**, never "skipped".
- A threshold change is a separate, recorded decision; it never happens inside a
  version.
- No champion → `NO_CHAMPION`; the release is held or refused.
- The gate never reads a number from prose; it reads artifacts.

### 22.6 Release qualification

Seven points before any tag:

1. **Version consistency** — `VERSION`, package metadata, changelog and tag
   agree.
2. **Release preparation** — `tools/prepare_release.py` completes cleanly.
3. **Evidence bundle** — every claim maps to a recorded artifact.
4. **Per-version gates** — all gates green at their recorded thresholds.
5. **Model tag** — only on a green promotion; source tags may precede it.
6. **Documentation consistency** — README, status and roadmap agree with the
   measured result.
7. **Owner authorisation** — an explicit, separate approval.

Refusal is a valid outcome: the seven points become a published refusal with the
failing rule ids.

### 22.7 Milestone roll-up (approved)

| Group | Versions | Responsible for | Close evidence |
|---|---|---|---|
| **M-A** | v1.1.0–v1.4.0 | Python Learning Foundation | Prompt identity 100%, reproducible runs, governed 100M-token corpus, syntax ≥ 0.50, completion top-5 ≥ 0.40 |
| **M-B** | v1.5.0–v1.9.0 | Python Developer Capability | Instruction format ≥ 0.90, functional ≥ 0.50, repair ≥ 0.30, workflow gates, alpha qualified or refused |
| **M-C** | v2.0.0–v2.6.0 | Multilingual Expansion | Nine non-Python languages (SQL, JavaScript, TypeScript, Shell, Java, Go, C#, C, C++) each measured, per-language retention, contamination clean |
| **M-D** | v2.7.0–v2.8.0 | Software Engineering Intelligence | Translation ≥ 0.25, repo tasks ≥ 0.30, tool safety enforced |
| **M-E** | v2.9.0–v3.0.0 | Production Qualification | SLOs met on a published profile, independent review, release manifest verified |

A group closes only when every version in it passes its own gates. Closing a
group does not release a version; releasing is always the separate decision in
§22.6.

### 22.8 Programme risk register

| Id | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Capability stalls below coding usefulness | Medium | High | Gate early at syntax and functional; publish refusal rather than overclaim | Owner |
| R2 | Data under target after governance | Medium | High | Approve documented shortfall or widen allowlist by decision | Owner |
| R3 | Licensing blocks a source or release | Medium | High | Exclude first; decide licences before acquisition | Owner |
| R4 | Compute budget exceeds approval | Medium | High | Staged pilots; stop rules; no silent rung change | Owner |
| R5 | Sandbox or tool escape | Low | High | No network, path confinement, security review, audits | Engineering |
| R6 | Benchmark contamination | Medium | High | Pinned suites, hashes, n-gram checks, private probes | Engineering |
| R7 | Python regression from multilingual breadth | High | Medium | Mixture floors, replay, retention gates | Engineering |
| R8 | SLO claims on inadequate hardware | Medium | Medium | Published-profile rule; no claim without provisioned hardware | Engineering |

### 22.9 Consolidation map

| Document | Role after this reconciliation |
|---|---|
| `docs/ROADMAP.md` | Historical M0–M7 record plus a navigation banner to this file |
| `docs/CODE_LLM_STRATEGY.md` | Strategy context; forward plans superseded here |
| `docs/VERSIONING.md` | Versioning rules; unchanged |
| `docs/RELEASING.md` | Release mechanics; unchanged |
| `docs/STATUS.md` | Current state; must not claim beyond evidence |
| `docs/OPEN_ISSUES.md` | Live issue register; §23 records owner decisions |
| `docs/DESIGN_ARCHITECTURE.md` | System design; implementations follow it |
| `evaluation/promotion.yaml` | The executable gate (§22.5) |

### 22.10 Decision ownership

The owner approves: version scope, model rung and compute, licence decisions,
thresholds marked **proposed**, promotion, refusal, and any release tag.
Engineering owns: implementation, evidence production, gate execution and
honest reporting. No agent, tool or document may substitute for the owner's
approval.

---

## 23. Open owner decisions

Seventeen decisions are recorded. Their status is explicit: none is approved by
this document. The categories below are deliberately separate — a confirmed fact
is not a decision, and a proposal is not an approval.

### 23.1 Confirmed implementation facts (not decisions)

Measured or shipped, and not open: M7 has 9,359,104 parameters; the locked suites
record 0/40 syntax, 0/8 functional and 0/29 cases; training used
`qm.plain_instruction/1`; serving used a different template; the diagnostic is
`MODEL_UNDERTRAINED` with `SERVING_PROMPT_MISMATCH`; v1.0.0 is a source release
with no weights. Config parameter counts are as listed in §22.1.

### 23.2 Unresolved licensing questions

| Id | Decision | Blocks | Status |
|---|---|---|---|
| D1 | `psf-black` subtree licence position (GPL-2.0) | v1.2.0 corpus | open |
| D2 | `pypa-setuptools` subtree licence position (AGPL-3.0) | v1.2.0 corpus | open |
| D3 | Allowlist widening beyond the current repositories | v1.2.0 scale target | open |
| D4 | Licence for distributed weights and derived data | v1.9.0, v3.0.0 | open |
| D5 | Benchmark-suite vendoring terms | v1.6.0 | open |

### 23.3 Proposed architectural decisions (await approval)

| Id | Decision | Blocks | Status |
|---|---|---|---|
| D6 | Tokenizer vocabulary and multilingual target | v1.2.0, v2.0.0 | proposed |
| D7 | Geometries and hyperparameters for new rungs (S4, S5) | v2.0.0, v3.0.0 | proposed |
| D8 | Sandbox toolchains and runtime pins | v2.0.0 onward | proposed |
| D9 | Repository-read residency posture (what may be read, where) | v2.8.0 | proposed |
| D10 | Tag naming and model-tag policy | every release | proposed |
| D11 | Thresholds marked `proposed` in this document | every gate | proposed |
| D12 | Promotion-runner enforcement timing | v1.8.0 onward | proposed |
| D13 | Human-evaluation annotator pool and rubric | v1.9.0 | open |

### 23.4 Future compute and scaling decisions

| Id | Decision | Blocks | Status |
|---|---|---|---|
| D14 | Compute class and budget per rung (S3–S5) | v1.3.0, v2.0.0, v3.0.0 | open |
| D15 | Production hardware profile and duration | v2.9.0, v3.0.0 | open |
| D16 | Whether S6 (6.07B) is ever attempted | post-v3.0.0 | open |
| D17 | Support window and maintenance policy post-v3.0.0 | v3.0.0 | open |

**Resolved within this change:** the earlier `docs/ROADMAP.md` authority question
is closed — the banner in that file now points here, and §0.1 defines precedence.
It is recorded rather than deleted so the count and history remain auditable.

Any decision approved by the owner must be recorded with date, decision text and
the affected versions; silence is not approval.

---

## 24. Traceability matrix

Every version maps to its section, milestone group and the primary artifact that
proves it.

| Version | § | Group | Primary evidence artifact | Deciding gate |
|---|---|---|---|---|
| v1.0.0 | 1 | baseline | `docs/source/roadmap/M7_EXPERIMENT_REPORT.rst` | source release accurate |
| v1.1.0 | 2 | M-A | prompt-identity and repeat-run transcripts | identity 100%, manifest complete |
| v1.2.0 | 3 | M-A | data contract, data card, rebuild hashes | ≥ 100M tokens or approved shortfall |
| v1.3.0 | 4 | M-A | loss/perplexity records, control comparison | beats M3/M4 controls |
| v1.4.0 | 5 | M-A | syntax and completion suites | syntax ≥ 0.50, top-5 ≥ 0.40 |
| v1.5.0 | 6 | M-B | instruction suite, format report | format ≥ 0.90 |
| v1.6.0 | 7 | M-B | pass@1 report with intervals | functional ≥ 0.50 |
| v1.7.0 | 8 | M-B | repair, localisation, mutation reports | repair ≥ 0.30, mutation ≥ 0.30 |
| v1.8.0 | 9 | M-B | latency report, promotion runner log | p95 first token ≤ 400 ms |
| v1.9.0 | 10 | M-B | qualification pack or refusal | reproduction + review |
| v2.0.0 | 11 | M-C | SQL/JS execution reports, retention report | SQL ≥ 0.50, JS ≥ 0.40 |
| v2.1.0 | 12 | M-C | `tsc` and web harness reports | type-check ≥ 0.60 |
| v2.2.0 | 13 | M-C | shell execution, safety report | safety recall ≥ 0.95 |
| v2.3.0 | 14 | M-C | compile and JUnit reports | compile ≥ 0.80 |
| v2.4.0 | 15 | M-C | build, test, race-detector reports | race-free ≥ 0.90 |
| v2.5.0 | 16 | M-C | compile, async, nullable reports | compile ≥ 0.80 |
| v2.6.0 | 17 | M-C | sanitizer matrix report | sanitizer-clean ≥ 0.90 |
| v2.7.0 | 18 | M-D | equivalence report over shared vectors | equivalence ≥ 0.25 |
| v2.8.0 | 19 | M-D | repo-task and tool-audit reports | repo tasks ≥ 0.30 |
| v2.9.0 | 20 | M-E | load/soak/failure reports, runbooks | SLOs on published profile |
| v3.0.0 | 21 | M-E | release manifest, independent review | all gates + level 5 |

## 25. Scope of this change

This reconciliation:

- Restores the **approved version responsibilities** for all 21 versions
  (§0.9) and the **approved milestone groups** (§0.5, §22.7).
- Applies the **ten-section schema** to every version (§0.4), including
  hardware/compute and explicit exclusions.
- **Moves** engineering detail into the version that owns it rather than
  deleting it: prompt identity and reproducibility to v1.1.0; corpus and
  tokenizer to v1.2.0; pretraining to v1.3.0; syntax and completion to v1.4.0;
  instruction tuning to v1.5.0; functional correctness to v1.6.0; debugging and
  tests to v1.7.0; workflows and serving maturity to v1.8.0; freeze and
  qualification to v1.9.0; SQL and JavaScript to v2.0.0, with the remaining
  languages and translation shifted accordingly.
- Makes this file the **canonical forward-development reference** and adds a
  navigation-and-authority banner to `docs/ROADMAP.md` without changing its
  history.
- Retains **all 17 owner decisions** (§23), separated into confirmed facts,
  proposed architecture, unresolved licensing and future compute/scaling.
- Changes **no** source code, dataset, checkpoint or training configuration; runs
  no training, inference or evaluation; preserves `reports/live-lab/` untouched;
  commits, tags and publishes nothing.

**Not decided here:** model sizes, benchmark thresholds as achieved, deployment
targets, licences and compute budgets. Those remain owner decisions (§23).
