# DLCAE — Dynamic Language Comprehension and Adaptation Engine

**Status:** architecture specification (L0.6.0-R1). **DLCAE is NOT implemented.**
Every component below is a *specified contract*, not working software. Nothing
in this document may be read as evidence of capability.

**Purpose.** TAKSH must progressively comprehend, learn, adapt to and qualify
arbitrary programming languages without a fixed catalogue and without a
separate foundation model per language.

**Position in the platform.** DLCAE is a *logical* engine cooperating with the
Learning, Knowledge, Code Intelligence, Engineering, Memory, Evaluation and
Agent engines. It owns nothing that would require changing the foundation
model's fixed configuration for each new language.

---

## 1. Design hypothesis (owner's original, preserved)

> Illustrative sequence: `C → C1 → C2 → C3 → C4`
> Another independent sequence: `D1 → D2 → D3`

The design hypothesis is that TAKSH can examine information *progressively* —
not merely ingest it wholesale — decomposing it into comprehensible stages,
tracking several such sequences at once, and driving them to terminal states.
This hypothesis is to be **tested**, not assumed. It is the design basis for the
Incremental Comprehension and Sequence Orchestration engines.

## 2. The six engines

### 2.1 Incremental Comprehension Engine

- **Responsibility:** transform raw input (bytes → tokens → identifiers →
  syntax trees → semantics → task decompositions) into progressively richer
  internal representations, *one level at a time, verifiably*.
- **Inputs:** raw source bytes; tokenizer identity (`taksh_code_bpe_v1.json`)
  + extension registry.
- **Outputs:** leveled representation bundles (L1–L7 below), each tagged with
  provenance and confidence; no bundle asserts capability by itself.
- **Dependencies:** tokenizer contract; Knowledge Engine; Evaluation Engine.
- **State:** per-document comprehension cursor; never global.
- **Failure conditions:** tokenizer mismatch → abort; level inconsistency →
  mark blocked, report, do not guess; resource exhaustion → checkpoint and suspend.

### 2.2 Sequence Orchestration Engine

- **Responsibility:** decompose an instruction into ordered sub-tasks
  (`C → C1 → C2 → C3 → C4`), track independent sequences (`D1 → D2 → D3`),
  resolve dependencies and trigger conditions, and drive every sequence to
  a terminal state.
- **Inputs:** instruction; contextual task graph; available tools.
- **Outputs:** task DAG; per-task state machine (PENDING → IN_PROGRESS →
  {DONE, FAILED, BLOCKED, SKIPPED}); resume checkpoints.
- **Dependencies:** Memory Engine (persistence); Evaluation Engine (definition
  of done); Agent Engine (execution).
- **State:** portable task ledger (JSON-serializable); resumable after crash.
- **Failure conditions:** cyclic dependency → report, do not loop; unbounded
  retries → cap; contradictory post-conditions → BLOCKED with reason; task
  fails twice → FAILED, record, continue or escalate per policy.

### 2.3 Knowledge Formation Engine

- **Responsibility:** construct versioned, reusable, *verified* language knowledge:
  concepts, rules, relationships, examples — stored separately from weights,
  shared across languages where valid.
- **Inputs:** approved sources (license-clean); comprehension bundles; test feedback.
- **Outputs:** knowledge records with provenance, verification result,
  version, and invalidation rules.
- **Dependencies:** Learning Engine (alignment); Evaluation Engine (gate for
  admission); Knowledge Engine (storage).
- **State:** append-only, versioned knowledge store; rollback by revert-to-mark.
- **Failure conditions:** source not on allowlist → reject; verification
  regression after add → automatic revert; conflicting knowledge → keep both, mark contested, never silently merge.

### 2.4 Language Adaptation Engine

- **Responsibility:** given a *new* language (or new version/dialect of an
  existing one), drive it from REGISTERED → DISCOVERED →
  COMPREHENSION_IN_PROGRESS → FUNCTIONALLY_QUALIFIED (or BLOCKED).
- **Inputs:** registry entry (name, extensions, spec reference, provenance,
  license authorization, toolchain contract).
- **Outputs:** extension record; adaptation metrics (time-to-first-correct,
  retention, interference vs prior languages); status promotion/demotion.
- **Dependencies:** Incremental Comprehension Engine; Knowledge Formation Engine;
  Evaluation Engine; toolchain integration.
- **State:** per-language adaptation ledger; no state is shared implicitly.
- **Failure conditions:** license gate failure → BLOCKED; toolchain missing →
  COMPREHENSION_IN_PROGRESS with explicit block reason; regression in held-out
  suite → demote one status, never silently accept.

### 2.5 Validation Engine

- **Responsibility:** validate every completed step: syntax, types, semantics,
  execution, tests, regressions vs previous capability; report, never fix silently.
- **Inputs:** candidate output; the authoritative toolchain contract
  (compiler/interpreter/test runner); benchmark gates.
- **Outputs:** PASS/FAIL + diagnostic codes; benchmark deltas; safety verdict.
- **Dependencies:** all other engines consume its verdicts.
- **State:** validation ledger; immutable verdicts.
- **Failure conditions:** toolchain unavailable → verdict NOT_APPLICABLE (never
  coerced to PASS); timeout → FAIL with timeout reason; partial check → BLOCKED.

### 2.6 Learning Feedback Engine

- **Responsibility:** convert verified outcomes into controlled learning signals:
  replay selection, forgetting measurement, curriculum updates — *never* raw
  ingestion of arbitrary outputs.
- **Inputs:** validation verdicts; provenance records; benchmarks.
- **Outputs:** replay buffer manifests; forgetting reports; approved SFT pairs
  (only after validation); drift alarms.
- **Dependencies:** Learning Engine; Evaluation Engine; Memory Engine.
- **State:** feedback manifests; rollback points.
- **Failure conditions:** unverifiable signal → discard; contaminating signal →
  quarantine; regression beyond tolerance → halt learning, revert last checkpoint.

## 3. The seven comprehension levels

| Level | Name | What is understood | Example check |
|---|---|---|---|
| L1 | Character representation | bytes → internal codes | tokenizer roundtrip |
| L2 | Token comprehension | tokens, boundaries, specials | ids across languages stable |
| L3 | Lexical/identifier recognition | keywords, identifiers, literals, operators | per-language keyword coverage |
| L4 | Syntax/structure | declarations, control flow, types, scopes | parser outcome |
| L5 | Semantic/type comprehension | types, ownership, effects, constraints | type-checker / compiler oracle |
| L6 | Instruction/task comprehension | decomposition of intent into steps | task-graph correctness |
| L7 | Execution and verification | runs, passes tests, debugs | benchmark execution |

**Character-level processing (L1) is a research hypothesis, not a default
design.** The correct algorithm for L1/L2 is to be determined by experiment,
not assumption. Token-level attention on a 5,822-vocab BPE may dominate any
character-level loop at plausible scales; the loop stays only if evidence
demands it.

## 4. Experiments required before any claim (DLCAE vs RAG vs hybrid)

See `DLCAE-RESEARCH-PLAN.md`. Absolutely none of these have been run; no
superiority claim is or may be made without them.

## 5. Continual-learning contracts (mandatory gates)

Every adaptation of TAKSH to something new — a language, version, dialect, framework, library/API, build system, repository or coding convention — must pass the same gates:

1. **Provenance & licensing** — every source record carries source id, license SPDX id (or `see-MANIFEST` + pointer to verified manifest), retrieval date, hash. Unknown license ⇒ rejected, never ingested.
2. **Quality qualification** — human triage or automated quality gates (license completeness, hash verification, secret scan, benchmark-leakage scan) must pass before a single token enters the training pool.
3. **Training-example validation** — every candidate training pair is executed (sandboxed) or otherwise verified BEFORE training; nothing model-generated enters training without validation.
4. **Benchmark contamination prevention** — no held-out benchmark task or near-duplicate may enter the corpus; enforced by the same shingle-Jaccard machinery used in L0.4.
5. **Replay and regression protection** — replay buffer samples previous languages every epoch; forgetting measured against the frozen suites; regression beyond tolerance halts the run.
6. **Catastrophic-forgetting measurement** — after every update, the full previously-qualified suite re-runs; delta recorded; any drop beyond tolerance triggers automatic rollback to the last validated checkpoint.
7. **Checkpoint versioning and rollback** — every accepted update produces a new versioned checkpoint with full manifest; rollback = revert-to-mark; rollback never requires re-training.
8. **Human approval where appropriate** — changes to production behavior, benchmark bars, license policy, or model defaults require explicit owner sign-off; routine adaptation does not.

**Execution results are evidence, not automatically authorized training data.** A failing run is a diagnostic, not a dataset.

## 6. What DLCAE must never do

- Never bundle a hidden hard-coded language list into the foundation weights.
- Never claim capability for a language not run through the qualification gates.
- Never swallow a compiler error into a guessed "fix" without disclosure.
- Never self-promote a status (statuses are earned through measurement).
- Never train on unvalidated model-generated code without explicit owner sign-off.

## 7. Implementation status (honest)

- **Specified:** everything in this document.
- **Partially implemented:** nothing.
- **Implemented:** nothing.
- **Test evidence:** none for DLCAE (by design, this milestone qualifies the
  architecture and its contracts; no DLCAE runtime exists).
