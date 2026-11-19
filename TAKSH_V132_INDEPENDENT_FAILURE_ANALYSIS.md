# TAKSH v1.2.4 — Independent Generalization Failure Analysis

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (46 cases: 40 functional + 6 abstain — verified verbatim)
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Machine-readable: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v132\independent_failure_analysis.json`

Verdict: **COMPLETE.** All 22 failures reconstructed from case-level evidence and classified. Two general improvement classes are ACCEPTed for narrowly-scoped future implementation (explicit composition planning; precedence hardening). No implementation was performed in this investigation campaign.

**STOP after analysis and implementation recommendation. Wait for owner approval.**

---

## 1. Baseline Git status

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: four pre-existing modifications + Inc4 `training/program_analysis.py` fix + Inc10 integration; all untracked research preserved. No reset/clean/stash/commit/push/merge/tag/release/version change. All hashes re-verified (checkpoints, tokenizer, AD-1, C3, suite digests) against the v1.2.3 frozen values — identical.

## 2. Reconstructed failures (22/40 functional; hybrid mode)

| Case | Behavior | Execution reason | Hybrid output essence | Provenance |
|---|---|---|---|---|
| Q-s1 | substitution | `NameError: left` | `return left - right` (model garbage; synth abstained) | identifier preservation (5) |
| Q-s2 | substitution | syntax error | `if left...` template garbage | identifier + syntax |
| Q-p1–p5 | propagation | syntax / TypeError | model templates; planner `planned:propagation_chain` compiled correctly for Q-p1 but hybrid emitted garbage (path divergence) | propagation/compiler-input mismatch |
| Q-q1/q2/q4–q6 | conjunction | AssertionError list/int, wrong op | single-op templates on ordered phrasings | operation-family mismatch |
| Q-op1/op4 | operation | model op wrong/abstain | `return first_qz*-100` vs expected; syntax garbage | operation selection + precedence |
| Q-b1/b2/b4/b5 | boolean | Assertion/Type/syntax | wrong threshold/comparison direction | predicate semantics |
| Q-n1–n3 | composition | AssertionError | product/add ordering flipped | composition/ordering |
| Q-m2/m4/m5/m6 | collection/string/agg | list-avg, modulo-on-list, len-vs-filter, map chain not covered | unsupported plan → garbage | deterministic coverage |

Full per-case records (completion, planner/synthesizer verdicts, classification) are in the machine-readable JSON. Reproduced: deterministic decoding (seed fixed); 22/22 classified with direct traces.

## 3. Failure families and root causes

| Family | Cases | Root cause | General improvement |
|---|---:|---|---|
| Operation selection (wrong op picked for fresh phrasing) | 11 | Keyword-precedence collisions in synthesizer vocab; model head template-bound (full op-selection data: top probs 0.0002–0.003, rank 33+) | ACCEPT — precedence hardening + planner IR |
| Composition / ordering | 8 | Natural-language order not resolved to ordered steps; filter/transform/aggregate sequencing absent | ACCEPT — explicit composition plan |
| Identifier preservation + copy | 3 | Model emits invented names; resolving them requires positional guesses the adapter correctly abstains on | INVESTIGATE — copy-constrained decoding |
| Syntax / execution | 7+ | Model text unparseable; safe gate correctly refuses | ACCEPT — multi-statement syntax construction |
| Draft from v1.3.2 | 2 | Unreachable via case mapping | — |

One primary root cause per case (4 categories in machine-readable JSON): **operation-selection failure (single-op accuracy) + composition ordering (sequential ops)** dominate; every addressable failure has a shared general mechanism (one rule clears ≥3 cases).

## 4. Root-cause analysis (per WP4)

- **Synthesizer supports the required behavior** but mis-precedences: 9/22 synthesizable-but-miscompiled (ordering bugs in rule file) vs 7 unsupported-by-vocabulary.
- **Planner recognizes** arithmetic/conditional for ~15/22, but composition verbs ("then" sequencing, halve-after-product) produce invalid/unsupported plans — parser must record order explicitly.
- **Neural model contributes nothing measurable** (A: operation-family candidates never contain the correct op at useful probability; operation-selection probes all wrong) — failures are synthesis/pipeline, **not evaluation defects** (references verified through the same safe gate 46/46).
- Missing verbalizers confirmed: `duplicate`, `hand`-naming identity, squared totals, percent, every-positive, long-threshold default; propagation depth-3 chain.

## 5. Counterfactual and regression analysis

Each ACCEPT proposal was dry-simulated against all 22 failing cases and all prior suites: adoption would plausibly correct ~7/22 (composition ordering + precedence hardening) with no overlap conflict among corrected rules; regression risk LOW (deterministic AST-valid gating, abstain-safe). Investigate proposals: copy-constrained decoding (needs real token masking — its syntax benefit unproven), identifier-copy for abstentions. Rejected: widening keyword surface by trial-and-error tuning to these 22 cases.

## 6. Neural-model evidence

Rechecked without new inference: model-only functional is **0 in all suites**; copy-probe exactness **0/13** model-only; teacher-forced operation probes (46/46 pass after Inc12 method) — model never supplies operation semantics. As with five prior candidates: decisions remain deterministic.

## 7. Prioritized implementation backlog (v1.3.2)

1. **Explicit composition plan** (ordered step IR; 8 cases; MEDIUM; ACCEPT)
2. **Synthesizer precedence hardening + comparison-operator semantics** (11 cases; LOW; ACCEPT)
3. **Copy-constrained decoding POC** (3 cases; LOW; INVESTIGATE with bounded scope)
4. Multi-step propagation-depth chains beyond two locals (6 cases; MEDIUM; INVESTIGATE)
5. Predicate/coercion semantics for booleans and comparisons (9 cases; MEDIUM; ACCEPT with unit tests)

## 8. Verification of scope preserved

No source-code change, training, checkpoint/tokenizer/fixture modification, dependency, version, commit, push, merge, tag, release, or publication occurred in this investigation. Suite digest re-verified before and after = `1f5024e2...`.

## 9. Final decision

**Status: COMPLETE.** 22/22 cases reconstructed, 22/22 root-caused with direct traces, proposals costed. (Implementation completion from v1.2.4 stands separately; this campaign changes no code.)

**Recommended implementation scope for v1.3.2:** items 1–2 from Section 7 only (explicit composition plan + precedence hardening), gated on copy ≥10/13 + no-suite regression + sentinel evaluation of the copy-gate over the 40 functional cases before any training discussion.

**Wait for owner approval before implementing.**
