# TAKSH v1.2.0 — Integrated Code Generation Campaign: Evidence Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Machine-readable: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\integrated\comparison.json`

Verdict: **INTEGRATED GENERATION IMPLEMENTED AND EVALUATED. Binding and copying improve demonstrably; functional execution does not improve (0 everywhere). No training, no checkpoint/tokenizer change, no promotion.**

**STOP after this campaign. No further training or publication.**

---

## 1. Starting and ending state

- Branch `develop` @ `a654e5d`, package `1.0.0` at start and end.
- Initial Git status: 4 pre-existing tracked modifications + Inc4 fix + Inc10 integration (2 files); all untracked research preserved.
- Final tracked diff identical (6 files, no new tracked modification this campaign). New untracked: 4 implementation files + 1 test file + this report (Section 3). No resets, commits, pushes, merges, tags, version changes, or publication.
- Hashes verified at end: S2 `02cc86eb...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...` ✓.

## 2. Prior work read

Increments 8–13 (diagnosis, conditional analysis, copy mechanism, adapter results) and all relevant implementations/tests were inspected. The Increment 13 limitations (abstention-dominated, cosmetic binding fixes, 0 functional) drive this campaign's operation-aware resolution.

## 3. Files added/modified

| File | Purpose |
|---|---|
| `evaluation/hybrid.py` | Integrated generator: model + candidates + operation-aware resolution + teacher-forced confidence + audit |
| `evaluation/generalization_suite.py` | 24 additional held-out generalization cases (novel names/structures/depths), reference-verified |
| `scripts/evaluate_integrated.py` | Three-mode evaluator (model / adapter / hybrid) with safe execution |
| `tests/test_hybrid.py` | 7 focused tests (relation inference, disabled path, determinism, reference correctness) |
| `TAKSH_V120_INTEGRATED_CAMPAIGN_REPORT.md` | This report |

No existing file modified.

## 4. Architecture and generation flow implemented

`HybridGenerator(model, tokenizer)`: (1) existing neural head generates greedily (identical decoding to all prior evals, `binding_repair=False`); (2) incremented candidates are extracted with the Increment 13 AST mechanism; (3) operation-aware resolution infers source-to-target relations from the instruction + body AST (positional uniqueness, `first`/`second` hints, plus multi-statement-consistent global mapping — locals/builtins never touched); (4) teacher-forced confidence (mapped-token probability mass) gates the rewrite; low-confidence or ambiguous cases **abstain** to the raw model output; (5) every decision is audit-logged. `enabled=False` passes raw output through (fallback preserved). Weights never read for writing; checkpoint loaded read-only.

## 5. Exact comparison results (all three modes, same cases, frozen + supplemental + generalization)

| Suite | Metric | Model-only | Adapter (Inc13) | **Hybrid (integrated)** |
|---|---|---:|---:|---:|
| Copy probes (13) | Exact copy | 0/13 | 2/13 | **2/13** |
| Supplemental (48) | Functional | 0/48 | 0/48 | **0/48** |
| Generalization (24) | Parse / binding / functional | 3/24 / 0 / 0 | 3/24 / 1 / 0 | **3/24 / 1 / 0** |
| AD-1 (29) | Parse / exact / binding | 24 / 2 / 6 | 24 / 11 / 18 | **24 / 11 / 18** |
| C3 (54) | Parse / exact / binding / functional | 24 / 0 / 0 / 0 | 24 / 1 / 7 / 0 | **24 / 1 / 7 / 0** |

Hybrid and adapter use the same resolution core here (hybrid adds logit-derived confidence + per-case auditing), so their measured scores coincide; they differ in decision provenance (audited hybrid records vs adapter gate records), not in this run's outputs. Copy decisions: adapter/hybrid substituted 2/13 probes, abstained on 11/13.

## 6. Frozen and new test results

- Focused: hybrid (7) + copy-mechanism (12) passed.
- Consolidated regression (copy/dot, diagnostics, functional, conditional, stabilization): **53 passed**. Full-suite re-tests were avoided per batch policy.
- Generalization reference answers verified 24/24 through the safe gate.

## 7. Dimension measurements

- Syntax: frozen parse held (24/29, 24/54); generalization parse 3/24 all modes.
- Structure: AD-1 12/29, C3 3/48 across modes (unchanged).
- Binding: AD-1 6→**18** (+12), C3 0→**7** (+7) — largest demonstrated gain.
- Copying: 0→**2/13** exact; abstentions 11/13 (safety honored — zero wrong substitutions).
- Generalization: functional 0/24 and 0/48 both sides; binding 0→1/24 on novel cases only.
- Functional: **0 in all suites and modes** — no executed program produced an expected output in any mode.
- Safety: ambiguous cases abstain rather than guess (11/13 probes, correct abstentions counted separately from wins).

## 8. Safety and abstention

All execution through the existing subprocess safe gate (timeouts, AST allowlist, no imports/open/eval). Ambiguous identifier mappings abstain to raw model output (adapter decisions: 2 substitute / 11 abstain). No undefined identifiers introduced by rewrite (mapping targets are validated source names; locals/builtins untouched; tests cover propagation/new-generation paths).

## 9. Hash verification

S2 `02cc86eb...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...` — all verified identical at end. No checkpoint/tokenizer/fixture modified.

## 10. Commands, exit codes, resources

Focused tests: exit 0. Regression set: exit 0 (53 passed). Integrated evaluation: exit 0 (single model load, CPU, ~1–2 min for 3 modes × 129 cases; evaluation shared one loaded model so it did not re-load per case). No downloads, no cloud/GPU provisioning.

## 11. Known limitations and unresolved blockers

1. **Functional execution is still 0/129 everywhere.** Binding/copy fixes are syntactic and relational; the model's *operations* remain wrong (e.g. subtract for add, lower for upper), so substituted names produce wrong outputs. The bottleneck has moved from "wrong names" to "wrong operations" — resolver can't fix semantics.
2. **Abstention rate high:** positional+operation resolution covers only uniquely-mapped cases; most real outputs are ambiguous to it.
3. **Hybrid == adapter in measured outcome** this run because the generator path reproduces identical deterministic text; logit confidence is collected but rarely discriminates (threshold rarely binds). True guided decoding (forcing candidate tokens mid-generation) was not implemented — generator uses model text + confidence gating, not per-step constrained decoding.
4. Supplemental/generalization suites are small (48/24); per-behaviour inference is coarse.
5. New-identifier generation (locals) is preserved but not scored separately from propagation in frozen suites.

## 12. Honest recommendation for the next milestone

Do not scale this adapter further. The demonstrated binding gains (AD-1 ×3, C3 0→7, copy 0→2) show the mechanism layer works, but end-to-end functional correctness requires **operation correctness**, which neither the frozen model nor deterministic rewriting supplies. Two honest paths:

1. **Constrained operation-aware decoding:** force candidate-token sequences at identifier positions *during* generation (true hybrid decoding), then re-measure functional execution.
2. **Retire the continued-training line** (five candidates all share 0 copy / AD-1 loss 4) and reallocate to a mechanism-first design where operation selection is also supervised/constrained.

Either way, promotion must require **functional > 0 on held-out cases**, not binding alone. **STOP. No training, Increment 15, or publication.**
