# TAKSH v1.2.0 — Increment 13 Copy Mechanism Report: Feasibility and Copy-Gate Prototype

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 10: `e68495931da7bf4d0dde666615f16757a87a01304c0a2358ac4fa05bda634c9b` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)

Verdict: **COPY MECHANISM FEASIBLE AS A DETERMINISTIC ADAPTER (NOT A MODEL-TRAINING SUCCESS).**

The mechanism correctly extracts, selects, substitutes, propagates, and abstains — and rescues real frozen bindings (AD-1 binding 6→18, C3 binding 0→7) without touching weights. But it abstains on most frozen probes, produces **zero functional passes** anywhere, and leaves the underlying model copying at 0/13. **No training was performed. NOT AUTHORIZED for training.**

**STOP after implementation, feasibility tests, and reporting. Awaiting owner review.**

---

## 1. Baseline Git status and HEAD

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: four pre-existing modifications + Inc4 `training/program_analysis.py` fix + Inc10 integration (two files). All hashes re-verified after all work (Section 9). No invariant failed. No commits/pushes/tags/releases/version changes.

## 2. Files added or changed

| File | Purpose |
|---|---|
| `evaluation/copy_mechanism.py` | WP1–WP3: AST source extraction, deterministic candidate selection, copy/generate gate with audit + fallback |
| `scripts/adapter_v120.py` | WP5 inference adapter evaluator (enabled/disabled, frozen + supplemental) |
| `tests/test_copy_mechanism.py` | 12 synthetic feasibility tests (10 required scenarios + disabled gate + adversarial) |
| `TAKSH_V120_INCREMENT13_COPY_MECHANISM_REPORT.md` | This report |

No existing file modified. No checkpoint/tokenizer/fixture touched.

## 3. Source-identifier extraction design (WP1)

`extract_source_identifiers(prompt, tokenizer)`: parses the signature with `ast`, takes **argument names in order** as the only candidates (keywords/builtins/literals excluded by construction), tokenizes each with round-trip verification (`decode(encode(name)) == name`), records token ids/counts/origin/validity. Familiar and unfamiliar names both extract (token counts 1–7). Function names excluded; instruction words never become candidates.

## 4. Copy-candidate selection design (WP2)

`select_substitutions`: body loads are classified (used params vs invented names vs locals/builtins via `analyse_body`). Unused source params match invented names in order of first appearance; relation hints (`first`/`leading`/`second`/`trailing`) override position. Counts mismatch/ambiguity → `action="abstain"` with reason + confidence 0.0 — never a guess. Repeated occurrences supported via whole-word rewrite of mapped names (longest-first).

## 5. Copy-gate decision logic (WP3)

`CopyGate(enabled, threshold=0.5)`: actions are `copy` (no invented names, no locals), `propagate` (locals present), `substitute` (unique positional mapping), `generate` (no source / gate disabled), `abstain` (ambiguous/unparseable/low-confidence). Each decision is recorded in `gate.audit` with action/mapping/reason/confidence/enabled/threshold — fully observable. Confidence is deterministic (1.0 unique, 0.0 ambiguous); uncertain decisions fall back to the raw model output, never to silent substitution. New-identifier generation remains possible (action `generate` passes the completion through unchanged).

## 6. Confidence and fallback behaviour

`enabled=False` passes every completion through untouched (`reason="gate disabled"`). Below-threshold confidence abstains. The model prediction head is never modified; syntax generation stays on the existing path (rewriting only touches mapped identifier tokens).

## 7. Synthetic test results (10 required + 2 extra)

**12 passed** (`tests/test_copy_mechanism.py`): familiar copy, unfamiliar copy (`probe_00`), multi-token identifiers (`left_value` etc. — boundaries preserved via whole-word rewrite), two-variable selection (order mapping), substitution hint (`first` → first param), propagation action (unchanged), repeated occurrences (all rewritten), new-identifier generation (abstain, unchanged), keyword/builtin/literal exclusion, ambiguity abstention (one invented vs two unused, no hint), gate-disabled passthrough, adversarial already-correct body (action `copy`, unchanged). Plus diagnostics/functional/conditional/stabilization suites: **46 passed total** across the focused selection.

## 8. Inference integration evidence (WP5)

`scripts/adapter_v120.py` (baseline S2, greedy, same prompts/decoding), adapter **enabled vs disabled**:

| Suite | Metric | Disabled (model only) | Enabled (adapter) |
|---|---|---:|---:|
| Copy probes (13) | Exact copy | 0/13 | **2/13** (both familiar subtract) |
| Copy decisions | — | all `generate` | 2 `substitute`, 11 `abstain` |
| Supplemental (48) | Functional | 0/48 | **0/48** (binding partials: scale 1, sum_positive 2) |
| AD-1 (29) | Parse / exact / binding | 24 / 2 / 6 | 24 / **11** / **18** |
| C3 (54) | Parse / binding / functional | 24 / 0 / 0 | 24 / **7** / **0** |

The adapter only substitutes when the mapping is uniquely determined; most model outputs are ambiguous (`left - right` style with 2 unused params and no hint), so it correctly abstains — the mechanism is safe but conservative. Copy decisions demonstrably connect to source identifiers and never modify weights. This is **mechanism-only evidence, not a model-training success**: functional execution is 0 everywhere.

Frozen gates status: copy probes 2/13 (<10), supplemental 0/48 (<3), AD-1 binding improved (6→18, so no regression), C3 functional unchanged at 0. **Mechanism-only success must not be reported as training success.**

## 9. Limitations and failure cases

1. **Abstention-dominated:** 11/13 probes abstain; the positional policy cannot resolve prompts where the model's invented-name count ≠ unused-param count and no hint exists. Richer relation inference (operation-aware) would lift coverage but risks wrong substitution.
2. **Rewrite is cosmetic for semantics:** substituting identifiers into a wrong-operation body (`left - right` → `left_value - right_value`) fixes binding, not correctness; functional stays 0.
3. **Syntax faults untouched:** unparseable bodies are abstained on; the adapter doesn't repair syntax.
4. **No learned gating:** confidence is a deterministic heuristic, not a calibrated probability; the gate distinguishes actions but cannot judge *which* operation the body implements.
5. **Source scope:** candidates come only from the signature; identifiers named in instructions but absent from the signature are not copyable.

## 10. Hashes at the end

Baseline `02cc86eb...`, Increment 10 `e6849593...`, tokenizer `90fd1c8b...`, AD-1 `a5bf9406...`, C3 `1b72f538...`, supplemental `05c0c657...` — all verified. HEAD `a654e5d`.

## 11. Training authorization status and recommendation

**Training authorization: NOT AUTHORIZED.** No training, optimizer step, checkpoint write, architecture/tokenizer change, or fixture modification occurred in this increment.

Recommended next decision: a copy-aware **decoding/generation redesign** (constrained decoding that forces identifier positions to copy source-token sequences, with the adapter's abstention semantics as the fallback contract), evaluated against the frozen gates — rather than another continued-training round, which four candidates have already shown does not teach copying. Any training-first alternative should be required to clear the ≥10/13 copy gate *before* further spend.

**STOP after implementation, feasibility tests, and reporting. Awaiting owner review.**
