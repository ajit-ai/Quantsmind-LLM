# TAKSH v1.3.1 — Planner Integration & Hybrid Reliability Consolidation

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Original S2: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Tokenizer: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (unchanged)
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (46 cases: 40 + 6)
Machine-readable: `Temp\opencode\v120\integrated\qualification_results.json` (+ comparison/acceptance JSONs)

**Implementation status: PARTIALLY COMPLETE.** Planner validation integrated and qualified as an audit layer with no execution regressions; independent-generalization target missed (18/40, 45% vs 80% bar). **STOP after this campaign. No training, promotion, or publication.**

---

## 1. Repository baseline (start and end identical)

`develop` @ `a654e5d`, package `1.0.0`. Tracked diff: 4 pre-existing + Inc4 fix + Inc10 integration; all untracked research preserved. No reset/clean/stash/discard/commit/push/merge/tag/release/version change.

## 2. Historical discrepancy reconciliation (WP2)

| Report | Independent figure | What it actually measured |
|---|---:|---|
| v1.2.4 report ("18/40") | 18/40 (45%, 5/6 abstain) | **Hybrid (v1.2.4-corrected synthesizer)** — from `qualify_v123.py` hybrid mode |
| v1.3.0 table ("6/40") | 8/40→18/40 planner head-to-head | **Structured planner prototype** (Architecture B, different code path, narrower vocabulary) |

Both numbers are accurate for the systems measured; they are not the same system. The v1.3.0 table juxtaposed planner-prototype results against hybrid results without that label. Nothing was filtered or removed in either suite. The v1.2.4 figure is the hybrid baseline for this campaign's targets.

## 3. Changes

| File | Purpose |
|---|---|
| `evaluation/hybrid.py` | **Modified.** Planner validation step integrated into `HybridGenerator.generate`: every hybrid generation records `planner` status (`VALIDATED` / `INVALID_PLAN` / `UNSUPPORTED_PLAN` / `VALIDATION_UNAVAILABLE`) with plan/reason; synthesis path unchanged and labeled `synthesized`, model raw text retained in records |
| `evaluation/operation_synth.py` | **Modified.** Coverage expansions from v1.2.4 residual failures (see v1.2.4 report for base vocabulary) |
| `evaluation/diagnostics.py` | **Modified.** (v1.2.2) Scorer attribute bug fix retained |
| `evaluation/acceptance_suite_v123.py` | **New.** 46-case independent suite (40 functional + 6 abstain), references pre-verified, digest frozen |
| `scripts/qualify_v123.py` | **New.** Four-mode qualification evaluator (model/adapter/hybrid) + op-selection probes + constrained POC |
| `tests/test_planner_integration.py` | 3 focused tests (planner outcomes, provenance records, gate still passes) |

No existing tracked file was modified beyond the two Integration touch-points (`evaluation/hybrid.py` lives untracked; planner integration does not change its public generation output).

## 4. Planner integration architecture & outcomes

Pipeline per generation: prompt → model draft → candidates (`CopyGate`) → planner (`plan()`: op family → steps with dependencies) → `validate()` (dependency resolution, literal-aware, output-defined) → `compile_plan()` (deterministic body) → provenance record → safe execution. Rules: only `VALIDATED`-eligible candidates pass to rewriting/execution; `INVALID_PLAN` blocks execution of proven-invalid plans but **never silently degrades the user output** (falls back to the raw hybrid candidate with status recorded); `UNSUPPORTED_PLAN` and `VALIDATION_UNAVAILABLE` are distinct from invalid and are not treated as validation-passed. Outcomes observed across suites: VALIDATED for synthesized/parsed repairs; UNSUPPORTED for the v1.2.4 residual phrasings (propagation depth, conjunction order, boolean edge cases) — all correctly recorded. No block on any existing hybrid success.

## 5. Four-mode comparison (final numbers)

| Suite | Model | Adapter | v1.2.0 hybrid | **v1.2.4/v1.3.1 hybrid** |
|---|---:|---:|---:|---:|
| Copy exact (13) | 0/13 | 2/13 | 8/13 → scorer-fixed | **10/13** |
| Supplemental functional (48) | 0/48 | 0/48 | 33/48 | **33/48 (68.8%)** |
| Generalization24 functional | 0/24 | 0/24 | 21/24 | **21/24 (87.5%)** |
| Independent (46: 40+6) | 0/40, abstain 0/6 | 0/40, 6/6 | 6/40 (via v1.2.2), — | **18/40 (45%), abstain 5/6** |
| AD-1 parse/binding | 24/6 | 24/18 | 26/23 | **26/23** |
| C3 parse/binding/functional | 24/0/0 | 24/7/0 | 39/26-28/6 | **39/28/8** |

New-suite composition (hybrid): passes cluster in arithmetic/string/filter; failures cluster in novel phrasing templates (e.g. "compute the reciprocal", "second value of the list") where synthesis vocabulary runs out and the model contributes 0.

## 6. Targets assessment

| Target | Result | Status |
|---|---:|---|
| Copy ≥10/13 | 10/13 | **MET** |
| Supplemental ≥33/48 | 33/48 | **MET** |
| Generalization24 ≥21/24 | 21/24 | **MET** |
| Independent ≥32/40 (80%) | **18/40 (45%)** | **MISSED** |
| C3 functional retained/improved (≥8) | 8 | **MET** |
| AD-1 binding retained (≥23) | 23 | **MET** |
| Abstention | 5/6 | **MET** (1 conjunction guessed) |
| Planner integrated + audited | yes, no execution regressions | **MET** |

## 7. Constrained decoding

Bounded POC (`evaluation/constrained.py`): bracket-balance EOS suppression + anti-loop suppression; unit-tested deterministic; disabled==unconstrained. **0 interventions over 13 probes** — efficacy unproven on this set; syntax-error reduction is unproven, semantics untouched. Scope stays a POC, not a grammar guarantee.

## 8. Verification

Focused: planner integration tests + copy/hybrid/reliability/diagnostics/conditional/stabilization slices **45 passed**. Consolidation regression slices **37+52 passed** across runs. Hashes verified at end; HEAD unchanged. All reference answers executed pre-tuning through the safe gate.

## 9. Limitations

1. Rule vocabulary remains the ceiling — unseen phrasings abstain or fall to the (0-functional) model.
2. Planner is validation/audit, not a generation backend; its typed coverage trails the synthesizer's rule surface (honest tradeoff, favors transparency).
3. Independent suite is small (46); per-behaviour uncertainty is wide.
4. No human semantic review; all functional judgements are mechanical.

## 10. Final decisions

- **Hybrid deployment: CONTROLLED PILOT ONLY** (reliable inside covered vocabulary; generalization bar missed at 45%)
- **Neural model: NOT QUALIFIED** (model-only functional: 0 in all suites; operation-selection top-tokens never correct)
- **Continued training: NOT JUSTIFIED** (five candidates share the 0-copy signature; objective labels/gradients proven correct — the head is template-bound)
- **Publication readiness: NOT READY**
- **Architecture: BOUNDED HYBRID WITH PLANNER** (primary: existing hybrid backend; planner as validation/audit layer)

**STOP after this campaign. Awaiting owner review.**
