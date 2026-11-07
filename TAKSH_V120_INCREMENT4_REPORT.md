# TAKSH v1.2.0 — Increment 4 Report: Functional Generalization Improvement

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Binding curriculum digest: `bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b` (unchanged)
Functional curriculum digest: `8c3f8d10c37a1f64add826bab759a4c9688ee4a7ec9661cfc65614ee01db0b60`
Baseline checkpoint: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 1 candidate: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80` (unchanged)
Increment 3 candidate: `6ff86ddcec6678e2d1f5a42a0a9ef07f2ff6e3c771b5fb5b3c228d9a72bbdfc6` (unchanged)

Verdict: **IMPLEMENTATION COMPLETE — NO TRAINING LAUNCHED. READY FOR REVIEW BEFORE ANY INCREMENT 5 EXPERIMENT.**

This increment added a safe functional evaluation layer, a functional synthetic curriculum with explicit I/O behaviour, functional regression tests, and separate conditional reporting. No model training or inference that modifies a checkpoint was run. Frozen AD-1/C3 cases were not altered or trained on.

---

## 1. Starting branch, HEAD and package version

- Branch: `develop`; HEAD: `a654e5d`; package version `1.0.0` from `taksh/__init__.py`.
- Git status at start: the four pre-existing tracked modifications (`serving/code_chat.py`, `serving/completion_server.py`, `tests/test_v200_training_stability.py`, `tests/test_v210_supervision_learning.py`) plus all pre-existing untracked research files.
- All three checkpoints verified at start and end with the hashes above (Section 9).

---

## 2. Root-cause findings from Increment 1 and 3

Failure classification of the Increment 3 evaluation (frozen suites, new classifier, Section 6), candidate/baseline:

| Suite | Checkpoint | Binding | Syntax | Operation | Collection | Plausible-but-wrong | Pass |
|---|---|---:|---:|---:|---:|---:|---:|
| AD-1 (29) | Baseline | 18 | 5 | 3 | 0 | 3 | 0* |
| AD-1 (29) | Increment 3 | 21 | 6 | 1 | 0 | 1 | 0* |
| C3 (54) | Baseline | 24 | 30 | 0 | 0 | 0 | 0 |
| C3 (54) | Increment 3 | 25 | 19 | 4 | 1 | 4 | 1 |

\*AD-1 has no executed functional tests in the frozen setup, so "pass" is not defined there. `incorrect_parameter_binding` is the dominant failure class in both suites and both checkpoints.

Why Increment 3 improved some C3 metrics but regressed AD-1:

- The gentler recipe (`2e-5`, 100 steps) plus diversified bodies removed Increment 1's parse catastrophe: C3 parses rose 24→35 (vs 24→12 in Increment 1).
- But AD-1 binding fell 6→2 and exact 2→1. The candidate lost the baseline's arithmetic/paraphrase wins (`E2-add-002`, `E2-subtract-001`) while shifting toward comprehension/loop templates that AD-1 arithmetic prompts do not want. At low LR the model moved its distribution without fitting the diversified target (validation loss 4.47 vs 1.14 in Increment 1), so binding on AD-1 shapes degraded.
- C3 "binding" rose 0→10 without functional correctness because the binding metric credits using the requested parameter names. The comprehension template `[value for value in values if value > 0]` uses the requested sequence name regardless of the requested task, harvesting binding credit on arithmetic, algorithm, edge-case, and mapping prompts. Only 1 of those 10 binding passes also passes functionally (`HELD-string-000`), i.e. **functional accuracy conditional on binding success is 10% (1/10)**. That is the Increment 3 failure pattern this increment makes explicit: binding without behaviour.

Root cause summary: the training signal still never contains an expected output. Syntax, structure, and name-copying can all improve while the input→output relationship stays untaught. Increment 4 therefore measures behaviour directly (Section 4) and teaches it explicitly (Section 5).

---

## 3. Source files added or changed

| File | Change |
|---|---|
| `evaluation/functional_eval.py` | **New.** Safe functional evaluation layer: `evaluate_functional`, `classify_failure`, `functional_report`, `safe_execution_available`. Never executes in-process. |
| `training/functional_curriculum.py` | **New.** Functional synthetic curriculum: 8 behaviours × 12 training / × 4 validation, four implementations each, explicit I/O metadata, deterministic seeds, rotated validation split. |
| `scripts/evaluate_v120_increment1.py` | **Extended (untracked, created in Increment 1).** Records now carry `failure_category`; `aggregate()` adds `functional_analysis` (parse/exact/structural/binding/functional, functional-given-parse, functional-given-binding, category distribution) per suite, separate from diversity/collapse. Prompts, fixtures, decoding unchanged. |
| `tests/test_functional_eval.py` | **New.** 15 focused tests (Section 7). |
| `training/program_analysis.py` | **One-line fix (tracked file).** Added `"enumerate"` to `_BUILTINS` so the analyzer recognizes the `enumerate` builtin (already in the safe gate's allowlist and used by the new curriculum). No scoring semantics change: no existing example uses `enumerate`. |
| `training/binding_curriculum.py` | Validator builtin set extended with `enumerate`/`range` (untracked Increment 2 file; no body or digest change — binding digest verified unchanged above). |

`training/v120_experiment.py` received one robustness fix discovered by the test suite: `check_memory_gate` now uses a sentinel for "measure real memory", so explicitly passing `available=None` correctly means "unknown" and fails closed instead of silently measuring the host.

---

## 4. Safe execution design and limitations

- **Isolation:** all execution goes through the existing `evaluation.safe_execute.execute_tests`, which runs `python -I -c <script>` as a subprocess with empty stdin, captured output, a **2-second timeout**, and a strict AST allowlist (single function, ≤240 nodes, allowed statements/operators/builtins/methods only; no imports, no `open`/`eval`/`exec`/`__import__`, no recursion).
- **Fail closed:** validator rejection, timeout (`TimeoutExpired`), non-zero exit, and assertions are all returned as failures with a category; there is no in-process fallback anywhere in the new layer. `evaluate_functional` contains no `exec`/`eval` of generated code.
- **Availability check:** `safe_execution_available()` starts the isolated interpreter; tests assert it is available on this platform. If it were unavailable, functional results would be reported as not-executed rather than executed unsafely (the safe gate itself refuses unknown/invalid sources).
- **Limitations (documented, not hidden):**
  - The subprocess boundary isolates code from the test process but is not an OS-level sandbox: the child runs as the same user. The AST allowlist is what prevents filesystem/network/env access; defence-in-depth relies on both layers.
  - `while` loops, lambdas, and dict comprehensions are **not** in the pre-existing allowlist, so functionally correct code using them is rejected by the gate (category `rejected_by_safe_execution_gate`). The new functional curriculum deliberately uses only allowlisted constructs (enumerated `for` loops instead of `while`) so every curriculum example passes the gate. Binding-curriculum `while` variants remain valid training targets but would not pass this execution gate if generated.
  - Timeout is 2 s per function test-set: correct-but-slow code can time out and be scored as failed. This is conservative (fail-closed) and recorded as its own category.

---

## 5. Functional curriculum additions

`training/functional_curriculum.py` — 8 new behaviours, **96 training** (8×12) and **32 validation** (8×4) examples. Every example carries `inputs`, `expected_outputs`, and a `relationship` string in its metadata. Four structurally distinct implementations per behaviour:

| Behaviour | Relationship | Implementation variants |
|---|---|---|
| `scale_value(a, b)` | `a * b` | direct, intermediate, augmented, swapped operands |
| `join_text(a, b)` | `a + " " + b` | concatenation, intermediate, f-string, `" ".join([...])` |
| `add_then_multiply(a, b)` | `(a + b) * 2` | direct, intermediate, staged augassign, `2 * (...)` |
| `multiply_then_add(a, b)` | `a * b + 2` | direct, intermediate, staged augassign, `2 + a * b` |
| `filter_and_count(values)` | count `> 0` | `for`+`if` counter, `len([...])`, `sum(1 for ...)`, `enumerate` counter |
| `keep_long_words(values)` | `len(word) > 3` | comprehension, `for`+append, `extend`, `enumerate` |
| `is_even(n)` | `n % 2 == 0` | modulo compare, if/else, intermediate, ternary |
| `sum_positive(values)` | sum of `> 0` | `for` total, `sum(genexp)`, `sum(genexp) if values else 0`, `enumerate` |

Sampling includes boundary cases first (zero, empty list/string, negatives, empty results). Validation rotates names, function names, and instructions with a distinct `F1V-` prefix. All examples are original synthetic data; the builder refuses `(instruction, signature)` overlap with frozen AD-1/C3 (verified by test, Section 6).

---

## 6. Before/after curriculum statistics

Combined corpus (Increment 2 binding + Increment 4 functional) via the unchanged Increment 2 quality checker:

| Statistic | Before (Increment 2) | After (+functional) |
|---|---:|---:|
| Train / validation | 156 / 52 | 252 / 84 |
| Operations | 13 | 21 (all ×12 train) |
| Distinct template families | 15 | 20 |
| Distinct normalised shapes | 52 | 87 |
| List-comprehension share | 6/156 (3.85%) | 9/252 (3.57%) |
| Max template share | 17.3% | 15.5% |
| Exact duplicates | 0 | 0 |
| Invalid examples | 0 | 0 |
| Train/validation leakage | 0 | 0 |
| Frozen held-out overlap | 0 | 0 |

Functionally verified: **all 128 functional examples execute correctly** through the isolated safe gate against their stored inputs/outputs (subprocess, 15/15 tests in Section 7 include this full sweep).

---

## 7. Test commands and actual results

```powershell
python -m pytest tests\test_functional_eval.py tests\test_binding_curriculum.py tests\test_curriculum_quality.py tests\test_v120_increment3.py -q -p no:cacheprovider
# 51 passed in 33.88s (15 new + 36 pre-existing)
```

New-test coverage: correct code passes; wrong arithmetic/string/list results fail; wrong binding fails; boundary cases (empty, zero) pass and wrong boundaries fail; plausible-but-wrong code rejected; infinite-work timeout handled safely; `import`/`open`/`eval`/`__import__` fail closed without execution; conditional metrics separated (`functional_given_parse`, `functional_given_binding`); Increment 3 pattern regression (binding 100%, functional-given-binding 10% exposed, not masked); curriculum reproducibility/digest; train/validation separation; full functional correctness sweep; combined-corpus quality + zero held-out overlap.

Two implementation defects were caught by the suite and fixed before final verification: a `classify_failure` operator-precedence bug and the memory-gate sentinel collision described in Section 3. The full repository suite was not run, per the focused-tests instruction.

---

## 8. AD-1/C3 fixture hash verification

- AD-1: `a5bf9406f49335fe0c2d8e1bff4be1c6cdbb4a88e8bd4a866f262457420ca45e` ✓
- C3: `1b72f5388d86372582f03ae8cdf78ca031bfe34d27c0e3d9bbdc7cac4532a26e` ✓
- Frozen prompts, cases, tests, and decoding settings untouched; nothing was trained on them.

---

## 9. Confirmation all three checkpoints remain unchanged

- Baseline `02cc86eb...837f8ac` ✓, Increment 1 `89c2d61c...b80` ✓, Increment 3 `6ff86ddc...dfc6` ✓ — all re-verified after all edits.
- `git diff --name-only`: the four pre-existing modifications plus the intentional one-line `training/program_analysis.py` builtin fix (Section 3). No reset, clean, stash, revert, discard, commit, push, merge, tag, or release. Package version still `1.0.0`, HEAD still `a654e5d`.

---

## 10. Confirmation no training was launched

No training command, model load for training, hosted inference, weight download, or checkpoint write was performed in this increment. The only executions are the isolated subprocesses of the safe gate (seconds each) used by tests and the static analyses. The Increment 3 dry-run runner was not re-invoked.

---

## 11. Recommendation for a separate Increment 5 experiment

Do not train yet. Increment 5 should be a *prepared* experiment (same gated pattern as Increment 3) with these design rules, for owner review:

1. **Train on the combined corpus** (252/84) only if the functional curriculum is first extended with AD-1-shaped arithmetic/paraphrase behaviours; Increment 3 showed those shapes are where binding regressed, and the current functional set under-covers them.
2. **Promotion gate must add functional floors:** require functional-given-binding ≥ baseline and no functional-rate regression in addition to the existing parse/collapse/hash gates, so template-emission binding credit cannot promote a candidate.
3. **Evaluate with the new reporting:** use `functional_analysis` (conditional rates + category distribution) alongside diversity/collapse; a candidate whose C3 binding gains remain functionally empty must be rejected.
4. **Mind the safe-gate allowlist:** training targets using `while`/lambdas cannot be functionally verified by the current gate; either extend the allowlist with a reviewed change or keep targets within it (as Increment 4 does).
5. **Replicate across seeds** before any capability claim; single-seed results produced both the Increment 1 collapse and the Increment 3 mixed verdict.

**STOP after Increment 4. Do not start training, Increment 5, or publication without review and explicit authorization.**
