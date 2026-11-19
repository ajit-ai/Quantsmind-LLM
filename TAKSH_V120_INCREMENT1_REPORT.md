# TAKSH v1.2.0 — Coding Capability Improvement, Increment 1 Report

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Architecture: `qm-code-m3-9m`, parameters `9,359,104`, vocabulary `5,822`, context `512`
Tokenizer: `tokenizer/qm_code_bpe_v0.json`
Baseline checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt`
Baseline checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`

Candidate checkpoint: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\checkpoints\trained.pt`
Candidate checkpoint SHA-256: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80`

Verdict: **NO SUPPORTED IMPROVEMENT — CANDIDATE NOT RECOMMENDED. DO NOT PROMOTE.**

This report describes one bounded implementation and training experiment only. No publication, commit, push, merge, tag, release, package-version change, external weights, external dataset, paid service, or Hugging Face action was performed. Increment 1 stops here; no further training run was started.

---

## 1. Baseline branch and HEAD

- Branch: `develop`
- HEAD: `a654e5dc13dc77e925b3981d09792bade43a970a` (`a654e5d`)
- Package version: `1.0.0` from `taksh/__init__.py`, unchanged.
- Working tree at start and end contained only pre-existing modified and untracked research files. The tracked modified files were unchanged by this increment:
  - `serving/code_chat.py`
  - `serving/completion_server.py`
  - `tests/test_v200_training_stability.py`
  - `tests/test_v210_supervision_learning.py`
- Existing selected checkpoint and artifact copy remain hash-identical to the recorded baseline SHA-256 above. The frozen baseline, artifact checkpoint, tokenizer files, datasets, and experiment outputs were not overwritten or deleted.
- All new work in this increment is in four new files listed in Section 3, plus this report, plus the isolated experiment directory under the approved temporary root.

### Known frozen baseline used for comparison

| Suite | Exact | Structural | Binding | Functional |
|---|---:|---:|---:|---:|
| AD-1 raw, frozen report | 2/29 | 12/29 | 6/29 | not scored |
| C3 held-out, frozen report | 0/54 | — | 0/54 | 0/45 |

The fresh fair re-evaluation in Section 6 reproduced the baseline counts exactly for AD-1 and C3 exact/binding/functional, confirming the evaluation harness and decoding settings match the frozen reference.

---

## 2. Root-cause findings

The diagnosis used only the existing S2 architecture, forward pass, tokenizer path, synthetic operation curriculum, training loop, and AD-1/C3 failure records. No architecture rewrite was performed because no concrete architecture defect was established.

### 2.1 What the baseline actually does

Baseline raw generation is deterministic, usually parseable, and often structurally plausible, but it frequently binds the wrong names:

- AD-1 baseline: 24/29 outputs parse, yet only 6/29 bind correctly.
- C3 baseline: 24/54 outputs parse, yet 0/54 bind correctly and 0/45 executable cases pass.
- Typical failure: a correct operation shape with invented names such as `left` / `right` when the requested signature uses different names, or the wrong string method (`lower` instead of `upper`).

This points to a supervision and data-diversity problem, not to the transformer shape: the model has learned operation templates but does not reliably copy the requested parameter names from the prompt into the body.

### 2.2 Why the existing curriculum is weak for binding

- The v2.1 operation curriculum has only 22 training examples (`T` partition) across 13 operations. Each operation is represented by a very small set of name and surface-form combinations.
- One template dominates each operation. The model can memorize the template while ignoring the prompt's parameter names.
- Several operations share almost identical bodies, so the model receives little pressure to condition on the instruction or signature.
- The training and evaluation prompts are consistent (instruction + signature), so the weakness is not explained by a tokenization mismatch. Tokenizer round-trip for a representative new example was verified in Section 6.4.

### 2.3 Increment 1 hypothesis and what actually happened

Increment 1 tested whether a larger, deterministic, binding-focused synthetic curriculum — same architecture, same tokenizer, same training implementation, continued from the frozen S2 checkpoint — would improve parameter-name copying and return-variable binding without touching held-out data.

The experiment falsified the optimistic version of that hypothesis. Training loss fell and validation loss fell, but held-out parse rate collapsed. The candidate learned a narrow surface template too strongly:

- It produced one genuine new exact match on `HELD-string-000` and one on `E2-upper-001`.
- Across most other held-out prompts it emitted degenerate list-comprehension text such as `return [value for item in in values if item > 0]`, regardless of the requested operation.
- The comprehension and counting templates together account for 4 of the 13 operations (48 of 156 examples) and share the tokens `for item in`. At `lr=1e-4` for 240 steps, that shared surface became an attractor for unrelated prompts.

Conclusion: the bottleneck is real, but this curriculum shape and these training settings caused **mode collapse**, not reliable binding. The candidate must not be promoted, and the result must not be described as an improvement.

---

## 3. Exact source files changed

No existing tracked file was modified by this increment. Four new files were created:

| File | Purpose |
|---|---|
| `training/binding_curriculum.py` | Deterministic, original, binding-focused synthetic curriculum builder. 156 training examples and 52 validation examples; AST-validated; fixed seeds; curriculum digest. |
| `tests/test_binding_curriculum.py` | Focused unit tests for validity, determinism, parameter-name binding, and train/validation disjointness. |
| `scripts/run_v120_increment1.py` | Dry-run default training runner. Real run requires `--train`, explicit isolated root, baseline hash check, RAM gate, AD-1 overlap check, and writes through the existing immutable `RunArtifacts` path. |
| `scripts/evaluate_v120_increment1.py` | Fair baseline-vs-candidate held-out comparison on frozen AD-1 and C3 with identical prompts and decoding. Raw model output only (`binding_repair=False`). Arbitrary generated code is executed only through the existing restricted `safe_execute` gate for `execution_safe` C3 cases. |

Plus:

| File | Purpose |
|---|---|
| `TAKSH_V120_INCREMENT1_REPORT.md` | This report. |

One implementation defect found during testing was fixed before training: the initial curriculum validator treated list-comprehension loop variables as undefined names. The validator was corrected to count store targets and comprehension targets as assigned. After the fix, `3 passed` (Section 6.4).

---

## 4. Curriculum changes and example categories

`training/binding_curriculum.py` builds original examples only. No external code or dataset was copied. Every example passes an AST check that it parses as one function and uses only its own parameters and locally assigned names.

Training set: **156 examples** = 13 operations × 12 examples.
Validation set: **52 examples** = 13 operations × 4 examples, with a different name offset and distinct `(instruction, signature)` pairs from training.
Curriculum digest: `6a531633ba62f76806bef8722c49bdde33140d3a2a584177b8b0e4f2cde77c9e`.

| Operation | Family | Training | Body pattern taught |
|---|---|---:|---|
| `add` | arithmetic | 12 | `return a + b` |
| `subtract` | arithmetic | 12 | `return a - b` |
| `multiply` | arithmetic | 12 | `return a * b` |
| `maximum` | conditioning | 12 | `if a > b: return a` else `return b` |
| `minimum` | conditioning | 12 | `if a < b: return a` else `return b` |
| `keep_positive` | collection | 12 | `[item for item in values if item > 0]` |
| `keep_negative` | collection | 12 | `[item for item in values if item < 0]` |
| `upper` | text | 12 | `return text.upper()` |
| `lower` | text | 12 | `return text.lower()` |
| `count_values` | aggregate | 12 | `count` loop over items |
| `sum_values` | aggregate | 12 | `total` loop over items |
| `choose_first` | binding | 12 | `return <first param>` |
| `choose_second` | binding | 12 | `return <second param>` |

Design choices:

- **Exact signatures first:** each operation cycles six function names and 20 parameter-name pairs (or 20 single names for unary operations), so the correct body must copy the names from the signature.
- **Return binding:** `choose_first` / `choose_second` directly teach "return the named parameter, ignore the other."
- **Adversarial pressure:** `choose_first` and `choose_second` use near-identical instructions and signatures differing only in the requested position; `upper` / `lower` and `keep_positive` / `keep_negative` are paired opposites.
- **Held-out separation:** the runner refuses to train if any training `(instruction, signature)` pair overlaps the frozen AD-1 suite. C3 cases were never used as training input. No check found overlap.
- **Reproducibility:** default seed `20261024`, validation seed offset by one, fixed name cycling, and a recorded curriculum digest. The digest is stable across rebuilds.

Known curriculum weakness, now evidenced by the result: surface diversity per operation is still one template. In particular, the four comprehension/loop operations share vocabulary and dominate the candidate's errors.

---

## 5. Training experiment details

### 5.1 Command

```powershell
python scripts/run_v120_increment1.py --train --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1 --run-id increment1-binding-240 --steps 240 --lr 0.0001 --seed 20261024
```

A dry run with the same arguments but without `--train` was executed first and printed the plan (156 train / 52 validation examples, curriculum digest above) without writing a checkpoint.

The runner enforces before training:

- steps between 1 and 300 (this run used exactly 240, inside the increment limit);
- baseline checkpoint exists and its SHA-256 equals `02cc86eb...`;
- tokenizer file exists;
- available RAM at least 0.5 GiB (the default gate). Two early attempts were refused by this gate at 0.13 GiB and 0.38 GiB available; the successful run proceeded at 0.503 GiB available;
- no AD-1 `(instruction, signature)` overlap with the training set.

### 5.2 Configuration

Recorded in `integrity/config.json` of the run:

- Initialization: continued from the frozen S2 checkpoint (existing `train_from_checkpoint` path; no pretrained weights, no new architecture).
- Tokenizer: `tokenizer/qm_code_bpe_v0.json`, vocab 5,822.
- Max length: 256. Gradient accumulation: 2. Pages: CPU only, 4 threads.
- Learning rate: `1e-4`, warmup 2, cosine schedule, gradient clip 1.0, weight decay 0.1 (existing continued-training defaults).
- Steps requested and completed: 240 / 240.
- Seed: `20261024`.
- Tokens consumed: 6,575. Supervised tokens per example: 27.4 average.

### 5.3 Outcomes

From `metrics/training.json` and `manifest.json`:

- Elapsed training time: **84.07 seconds**.
- Initial loss (step 1): **3.2122**. Final loss (step 240): **1.4715**.
- Validation loss: **1.1446**.
- Parameters: 9,359,104 (unchanged). All 240 history entries have `finite_loss: true`.
- Experiment root is new and exclusive: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240`. The run was finalized through the existing immutable artifacts path; the checkpoint was written once.
- Baseline and artifact checkpoint hashes after training still equal `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`.

Lower training and validation loss did **not** translate into held-out capability. Section 6 is the deciding evidence.

---

## 6. Baseline versus candidate metrics

### 6.1 Evaluation protocol

Script: `scripts/evaluate_v120_increment1.py`
Output: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\evaluations\heldout_comparison.json`

- Suites: frozen AD-1 (`ad1_suite.json`, 29 cases) and C3 held-out (`evaluation/python_benchmark_heldout_v1.json`, 54 cases). At least 20 held-out tasks is satisfied by both suites.
- Identical prompts for both checkpoints: `instruction + "\n" + signature + "\n"`.
- Identical decoding: greedy (`temperature=0.0, top_k=1, top_p=1.0`), `max_new_tokens=64`, seed `20261023`, device CPU.
- Raw model output only: `binding_repair=False`, so the checkpoint is measured rather than the post-processing repair.
- C3 functional execution uses the existing restricted `safe_execute` gate and only for `execution_safe`, non-negative cases. AD-1 is scored statically (parse / exact AST / structural / binding).
- Models were loaded sequentially and released between evaluations.

### 6.2 AD-1 results (29 cases)

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Parses | 24/29 | 15/29 | −9 |
| Exact | 2/29 | 3/29 | +1 |
| Structural | 12/29 | 13/29 | +1 |
| Binding | 6/29 | 5/29 | −1 |

Candidate exact wins on AD-1: `E2-upper-001`, `E2-add-002`, `E2-subtract-001`. Candidate binding wins include `E2-keep_positive-001` and `E4-filter_positive_count-0`, but 14 candidate outputs do not parse, including degenerate comprehensions on operations that are not comprehensions.

### 6.3 C3 held-out results (54 cases)

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Parses | 24/54 | 12/54 | −12 |
| Exact | 0/54 | 1/54 | +1 |
| Structural | 3/48 scored | 2/48 scored | −1 |
| Binding | 0/54 | 2/54 | +2 |
| Functional passed | 0/45 scored | 1/45 scored | +1 |

The single candidate C3 functional pass is `HELD-string-000` (`return text.upper()`). The two candidate binding wins are `HELD-string-000` and `HELD-negative-004` (the latter is a negative control and was not executed). 42 candidate outputs do not parse, most of them the same degenerate comprehension pattern.

Baseline reproduction check: the fresh baseline numbers above match the frozen report exactly (AD-1 2/12/6, C3 0 exact, 0 binding, 0 functional), so the comparison is fair and the harness is faithful.

### 6.4 Verification checks

- Focused unit tests: `python -m pytest tests/test_binding_curriculum.py -q -p no:cacheprovider` → **3 passed**.
- Compilation: `python -m py_compile training/binding_curriculum.py scripts/run_v120_increment1.py scripts/evaluate_v120_increment1.py` → OK.
- Curriculum validity: all 156 training and 52 validation examples parse as one function with only parameter/local names (covered by the unit tests and the builder's own AST validator).
- Tokenizer compatibility: `Tokenizer.from_file(tokenizer/qm_code_bpe_v0.json)` reports vocab 5,822. A representative example, `def adds_pair(alpha_value, beta_value):\n    return alpha_value + beta_value`, encodes to 26 ids and round-trips exactly.
- Held-out comparison: completed as described in Section 6.1; full per-case records are in the JSON output above.
- Full repository test suite: not run, per the increment instruction to avoid unnecessary full-suite runs.

### 6.5 Interpretation

The candidate is **not** an improvement:

- Parse rate fell by 9/29 on AD-1 and 12/54 on C3.
- AD-1 binding fell by 1; C3 binding rose by only 2, from a zero base.
- Exact rose by 1 on each suite, driven by string-case examples.
- Functional rose by 1/45 on C3.

A single new exact match in each suite does not support a capability claim when half the C3 outputs no longer parse. The correct conclusion is that this curriculum and training setting induced mode collapse toward the comprehension/loop template.

---

## 7. Generated Python examples and functional results

All examples are raw generations under the identical prompt/decoding protocol of Section 6.1. Functional results are the restricted-gate outcomes for C3.

### 7.1 `HELD-string-000` — "Write a function that returns the uppercase form of a string." / `def upper_phrase(text):` (expected `upper_phrase("text") == "TEXT"`)

Baseline:

```python
def upper_phrase(text):
    return left + right
```

- Parses: yes. Exact: no. Binding: no. Functional: **fail** (`NameError: name 'left' is not defined`).

Candidate:

```python
def upper_phrase(text):
    return text.upper()
```

- Parses: yes. Exact: yes. Binding: yes. Functional: **pass**.

This is the one clear candidate win, and it is also the exact kind of binding the increment targeted: copy the requested parameter name `text` and apply the requested operation.

### 7.2 `HELD-arith-000` — "Write a function that returns three times a value." / `def triple_value(n):` (expected `triple_value(5) == 15`)

Baseline:

```python
def triple_value(n):
    if left < right:
        return left - right
    return left
```

- Parses: yes. Exact: no. Binding: no. Functional: **fail** (`NameError: name 'left' is not defined`).

Candidate:

```python
def triple_value(n):
    return [value for item in in values if item > 0]
```

- Parses: no. Exact: no. Binding: no. Functional: **fail** (syntax error). The candidate applied the comprehension template to an arithmetic prompt.

### 7.3 `HELD-binding-000` — "Write a function that joins two strings with a space." / `def join_words(first, second):` (expected `join_words("blue", "sky") == "blue sky"`)

Baseline:

```python
def join_words(first, second):
    if left < right:
        return left + right
    return left + right
```

- Parses: yes. Exact: no. Binding: no. Functional: **fail** (`NameError: name 'left' is not defined`).

Candidate:

```python
def join_words(first, second):
    return [value for item in in in in in values if item < 0]
    return 0]
```

- Parses: no. Exact: no. Binding: no. Functional: **fail** (syntax error, unmatched `]`).

### 7.4 AD-1 examples

`E1-add-000` ("Write a function that returns the sum of two integers." / `def compute_total(x, y):`): both checkpoints emit `return left + right`. Neither binds `x` / `y`. No change.

`E2-add-002` ("Produce a function that result of adding the two inputs." / `def add_values(left, right):`): both checkpoints emit `return left + right`, exact and binding correct. No change.

`E2-upper-001` ("Return the input converted to capital letters." / `def upper_text(text):`):

- Baseline: `return text.lower()` — parses, binding name correct, wrong operation.
- Candidate: `return text.upper()` — exact, binding correct.

The pattern is consistent: the candidate learned the string-case operation correctly but generalized the comprehension template far beyond its intended operations.

---

## 8. Checkpoint path and SHA-256

- Baseline (unchanged): `C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt`
  - SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- Artifact copy (unchanged): `artifacts/taksh-s2-experimental/taksh-s2.pt`
  - SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- Candidate: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\checkpoints\trained.pt`
  - SHA-256: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80`
  - Size: 37,463,173 bytes. Parameters: 9,359,104. Tensors: 75. Architecture: unchanged `qm-code-m3-9m`.
- Run manifest: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\manifest.json`
- Run metrics: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\metrics\training.json`
- Held-out comparison: `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\evaluations\heldout_comparison.json`

The candidate checkpoint is preserved for inspection only. It was not copied into `artifacts/`, not registered as a release, and must not be promoted.

---

## 9. Resource usage and limitations

- Training: CPU only, 4 threads, 240 steps, 84.07 seconds, 6,575 tokens consumed. No GPU stack was installed and no cloud hardware was provisioned.
- Memory: the host has 3.911 GiB total RAM. The training runner's 0.5 GiB available-RAM gate refused two early starts (0.13 GiB and 0.38 GiB available) and allowed the run at 0.503 GiB available. Evaluation later ran at about 0.45 GiB available by loading checkpoints sequentially and releasing each model before loading the next.
- Evaluation latency (mean per generation): AD-1 baseline 0.3455 s, candidate 0.3816 s; C3 baseline 0.4774 s, candidate 0.6477 s.
- Limitations:
  - The curriculum has one surface template per operation, so "more examples" did not mean "more diversity."
  - Training continued from a tiny checkpoint with a high learning rate (1e-4) over only 156 examples, so template memorization dominated.
  - Evaluation covers AD-1 and C3 only; it does not establish general Python capability.
  - C3 functional scoring can only run `execution_safe` cases (45/54); negative and unsafe cases are scored statically or not executed.
  - The candidate was evaluated once, deterministically. No repeated-seed variance study was performed, and none is needed to reject promotion given the parse-rate collapse.

---

## 10. Recommendation for Increment 2

**Do not carry the candidate checkpoint forward.** Keep the frozen S2 checkpoint as the only baseline.

Increment 2 should be a data-diversity and training-strength fix, not another run of the same recipe:

1. **Diversify surface forms per operation before retraining.** Give each operation at least 3–4 syntactically distinct correct bodies (for example, arithmetic via direct return, via a named intermediate, and via a conditional guard; filtering via comprehension, via explicit loop, and via `list(filter(...))`). The goal is to break the single-template attractor, especially the `for item in` comprehension.
2. **Balance token surface across operations.** Reduce the share of examples containing the tokens `for item in`, or add a held-out "template confusion" probe that fails the run if unrelated prompts emit comprehension text.
3. **Lower the training strength and add a parse-rate early stop.** Try `lr=3e-5` with fewer steps (for example 120) and stop if validation parse rate or held-out parse rate drops below the baseline. A lower training loss is not a success criterion by itself.
4. **Keep the binding probes.** Retain `choose_first` / `choose_second` and the paired opposites (`upper`/`lower`, `keep_positive`/`keep_negative`); they produced the only genuine wins. Add string-join and simple scaling operations, since `HELD-binding-000` and `HELD-arith-000` expose gaps the current curriculum does not cover.
5. **Gate promotion on parse rate first.** Require candidate parse rate at least equal to baseline on both suites before considering exact/binding gains. The Increment 1 candidate would have failed this gate immediately.
6. **Pre-register the comparison.** Reuse `scripts/evaluate_v120_increment1.py` (or its successor) with the same prompts, decoding, and suites so Increment 2 is directly comparable to the numbers in Section 6.

No further training run or implementation increment should begin without review of this report.

---

## Stop

This increment ends here. No commit, push, merge, tag, release, package-version change, publication, license declaration, or additional training run was performed.
