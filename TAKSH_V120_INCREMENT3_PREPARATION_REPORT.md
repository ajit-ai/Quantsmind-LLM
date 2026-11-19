# TAKSH v1.2.0 — Increment 3 Preparation Report: Controlled Training Experiment Preparation

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`

Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a`, package version `1.0.0`
Baseline checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (unchanged)
Increment 1 candidate SHA-256: `89c2d61c8a34198cf1ef827e9482c7994eea0fcabc215d56a5745c70d18c1b80` (unchanged)
Increment 2 curriculum digest: `bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b`
Experiment identity: `47faeeaa94c0354280f29a305d6dec5f3c230ff95db6b3b0d0bfd1e65bde245b`

Verdict: **PREPARATION COMPLETE — NO TRAINING LAUNCHED. AWAITING EXPLICIT OWNER AUTHORIZATION.**

This increment prepared an isolated, reproducible, resource-bounded experiment configuration with strict pre-training gates and automated promotion safeguards. No training command was invoked, no training loop was initialized, and no checkpoint was written or modified.

---

## 1. Verified starting state

Verified before any work:

- Branch: `develop`; HEAD: `a654e5d`; package version `1.0.0` from `taksh/__init__.py`.
- Git status: only the four pre-existing tracked modifications (`serving/code_chat.py`, `serving/completion_server.py`, `tests/test_v200_training_stability.py`, `tests/test_v210_supervision_learning.py`), unchanged at the end (`git diff --name-only` identical).
- Baseline checkpoint `02cc86eb...`, Increment 1 candidate `89c2d61c...`, artifact copy `02cc86eb...` — all re-verified after all edits with the same values.
- Frozen fixtures: AD-1 `a5bf9406f49335fe0c2d8e1bff4be1c6cdbb4a88e8bd4a866f262457420ca45e` (29 cases), C3 `1b72f5388d86372582f03ae8cdf78ca031bfe34d27c0e3d9bbdc7cac4532a26e` (54 cases).
- Increment 2 curriculum digest `bc533a18...`.

Reviewed before editing: `training/binding_curriculum.py`, `training/curriculum_quality.py`, `scripts/run_v120_increment1.py`, `scripts/evaluate_v120_increment1.py`, the Increment 1 and Increment 2 reports, `training/continued_training.py` (`train_from_checkpoint`), and `training/experiment_artifacts.py` (`RunArtifacts`, `file_sha256`). The Increment 3 runner reuses the existing `train_from_checkpoint` and `RunArtifacts` implementations; no training pipeline or architecture was duplicated or rewritten.

---

## 2. Files changed

No existing tracked file was modified. All Increment 3 work is in five new files:

| File | Purpose |
|---|---|
| `configs/v120_increment3.json` | Isolated experiment configuration: architecture, hashes, curriculum digest, steps/lr/seed/device, output root/run-id, memory minimum, frozen fixture hashes and case counts, promotion rules. |
| `training/v120_experiment.py` | Configuration validation, experiment identity, nine pre-training gates (`run_preflight`), and the pure promotion decision (`decide_promotion`). Preparation only — never initializes training. |
| `scripts/run_v120_increment3.py` | Runner. Dry run by default (validates, runs gates, prints plan, writes nothing). Training path requires both `--train` and `--authorize` and reuses the existing pipeline. |
| `scripts/promote_v120_candidate.py` | Reads an `evaluate_v120_increment1.py` JSON, verifies checkpoint and fixture hashes, prints the promote/reject decision. Writes nothing to any checkpoint directory. |
| `tests/test_v120_increment3.py` | 23 focused tests (Section 5). |

---

## 3. Experiment configuration

From `configs/v120_increment3.json` (experiment identity `47faeeaa...`):

| Setting | Value |
|---|---|
| Architecture | `qm-code-m3-9m`, 9,359,104 parameters, vocab 5,822, context 512 |
| Initialization | Continued from frozen S2 baseline `02cc86eb...` |
| Curriculum | Increment 2 diversified curriculum, digest `bc533a18...`, 156 train / 52 validation |
| Maximum steps | **100** (validation rejects >100) |
| Learning rate | **2e-05** (validation rejects ≤0 or >1e-4) |
| Seed | **20261106** |
| Device | `cpu` (validation rejects anything else) |
| Max length / grad accum / warmup / weight decay / grad clip | 256 / 2 / 2 / 0.1 / 1.0 |
| Minimum available RAM | 0.5 GiB |
| Experiment root | `C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3` (outside the repository; validation rejects a root inside the source tree) |
| Run ID | `increment3-diverse-100` (new; exclusive via existing `RunArtifacts`) |
| Tokenizer | `tokenizer/qm_code_bpe_v0.json` |

These are conservative starting values, not proven optimal. The step and learning-rate bounds are enforced by `validate_config`, so the exact approved settings cannot be silently widened.

---

## 4. Gate implementation and test evidence

`run_preflight(config, authorized, for_training)` implements all nine required gates; with `for_training=True` any failure raises `GateError` and training cannot start. A dry run returns the same report without writing anything.

| # | Gate | Implementation | Evidence |
|---|---|---|---|
| 1 | Baseline checkpoint hash | `check_checkpoint_gate`: file exists and SHA-256 equals the recorded value | Real pass; tampered hash raises |
| 2 | Curriculum digest | `check_curriculum_gates`: recomputed digest equals the approved Increment 2 digest | Real pass; zero digest raises |
| 3 | Curriculum quality | Increment 2 `check_curriculum` (balance, diversity, dominance, binding) | 156/52 counts, all checks pass |
| 4 | Train/validation separation | Same check: `(instruction, signature)` and `(signature, body)` overlap must be 0; AD-1 overlap must be 0 | Leakage test fires on planted overlap |
| 5 | Frozen fixtures | AD-1/C3 hashes and case counts (29/54) match config | Real pass; tampered C3 hash raises |
| 6 | Configuration and seed recorded | `validate_config` + `experiment_identity` (canonical hash of config) | Identity stable across loads; unsafe values rejected (7 parametrized cases) |
| 7 | Memory minimum | `check_memory_gate(min, available)`; refuses on low or unknown memory for training | 0.6 passes vs 0.5; 0.4 and unknown raise |
| 8 | New/empty output | Run directory must not exist; root must be outside repository | Existing directory raises |
| 9 | Explicit authorization | `--train` without `--authorize` prints `REFUSED` and exits 2; `run_preflight(for_training=True, authorized=False)` raises | Subprocess test: exit 2, no directory created |

Promotion (`decide_promotion`, used by `scripts/promote_v120_candidate.py`) reports parse rate, exact-match, structural validity, binding, functional accuracy, template diversity (via the Increment 2 `aggregate` keys), and mode collapse, and rejects when:

- parse rate in AD-1 or C3 falls below baseline,
- functional rate in C3 falls below baseline,
- mode collapse is detected in either suite,
- required cases are missing (counts must be exactly 29 AD-1 / 54 C3 for both checkpoints),
- checkpoint or fixture hashes fail verification.

Training loss is never consulted. End-to-end evidence: running the promoter on the real Increment 1 evaluation JSON returns `promote: false` with a parse-rate-fell reason, as expected.

---

## 5. Test commands and actual results

```powershell
python -m pytest tests\test_v120_increment3.py -q -p no:cacheprovider
# 23 passed in 68.94s

python -m pytest tests\test_binding_curriculum.py tests\test_curriculum_quality.py -q -p no:cacheprovider
# 13 passed in 9.80s (no regression from Increments 1-2)
```

Coverage: configuration validation (7 unsafe mutations), reproducible identity, checkpoint/digest/fixture gates (pass and tamper), memory thresholds, output-directory exclusivity, authorization enforcement, dry-run non-mutation (subprocess: exit 0, no run directory, baseline hash unchanged), `--train`-without-`--authorize` refusal (exit 2), all five promotion rejection conditions plus acceptance, and the real Increment 1 candidate rejection. No test invokes training; no test passes both `--train` and `--authorize`.

---

## 6. Dry-run results

`python scripts/run_v120_increment3.py` (no flags) printed mode `dry_run` with preflight gates:

- configuration: **pass** (identity `47faeeaa...`)
- baseline checkpoint hash: **pass** (`02cc86eb...`)
- curriculum: **pass** (digest `bc533a18...`, 156 train)
- frozen fixtures: **pass** (AD-1 29, C3 54)
- memory: **fail, as expected on this host** — available `0.16 GiB` below the required `0.50 GiB`. This only blocks an actual training start; the dry run itself completed and wrote nothing (`Test-Path` for the run directory returned `False`).
- output directory: **pass** (new run directory, not created)
- authorization: not authorized (dry run only)

The memory result is a genuine operational risk (Section 8), not a configuration error: the gate correctly reports that the host cannot presently meet the conservative minimum.

---

## 7. Confirmation that no training was launched

- The only run of the Increment 3 runner was the dry run above. The `--train` flag was used solely in one test that omitted `--authorize` and was refused with exit 2 before any training code was reached.
- No `increment3` run directory exists; no new checkpoint file was created anywhere.
- Baseline, candidate, and artifact checkpoint hashes are byte-identical to their starting values (Section 1).
- No commit, push, merge, tag, release, package-version change, external weights, external dataset, or paid service occurred.

---

## 8. Exact command that would launch the experiment after approval

Only after explicit owner authorization, and only once the host meets the memory minimum:

```powershell
python scripts/run_v120_increment3.py --train --authorize --root C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3 --run-id increment3-diverse-100
```

Immediately afterwards, the candidate must be evaluated and gated with:

```powershell
python scripts/evaluate_v120_increment1.py --candidate <candidate-checkpoint> --baseline C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt --out <evaluation-json>
python scripts/promote_v120_candidate.py --evaluation <evaluation-json>
```

The candidate may be promoted only if the promoter prints `promote: true`. Note that the current configuration binds `--root` and `--run-id` to the approved values; overrides are supported for testing but change the experiment identity recorded in the run.

---

## 9. Remaining risks and recommended next action

1. **Host memory (immediate blocker):** 0.16 GiB available vs the 0.5 GiB minimum. The gate will refuse training until memory is freed or the minimum is consciously revised by the owner. Do not lower the minimum without recording the reason.
2. **Conservative settings are unproven:** 100 steps at `2e-5` may undertrain (Increment 1 at `1e-4` / 240 collapsed; too little training could leave binding unchanged). The promotion gate, not training loss, is the safeguard — a null result (no promotion) is an acceptable, informative outcome.
3. **Coverage gaps remain:** string-join and simple scaling are still absent from the curriculum; adding them changes operation counts and should be a separate reviewed change.
4. **Single-host, single-seed:** one seed (`20261106`) cannot estimate variance. A follow-up should vary the seed before any capability claim.
5. **Template watch item:** `assign_name_return` is 17.3% of the Increment 2 curriculum, close to the 20% quality cap. If Increment 3 drifts toward intermediates, split that variant.

**Recommended next action:** owner reviews this report and the frozen configuration, frees host memory (or revises the minimum explicitly), and only then authorizes the exact command in Section 8. **STOP — do not launch training. Waiting for explicit owner authorization.**
