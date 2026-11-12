# TAKSH Capability Evaluation Contract

**Status:** `PARTIAL` - contract design only. Capability remains `NOT EVALUATED`.

**Verdict:** `CAPABILITY_NOT_EVALUATED`

**Machine-readable inputs:**
- `configs/roadmap/capability_evaluation_contract.yaml`
- `reports/taksh_capability_evaluation.json`
- `python -m evaluation.capability_evaluation`

---

## 1. What is ready

- Fail-closed contract schema.
- Model identity check (`taksh-foundation-111m`, 110,658,816 parameters).
- Policy requiring raw outputs and preserving historical results.

## 2. Why capability is not evaluated

| Prerequisite | Status | Detail |
|---|---|---|
| Canonical checkpoint | `BLOCKED` | `checkpoints/taksh-foundation-111m` is absent |
| Frozen benchmarks | `BLOCKED` | Both `l02-python-v1` and `l051-multilingual-v1` are `ABSENT` |
| Owner authorization | `NOT GRANTED` | `D-EVAL-5` is not granted |
| Implementation | `BLOCKED` | No evaluation has been executed |

No partial score is reported. A missing prerequisite blocks the whole evaluation.

## 3. Historical results

The historical Python result `0/30` and the unmeasured 64-task multilingual suite
are preserved and are not reused as new scores.

## 4. Fail-closed rules

- Missing checkpoint blocks evaluation.
- Missing benchmark blocks evaluation.
- Missing authorization blocks evaluation.
- No partial scores.
- No capability claim before evaluation.
- Historical results are never reused as new scores.

## 5. Regenerate the evidence

```bash
python -m evaluation.capability_evaluation
python -m pytest tests/test_capability_evaluation.py -q
```

The command exits non-zero while capability is not established.

## 6. Non-actions

This contract does not:

- load or run a model;
- run or re-score a benchmark;
- train, create a checkpoint, upload, publish, merge, tag or release anything.
