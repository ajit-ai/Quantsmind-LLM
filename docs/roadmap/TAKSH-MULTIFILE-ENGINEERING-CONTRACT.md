# TAKSH Multi-File Engineering Contract

**Status:** `PARTIAL` - contract design only. Multi-file editing is not implemented.

**Verdict:** `MULTIFILE_ENGINEERING_NOT_IMPLEMENTED`

**Machine-readable inputs:**
- `configs/roadmap/multifile_engineering_contract.yaml`
- `reports/taksh_multifile_engineering.json`
- `python -m evaluation.multifile_engineering`

---

## 1. What is ready

- Fail-closed contract schema.
- Required workflow stages: discover, plan, patch, review, apply, validate, rollback.
- Approval, sandbox, rollback and capability prerequisites recorded.

## 2. Why multi-file engineering is not implemented

| Gate | Status | Detail |
|---|---|---|
| Implementation | `BLOCKED` | No multi-file apply workflow exists |
| Sandbox approval | `BLOCKED` | No sandbox approval recorded |
| Capability prerequisite | `BLOCKED` | Capability is not evaluated |
| Owner authorization | `BLOCKED` | Multi-file engineering is not authorized |

No repository mutation, patch application or test execution is performed by this
contract.

## 3. Fail-closed rules

- No unapproved patch application.
- No partial apply.
- No arbitrary code execution without a sandbox.
- No capability claim without evaluation.
- Automatic rollback on validation failure.
- Original tree preserved.

## 4. Regenerate the evidence

```bash
python -m evaluation.multifile_engineering
python -m pytest tests/test_multifile_engineering.py -q
```

The command exits non-zero while multi-file engineering is not implemented.

## 5. Non-actions

This contract does not:

- modify a repository;
- apply or execute a patch;
- run tests or commands;
- train, create a checkpoint, upload, publish, merge, tag or release anything.
