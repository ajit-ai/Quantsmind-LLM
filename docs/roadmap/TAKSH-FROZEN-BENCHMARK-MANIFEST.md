# TAKSH Frozen Benchmark Manifest

**Status:** `BLOCKED` - both frozen artifacts are `ABSENT` on this ref.

**Verdict:** `FROZEN_BENCHMARKS_ABSENT`

**Machine-readable inputs:**
- `configs/roadmap/frozen_benchmarks.yaml`
- `reports/taksh_frozen_benchmarks.json`
- `python -m evaluation.frozen_benchmarks`

---

## 1. Pinned artifacts

| ID | Path | Expected SHA-256 | Tasks | Availability |
|---|---|---|---|---|
| `l02-python-v1` | `evaluation/l02_python_bench_v1.json` | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` | 30 | `ABSENT` |
| `l051-multilingual-v1` | `evaluation/l051_multilingual_benchmark_v1.json` | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` | 64 | `ABSENT` |

Both hashes are independently pinned in:

- `.github/workflows/ci-l0704-manual-qualification.yml`
- `evaluation/taksh_cicd_recovery_check.py`

The manifest verifies that each source still contains the expected hash before
trusting it. A missing or edited hash source fails closed.

## 2. Historical results

| ID | Historical result | Measured | Preserved | Re-scored |
|---|---|---|---|---|
| `l02-python-v1` | `0/30` | yes | yes | **no** |
| `l051-multilingual-v1` | `64 tasks, not measured` | no | yes | **no** |

The Python benchmark's historical `0/30` stands. It is not improved, weakened
or recomputed here. The multilingual suite has never been measured against a
canonical TAKSH checkpoint; absence is not a score.

## 3. What `ABSENT` means

`ABSENT` means the artifact is not present on this ref, so its SHA-256 cannot
be checked here. It does **not** mean the benchmark failed, passed, or was
deleted. The artifacts are feature-branch evidence and are intentionally absent
from `develop`/`main`.

If an artifact becomes present:

| Observed state | Result |
|---|---|
| SHA-256 matches and provenance is confirmed | `PRESENT_VERIFIED`, gate `PASS` |
| SHA-256 differs | `PRESENT_HASH_MISMATCH`, gate `BLOCKED`, verdict `FROZEN_BENCHMARK_INTEGRITY_FAILURE` |
| Hash source missing or edited | gate `BLOCKED`, verdict `FROZEN_BENCHMARK_PROVENANCE_UNVERIFIED` |

Even `PRESENT_VERIFIED` proves integrity only. It does not authorize a run and
does not prove model capability.

## 4. Policy

- No download or acquisition.
- No modification of benchmark content.
- No re-scoring of historical results.
- No capability claim from manifest inspection.
- Evaluation requires separate owner authorization and a canonical checkpoint.

## 5. Regenerate the evidence

```bash
python -m evaluation.frozen_benchmarks
python -m pytest tests/test_frozen_benchmarks.py -q
```

The command exits non-zero while the gate is closed. A missing benchmark must
not look like a passing qualification.

## 6. Non-actions

This manifest does not:

- download or fetch a benchmark;
- modify, truncate or re-score a benchmark;
- evaluate any model;
- train, create a checkpoint, upload, publish, merge, tag or release anything.
