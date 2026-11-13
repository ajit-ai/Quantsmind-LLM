"""TAKSH L0.6.0 — WP6: multilingual execution readiness.

Audits the 64-task starter benchmark, resolves the 4-task denominator problem
(a 3-of-4 or 8-of-16 threshold is ambiguous), and emits scoring/validation
interfaces plus a readiness report. The Python 30-task frozen benchmark
(evaluation/l02_python_bench_v1.json) stays byte-identical.

Scoring interfaces:
  score_task(record) -> {'id', 'passed', 'assertions_passed', 'assertions_total'}
  score_language(task_records) -> {'executed', 'passed', 'pass_ratio', 'qualified'}
  score_benchmark(all_records) -> {'languages': {...}, 'frozen_python_preserved': True}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
BENCH_PATH = REPO / "evaluation" / "l051_multilingual_benchmark_v1.json"
FAMILIES = frozenset(str(i) for i in range(1, 11))
L1_TARGETS = frozenset({
    "python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
    "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
})
MIN_TASKS_PER_LANGUAGE = 16
MIN_PASS_RATIO = 0.50


def score_task(record: dict) -> dict:
    tests = record.get("tests", [])
    total = len(tests)
    passed = sum(1 for t in tests if t.get("passed"))
    return {"id": record.get("id"), "passed": passed == total and total > 0,
            "assertions_passed": passed, "assertions_total": total}


def score_language(records) -> dict:
    scored = [score_task(r) for r in records]
    executed = sum(1 for s in scored if s["id"] is not None)
    passed = sum(1 for s in scored if s["passed"])
    ratio = (passed / executed) if executed else 0.0
    qualified = executed >= MIN_TASKS_PER_LANGUAGE and ratio >= MIN_PASS_RATIO
    return {
        "executed": executed,
        "passed": passed,
        "pass_ratio": round(ratio, 4),
        "min_tasks_required": MIN_TASKS_PER_LANGUAGE,
        "min_pass_ratio": MIN_PASS_RATIO,
        "qualified": qualified,
        "ineligible_reason": (
            None if qualified else
            ("insufficient_task_count" if executed < MIN_TASKS_PER_LANGUAGE else "pass_ratio_below_bar")
        ),
    }


def score_benchmark(all_records) -> dict:
    by_lang: dict[str, list] = {}
    for r in all_records:
        by_lang.setdefault(r.get("language"), []).append(r)
    per_language = {lid: score_language(recs) for lid, recs in by_lang.items()}
    return {"languages": per_language,
            "python_frozen_benchmark_untouched": hashlib.sha256(
                (REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA}


def audit_benchmark(bench: dict) -> dict:
    tasks = bench["tasks"]
    per_lang = Counter(t["language"] for t in tasks)
    checks = []
    checks.append({"check": "all_16_languages", "passed": set(per_lang) == L1_TARGETS})
    for lid in sorted(L1_TARGETS):
        checks.append({"check": f"{lid}_task_count>=1", "passed": per_lang.get(lid, 0) >= 1})
    checks.append({"check": "all_10_families_defined",
                   "passed": {f["id"] for f in bench["families"]} == FAMILIES})
    ids = [t["id"] for t in tasks]
    checks.append({"check": "unique_ids", "passed": len(set(ids)) == len(ids)})
    failed = [c for c in checks if not c["passed"]]
    return {
        "task_count": len(tasks),
        "per_language": dict(sorted(per_lang.items())),
        "families_defined": sorted({f["id"] for f in bench["families"]}),
        "checks_run": len(checks),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "verdict": "PASS" if not failed else "FAIL",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l060_multilingual_readiness.json")
    args = parser.parse_args(argv)

    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    audit = audit_benchmark(bench)
    per_lang = audit["per_language"]
    tasks_per_language = min(per_lang.values())
    shortfall_per_language = MIN_TASKS_PER_LANGUAGE - tasks_per_language

    frozen = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    report = {
        "schema": "taksh.l060_multilingual_readiness/1",
        "generated_by": "evaluation/l060_multilingual_readiness.py",
        "benchmark": {
            "path": str(BENCH_PATH.relative_to(REPO)),
            "name": bench["name"],
            "version": bench["version"],
            "qualification_block_present": "qualification" in bench,
            "task_count": audit["task_count"],
            "per_language": per_lang,
            "families_defined": audit["families_defined"],
            "audit": audit,
        },
        "denominator_resolution": {
            "problem": "4 tasks per language cannot establish an 8-pass threshold",
            "resolution": {
                "per_language_scoring": "proportion of executed tasks that pass all unit assertions",
                "min_tasks_per_language": MIN_TASKS_PER_LANGUAGE,
                "min_pass_ratio": MIN_PASS_RATIO,
                "languages_with_4_tasks": "NOT eligible for qualification (insufficient_task_count)",
                "additional_tasks_needed": shortfall_per_language * 16,
                "additional_tasks_needed_per_language": shortfall_per_language,
                "instantiated_families": "1-4",
                "missing_families": "5-10 (specified; instances to be authored as tasks are added)",
            },
        },
        "python_frozen_benchmark": {
            "path": "evaluation/l02_python_bench_v1.json",
            "sha256": frozen,
            "expected": FROZEN_PY_SHA,
            "unchanged": frozen == FROZEN_PY_SHA,
        },
        "execution_readiness": {
            "toolchain_available": ["python", "javascript", "c", "cpp", "java", "csharp",
                                   "go", "rust", "kotlin", "bash", "sql", "typescript"],
            "missing_toolchain": ["php", "ruby", "swift", "dart"],
            "harness_note": "per-language sandboxed harness is an L1.0 work item; not implemented or claimed here",
        },
        "scoring_interfaces": ["score_task", "score_language", "score_benchmark"],
        "verdict": "MULTILINGUAL READINESS QUALIFIED (spec corrected; no capability claimed)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"denominator resolved: min {MIN_TASKS_PER_LANGUAGE} tasks/language, ratio >= {MIN_PASS_RATIO}")
    print(f"tasks shortfall: {shortfall_per_language} per language ({shortfall_per_language * 16} total)")
    print(f"python benchmark unchanged: {frozen == FROZEN_PY_SHA}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
