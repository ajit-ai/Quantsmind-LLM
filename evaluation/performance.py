"""Focused performance benchmarks for TAKSH engineering operations."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taksh.engineering import inspect_project, validate_project, run_tests, apply_plan, ChangePlan, FileChange
from taksh.memory import ProjectMemory


def _measure(fn: Callable[[], Any], iterations: int = 5) -> dict[str, Any]:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return {
        "iterations": iterations,
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
    }


def run_benchmarks(root: str | Path, iterations: int = 5) -> dict[str, Any]:
    root_path = Path(root).resolve()
    results = {
        "project_indexing": _measure(lambda: inspect_project(root_path), iterations),
        "validation": _measure(lambda: validate_project(root_path), iterations),
    }

    # Memory retrieval benchmark.
    memory_path = root_path / ".taksh" / "memory.json"
    memory = ProjectMemory(memory_path)
    memory.store("bench-project", "fact", "value", source="benchmark", verified=True)
    results["memory_retrieval"] = _measure(lambda: memory.inspect("bench-project"), iterations)

    # Patch application benchmark.
    def apply_patch():
        changes = [FileChange(path="bench.txt", operation="create", content="x = 1\n", reason="bench")]
        plan = ChangePlan(task_id="bench", root=str(root_path), changes=changes, approved=True)
        apply_plan(root_path, plan, approved=True)
        # cleanup
        path = root_path / "bench.txt"
        if path.exists():
            path.unlink()

    results["patch_application"] = _measure(apply_patch, iterations)

    return {
        "schema": "taksh.performance_benchmark/1",
        "root": str(root_path),
        "iterations": iterations,
        "results": results,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "note": "Baseline measurements only; no performance claims are made.",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--out", default="reports/taksh_performance.json")
    args = parser.parse_args()

    report = run_benchmarks(args.root, args.iterations)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
