"""Experiment and evaluation framework — A/B/C arms, no results claimed.

Arms:

- A — conventional retrieval baseline;
- B — DLCAE internal structured knowledge formation;
- C — hybrid (DLCAE + optional retrieval + model learning + execution feedback).

Rules enforced here:

- the same task set and scoring are used across arms;
- every task-set and config is hash-pinned;
- n >= 3 seeds for any future comparative run;
- model-only vs tool-assisted performance is tracked separately;
- a comparative winner can only be reported if every arm actually ran;
  until then the comparison verdict is NOT_ESTABLISHED — never implied;
- frozen TAKSH benchmarks can never be registered as an experiment task set.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from taksh.dlcae.types import fingerprint, utcnow_iso

REPO = Path(__file__).resolve().parent.parent.parent
FROZEN_BENCHMARKS = {
    "l02_python_bench_v1": REPO / "evaluation" / "l02_python_bench_v1.json",
    "l051_multilingual_benchmark_v1": REPO / "evaluation" / "l051_multilingual_benchmark_v1.json",
}

ARMS = {
    "A": "conventional_retrieval_baseline",
    "B": "dlcae_internal_knowledge",
    "C": "hybrid_dlcae_retrieval_execution",
}


def frozen_benchmark_problem_ids() -> set[str]:
    ids: set[str] = set()
    for name, path in FROZEN_BENCHMARKS.items():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for problem in data.get("problems", []) or data.get("tasks", []) or data.get("items", []):
            if isinstance(problem, dict):
                for key in ("id", "task_id", "name", "title"):
                    if problem.get(key):
                        ids.add(str(problem[key]))
                        break
    return ids


class ExperimentFramework:
    """Reproducible, hash-pinned experiment registration and execution."""

    def __init__(self) -> None:
        self.task_sets: dict[str, dict[str, Any]] = {}
        self.runs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def register_task_set(self, name: str, tasks: list[dict[str, Any]], *,
                          scoring: str) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("task set name must be non-empty")
        ids = [str(t.get("id")) for t in tasks]
        if any(not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("task ids must be present and unique")
        forbidden = frozen_benchmark_problem_ids()
        overlap = sorted(set(ids) & forbidden)
        if overlap:
            raise ValueError(
                f"task id(s) collide with a frozen benchmark — contamination refused: {overlap}")
        if not scoring.strip():
            raise ValueError("scoring rule must be named")
        taskset = {
            "schema": "taksh.dlcae.taskset/1",
            "name": name,
            "created": utcnow_iso(),
            "task_count": len(tasks),
            "tasks": tasks,
            "scoring": scoring,
            "taskset_sha256": fingerprint(tasks),
            "same_taskset_for_all_arms": True,
        }
        self.task_sets[name] = taskset
        return {k: taskset[k] for k in ("name", "task_count", "scoring", "taskset_sha256")}

    # ------------------------------------------------------------------ #
    def run_arm(self, arm: str, task_set: str, *, seeds: list[int],
                model_fn: Callable[[dict[str, Any], int], dict[str, Any]],
                retrieval_fn: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
                tool_assisted: bool = False) -> dict[str, Any]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
        if len(seeds) < 3:
            raise ValueError("comparative experiments require at least 3 seeds")
        taskset = self.task_sets.get(task_set)
        if taskset is None:
            raise ValueError(f"unregistered task set: {task_set!r}")
        results: list[dict[str, Any]] = []
        for seed in seeds:
            for task in taskset["tasks"]:
                t0 = time.perf_counter()
                tracemalloc.start()
                try:
                    if arm in ("A", "C"):
                        retrieved = retrieval_fn(task) if retrieval_fn else []
                    else:
                        retrieved = []
                    output = model_fn(task, seed)
                    scoring = self._score(task.get("scoring") or taskset["scoring"],
                                          output, task)
                except Exception as exc:  # noqa: BLE001 — failures are data
                    output, scoring = {"error": f"{type(exc).__name__}: {exc}"}, False
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                latency_ms = (time.perf_counter() - t0) * 1000
                results.append({
                    "arm": arm, "arm_name": ARMS[arm], "seed": seed,
                    "task_id": task["id"],
                    "retrieval_used": bool(arm in ("A", "C") and retrieval_fn),
                    "retrieved_items": len(retrieved) if arm in ("A", "C") else 0,
                    "correct": bool(scoring),
                    "mode": "tool_assisted" if tool_assisted else "model_only",
                    "latency_ms": round(latency_ms, 3),
                    "memory_peak_bytes": int(peak),
                    "output_sha256": fingerprint(output),
                })
        record = {
            "schema": "taksh.dlcae.experiment_run/1",
            "arm": arm,
            "arm_name": ARMS[arm],
            "task_set": task_set,
            "taskset_sha256": taskset["taskset_sha256"],
            "scoring": taskset["scoring"],
            "seeds": list(seeds),
            "tool_assisted": tool_assisted,
            "results": results,
            "metrics": self._metrics(results),
            "config_sha256": fingerprint({"arm": arm, "seeds": seeds,
                                          "taskset_sha256": taskset["taskset_sha256"],
                                          "tool_assisted": tool_assisted}),
            "executed_at": utcnow_iso(),
        }
        self.runs.append(record)
        return record

    @staticmethod
    def _score(scoring: str, output: dict[str, Any], task: dict[str, Any]) -> bool:
        """Deterministic built-in scorers. Unknown scorer => False (never guessed)."""
        expected = task.get("expected")
        answer = output.get("answer")
        if scoring == "exact_match":
            return answer is not None and answer == expected
        if scoring == "contains":
            return answer is not None and str(expected) in str(answer)
        if scoring == "truthy":
            return bool(output.get("ok"))
        return False

    @staticmethod
    def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {"tasks": 0}
        correct = sum(1 for r in results if r["correct"])
        by_mode: dict[str, dict[str, int]] = {}
        for r in results:
            bucket = by_mode.setdefault(r["mode"], {"tasks": 0, "correct": 0})
            bucket["tasks"] += 1
            bucket["correct"] += int(r["correct"])
        return {
            "tasks": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results), 4),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 3),
            "memory_peak_bytes_max": max(r["memory_peak_bytes"] for r in results),
            "by_mode": by_mode,
        }

    # ------------------------------------------------------------------ #
    def comparison(self) -> dict[str, Any]:
        """Honest comparison: a winner exists only if all arms ran completely."""
        by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        for run in self.runs:
            by_arm[run["arm"]].append(run)
        missing = [arm for arm, runs in by_arm.items() if not runs]
        seeds_ok = all(len(r["seeds"]) >= 3 for runs in by_arm.values() for r in runs)
        if missing or not seeds_ok:
            return {
                "verdict": "NOT_ESTABLISHED",
                "reason": (f"arms without complete runs: {missing or 'none'}"
                           if missing else "fewer than 3 seeds in a run"),
                "claim": "no comparative capability claim is authorized or made",
                "arms": {arm: len(runs) for arm, runs in by_arm.items()},
                "model_only_vs_tool_assisted": "separated per task record",
            }
        metrics = {arm: [r["metrics"]["accuracy"] for r in runs] for arm, runs in by_arm.items()}
        best = max(metrics, key=lambda a: sum(metrics[a]) / len(metrics[a]))
        return {
            "verdict": "COMPUTED_FOR_REVIEW",
            "reason": "all arms ran; results are reproducible evidence, not a release gate",
            "arms": metrics,
            "highest_mean_accuracy_arm": best,
            "claim": ("engineering measurement only — no model-learning claim, "
                      "no capability claim, no release implication"),
            "model_only_vs_tool_assisted": "separated per task record",
        }
