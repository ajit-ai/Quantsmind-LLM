"""Bounded coding-task orchestrator (ENG3).

Pipeline per task:

    accept → generate (gated) → comprehension → symbols/types/dependencies
    → sandboxed execution of tests → structured diagnostics → correlate
    → repair candidates → (optional model repair) → revalidate → verdict

Hard rules:

- never completes a task without the validation criteria being met;
- stops immediately when the model is unavailable/blocked;
- stops when validation is BLOCKED;
- repair suggestions never mutate source; a replacement candidate is generated
  only via a fresh model request, in-memory, never written over user files;
- maximum repair attempts enforced; cancellation and interruption are
  first-class; ledgers are persistable for resume;
- model doubles are labelled; nothing here confers capability.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from taksh.dlcae.coding.model import ModelSession, ModelCapabilityState, StaticModelDouble
from taksh.dlcae.coding.task import CodingTask, render_task_prompt
from taksh.dlcae.persistence import PersistenceStore
from taksh.dlcae.runtime import DLCAERuntime
from taksh.dlcae.semantic.engine import SemanticAnalysisEngine
from taksh.dlcae.semantic.feedback import correlate, repair_candidates
from taksh.dlcae.semantic.tools import PythonToolchain
from taksh.dlcae.types import fingerprint, utcnow_iso

class TaskState:
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    REPAIRING = "REPAIRING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"

    TERMINAL = frozenset({COMPLETED, FAILED, BLOCKED, INTERRUPTED})


class OrchestratorError(Exception):
    pass


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def extract_code(text: str) -> str:
    """Strip markdown fences if present; otherwise return text unchanged."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _runner_script(task: CodingTask, entry_point: str) -> str:
    lines = [
        "import sys",
        "sys.path.insert(0, '.')",
        "import json",
        "import candidate",
        "results = []",
        f"entry = getattr(candidate, {entry_point!r})",
    ]
    for test in task.tests:
        ep = test.entry_point or entry_point
        args = tuple(test.args)
        if test.check == "call_equals":
            lines += [
                "try:",
                f"    got = entry(*{args!r})",
                f"    ok = (got == {test.expected!r})",
                "except Exception:",
                "    ok = False",
                "    got = None",
                f"results.append({{'name': {test.name!r}, 'ok': bool(ok), 'got': repr(got)}})",
            ]
        elif test.check == "raises":
            lines += [
                "try:",
                f"    got = entry(*{args!r})",
                "except Exception as exc:",
                f"    ok = type(exc).__name__ == {test.expected!r}",
                "else:",
                "    ok = False",
                f"results.append({{'name': {test.name!r}, 'ok': bool(ok)}})",
            ]
        elif test.check == "contains":
            lines += [
                "try:",
                f"    got = entry(*{args!r})",
                f"    ok = {test.expected!r} in str(got)",
                "except Exception:",
                "    ok = False",
                f"results.append({{'name': {test.name!r}, 'ok': bool(ok)}})",
            ]
    lines += ["print(json.dumps(results))"]
    return "\n".join(lines) + "\n"


class CodingTaskOrchestrator:
    def __init__(self, model: ModelSession | StaticModelDouble, *,
                 runtime: DLCAERuntime | None = None,
                 max_repair_attempts: int = 2,
                 execution_timeout_s: float = 10.0,
                 state_dir: str | None = None,
                 tool_assisted: bool = True) -> None:
        if not 0 <= max_repair_attempts <= 5:
            raise OrchestratorError("max_repair_attempts must be in [0, 5]")
        self.model = model
        if state_dir:
            self.state_dir = state_dir
        else:
            # no repository writes by default; smoke runs stay ephemeral
            import tempfile

            self.state_dir = tempfile.mkdtemp(prefix="l100eng3_runtime_")
        self.runtime = runtime or DLCAERuntime(self.state_dir)
        self.max_repair_attempts = max_repair_attempts
        self.execution_timeout_s = execution_timeout_s
        self.tool_assisted = tool_assisted
        self.state_store = PersistenceStore(state_dir) if state_dir else None
        self.python = PythonToolchain()

    # ------------------------------------------------------------------ #
    def run(self, task: CodingTask, *, context: str = "",
            cancel_event=None) -> dict[str, Any]:
        started = time.perf_counter()
        ledger: dict[str, Any] = {
            "schema": "taksh.l100eng3_coding_task/1",
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "language": task.language,
            "mode": "tool_assisted" if self.tool_assisted else "model_only",
            "opened_at": utcnow_iso(),
            "state": TaskState.PENDING,
            "model_availability": self.model.assess(),
            "attempts": [],
            "budget": {"max_repair_attempts": self.max_repair_attempts},
            "evidence": {"model_generated": [], "tool_observed": [],
                         "tool_verified": [], "invalidated": []},
        }
        self._persist(ledger)
        ledger["state"] = TaskState.GENERATING

        completed_tests = None
        for attempt_n in range(1, self.max_repair_attempts + 2):
            if cancel_event is not None and cancel_event.is_set():
                ledger["state"] = TaskState.INTERRUPTED
                ledger["interruption"] = {"reason": "cancelled by caller",
                                          "at": utcnow_iso()}
                return self._finalize(ledger, started)
            prompt = render_task_prompt(task, context=context)
            if attempt_n > 1:
                prompt = self._repair_prompt(task, attempt_n, ledger)
            attempt = self._attempt(task, attempt_n, prompt, cancel_event)
            ledger["attempts"].append(attempt)
            self._persist(ledger)
            if attempt["state"] == ModelCapabilityState.GENERATION_BLOCKED:
                ledger["state"] = TaskState.BLOCKED
                ledger["blocking_reason"] = attempt.get("reason", "model unavailable or blocked")
                return self._finalize(ledger, started)
            if attempt["state"] == ModelCapabilityState.GENERATION_FAILED:
                ledger["state"] = TaskState.FAILED
                ledger["blocking_reason"] = attempt.get("error", "generation failed")
                return self._finalize(ledger, started)
            if attempt.get("validation", {}).get("state") == "BLOCKED":
                ledger["state"] = TaskState.BLOCKED
                ledger["blocking_reason"] = attempt["validation"].get(
                    "reason", "validation BLOCKED — stopping immediately")
                return self._finalize(ledger, started)
            if attempt["validation"].get("execution_verified", False):
                ledger["state"] = TaskState.COMPLETED
                return self._finalize(ledger, started)
            if attempt_n > self.max_repair_attempts:
                ledger["state"] = TaskState.FAILED
                ledger["blocking_reason"] = ("max_repair_attempts exhausted without "
                                             "passing validation")
                return self._finalize(ledger, started)
            ledger["state"] = TaskState.REPAIRING

        ledger["state"] = TaskState.FAILED
        return self._finalize(ledger, started)

    # ------------------------------------------------------------------ #
    def _attempt(self, task: CodingTask, attempt_n: int, prompt: str,
                 cancel_event) -> dict[str, Any]:
        generation = self.model.generate(prompt, seed=1100 + attempt_n,
                                         max_new_tokens=128, task_id=task.task_id,
                                         mode="tool_assisted" if self.tool_assisted
                                         else "model_only")
        source = extract_code(generation.get("text", "")) if generation.get(
            "state") == ModelCapabilityState.GENERATION_COMPLETED else ""
        attempt: dict[str, Any] = {
            "attempt_n": attempt_n,
            "created": utcnow_iso(),
            "model_kind": generation.get("model_kind"),
            "model_id": generation.get("model_id"),
            "model_identity": generation.get("model_identity"),
            "generation": generation.get("generation"),
            "trace_fingerprint": generation.get("trace_fingerprint"),
            "state": generation.get("state"),
            "source_sha256": _source_sha256(source) if source else None,
            "source_present": bool(source),
            "source_lines": source.count("\n") + 1 if source else 0,
            "evidence_category": "MODEL_GENERATED",
        }
        if generation.get("state") != ModelCapabilityState.GENERATION_COMPLETED or not source:
            attempt["validation"] = {"state": TaskState.BLOCKED,
                                     "reason": "no generation output to validate"}
            attempt["repair_candidates"] = {"candidate_count": 0, "candidates": [],
                                            "origin": "suggestion-not-verified-fix",
                                            "source_mutation": False}
            return attempt

        attempt["validation"] = self._validate_and_execute(task, source, cancel_event)
        attempt["diagnostics"] = attempt["validation"].get("diagnostics", [])
        attempt["repair_candidates"] = repair_candidates(
            attempt["diagnostics"], source_sha256=attempt["source_sha256"])
        return attempt

    # ------------------------------------------------------------------ #
    def _validate_and_execute(self, task: CodingTask, source: str,
                              cancel_event) -> dict[str, Any]:
        if task.language.lower() != "python":
            return {"state": "BLOCKED",
                    "reason": f"no execution oracle qualified for {task.language!r}; "
                              f"Karkain/tooling stays BLOCKED",
                    "execution_verified": False}
        # DLCAE comprehension + ENG2 semantic pipeline (additive path)
        runtime_result = self.runtime.semantic_analysis(
            f"coding-{task.task_id}-{fingerprint(source)[:12]}",
            task.language, source, file=f"candidate_{task.task_id.replace('-', '_')}.py",
            mode="tool_assisted" if self.tool_assisted else "model_only")
        diagnostics: list[dict[str, Any]] = list(runtime_result.get("diagnostics", []))
        execution = self._execute_tests(task, source, cancel_event)
        diagnostics.extend(execution.pop("diagnostics", []))
        feedback = self._feedback_records(diagnostics, source, execution)
        feedback_correlation = correlate(feedback) if feedback else {"verdict": "NO_EVIDENCE"}
        execution_verified = execution["all_passed"] and runtime_result["status"] in (
            "PASS", "PARTIAL")
        return {
            "state": "COMPLETED" if execution["all_passed"] else "FAILED",
            "semantic": {"status": runtime_result["status"],
                         "contract_problems": runtime_result.get("contract_problems", []),
                         "stage_status": runtime_result.get("stage_status")},
            "execution": execution,
            "diagnostics": diagnostics,
            "feedback_records": len(feedback),
            "correlation": feedback_correlation,
            "execution_verified": bool(execution_verified),
        }

    # ------------------------------------------------------------------ #
    def _execute_tests(self, task: CodingTask, source: str,
                       cancel_event) -> dict[str, Any]:
        from taksh.dlcae.semantic.tools import Sandbox

        with Sandbox(prefix="l100eng3_candidate_") as sandbox:
            path = sandbox.write("candidate.py", source)
            runner = sandbox.write("_runner.py", _runner_script(task, task.entry_point))
            if cancel_event is not None and cancel_event.is_set():
                return {"state": "INTERRUPTED", "all_passed": False,
                        "tests": [], "duration_ms": 0.0, "diagnostics": []}
            t0 = time.perf_counter()
            # execution goes through the sandbox runner — never in-process exec
            import subprocess
            import sys as _sys
            try:
                proc = subprocess.run([_sys.executable, "-I", str(runner)],
                                      capture_output=True, text=True,
                                      timeout=self.execution_timeout_s, cwd=sandbox.root)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc, timed_out = None, True
            duration_ms = round((time.perf_counter() - t0) * 1000, 3)
            diagnostics: list[dict[str, Any]] = []
            tests: list[dict[str, Any]] = []
            stdout_tail = (proc.stdout if proc else "")[-4096:]
            stderr_tail = (proc.stderr if proc else "")[-4096:]
            if timed_out:
                diagnostics.append({"category": "timeout",
                                    "message": f"test execution exceeded "
                                               f"{self.execution_timeout_s}s",
                                    "severity": "ERROR", "evidence_classification": "TOOL_VERIFIED"})
            else:
                try:
                    results = json.loads(stdout_tail.strip().splitlines()[-1])
                    tests = results
                    for test in results:
                        if not test.get("ok"):
                            diagnostics.append({"category": "test_failure",
                                                "message": test.get("name", "unnamed test"),
                                                "severity": "ERROR",
                                                "evidence_classification": "TOOL_VERIFIED"})
                except (ValueError, IndexError):
                    diagnostics.append({"category": "runtime_exception",
                                        "message": ("test runner crashed: "
                                                    + (stderr_tail or stdout_tail).strip()[:400]),
                                        "severity": "ERROR",
                                        "evidence_classification": "TOOL_VERIFIED"})
            all_passed = bool(tests) and all(t.get("ok") for t in tests) and not timed_out
            return {"state": "COMPLETED" if all_passed else "FAILED",
                    "all_passed": all_passed,
                    "tests": tests,
                    "tests_total": len(task.tests),
                    "tests_passed": sum(1 for t in tests if t.get("ok")),
                    "duration_ms": duration_ms,
                    "timed_out": timed_out,
                    "exit_code": (proc.returncode if proc else None),
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                    "diagnostics": diagnostics,
                    "workspace": "isolated-tmp-workspace (auto-cleaned)"}

    # ------------------------------------------------------------------ #
    def _feedback_records(self, diagnostics, source: str, execution):
        from taksh.dlcae.semantic.feedback import FeedbackRecord

        sha = _source_sha256(source)
        records = []
        for diagnostic in diagnostics:
            records.append(FeedbackRecord.build(
                source_sha256=sha, language_id="python",
                adapter="coding-orchestrator", adapter_version="1.0.0",
                toolchain_name="python", toolchain_version=self.python.tool_version() or "unknown",
                category=str(diagnostic.get("category", "compiler_error")),
                message=str(diagnostic.get("message", "unspecified")),
                line=diagnostic.get("line"),
                evidence_classification=str(diagnostic.get("evidence_classification",
                                                           "TOOL_VERIFIED"))))
        return records

    # ------------------------------------------------------------------ #
    def _repair_prompt(self, task: CodingTask, attempt_n: int,
                       ledger: dict[str, Any]) -> str:
        last = ledger["attempts"][-1] if ledger["attempts"] else {}
        semantic_status = last.get("validation", {}).get("semantic", {}).get("status", "UNKNOWN")
        test_errors = [d.get("message") for d in last.get("diagnostics", [])][:3]
        wanted = "\n".join(f"- {e}" for e in test_errors) or "- unknown failure"
        return (render_task_prompt(task)
                + f"\n\nYour previous candidate failed validation "
                  f"(semantic={semantic_status}). Repair the program:\n{wanted}\n"
                  f"Return code only.")

    # ------------------------------------------------------------------ #
    def _persist(self, ledger: dict[str, Any]) -> None:
        if self.state_store is None:
            return
        try:
            self.state_store.save_json_idempotent(
                f"coding/{ledger['task_id']}-{ledger.get('state', 'pending')}.json", ledger)
        except Exception:  # noqa: BLE001 — persistence is best-effort; never mask the run
            pass

    def _finalize(self, ledger: dict[str, Any], started: float) -> dict[str, Any]:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        attempts = ledger["attempts"]
        execution = ((attempts[-1].get("validation") or {}).get("execution") or {}) if attempts else {}
        ledger["final_result"] = {
            "state": ledger["state"],
            "tests_passed": execution.get("tests_passed", 0),
            "tests_total": execution.get("tests_total", len(execution.get("tests", [])) or 0),
            "all_passed": execution.get("all_passed", False),
            "execution_verified": bool(execution.get("all_passed")),
            "attempts_used": len(attempts),
        }
        ledger["resources"] = {
            "elapsed_ms": elapsed,
            "generator": self.model.model_kind,
        }
        ledger["capability_claim"] = None
        ledger["model_kind"] = self.model.model_kind
        ledger["closed_at"] = utcnow_iso()
        self._persist(ledger)
        return ledger
