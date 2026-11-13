"""TAKSH integrated coding-engineering platform.

This module unifies project understanding, task orchestration, specialist
routing, validation, safe multi-file patch application, test execution, bounded
repair and reporting. It is usable without a native model: provider-unavailable
states are explicit and fail closed.

It never trains, creates checkpoints, downloads models or datasets, publishes,
merges, tags or releases anything.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent


class EngineeringError(Exception):
    """Base error for the engineering platform."""


class SecurityError(EngineeringError):
    """A path or operation violated workspace security rules."""


class ProviderUnavailable(EngineeringError):
    """The requested provider cannot generate changes."""


def _is_model_provider_error(exc: Exception) -> bool:
    """Recognize serving-layer provider failures without importing serving eagerly."""
    try:
        from serving.providers import ProviderError
    except Exception:  # noqa: BLE001 - optional serving layer
        return False
    return isinstance(exc, ProviderError)


class TaskState(str, Enum):
    CREATED = "CREATED"
    UNDERSTOOD = "UNDERSTOOD"
    PLANNED = "PLANNED"
    GENERATED = "GENERATED"
    REVIEWED = "REVIEWED"
    APPLIED = "APPLIED"
    VALIDATED = "VALIDATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ROLLED_BACK = "ROLLED_BACK"
    BLOCKED = "BLOCKED"


class ChangeOperation(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass
class FileChange:
    path: str
    operation: str
    content: str | None = None
    reason: str = ""
    dependencies: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    expected_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangePlan:
    task_id: str
    root: str
    changes: list[FileChange] = field(default_factory=list)
    summary: str = ""
    specialist: str | None = None
    provider: str | None = None
    approved: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "root": self.root,
            "summary": self.summary,
            "specialist": self.specialist,
            "provider": self.provider,
            "approved": self.approved,
            "provenance": self.provenance,
            "changes": [change.to_dict() for change in self.changes],
        }


@dataclass
class EngineeringTask:
    task_id: str
    objective: str
    root: str
    language: str = "python"
    constraints: list[str] = field(default_factory=list)
    max_repair_iterations: int = 3
    state: TaskState = TaskState.CREATED
    specialist: str | None = None
    provider: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    plan: ChangePlan | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    test_result: dict[str, Any] = field(default_factory=dict)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    capability_claim: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "root": self.root,
            "language": self.language,
            "constraints": self.constraints,
            "max_repair_iterations": self.max_repair_iterations,
            "state": self.state.value,
            "specialist": self.specialist,
            "provider": self.provider,
            "context": self.context,
            "plan": self.plan.to_dict() if self.plan else None,
            "validation": self.validation,
            "test_result": self.test_result,
            "repair_history": self.repair_history,
            "capability_claim": None,
        }


@dataclass
class ApplyResult:
    status: str
    applied: list[str] = field(default_factory=list)
    rolled_back: bool = False
    error: str | None = None
    snapshot: dict[str, bytes | None] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "error": self.error,
        }


class ChangeProvider(Protocol):
    name: str

    def generate_changes(self, task: EngineeringTask, context: dict[str, Any]) -> ChangePlan | None: ...


def native_change_provider(**kwargs: Any) -> "ChangeProvider":
    """Return the canonical native provider behind the change-planning adapter.

    The serving layer owns native inference; the engineering layer only knows
    the provider-agnostic ``ChangeProvider`` seam.
    """
    from serving.model_change_provider import ModelChangeProvider
    from serving.native_provider import NativeTakshProvider as NativeProvider

    return ModelChangeProvider(NativeProvider(**kwargs))


class DeterministicTestProvider:
    """Explicitly identified deterministic provider for tests and demonstrations."""

    name = "deterministic-test"

    def __init__(self, changes: list[FileChange], summary: str = "deterministic test change") -> None:
        self.changes = changes
        self.summary = summary

    def generate_changes(self, task: EngineeringTask, context: dict[str, Any]) -> ChangePlan:
        return ChangePlan(
            task_id=task.task_id,
            root=task.root,
            changes=list(self.changes),
            summary=self.summary,
            specialist=task.specialist,
            provider=self.name,
            provenance={"provider": self.name, "deterministic": True, "model_capability_claim": None},
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve(root: str | Path, relative: str) -> Path:
    root_path = Path(root).resolve()
    if not relative or Path(relative).is_absolute():
        raise SecurityError(f"absolute or empty path is not allowed: {relative!r}")
    if ".." in Path(relative).parts:
        raise SecurityError(f"path traversal is not allowed: {relative!r}")
    candidate = (root_path / relative).resolve()
    if not candidate.is_relative_to(root_path):
        raise SecurityError(f"workspace escape is not allowed: {relative!r}")
    # Symlink escape protection: no resolved path may leave the root.
    if candidate != (root_path / relative).resolve():
        raise SecurityError(f"symlink escape is not allowed: {relative!r}")
    return candidate


def _safe_relative(root: str | Path, path: Path) -> str:
    root_path = Path(root).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root_path):
        raise SecurityError(f"path is outside the workspace: {path}")
    return resolved.relative_to(root_path).as_posix()


def discover_tests(root: str | Path) -> list[Path]:
    root_path = Path(root).resolve()
    tests: list[Path] = []
    excluded = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for name in filenames:
            if name.startswith("test_") and name.endswith(".py"):
                tests.append(Path(dirpath) / name)
            elif name.endswith("_test.py"):
                tests.append(Path(dirpath) / name)
    return sorted(tests)


def inspect_project(root: str | Path) -> dict[str, Any]:
    from taksh.project_understanding import discover_files, index_symbols, dependency_graph, project_summary

    root_path = Path(root).resolve()
    summary = project_summary(root_path)
    symbols = index_symbols(root_path)
    dependencies = dependency_graph(root_path)
    tests = discover_tests(root_path)
    return {
        "root": str(root_path),
        "summary": summary,
        "symbols": symbols,
        "dependencies": dependencies,
        "tests": [p.relative_to(root_path).as_posix() for p in tests],
        "test_count": len(tests),
    }


def route_specialists(objective: str, context: dict[str, Any] | None = None) -> list[str]:
    lower = objective.lower()
    selected: list[str] = []
    if any(word in lower for word in ("test", "pytest", "coverage")):
        selected.append("test_generation")
    if any(word in lower for word in ("debug", "fix", "repair", "error", "failure")):
        selected.append("test_failure_diagnosis")
    if any(word in lower for word in ("explain", "understand", "document")):
        selected.append("python_explanation")
    if any(word in lower for word in ("refactor", "clean")):
        selected.append("python_refactoring")
    if any(word in lower for word in ("import", "dependency")):
        selected.append("dependency_analysis")
    if not selected:
        selected = ["python_generation", "code_review", "dependency_analysis"]
    # Preserve order and uniqueness.
    unique: list[str] = []
    for name in selected:
        if name not in unique:
            unique.append(name)
    return unique


def _run_specialists(objective: str, context: dict[str, Any], names: list[str]) -> dict[str, Any]:
    from taksh.specialists import SPECIALISTS

    results: dict[str, Any] = {}
    for name in names:
        specialist = SPECIALISTS.get(name)
        if specialist is None:
            results[name] = {"status": "MISSING", "output": "", "findings": ["specialist not registered"]}
            continue
        source = context.get("primary_source", "") if isinstance(context, dict) else ""
        result = specialist.run(source)
        results[name] = {
            "status": result.status,
            "output": result.output,
            "findings": result.findings,
        }
    return results


def validate_source(source: str, extra_allowed: set[str] | None = None) -> dict[str, Any]:
    from taksh.coding_validation import check_syntax, check_imports, check_semantics

    syntax = check_syntax(source)
    imports = check_imports(source, extra_allowed=extra_allowed)
    semantics = check_semantics(source)
    issues = syntax.issues + imports.issues + semantics.issues
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": [issue.to_dict() for issue in issues],
        "syntax": syntax.status,
        "imports": imports.status,
        "semantics": semantics.status,
    }


def validate_project(root: str | Path, files: Iterable[str] | None = None, extra_allowed: set[str] | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    targets = list(files) if files else None
    if targets is None:
        targets = [p.relative_to(root_path).as_posix() for p in root_path.rglob("*.py")]
    if extra_allowed is None:
        extra_allowed = {Path(p).parts[0] for p in targets if Path(p).parts}
    results: dict[str, Any] = {}
    all_pass = True
    for relative in targets:
        path = _resolve(root_path, relative)
        if not path.is_file():
            results[relative] = {"status": "FAIL", "issues": [{"category": "io", "message": "file not found", "line": None, "severity": "error"}]}
            all_pass = False
            continue
        source = path.read_text(encoding="utf-8")
        result = validate_source(source, extra_allowed=extra_allowed)
        results[relative] = result
        if result["status"] != "PASS":
            all_pass = False
    return {"status": "PASS" if all_pass else "FAIL", "files": results}


def discover_tests_in_project(root: str | Path) -> list[str]:
    return [p.relative_to(Path(root).resolve()).as_posix() for p in discover_tests(root)]


def run_tests(root: str | Path, test_files: Iterable[str] | None = None, timeout: int = 120) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if test_files is None:
        tests = [p.relative_to(root_path).as_posix() for p in discover_tests(root_path)]
    else:
        tests = [str(item) for item in test_files]
    if not tests:
        return {"status": "NO_TESTS", "returncode": 0, "stdout": "", "stderr": "no tests discovered", "tests": []}
    cmd = [sys.executable, "-m", "pytest", *tests, "-q"]
    try:
        proc = subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "TIMEOUT", "returncode": None, "stdout": (exc.stdout or "")[:4000], "stderr": (exc.stderr or "")[:4000], "tests": tests}
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return {
        "status": status,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[:4000],
        "stderr": (proc.stderr or "")[:4000],
        "tests": tests,
    }


class PatchTransaction:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.snapshots: dict[Path, bytes | None] = {}
        self.applied: list[Path] = []
        self.created: list[Path] = []

    def _snapshot(self, path: Path) -> None:
        if path not in self.snapshots:
            self.snapshots[path] = path.read_bytes() if path.exists() else None

    def apply_change(self, change: FileChange) -> None:
        path = _resolve(self.root, change.path)
        self._snapshot(path)

        if change.operation == ChangeOperation.CREATE.value:
            if path.exists():
                if change.expected_sha256 and _sha256_bytes(path.read_bytes()) != change.expected_sha256:
                    raise SecurityError(f"conflict: {change.path} already exists with different content")
                if not change.expected_sha256:
                    raise SecurityError(f"conflict: refusing to overwrite existing file {change.path}")
            if change.content is None:
                raise EngineeringError(f"create operation requires content: {change.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                self.created.append(path)
            path.write_bytes(change.content.encode("utf-8"))

        elif change.operation == ChangeOperation.MODIFY.value:
            if not path.exists():
                raise SecurityError(f"modify operation requires existing file: {change.path}")
            if change.expected_sha256 is not None:
                actual = _sha256_bytes(path.read_bytes())
                if actual != change.expected_sha256:
                    raise SecurityError(f"conflict: {change.path} changed since planning")
            if change.content is None:
                raise EngineeringError(f"modify operation requires content: {change.path}")
            path.write_bytes(change.content.encode("utf-8"))

        elif change.operation == ChangeOperation.DELETE.value:
            if not path.exists():
                raise SecurityError(f"delete operation requires existing file: {change.path}")
            if change.expected_sha256 is not None:
                actual = _sha256_bytes(path.read_bytes())
                if actual != change.expected_sha256:
                    raise SecurityError(f"conflict: {change.path} changed since planning")
            path.unlink()

        else:
            raise EngineeringError(f"unknown operation: {change.operation}")

        self.applied.append(path)

    def apply_plan(self, plan: ChangePlan, *, approved: bool = False) -> ApplyResult:
        if not approved and not plan.approved:
            return ApplyResult(status="BLOCKED", error="plan is not approved")
        applied: list[str] = []
        try:
            for change in plan.changes:
                self.apply_change(change)
                applied.append(change.path)
            return ApplyResult(status="APPLIED", applied=applied, snapshot=dict(self.snapshots))
        except Exception as exc:  # noqa: BLE001
            self.rollback()
            return ApplyResult(status="FAILED", applied=applied, rolled_back=True, error=str(exc), snapshot=dict(self.snapshots))

    def rollback(self) -> None:
        for path, data in self.snapshots.items():
            try:
                if data is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            except OSError:
                pass
        self.applied.clear()


def apply_plan(root: str | Path, plan: ChangePlan, *, approved: bool = False) -> ApplyResult:
    return PatchTransaction(root).apply_plan(plan, approved=approved)


def plan_task(task: EngineeringTask, provider: ChangeProvider, context: dict[str, Any] | None = None) -> tuple[EngineeringTask, ChangePlan | None]:
    if context is None:
        context = inspect_project(task.root)
    task.context = context
    task.state = TaskState.UNDERSTOOD
    specialist_names = route_specialists(task.objective, context)
    task.specialist = specialist_names[0] if specialist_names else None
    specialist_results = _run_specialists(task.objective, context, specialist_names)
    context = dict(context)
    context["specialists"] = specialist_results
    task.context = context

    try:
        plan = provider.generate_changes(task, context)
    except ProviderUnavailable:
        task.state = TaskState.BLOCKED
        task.provider = getattr(provider, "name", None)
        return task, None
    except Exception as exc:  # noqa: BLE001
        task.state = TaskState.BLOCKED if _is_model_provider_error(exc) else TaskState.FAILED
        task.provider = getattr(provider, "name", None)
        if task.state == TaskState.BLOCKED:
            return task, None
        raise EngineeringError(f"provider failed: {exc}") from exc

    task.provider = getattr(provider, "name", None)
    if plan is None:
        task.state = TaskState.BLOCKED
        return task, None
    plan.root = task.root
    plan.provider = task.provider
    task.plan = plan
    task.state = TaskState.PLANNED
    return task, plan


def review_plan(plan: ChangePlan, reviewer: Callable[[ChangePlan], bool] | None = None) -> bool:
    if reviewer is None:
        plan.approved = True
        return True
    approved = bool(reviewer(plan))
    plan.approved = approved
    return approved


def run_validation(task: EngineeringTask, files: Iterable[str] | None = None) -> dict[str, Any]:
    result = validate_project(task.root, files)
    task.validation = result
    task.state = TaskState.VALIDATED if result["status"] == "PASS" else TaskState.FAILED
    return result


def run_repair_loop(
    task: EngineeringTask,
    plan: ChangePlan,
    apply_fn: Callable[[ChangePlan], ApplyResult],
    validate_fn: Callable[[], dict[str, Any]],
    repair_provider: Callable[[EngineeringTask, ChangePlan, dict[str, Any]], ChangePlan | None],
    max_iterations: int | None = None,
) -> dict[str, Any]:
    max_iterations = max_iterations if max_iterations is not None else task.max_repair_iterations
    history: list[dict[str, Any]] = []
    current_plan = plan
    apply_result = apply_fn(current_plan)
    if apply_result.status != "APPLIED":
        result = {"status": "FAILED", "reason": apply_result.error, "iterations": 0, "history": history}
        task.state = TaskState.FAILED
        task.repair_history = history
        return result

    validation = validate_fn()
    if validation["status"] == "PASS":
        history.append({"iteration": 0, "status": "PASS", "validation": validation})
        task.repair_history = history
        task.state = TaskState.VALIDATED
        return {"status": "PASS", "iterations": 0, "history": history, "validation": validation}

    for iteration in range(1, max_iterations + 1):
        repair_plan = repair_provider(task, current_plan, validation)
        if repair_plan is None:
            history.append({"iteration": iteration, "status": "NO_REPAIR", "validation": validation})
            break
        apply_result = apply_fn(repair_plan)
        if apply_result.status != "APPLIED":
            history.append({"iteration": iteration, "status": "APPLY_FAILED", "error": apply_result.error, "validation": validation})
            break
        current_plan = repair_plan
        validation = validate_fn()
        history.append({"iteration": iteration, "status": validation["status"], "validation": validation})
        if validation["status"] == "PASS":
            task.repair_history = history
            task.state = TaskState.VALIDATED
            return {"status": "PASS", "iterations": iteration, "history": history, "validation": validation}

    task.repair_history = history
    task.state = TaskState.FAILED
    return {"status": "FAIL", "iterations": len(history), "history": history, "validation": validation}


def run_engineering_task(
    task: EngineeringTask,
    provider: ChangeProvider,
    *,
    apply: bool = False,
    validate: bool = True,
    run_project_tests: bool = False,
    test_timeout: int = 120,
    reviewer: Callable[[ChangePlan], bool] | None = None,
    repair_provider: Callable[[EngineeringTask, ChangePlan, dict[str, Any]], ChangePlan | None] | None = None,
) -> EngineeringTask:
    task, plan = plan_task(task, provider)
    if plan is None:
        return task

    if reviewer is None and apply:
        raise SecurityError(
            "explicit reviewer approval is required before applying changes; "
            "pass reviewer=callable or run with apply=False"
        )
    if reviewer is None:
        plan.approved = False
        task.state = TaskState.PLANNED
        return task
    if not review_plan(plan, reviewer):
        task.state = TaskState.PLANNED
        return task
    task.state = TaskState.REVIEWED

    if not apply:
        task.state = TaskState.PLANNED
        return task

    transaction = PatchTransaction(task.root)
    apply_result = transaction.apply_plan(plan, approved=True)
    if apply_result.status != "APPLIED":
        task.state = TaskState.FAILED
        task.validation = {"status": "FAIL", "error": apply_result.error, "rolled_back": apply_result.rolled_back}
        return task

    task.state = TaskState.APPLIED
    if validate:
        validation = validate_project(task.root, [change.path for change in plan.changes])
        task.validation = validation
        if validation["status"] != "PASS":
            if repair_provider is not None:
                def apply_fn(p: ChangePlan) -> ApplyResult:
                    return transaction.apply_plan(p, approved=True)

                def validate_fn() -> dict[str, Any]:
                    return validate_project(task.root, [change.path for change in (task.plan.changes if task.plan else [])])

                repair_result = run_repair_loop(task, plan, apply_fn, validate_fn, repair_provider, task.max_repair_iterations)
                task.validation = repair_result.get("validation", task.validation)
            else:
                transaction.rollback()
                task.validation = {**validation, "rolled_back": True}
                task.state = TaskState.FAILED
                return task

    if task.validation and task.validation.get("status") != "PASS":
        transaction.rollback()
        task.validation = {**task.validation, "rolled_back": True}
        task.state = TaskState.FAILED
        return task

    task.state = TaskState.VALIDATED
    if run_project_tests:
        test_result = run_tests(task.root, timeout=test_timeout)
        task.test_result = test_result
        if test_result["status"] not in ("PASS", "NO_TESTS"):
            task.state = TaskState.FAILED
            return task

    task.state = TaskState.COMPLETED
    return task


def create_task(task_id: str, objective: str, root: str | Path, **kwargs: Any) -> EngineeringTask:
    return EngineeringTask(task_id=task_id, objective=objective, root=str(Path(root).resolve()), **kwargs)


def engineering_report(task: EngineeringTask) -> dict[str, Any]:
    return {
        "schema": "taksh.engineering_task_report/1",
        "task": task.to_dict(),
        "native_model_capability": "NOT_EVALUATED",
        "capability_claim": None,
    }


def run_coding_task(
    root: str | Path,
    objective: str,
    provider: Any,
    *,
    task_id: str = "coding-task",
    reviewer: Callable[[ChangePlan], bool] | None = None,
    apply: bool = False,
    allow_file_contents: bool = False,
    validate: bool = True,
    run_project_tests: bool = False,
    test_timeout: int = 120,
) -> dict[str, Any]:
    """Run the canonical provider-independent coding workflow.

    The provider may be a serving ``ModelProvider`` or an existing
    ``ChangeProvider``. Nothing is written unless ``apply`` is true and the
    explicit ``reviewer`` returns true.
    """
    from serving.model_change_provider import ModelChangeProvider

    change_provider = (
        provider
        if hasattr(provider, "generate_changes")
        else ModelChangeProvider(provider, allow_file_contents=allow_file_contents)
    )
    task = create_task(task_id, objective, root)
    try:
        task = run_engineering_task(
            task,
            change_provider,
            apply=apply,
            validate=validate,
            run_project_tests=run_project_tests,
            test_timeout=test_timeout,
            reviewer=reviewer,
        )
    except SecurityError as exc:
        report = engineering_report(task)
        report["workflow"] = {
            "status": "BLOCKED",
            "error": str(exc),
            "requested_apply": bool(apply),
            "approved": False,
        }
        return report

    report = engineering_report(task)
    report["workflow"] = {
        "status": task.state.value,
        "requested_apply": bool(apply),
        "approved": bool(task.plan.approved) if task.plan else False,
        "allow_file_contents": bool(allow_file_contents),
        "native_model_capability": "NOT_EVALUATED",
    }
    return report


# SDK-friendly aliases
inspect_project_sdk = inspect_project
plan_task_sdk = plan_task
apply_plan_sdk = apply_plan
validate_project_sdk = validate_project
run_tests_sdk = run_tests
run_task_sdk = run_engineering_task
