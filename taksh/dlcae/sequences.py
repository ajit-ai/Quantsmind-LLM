"""Sequence Orchestration Engine — resumable, isolated comprehension sequences.

Supports the owner's illustrative shapes verbatim:

- one sequence decomposing into sub-tasks: ``C -> C1 -> C2 -> C3 -> C4``
- independent sequences running side by side: ``D1 -> D2 -> D3``

Guarantees:

- topological ordering; cyclic dependencies and unknown dependencies are
  rejected at creation;
- per-sequence ledger; a failure in one sequence never corrupts another;
- retries are bounded by an explicit policy;
- checkpoint/resume is JSON-serializable (portable task ledger);
- interrupted sequences resume from the ledger;
- completed idempotent tasks are not re-executed on resume;
- execution is bounded (``max_tasks_per_run``); cancellation is first-class;
- a terminal sequence state is reached or an explicit reason is recorded.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from taksh.dlcae.types import TaskState, fingerprint, utcnow_iso


class SequenceError(Exception):
    """Base error for sequence orchestration."""


class CycleError(SequenceError):
    """Cyclic dependency detected — reported, never looped."""


class DependencyError(SequenceError):
    """Dependency refers to a task that does not exist in the sequence."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    backoff_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts,
                "backoff_seconds": self.backoff_seconds}


@dataclass
class Task:
    task_id: str
    action: str
    depends_on: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = True
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    last_error: str | None = None
    result: dict[str, Any] | None = None
    result_fingerprint: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "depends_on": list(self.depends_on),
            "payload": self.payload,
            "retry": self.retry.to_dict(),
            "idempotent": self.idempotent,
            "state": self.state.value,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "result": self.result,
            "result_fingerprint": self.result_fingerprint,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        retry = data.get("retry", {})
        return cls(
            task_id=data["task_id"],
            action=data["action"],
            depends_on=tuple(data.get("depends_on", ())),
            payload=dict(data.get("payload", {})),
            retry=RetryPolicy(max_attempts=int(retry.get("max_attempts", 2)),
                              backoff_seconds=float(retry.get("backoff_seconds", 0.0))),
            idempotent=bool(data.get("idempotent", True)),
            state=TaskState(data.get("state", "PENDING")),
            attempts=int(data.get("attempts", 0)),
            last_error=data.get("last_error"),
            result=data.get("result"),
            result_fingerprint=data.get("result_fingerprint"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class Sequence:
    sequence_id: str
    name: str
    tasks: dict[str, Task] = field(default_factory=dict)
    created: str = field(default_factory=utcnow_iso)
    cancelled: bool = False
    cancel_reason: str | None = None

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        if self.cancelled:
            return "CANCELLED"
        states = [t.state for t in self.tasks.values()]
        if not states:
            return "COMPLETED"
        if any(s is TaskState.RUNNING for s in states):
            return "RUNNING"
        if any(s is TaskState.INTERRUPTED for s in states):
            return "INTERRUPTED"
        if all(s is TaskState.COMPLETED for s in states):
            return "COMPLETED"
        if any(s is TaskState.FAILED for s in states):
            if all(s in (TaskState.FAILED, TaskState.BLOCKED, TaskState.COMPLETED)
                   for s in states):
                return "FAILED"
            return "RUNNING"
        if all(s in (TaskState.BLOCKED,) for s in states):
            return "BLOCKED"
        return "PENDING"

    def ordered_task_ids(self) -> list[str]:
        """Kahn's algorithm over the task DAG. Raises CycleError."""
        indeg = {t: len(task.depends_on) for t, task in self.tasks.items()}
        dependents: dict[str, list[str]] = {t: [] for t in self.tasks}
        for t, task in self.tasks.items():
            for dep in task.depends_on:
                dependents[dep].append(t)
        queue = [t for t in sorted(self.tasks) if indeg[t] == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dependent in dependents[node]:
                indeg[dependent] -= 1
                if indeg[dependent] == 0:
                    queue.append(dependent)
        if len(order) != len(self.tasks):
            cyclic = sorted(set(self.tasks) - set(order))
            raise CycleError(f"cyclic dependency among tasks: {cyclic}")
        return order

    def runnable(self) -> list[str]:
        return [tid for tid, task in sorted(self.tasks.items())
                if task.state is TaskState.PENDING
                and all(self.tasks[dep].state is TaskState.COMPLETED
                        for dep in task.depends_on)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.sequence/1",
            "sequence_id": self.sequence_id,
            "name": self.name,
            "created": self.created,
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
            "state": self.state,
            "tasks": {tid: task.to_dict() for tid, task in sorted(self.tasks.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sequence":
        seq = cls(sequence_id=data["sequence_id"], name=data["name"],
                  created=data.get("created", utcnow_iso()))
        seq.cancelled = bool(data.get("cancelled", False))
        seq.cancel_reason = data.get("cancel_reason")
        seq.tasks = {tid: Task.from_dict(tdata) for tid, tdata in data["tasks"].items()}
        return seq


class SequenceOrchestrationEngine:
    """Executes and persists comprehension sequences.

    Handlers are callables ``handler(payload: dict) -> dict``. Handlers must be
    deterministic test doubles in tests; real tool integration flows through
    the Validation Engine and never silently succeeds.
    """

    def __init__(self, *, handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
                 max_tasks_per_run: int = 256) -> None:
        self.handlers = dict(handlers or {})
        self.max_tasks_per_run = max_tasks_per_run
        self._counter = itertools.count(1)
        self._sequences: dict[str, Sequence] = {}

    # ------------------------------------------------------------------ #
    def create_sequence(self, name: str, tasks: Iterable[dict[str, Any]]) -> str:
        if not name.strip():
            raise ValueError("sequence name must be non-empty")
        seq_id = f"seq-{fingerprint({'name': name, 'n': next(self._counter)})[:12]}"
        sequence = Sequence(sequence_id=seq_id, name=name)
        for spec in tasks:
            task_id = str(spec["task_id"]).strip()
            if not task_id:
                raise ValueError("task_id must be non-empty")
            if task_id in sequence.tasks:
                raise ValueError(f"duplicate task_id: {task_id}")
            retry = spec.get("retry", {})
            sequence.tasks[task_id] = Task(
                task_id=task_id,
                action=str(spec["action"]),
                depends_on=tuple(str(d) for d in spec.get("depends_on", ())),
                payload=dict(spec.get("payload", {})),
                retry=RetryPolicy(max_attempts=int(retry.get("max_attempts", 2)),
                                  backoff_seconds=float(retry.get("backoff_seconds", 0.0))),
                idempotent=bool(spec.get("idempotent", True)),
            )
        self._validate(sequence)
        sequence.ordered_task_ids()  # raises CycleError early
        self._sequences[seq_id] = sequence
        return seq_id

    def register_handler(self, action: str,
                         handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.handlers[action] = handler

    def _validate(self, sequence: Sequence) -> None:
        for task in sorted(sequence.tasks.values(), key=lambda t: t.task_id):
            for dep in task.depends_on:
                if dep not in sequence.tasks:
                    raise DependencyError(
                        f"task {task.task_id!r} depends on unknown task {dep!r}")

    # ------------------------------------------------------------------ #
    def get(self, sequence_id: str) -> Sequence:
        try:
            return self._sequences[sequence_id]
        except KeyError:
            raise SequenceError(f"unknown sequence: {sequence_id}") from None

    def run(self, sequence_id: str) -> dict[str, Any]:
        sequence = self.get(sequence_id)
        executed = 0
        while not sequence.cancelled:
            if executed >= self.max_tasks_per_run:
                return self._report(sequence, interrupted=True,
                                    reason="max_tasks_per_run reached; sequence left resumable")
            runnable = sequence.runnable()
            if not runnable:
                break
            for task_id in runnable:
                if executed >= self.max_tasks_per_run:
                    break
                self._execute(sequence, sequence.tasks[task_id])
                executed += 1
                if sequence.cancelled:
                    break
            if executed >= self.max_tasks_per_run:
                return self._report(sequence, interrupted=True,
                                    reason="max_tasks_per_run reached; sequence left resumable")
        # propagate dependency failure into BLOCKED for dependents
        self._propagate_blocked(sequence)
        return self._report(sequence)

    def _execute(self, sequence: Sequence, task: Task) -> None:
        if task.state is TaskState.COMPLETED and task.idempotent:
            return  # idempotent resume: never re-executed
        handler = self.handlers.get(task.action)
        if handler is None:
            task.attempts += 1
            task.last_error = f"no handler registered for action {task.action!r}"
            task.state = TaskState.FAILED
            return
        task.state = TaskState.RUNNING
        while task.attempts < task.retry.max_attempts:
            task.attempts += 1
            try:
                result = handler(dict(task.payload))
            except Exception as exc:  # noqa: BLE001 — handler failures are data, not crashes
                task.last_error = f"{type(exc).__name__}: {exc}"
                if task.retry.backoff_seconds:
                    time.sleep(task.retry.backoff_seconds)
                continue
            task.result = result if isinstance(result, dict) else {"value": result}
            task.result_fingerprint = fingerprint(task.result)
            task.state = TaskState.COMPLETED
            task.completed_at = utcnow_iso()
            task.last_error = None
            return
        task.state = TaskState.FAILED

    def _propagate_blocked(self, sequence: Sequence) -> None:
        changed = True
        while changed:
            changed = False
            for task in sequence.tasks.values():
                if task.state is TaskState.PENDING:
                    blocked = [dep for dep in task.depends_on
                               if sequence.tasks[dep].state in
                               (TaskState.FAILED, TaskState.BLOCKED, TaskState.CANCELLED)]
                    if blocked:
                        task.state = TaskState.BLOCKED
                        task.last_error = f"blocked by: {sorted(blocked)}"
                        changed = True

    # ------------------------------------------------------------------ #
    def checkpoint(self, sequence_id: str) -> dict[str, Any]:
        """Portable, JSON-serializable ledger for this sequence."""
        sequence = self.get(sequence_id)
        return sequence.to_dict()

    def restore(self, ledger: dict[str, Any]) -> str:
        """Resume a sequence from a ledger. Checkpoints are read-only inputs."""
        sequence = Sequence.from_dict(ledger)
        self._validate(sequence)
        if sequence.cancelled:
            raise SequenceError(
                "ledger belongs to a cancelled sequence; refusing to resume a cancelled sequence")
        for task in sequence.tasks.values():
            if task.state is TaskState.RUNNING:
                # crash mid-execution: not completed, resume as pending
                task.state = TaskState.PENDING
                task.last_error = (task.last_error or "interrupted mid-execution") + " — resumed as pending"
            elif task.state is TaskState.INTERRUPTED:
                task.state = TaskState.PENDING
                task.last_error = (task.last_error or "interrupted") + " — resumed as pending"
        # deterministic re-validation of the graph (catches hand-edited ledgers)
        sequence.ordered_task_ids()
        self._sequences[sequence.sequence_id] = sequence
        return sequence.sequence_id

    def mark_interrupted(self, sequence_id: str, reason: str = "interrupted") -> None:
        sequence = self.get(sequence_id)
        for task in sequence.tasks.values():
            if task.state in (TaskState.RUNNING, TaskState.PENDING):
                task.state = TaskState.INTERRUPTED
                task.last_error = reason

    def cancel(self, sequence_id: str, reason: str = "cancelled by caller") -> dict[str, Any]:
        sequence = self.get(sequence_id)
        sequence.cancelled = True
        sequence.cancel_reason = reason
        for task in sequence.tasks.values():
            if task.state in (TaskState.PENDING, TaskState.RUNNING):
                task.state = TaskState.CANCELLED
                task.last_error = reason
        return self._report(sequence)

    def status_map(self) -> dict[str, str]:
        return {sid: seq.state for sid, seq in sorted(self._sequences.items())}

    def _report(self, sequence: Sequence, *, interrupted: bool = False,
                reason: str | None = None) -> dict[str, Any]:
        failed = [t.task_id for t in sequence.tasks.values() if t.state is TaskState.FAILED]
        blocked = [t.task_id for t in sequence.tasks.values() if t.state is TaskState.BLOCKED]
        done = [t.task_id for t in sequence.tasks.values() if t.state is TaskState.COMPLETED]
        report = {
            "sequence_id": sequence.sequence_id,
            "name": sequence.name,
            "state": sequence.state,
            "cancelled": sequence.cancelled,
            "cancel_reason": sequence.cancel_reason,
            "completed": sorted(done),
            "failed": sorted(failed),
            "blocked": sorted(blocked),
            "ledger_fingerprint": fingerprint(sequence.to_dict()),
        }
        if interrupted:
            report["interrupted"] = True
            report["interruption_reason"] = reason
        return report
