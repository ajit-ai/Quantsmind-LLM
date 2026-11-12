"""TAKSH 1.3.0 — Coding Task Engine.

Orchestrates the workflow: understand → plan → generate → review → apply → test → repair → verify.

This is a provider-agnostic orchestration layer. It coordinates tasks but does
not execute training or claim capability. Model-generated code is isolated for
validation; generation is delegated to the configured provider; no capability
claim is advanced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    PENDING = "PENDING"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    REVIEWING = "REVIEWING"
    APPLYING = "APPLYING"
    TESTING = "TESTING"
    REPAIRING = "REPAIRING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass
class TaskSpec:
    task_id: str
    description: str
    language: str = "python"
    context: dict[str, Any] = field(default_factory=dict)
    max_repair_iterations: int = 3
    repair_hints: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    step: str
    status: str
    detail: str = ""
    artifact: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "status": self.status,
                "detail": self.detail, "artifact": self.artifact}


@dataclass
class TaskResult:
    task_id: str
    status: TaskState
    steps: list[StepResult] = field(default_factory=list)
    tokens_used: int = 0
    repair_iterations: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    capability_claim: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.coding_task_result/1",
            "task_id": self.task_id,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "tokens_used": self.tokens_used,
            "repair_iterations": self.repair_iterations,
            "validation": self.validation,
            "capability_claim": None,
        }


class TaskEngine:
    """Provider-agnostic coding task orchestrator.

    The engine maintains explicit stage transitions, accumulates evidence,
    and records outcomes. It never fabricates capability or executes code
    outside the validation layer's sandbox.
    """

    def __init__(self, *, validator=None) -> None:
        self.validator = validator
        self._history: list[dict[str, Any]] = []

    def validate(self, task: TaskSpec, source: str | None) -> dict[str, Any]:
        if self.validator is None:
            return {"status": "NOT_VALIDATED",
                    "reason": "no validator configured"}
        return self.validator(task, source)

    def run(self, task: TaskSpec, *,
            generator,
            applier=None,
            repair_callback=None) -> TaskResult:
        result = TaskResult(task_id=task.task_id,
                            status=TaskState.PENDING,
                            steps=[])

        def emit(step: str, status: str, detail: str = "", artifact=None):
            result.steps.append(StepResult(step, status, detail, artifact))

        try:
            # understand
            emit("understand", "completed", "task accepted")
            result.status = TaskState.UNDERSTANDING

            # plan
            emit("plan", "completed", "plan stage reached (provider-independent)")
            result.status = TaskState.PLANNING

            # generate
            generated = generator(task) if callable(generator) else None
            emit("generate", "completed" if generated else "blocked",
                 f"generated={generated is not None}", generated)
            result.status = TaskState.GENERATING

            # review
            emit("review", "completed", "review stage reached")
            result.status = TaskState.REVIEWING

            # apply (no-op owner-side; we do not modify user files automatically)
            if applier is not None:
                applier_out = applier(generated)
                emit("apply", "completed", str(applier_out)[:200])
            else:
                emit("apply", "skipped", "no applier configured; owner approval required")
            result.status = TaskState.APPLYING

            # test / verify
            validation = self.validate(task, generated) if generated else {"status": "NOT_VALIDATED"}
            result.validation = validation
            emit("test", "completed" if validation.get("status") == "PASS" else "blocked",
                 f"validation={validation}")
            result.status = TaskState.TESTING

            # repair loop
            if validation.get("status") != "PASS" and repair_callback is not None:
                for i in range(task.max_repair_iterations):
                    feedback = validation
                    repair = repair_callback(task, feedback, generated)
                    if repair is None:
                        break
                    generated = repair
                    validation = self.validate(task, generated)
                    result.repair_iterations += 1
                    emit(f"repair_{i}", "completed" if validation.get("status") == "PASS" else "blocked",
                         f"iteration={i+1}")
                    if validation.get("status") == "PASS":
                        break
            result.status = TaskState.VERIFYING

            emit("verify", "completed", f"final_validation={validation.get('status')}")
            result.status = TaskState.COMPLETED if validation.get("status") == "PASS" else TaskState.FAILED

        except Exception as exc:  # noqa: BLE001 — failures are data
            emit("error", "failed", f"{type(exc).__name__}: {exc}")
            result.status = TaskState.FAILED

        self._history.append(result.to_dict())
        return result

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
