"""Coding-task identity, constraints and an isolated Python dev suite (ENG3).

Synthetic development tasks with known expected behavior. Never frozen
benchmark tasks, never KARK-BENCH tasks. Generated code is only ever
*executed* inside the sandboxed toolchain workspace (see orchestrator).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taksh.dlcae.types import fingerprint

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEV_SUITE_PATH = REPO / "configs" / "l100eng3" / "dev_tasks.json"


@dataclass
class DevTaskTests:
    """Deterministic, in-suite test definitions (not frozen benchmarks)."""

    name: str
    check: str                # "call_equals" | "contains" | "raises"
    args: tuple = ()
    expected: Any = None
    entry_point: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "check": self.check, "args": list(self.args),
                "expected": self.expected, "entry_point": self.entry_point}


@dataclass
class CodingTask:
    task_id: str
    language: str
    description: str
    constraints: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    tests: list[DevTaskTests] = field(default_factory=list)
    entry_point: str = "solve"
    suite: str = "dev"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "language": self.language,
            "description": self.description,
            "constraints": list(self.constraints),
            "expected_behavior": self.expected_behavior,
            "tests": [t.to_dict() for t in self.tests],
            "entry_point": self.entry_point,
            "suite": self.suite,
        }

    @property
    def task_sha256(self) -> str:
        return fingerprint(self.to_dict())

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CodingTask":
        tests = [DevTaskTests(name=t["name"], check=t["check"],
                              args=tuple(t.get("args", ())),
                              expected=t.get("expected"),
                              entry_point=t.get("entry_point"))
                 for t in data.get("tests", [])]
        return CodingTask(
            task_id=data["task_id"],
            language=data.get("language", "python"),
            description=data["description"],
            constraints=list(data.get("constraints", [])),
            expected_behavior=data.get("expected_behavior", ""),
            tests=tests,
            entry_point=data.get("entry_point", "solve"),
            suite=data.get("suite", "dev"))


class DevSuite:
    """Loads the isolated development-task suite (never a benchmark)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEV_SUITE_PATH
        self._tasks = self._load()

    def _load(self) -> dict[str, CodingTask]:
        if not self.path.is_file():
            raise FileNotFoundError(f"dev task suite not found: {self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        assert data.get("schema") == "taksh.l100eng3_dev_tasks/1"
        tasks = [CodingTask.from_dict(item) for item in data["tasks"]]
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids)) == len(tasks)
        return {t.task_id: t for t in tasks}

    def tasks(self) -> list[CodingTask]:
        return [self._tasks[k] for k in sorted(self._tasks)]

    def get(self, task_id: str) -> CodingTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"unknown dev task: {task_id!r}") from None

    def suite_sha256(self) -> str:
        return fingerprint({tid: t.task_sha256 for tid, t in sorted(self._tasks.items())})

    def __len__(self) -> int:
        return len(self._tasks)


def render_task_prompt(task: CodingTask, *, context: str = "") -> str:
    parts = [f"Language: {task.language}",
             f"Task ({task.task_id}): {task.description}"]
    if task.constraints:
        parts.append("Constraints: " + "; ".join(task.constraints))
    if task.expected_behavior:
        parts.append(f"Expected behavior: {task.expected_behavior}")
    if context:
        parts.append(context.strip())
    parts.append(f"Implement the function `{task.entry_point}`. Reply with code only.")
    return "\n\n".join(parts)
