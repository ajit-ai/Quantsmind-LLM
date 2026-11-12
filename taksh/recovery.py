"""Recovery helpers for interrupted agent tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taksh.agent import AgentWorkflow, AgentState


def recover_tasks(tasks_dir: str | Path) -> dict[str, Any]:
    directory = Path(tasks_dir).resolve()
    if not directory.is_dir():
        return {"status": "missing", "tasks": [], "directory": str(directory)}

    recovered: list[dict[str, Any]] = []
    workflow = AgentWorkflow()
    for path in sorted(directory.glob("*.json")):
        try:
            task = workflow.load(path)
            interrupted = task.state not in {
                AgentState.REPORTED,
                AgentState.FAILED,
                AgentState.CANCELLED,
            }
            recovered.append({
                "path": str(path),
                "task_id": task.task_id,
                "state": task.state.value,
                "interrupted": interrupted,
            })
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            recovered.append({
                "path": str(path),
                "task_id": None,
                "state": "corrupt",
                "interrupted": True,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "status": "ok",
        "directory": str(directory),
        "recovered": recovered,
        "interrupted_count": sum(1 for item in recovered if item["interrupted"]),
        "corrupt_count": sum(1 for item in recovered if item.get("state") == "corrupt"),
    }
