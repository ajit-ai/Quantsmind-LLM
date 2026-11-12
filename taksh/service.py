"""Shared service layer for TAKSH engineering interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from taksh.engineering import (
    create_task,
    inspect_project,
    plan_task,
    apply_plan,
    validate_project,
    run_tests,
    run_engineering_task,
    engineering_report,
    ChangePlan,
    FileChange,
)
from taksh.memory import ProjectMemory
from taksh.permissions import PermissionPolicy
from taksh.tools import ToolRuntime, build_engineering_tool_registry


class EngineeringService:
    def __init__(self, root: str | Path, memory_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.memory = ProjectMemory(memory_path) if memory_path else None
        self.permissions = PermissionPolicy()
        self.tools = ToolRuntime(build_engineering_tool_registry(), self.permissions)

    def inspect_project(self) -> dict[str, Any]:
        return inspect_project(self.root)

    def create_task(self, task_id: str, objective: str, **kwargs: Any):
        return create_task(task_id, objective, self.root, **kwargs)

    def plan_task(self, task, provider, context=None):
        return plan_task(task, provider, context)

    def apply_patch(self, plan: ChangePlan, *, approved: bool = False):
        return apply_plan(self.root, plan, approved=approved)

    def validate(self, files: list[str] | None = None):
        return validate_project(self.root, files)

    def run_tests(self, files: list[str] | None = None, timeout: int = 120):
        return run_tests(self.root, files, timeout=timeout)

    def run_task(self, task, provider, **kwargs):
        return run_engineering_task(task, provider, **kwargs)

    def report(self, task):
        return engineering_report(task)

    def invoke_tool(self, name: str, arguments: dict[str, Any], approved: bool = False):
        from taksh.tools import ToolContext

        return self.tools.invoke(name, arguments, ToolContext(root=str(self.root), approved=approved)).to_dict()

    def tool_catalog(self):
        return [tool.to_dict() for tool in self.tools.registry.list()]

    def memory_inspect(self, project_id: str | None = None):
        if self.memory is None:
            return {"facts": [], "fact_count": 0, "storage_path": None}
        return self.memory.inspect(project_id)

    def memory_store(self, project_id: str, kind: str, value: Any, **kwargs):
        if self.memory is None:
            raise RuntimeError("memory is not configured")
        return self.memory.store(project_id, kind, value, **kwargs).to_dict()

    def memory_invalidate(self, project_id: str, reason: str) -> int:
        if self.memory is None:
            return 0
        return self.memory.invalidate_project(project_id, reason)
