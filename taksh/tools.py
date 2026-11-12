"""Tool-calling runtime for TAKSH engineering.

Tools are explicit, typed and independently validated. A model may propose a
tool invocation, but this runtime validates schema, permissions, workspace
scope and execution limits before running anything. Unknown or invalid tools
are rejected fail-closed.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from taksh.engineering import (
    _resolve,
    inspect_project,
    validate_project,
    run_tests,
    apply_plan,
    ChangePlan,
    FileChange,
)
from taksh.permissions import PermissionPolicy, PermissionDecision, OperationClass


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permission: str
    handler: Callable[[dict[str, Any], "ToolContext"], Any]
    timeout_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permission": self.permission,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ToolContext:
    root: str
    task_id: str | None = None
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolInvocation:
    invocation_id: str
    tool: str
    arguments: dict[str, Any]
    started_at: float
    finished_at: float | None = None
    status: str = "pending"
    result: Any = None
    error: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return sorted(self._tools.values(), key=lambda item: item.name)

    def names(self) -> list[str]:
        return sorted(self._tools)


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, permissions: PermissionPolicy | None = None) -> None:
        self.registry = registry
        self.permissions = permissions or PermissionPolicy()
        self.history: list[ToolInvocation] = []
        self._max_calls = 100
        self._call_count = 0

    def invoke(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolInvocation:
        invocation = ToolInvocation(
            invocation_id=str(uuid.uuid4()),
            tool=name,
            arguments=dict(arguments or {}),
            started_at=time.time(),
            provenance={"task_id": context.task_id, "root": context.root},
        )
        try:
            self._call_count += 1
            if self._call_count > self._max_calls:
                raise PermissionError("tool call limit exceeded")

            tool = self.registry.get(name)
            if tool is None:
                raise ValueError(f"unknown tool: {name}")

            self._validate_arguments(tool, invocation.arguments)
            permission = self.permissions.evaluate(tool.permission, context.approved, invocation.arguments)
            if not permission.allowed:
                raise PermissionError(permission.reason)

            result = tool.handler(invocation.arguments, context)
            invocation.result = result
            invocation.status = "success"
        except Exception as exc:  # noqa: BLE001
            invocation.status = "error"
            invocation.error = f"{type(exc).__name__}: {exc}"
        finally:
            invocation.finished_at = time.time()
            self.history.append(invocation)
        return invocation

    def _validate_arguments(self, tool: ToolDefinition, arguments: dict[str, Any]) -> None:
        schema = tool.input_schema or {}
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ValueError(f"missing required argument: {key}")
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            if key not in properties:
                raise ValueError(f"unknown argument: {key}")
            expected = properties[key].get("type")
            if expected and not _type_matches(value, expected):
                raise ValueError(f"argument {key} must be {expected}")

    def audit(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.history]


def _type_matches(value: Any, expected: str) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    if expected not in mapping:
        return True
    return isinstance(value, mapping[expected])


def build_engineering_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def inspect_project_handler(arguments, context):
        return inspect_project(context.root)

    def file_lookup_handler(arguments, context):
        root = Path(context.root).resolve()
        relative = arguments["path"]
        path = _resolve(context.root, relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        return {"path": relative, "content": path.read_text(encoding="utf-8"), "sha256": _sha256(path.read_bytes())}

    def symbol_lookup_handler(arguments, context):
        info = inspect_project(context.root)
        needle = arguments["symbol"]
        matches = []
        for file_path, symbols in info["symbols"].get("symbols", {}).items():
            for symbol in symbols:
                if symbol.get("name") == needle:
                    matches.append({"file": file_path, **symbol})
        return {"symbol": needle, "matches": matches}

    def dependency_handler(arguments, context):
        info = inspect_project(context.root)
        return {"dependencies": info["dependencies"]}

    def plan_handler(arguments, context):
        from taksh.engineering import create_task, plan_task

        # Planning without a provider returns a blocked task; callers must supply
        # a provider through the service layer for real planning.
        task = create_task(arguments.get("task_id", "tool-plan"), arguments["objective"], context.root)
        return {"task_id": task.task_id, "state": task.state.value, "note": "plan requires an explicit provider"}

    def patch_review_handler(arguments, context):
        plan_data = arguments["plan"]
        return {"approved": False, "change_count": len(plan_data.get("changes", [])), "review_required": True}

    def patch_apply_handler(arguments, context):
        plan_data = arguments["plan"]
        plan = ChangePlan(
            task_id=plan_data["task_id"],
            root=context.root,
            changes=[FileChange(**item) for item in plan_data.get("changes", [])],
            approved=bool(arguments.get("approved", False)),
        )
        return apply_plan(context.root, plan, approved=bool(arguments.get("approved", False))).to_dict()

    def validation_handler(arguments, context):
        files = arguments.get("files")
        return validate_project(context.root, files)

    def test_discovery_handler(arguments, context):
        from taksh.engineering import discover_tests_in_project

        return {"tests": discover_tests_in_project(context.root)}

    def test_execution_handler(arguments, context):
        files = arguments.get("files")
        timeout = int(arguments.get("timeout", 120))
        return run_tests(context.root, files, timeout=timeout)

    def report_handler(arguments, context):
        from taksh.engineering import inspect_project, engineering_report, create_task

        task = create_task(arguments.get("task_id", "tool-report"), arguments.get("objective", "engineering task"), context.root)
        task.context = inspect_project(context.root)
        return engineering_report(task)

    tools = [
        ToolDefinition("inspect_project", "Inspect a project root", "read", {"required": [], "properties": {}}, {"type": "object"}, "read", inspect_project_handler),
        ToolDefinition("file_lookup", "Read a file within the workspace", "read", {"required": ["path"], "properties": {"path": {"type": "string"}}}, {"type": "object"}, "read", file_lookup_handler),
        ToolDefinition("symbol_lookup", "Find symbol definitions", "read", {"required": ["symbol"], "properties": {"symbol": {"type": "string"}}}, {"type": "object"}, "read", symbol_lookup_handler),
        ToolDefinition("dependency_inspection", "Inspect project dependencies", "read", {"required": [], "properties": {}}, {"type": "object"}, "read", dependency_handler),
        ToolDefinition("change_plan", "Create a task planning placeholder", "read", {"required": ["objective"], "properties": {"objective": {"type": "string"}, "task_id": {"type": "string"}}}, {"type": "object"}, "read", plan_handler),
        ToolDefinition("patch_review", "Review a proposed patch without applying it", "read", {"required": ["plan"], "properties": {"plan": {"type": "object"}}}, {"type": "object"}, "read", patch_review_handler),
        ToolDefinition("patch_apply", "Apply an approved patch", "mutation", {"required": ["plan", "approved"], "properties": {"plan": {"type": "object"}, "approved": {"type": "boolean"}}}, {"type": "object"}, "mutation", patch_apply_handler),
        ToolDefinition("validate_project", "Validate project files", "controlled", {"required": [], "properties": {"files": {"type": "array"}}}, {"type": "object"}, "controlled", validation_handler),
        ToolDefinition("test_discovery", "Discover project tests", "read", {"required": [], "properties": {}}, {"type": "object"}, "read", test_discovery_handler),
        ToolDefinition("test_execution", "Run targeted tests", "controlled", {"required": [], "properties": {"files": {"type": "array"}, "timeout": {"type": "integer"}}}, {"type": "object"}, "controlled", test_execution_handler),
        ToolDefinition("engineering_report", "Generate an engineering report", "read", {"required": [], "properties": {"task_id": {"type": "string"}, "objective": {"type": "string"}}}, {"type": "object"}, "read", report_handler),
    ]
    for tool in tools:
        registry.register(tool)
    return registry


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
