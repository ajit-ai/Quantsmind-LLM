"""REST API for TAKSH engineering services with deny-by-default security."""

from __future__ import annotations

import hmac
import json
import os
import time
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional dependency
    APIRouter = None
    BaseModel = object
    Field = None
    Request = None
    HTTPException = Exception

from taksh.service import EngineeringService
from taksh.engineering import create_task, engineering_report
from taksh.memory import ProjectMemory

_MAX_BODY_BYTES = 1_000_000
_audit_log: list[dict[str, Any]] = []


def _configured_token() -> str | None:
    return os.environ.get("TAKSH_ENGINEERING_TOKEN") or os.environ.get("QM_API_KEY") or None


def _configured_scopes() -> set[str]:
    raw = os.environ.get("TAKSH_TOKEN_SCOPES", "read")
    return {scope.strip() for scope in raw.split(",") if scope.strip()}


def _authenticate(request):
    if request is None:
        return {"identity": "anonymous", "scopes": set()}
    token = _configured_token()
    if token is None:
        return {"identity": "unauthenticated", "scopes": set(), "error": "engineering authentication is not configured"}
    header = request.headers.get("authorization", "")
    supplied = header[7:] if header.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied, token):
        return {"identity": "invalid-token", "scopes": set(), "error": "invalid or missing bearer token"}
    return {"identity": "token", "scopes": _configured_scopes(), "error": None}


def _authorize(auth, required_scope: str):
    scopes = auth.get("scopes", set())
    if required_scope in scopes:
        return True, "authorized"
    return False, f"missing required scope: {required_scope}"


def _audit(identity, operation, decision, outcome):
    _audit_log.append({
        "identity": identity,
        "operation": operation,
        "decision": decision,
        "outcome": outcome,
        "timestamp": time.time(),
    })
    if len(_audit_log) > 1000:
        del _audit_log[:100]


def _require(request, required_scope, operation):
    if request is not None:
        length = request.headers.get("content-length")
        if length and int(length) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
    auth = _authenticate(request)
    identity = auth.get("identity", "unknown")
    if auth.get("error"):
        _audit(identity, operation, "deny", auth["error"])
        raise HTTPException(status_code=503 if auth["error"] == "engineering authentication is not configured" else 401, detail=auth["error"])
    ok, reason = _authorize(auth, required_scope)
    if not ok:
        _audit(identity, operation, "deny", reason)
        raise HTTPException(status_code=403, detail=reason)
    _audit(identity, operation, "allow", "authorized")
    return auth


class TaskRequest(BaseModel):
    task_id: str
    objective: str
    root: str
    provider: str = "native"


class ApplyRequest(BaseModel):
    plan: dict[str, Any]
    approved: bool = False


class ValidateRequest(BaseModel):
    files: list[str] | None = None


class TestRequest(BaseModel):
    files: list[str] | None = None
    timeout: int = 120


_tasks: dict[str, dict[str, Any]] = {}


def build_router() -> Any:
    if APIRouter is None:
        return None
    router = APIRouter(prefix="/v1", tags=["engineering"])

    @router.get("/health")
    async def health():
        return {"status": "ok", "service": "taksh-engineering"}

    @router.get("/tools")
    async def tools(request: Request, root: str):
        _require(request, "read", "tools")
        service = EngineeringService(root)
        return {"tools": service.tool_catalog()}

    @router.get("/projects/inspect")
    async def inspect_project(request: Request, root: str):
        _require(request, "read", "project_inspect")
        service = EngineeringService(root)
        return service.inspect_project()

    @router.post("/tasks")
    async def create_task_endpoint(request: Request, task_request: TaskRequest):
        _require(request, "read", "task_create")
        service = EngineeringService(task_request.root)
        task = service.create_task(task_request.task_id, task_request.objective)
        _tasks[task_request.task_id] = {"task": task, "service": service, "provider": task_request.provider}
        return {"task_id": task_request.task_id, "state": task.state.value}

    @router.get("/tasks/{task_id}")
    async def get_task(request: Request, task_id: str):
        _require(request, "read", "task_get")
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["task"].to_dict()

    @router.post("/tasks/{task_id}/validate")
    async def validate_task(request: Request, task_id: str, body: ValidateRequest):
        _require(request, "read", "task_validate")
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["service"].validate(body.files)

    @router.post("/tasks/{task_id}/tests")
    async def test_task(request: Request, task_id: str, body: TestRequest):
        _require(request, "execute", "task_tests")
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["service"].run_tests(body.files, body.timeout)

    @router.post("/tasks/{task_id}/apply")
    async def apply_task(request: Request, task_id: str, body: ApplyRequest):
        _require(request, "mutate", "task_apply")
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        from taksh.engineering import ChangePlan, FileChange

        plan = ChangePlan(
            task_id=task_id,
            root=record["task"].root,
            changes=[FileChange(**item) for item in body.plan.get("changes", [])],
            approved=body.approved,
        )
        return record["service"].apply_patch(plan, approved=body.approved).to_dict()

    @router.get("/memory/{project_id}")
    async def memory(request: Request, project_id: str, storage: str):
        _require(request, "memory", "memory_read")
        memory = ProjectMemory(storage)
        return memory.inspect(project_id)

    @router.get("/tasks/{task_id}/report")
    async def report(request: Request, task_id: str):
        _require(request, "read", "task_report")
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return engineering_report(record["task"])

    @router.get("/audit")
    async def audit(request: Request):
        _require(request, "audit", "audit_read")
        return {"records": list(_audit_log)}

    return router
