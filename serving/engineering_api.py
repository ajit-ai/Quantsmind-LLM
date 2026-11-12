"""REST API for TAKSH engineering services."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional dependency
    APIRouter = None
    BaseModel = object
    Field = None
    HTTPException = Exception

from taksh.service import EngineeringService
from taksh.engineering import create_task, engineering_report
from taksh.memory import ProjectMemory


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
    async def tools(root: str):
        service = EngineeringService(root)
        return {"tools": service.tool_catalog()}

    @router.get("/projects/inspect")
    async def inspect_project(root: str):
        service = EngineeringService(root)
        return service.inspect_project()

    @router.post("/tasks")
    async def create_task_endpoint(request: TaskRequest):
        service = EngineeringService(request.root)
        task = service.create_task(request.task_id, request.objective)
        _tasks[request.task_id] = {"task": task, "service": service, "provider": request.provider}
        return {"task_id": request.task_id, "state": task.state.value}

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["task"].to_dict()

    @router.post("/tasks/{task_id}/validate")
    async def validate_task(task_id: str, request: ValidateRequest):
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["service"].validate(request.files)

    @router.post("/tasks/{task_id}/tests")
    async def test_task(task_id: str, request: TestRequest):
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return record["service"].run_tests(request.files, request.timeout)

    @router.post("/tasks/{task_id}/apply")
    async def apply_task(task_id: str, request: ApplyRequest):
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        from taksh.engineering import ChangePlan, FileChange

        plan = ChangePlan(
            task_id=task_id,
            root=record["task"].root,
            changes=[FileChange(**item) for item in request.plan.get("changes", [])],
            approved=request.approved,
        )
        return record["service"].apply_patch(plan, approved=request.approved).to_dict()

    @router.get("/memory/{project_id}")
    async def memory(project_id: str, storage: str):
        memory = ProjectMemory(storage)
        return memory.inspect(project_id)

    @router.get("/tasks/{task_id}/report")
    async def report(task_id: str):
        record = _tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return engineering_report(record["task"])

    return router
