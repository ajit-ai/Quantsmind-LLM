"""Deployment and operational helpers for TAKSH."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from taksh.engineering import inspect_project, validate_project


def validate_environment() -> dict[str, Any]:
    checks = {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "project_root_exists": Path(".").resolve().is_dir(),
    }
    required_files = [
        "configs/model/foundation_111m.yaml",
        "tokenizer/qm_code_bpe_v0.json",
    ]
    for relative in required_files:
        checks[relative] = Path(relative).is_file()
    checks["all_required_files_present"] = all(checks[relative] for relative in required_files)
    return checks


def validate_deployment_config(config: dict) -> dict[str, Any]:
    errors: list[str] = []
    token = config.get("TAKSH_ENGINEERING_TOKEN", "")
    if not isinstance(token, str) or len(token) < 16:
        errors.append("TAKSH_ENGINEERING_TOKEN must be set to at least 16 non-placeholder characters")
    if token in {"replace-with-strong-random-token", "changeme", "dev-token", "test-token"}:
        errors.append("TAKSH_ENGINEERING_TOKEN is a placeholder")
    scopes = config.get("TAKSH_TOKEN_SCOPES", "")
    allowed = {"read", "execute", "mutate", "memory", "audit"}
    if not scopes:
        errors.append("TAKSH_TOKEN_SCOPES must be set explicitly")
    else:
        unknown = {scope.strip() for scope in scopes.split(",") if scope.strip()} - allowed
        if unknown:
            errors.append(f"unsupported scopes: {sorted(unknown)}")
    root = config.get("TAKSH_WORKSPACE", ".")
    if not Path(root).is_dir():
        errors.append(f"TAKSH_WORKSPACE does not exist: {root}")
    return {"status": "valid" if not errors else "invalid", "errors": errors}


def backup_state(memory_path: str | Path, backup_path: str | Path) -> dict[str, Any]:
    import hashlib, shutil

    source = Path(memory_path).resolve()
    destination = Path(backup_path).resolve()
    if not source.is_file():
        return {"status": "missing", "detail": f"no state file at {source}"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "status": "backed_up",
        "source": str(source),
        "destination": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
    }


def restore_state(backup_path: str | Path, memory_path: str | Path) -> dict[str, Any]:
    import hashlib, json, shutil

    source = Path(backup_path).resolve()
    destination = Path(memory_path).resolve()
    if not source.is_file():
        return {"status": "missing", "detail": f"no backup at {source}"}
    try:
        json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "invalid", "detail": f"backup is not valid JSON: {exc}"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "status": "restored",
        "source": str(source),
        "destination": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def health_check(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    return {
        "status": "ok",
        "root": str(root_path),
        "root_exists": root_path.is_dir(),
        "version": _version(),
    }


def readiness_check(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    env = validate_environment()
    project = inspect_project(root_path)
    return {
        "status": "ready" if env["all_required_files_present"] and root_path.is_dir() else "not_ready",
        "environment": env,
        "project": {
            "files": project["summary"]["total_files"],
            "python_files": project["summary"]["python_files"],
            "tests": project["test_count"],
        },
    }


def graceful_shutdown_report(active_tasks: int = 0) -> dict[str, Any]:
    return {
        "status": "shutdown",
        "active_tasks": active_tasks,
        "timestamp": time.time(),
    }


def _version() -> str:
    try:
        from taksh import __version__

        return __version__
    except Exception:
        return "unknown"
