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
