"""Version and release-artifact reproducibility checks for TAKSH."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from taksh import __version__

REPO = Path(__file__).resolve().parent.parent


def check_version_consistency() -> dict[str, Any]:
    pyproject = REPO / "pyproject.toml"
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = document.get("project", {}).get("dependencies", [])
    return {
        "package_version": __version__,
        "pyproject_name": document.get("project", {}).get("name"),
        "requires_python": document.get("project", {}).get("requires-python"),
        "dependency_count": len(dependencies),
        "release_status": "not_released",
    }


def source_manifest() -> dict[str, Any]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", "runs", "dist", "build", ".taksh"}
    files = []
    for path in sorted(REPO.rglob("*")):
        if path.is_dir():
            continue
        if any(part in ignored for part in path.parts):
            continue
        relative = path.relative_to(REPO).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return {
        "schema": "taksh.source_manifest/1",
        "package_version": __version__,
        "file_count": len(files),
        "files": files,
    }


def verify_release_artifacts() -> dict[str, Any]:
    required = ["LICENSE", "NOTICE", "README.md", "pyproject.toml"]
    result = {item: (REPO / item).is_file() for item in required}
    result["all_present"] = all(result.values())
    result["package_version"] = __version__
    return result
