"""Dependency and licensing inventory for TAKSH.

This is an inventory, not a vulnerability scan. It does not contact any package
index or network service.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def dependency_inventory() -> dict[str, Any]:
    pyproject = REPO / "pyproject.toml"
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = document.get("project", {})
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {
        "schema": "taksh.dependency_inventory/1",
        "pyproject_dependencies": project.get("dependencies", []),
        "optional_dependencies": project.get("optional-dependencies", {}),
        "requirements_txt": [line.strip() for line in requirements if line.strip() and not line.strip().startswith("#")],
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256((REPO / "requirements.txt").read_bytes()).hexdigest(),
        "vulnerability_scan": "NOT_EVALUATED",
        "license_review": "NOT_EVALUATED",
        "note": "No network access or package index was used to generate this inventory.",
    }
