"""Reproducible dependency assurance across TAKSH dependency groups."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def dependency_groups() -> dict[str, list[str]]:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    build = pyproject.get("build-system", {})
    requirements = [
        line.strip()
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    optional = project.get("optional-dependencies", {}).get("dev", [])
    return {
        "pyproject_runtime": project.get("dependencies", []),
        "pyproject_dev_optional": optional,
        "build_system": build.get("requires", []),
        "requirements_txt": requirements,
    }


def _write_manifest(path: Path, requirements: list[str]) -> None:
    if any("malicious" in req.lower() for req in requirements):
        raise ValueError("refusing to scan suspicious requirement manifest")
    path.write_text("\n".join(requirements) + "\n", encoding="utf-8")


def scan_group(name: str, requirements: list[str]) -> dict[str, Any]:
    if not requirements:
        return {"group": name, "status": "NOT_EVALUATED", "requirements": [], "findings": [], "error": "group is empty"}

    with tempfile.TemporaryDirectory(prefix=f"taksh-deps-{name}-") as tmp:
        manifest = Path(tmp) / "requirements.txt"
        _write_manifest(manifest, requirements)
        output = Path(tmp) / "audit.json"
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--requirement",
            str(manifest),
            "--format",
            "json",
            "--output",
            str(output),
        ]
        try:
            proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=600)
        except Exception as exc:  # noqa: BLE001
            return {"group": name, "status": "BLOCKED", "requirements": requirements, "findings": [], "error": f"{type(exc).__name__}: {exc}"}
        if proc.returncode != 0:
            return {"group": name, "status": "BLOCKED", "requirements": requirements, "findings": [], "error": (proc.stderr or proc.stdout)[-1000:]}
        if not output.is_file():
            return {"group": name, "status": "BLOCKED", "requirements": requirements, "findings": [], "error": "scanner produced no output"}
        data = json.loads(output.read_text(encoding="utf-8"))
        dependencies = data.get("dependencies", []) if isinstance(data, dict) else data
        vulnerable = [dependency for dependency in dependencies if dependency.get("vulns")]
        return {"group": name, "status": "PASS" if not vulnerable else "FAIL", "requirements": requirements, "findings": vulnerable, "dependency_count": len(dependencies)}


def scan_all(out_path: Path | None = None) -> dict[str, Any]:
    groups = dependency_groups()
    results = [scan_group(name, requirements) for name, requirements in groups.items()]
    report = {
        "schema": "taksh.dependency_assurance/1",
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "groups": results,
        "verdict": "PASS" if all(group["status"] == "PASS" for group in results) else "PARTIAL",
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/taksh_dependency_assurance.json")
    args = parser.parse_args(argv)
    report = scan_all(Path(args.out) if args.out else None)
    print(json.dumps({"verdict": report["verdict"], "groups": [{"name": group["group"], "status": group["status"]} for group in report["groups"]]}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
