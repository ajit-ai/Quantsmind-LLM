"""Record Python 3.11 compatibility evidence without installing dependencies.

The project targets Python >=3.10 and CI runs its unit suite on Python 3.11.
This module adds a reproducible local syntax check under a chosen interpreter
and records the result honestly: compilation is not a runtime-test claim.

It never installs packages, contacts a network service, trains, creates a
checkpoint, or publishes anything.

Run:

    python -m evaluation.python_compat --interpreter C:/Python311/python.exe
    python -m evaluation.python_compat --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_OUT = "reports/taksh_python311_compat.json"
CI_WORKFLOW = ".github/workflows/ci.yml"
SOURCE_ROOTS = ("taksh", "evaluation", "serving", "training", "scripts", "tests")

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _display_path(value: str | Path) -> str:
    """Render a path without embedding a machine-specific home directory."""
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except (ValueError, OSError):
        pass
    try:
        relative = path.resolve().relative_to(Path.home().resolve())
        return "~/" + relative.as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _read_requires_python() -> str:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python <3.11
        raise RuntimeError("Python 3.11+ is required to read pyproject.toml") from exc
    document = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document.get("project", {}).get("requires-python", "unknown"))


def _ci_python_versions() -> list[str]:
    path = REPO / CI_WORKFLOW
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    values = re.findall(
        r"python-version:\s*[\"']?([0-9]+(?:\.[0-9]+)*)", text
    )
    return sorted(set(values))


def _version_of(interpreter: str) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [interpreter, "--version"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    text = (completed.stdout or completed.stderr or "").strip()
    return (text or None), None


def _compileall(
    interpreter: str,
    roots: Sequence[str],
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    command = [interpreter, "-m", "compileall", "-q", *roots]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "status": "BLOCKED",
            "exit_code": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    stderr = (completed.stderr or "").strip()
    return {
        "command": command,
        "status": "PASS" if completed.returncode == 0 else "BLOCKED",
        "exit_code": completed.returncode,
        "stderr_tail": stderr[-2000:] if stderr else None,
    }


def check(
    *,
    interpreter: str | None = None,
    roots: Sequence[str] = SOURCE_ROOTS,
) -> dict[str, Any]:
    """Return deterministic compatibility evidence; never installs packages."""
    interpreter = interpreter or sys.executable
    interpreter_version, interpreter_error = _version_of(interpreter)
    compile_result = _compileall(interpreter, roots)
    displayed_interpreter = _display_path(interpreter)
    compile_result["command"] = [
        displayed_interpreter,
        *compile_result["command"][1:],
    ]

    if compile_result["status"] == "PASS":
        verdict = "SYNTAX_COMPATIBLE_RUNTIME_UNVERIFIED"
        overall = "PARTIAL"
    else:
        verdict = "SYNTAX_CHECK_BLOCKED"
        overall = "BLOCKED"
    if overall not in ALLOWED_STATUSES:
        raise RuntimeError(f"invalid status: {overall}")

    return {
        "schema": "taksh.python_compat/1",
        "generated_by": "evaluation/python_compat.py",
        "target": "Python 3.11",
        "requires_python": _read_requires_python(),
        "ci": {
            "workflow": CI_WORKFLOW,
            "python_versions": _ci_python_versions(),
            "status": "CONFIGURED",
            "note": (
                "CI unit-and-smoke is configured for Python 3.11, but this "
                "feature-branch push does not trigger remote CI."
            ),
        },
        "syntax_check": {
            "interpreter": displayed_interpreter,
            "interpreter_version": interpreter_version,
            "interpreter_error": interpreter_error,
            "roots": list(roots),
            **compile_result,
        },
        "runtime_tests": {
            "status": "NOT EVALUATED",
            "reason": (
                "Local Python 3.11 lacks the project test dependencies; no "
                "packages were installed and no remote branch CI was run."
            ),
        },
        "status": overall,
        "verdict": verdict,
        "policy": {
            "dependency_install": False,
            "network_access": "none",
            "training_authorized": False,
            "compute_authorized": False,
            "publication_authorized": False,
            "syntax_check_is_not_runtime_qualification": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpreter",
        default=sys.executable,
        help="Python interpreter to syntax-check with (default: current)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = check(interpreter=args.interpreter)
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        syntax = report["syntax_check"]
        print(f"target: {report['target']}")
        print(f"interpreter: {syntax['interpreter_version'] or syntax['interpreter']}")
        print(f"syntax check: {syntax['status']}")
        print(f"runtime tests: {report['runtime_tests']['status']}")
        print(f"verdict: {report['verdict']}")
        print(f"report: {out_path}")
    return 0 if report["syntax_check"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
