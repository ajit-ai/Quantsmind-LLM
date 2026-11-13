"""Dependency vulnerability scan workflow.

Attempts local scans and records results honestly. Network-unavailable or
scanner failures are recorded as BLOCKED rather than reported as clean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _run(command: list[str], output: Path, timeout: int = 300) -> tuple[int | None, str | None]:
    try:
        proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        output.write_text(f"scan failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return None, f"{type(exc).__name__}: {exc}"
    output.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return proc.returncode, None


def scan() -> dict[str, Any]:
    pip_report = REPO / "reports" / "taksh_pip_audit.json"
    npm_report = REPO / "reports" / "taksh_npm_audit.json"
    pip_error_file = REPO / "reports" / "taksh_pip_audit.error.log"

    pip_rc, pip_error = _run(
        [sys.executable, "-m", "pip_audit", "--requirement", "requirements.txt", "--format", "json", "--output", str(pip_report)],
        pip_error_file,
    )

    pip_result = "PASS" if pip_rc == 0 and pip_report.is_file() else "BLOCKED"
    if pip_rc is not None and pip_report.is_file():
        try:
            data = json.loads(pip_report.read_text(encoding="utf-8"))
            findings = data.get("dependencies", []) if isinstance(data, dict) else data
        except Exception:
            findings = []
    else:
        findings = []

    npm = None
    if npm_report.is_file():
        try:
            npm = json.loads(npm_report.read_text(encoding="utf-8"))
        except Exception:
            npm = {"error": "unreadable npm audit output"}

    return {
        "schema": "taksh.dependency_scan/1",
        "pip_audit": {
            "status": pip_result,
            "error": pip_error,
            "findings": findings,
            "requirements": "requirements.txt",
            "note": "No vulnerability scan result is claimed if the advisory source is unreachable.",
        },
        "npm_audit": npm,
        "verdict": "SCAN_BLOCKED" if pip_result == "BLOCKED" else "SCAN_RECORDED",
        "note": "This workflow is reproducible and does not transmit source code or credentials.",
    }


def main() -> int:
    out_path = REPO / "reports" / "taksh_dependency_scan.json"
    report = scan()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pip_audit": report["pip_audit"]["status"], "verdict": report["verdict"]}, indent=2))
    return 0 if report["verdict"] == "SCAN_RECORDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
