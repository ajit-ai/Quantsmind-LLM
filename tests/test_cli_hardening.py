"""Tests for v1.5.0 CLI hardening.

Verifies that the ``taksh`` CLI:
1. Reports version correctly via ``--version``.
2. Exits cleanly (code 2, no traceback) on unrecognized arguments.
3. Exits cleanly (code 0) with no arguments.
4. Package version remains ``1.0.0`` (unchanged for v1.5.0).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import taksh


PYTHON = sys.executable
REPO = Path(__file__).resolve().parent.parent


def run_taksh(*args):
    """Run ``python -m taksh <args>`` and return (returncode, stdout, stderr)."""
    completed = subprocess.run(
        [PYTHON, "-m", "taksh", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.returncode, completed.stdout, completed.stderr


def test_version_flag():
    """``taksh --version`` prints version and exits 0."""
    rc, out, err = run_taksh("--version")
    assert rc == 0, f"expected exit 0, got {rc}; stderr={err!r}"
    assert "Taksh 1.0.0 (final)" in out, f"unexpected version output: {out!r}"


def test_invalid_args_exit_clean():
    """Unrecognized arguments exit code 2 with a usage message, no traceback."""
    rc, out, err = run_taksh("--invalid-flag-xyz")
    assert rc == 2, f"expected exit 2, got {rc}; stdout={out!r}"
    combined = out + err
    assert "unrecognized arguments" in combined or "usage:" in combined, (
        f"expected usage/error message, got: {combined!r}"
    )
    assert "Traceback" not in combined, f"traceback leaked: {combined!r}"


def test_no_args_clean_exit():
    """No arguments exits 0 without crashing or printing a traceback."""
    rc, out, err = run_taksh()
    assert rc == 0, f"expected exit 0, got {rc}; stderr={err!r}"
    assert "Traceback" not in out + err, f"traceback leaked: {(out + err)!r}"


def test_package_version_unchanged():
    """Package version remains 1.0.0 (v1.5.0 does not change package version)."""
    assert taksh.__version__ == "1.0.0"
    assert taksh.__release_status__ == "final"


def test_entry_point_is_main():
    """``taksh.main`` exists and is callable (CLI entry point)."""
    from taksh import main

    assert callable(main)
