"""Focused tests for Python 3.11 compatibility evidence.

The tests never install packages, contact a network service, train, or publish.
They verify that the report records syntax evidence without overclaiming
runtime qualification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def test_requires_python_and_ci_target_are_recorded():
    from evaluation.python_compat import _ci_python_versions, _read_requires_python

    assert _read_requires_python().startswith(">=")
    assert "3.11" in _ci_python_versions()


def test_current_interpreter_syntax_check_passes():
    from evaluation.python_compat import check

    report = check(roots=("taksh",))
    assert report["syntax_check"]["status"] == "PASS"
    assert report["syntax_check"]["exit_code"] == 0
    assert report["syntax_check"]["interpreter_version"]
    assert report["status"] in ALLOWED
    assert report["status"] == "PARTIAL"
    assert report["verdict"] == "SYNTAX_COMPATIBLE_RUNTIME_UNVERIFIED"


def test_runtime_tests_are_not_overclaimed():
    from evaluation.python_compat import check

    report = check(roots=("taksh",))
    assert report["runtime_tests"]["status"] == "NOT EVALUATED"
    assert "no packages were installed" in report["runtime_tests"]["reason"]
    assert report["policy"] == {
        "dependency_install": False,
        "network_access": "none",
        "training_authorized": False,
        "compute_authorized": False,
        "publication_authorized": False,
        "syntax_check_is_not_runtime_qualification": True,
    }


def test_interpreter_path_is_portable():
    from evaluation.python_compat import _display_path

    home = Path.home()
    shown = _display_path(home / "tools" / "python311" / "python.exe")

    assert shown.startswith("~/")
    assert home.as_posix() not in shown


def test_report_does_not_embed_absolute_home_path():
    from evaluation.python_compat import check

    report = check(roots=("taksh",))
    syntax = report["syntax_check"]

    assert Path.home().as_posix() not in syntax["interpreter"]
    assert Path.home().as_posix() not in " ".join(syntax["command"])


def test_missing_interpreter_is_blocked_not_a_crash():
    from evaluation.python_compat import check

    report = check(interpreter="definitely-not-a-python", roots=("taksh",))
    assert report["syntax_check"]["status"] == "BLOCKED"
    assert report["status"] == "BLOCKED"
    assert report["verdict"] == "SYNTAX_CHECK_BLOCKED"
    assert report["syntax_check"]["interpreter_error"]


def test_check_is_deterministic():
    from evaluation.python_compat import check

    first = check(roots=("taksh",))
    second = check(roots=("taksh",))
    assert first == second


def test_cli_writes_report(tmp_path, capsys):
    from evaluation.python_compat import main

    out = tmp_path / "compat.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 0
    assert out.is_file()
    assert "syntax check: PASS" in captured.out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "taksh.python_compat/1"
    assert report["target"] == "Python 3.11"
    assert report["runtime_tests"]["status"] == "NOT EVALUATED"
