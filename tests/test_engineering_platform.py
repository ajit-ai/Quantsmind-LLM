"""Focused tests for the integrated TAKSH engineering platform.

These tests exercise real multi-file workflows in temporary workspaces. They use
an explicitly identified deterministic provider and do not claim model
capability.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.engineering import (
    ChangeOperation,
    ChangePlan,
    DeterministicTestProvider,
    EngineeringTask,
    FileChange,
    SecurityError,
    native_change_provider,
    apply_plan,
    create_task,
    discover_tests,
    inspect_project,
    plan_task,
    review_plan,
    run_engineering_task,
    run_tests,
    validate_project,
    validate_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "pkg" / "math.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (root / "pkg" / "utils.py").write_text(
        "def shout(text):\n    return text.upper()\n", encoding="utf-8"
    )
    (root / "tests" / "test_math.py").write_text(
        "from pkg.math import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("unrelated\n", encoding="utf-8")
    return root


def test_project_inspection(tmp_path):
    root = _make_project(tmp_path)
    info = inspect_project(root)
    assert info["summary"]["python_files"] == 3
    assert "pkg/math.py" not in info["tests"]
    assert "tests/test_math.py" in info["tests"]
    assert "pkg/math.py" in info["symbols"]["symbols"]
    assert info["dependencies"]


def test_safe_path_resolution_blocks_traversal_and_escape(tmp_path):
    root = _make_project(tmp_path)
    from taksh.engineering import _resolve

    with pytest.raises(SecurityError):
        _resolve(root, "../secret.txt")
    with pytest.raises(SecurityError):
        _resolve(root, "/etc/passwd")
    assert _resolve(root, "pkg/math.py").is_file()


def test_symlink_escape_is_blocked(tmp_path):
    root = _make_project(tmp_path)
    from taksh.engineering import _resolve

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "pkg" / "escape.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(SecurityError):
        _resolve(root, "pkg/escape.py")


def test_patch_application_creates_modifies_and_deletes(tmp_path):
    root = _make_project(tmp_path)
    changes = [
        FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b + 0\n", reason="adjust"),
        FileChange(path="pkg/new.py", operation="create", content="VALUE = 1\n", reason="add module"),
        FileChange(path="pkg/utils.py", operation="delete", reason="remove"),
    ]
    plan = ChangePlan(task_id="t1", root=str(root), changes=changes, approved=True)
    result = apply_plan(root, plan, approved=True)
    assert result.status == "APPLIED"
    assert (root / "pkg" / "new.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (root / "pkg" / "utils.py").exists()
    assert "return a + b + 0" in (root / "pkg" / "math.py").read_text(encoding="utf-8")


def test_conflicting_modification_is_detected(tmp_path):
    root = _make_project(tmp_path)
    original = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    expected = __import__("hashlib").sha256(original.encode()).hexdigest()
    change = FileChange(path="pkg/math.py", operation="modify", content="def add(a,b): return a+b\n", expected_sha256=expected, reason="change")
    plan = ChangePlan(task_id="t2", root=str(root), changes=[change], approved=True)
    # Mutate file after planning.
    (root / "pkg" / "math.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    result = apply_plan(root, plan, approved=True)
    assert result.status == "FAILED"
    assert "conflict" in (result.error or "").lower()
    # Original mutated content remains (rollback restored snapshot).
    assert "return a - b" in (root / "pkg" / "math.py").read_text(encoding="utf-8")


def test_failed_multi_file_application_rolls_back(tmp_path):
    root = _make_project(tmp_path)
    before_math = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    before_utils = (root / "pkg" / "utils.py").read_text(encoding="utf-8")
    changes = [
        FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b + 1\n", reason="first"),
        FileChange(path="pkg/missing.py", operation="modify", content="x = 1\n", reason="second fails"),
    ]
    plan = ChangePlan(task_id="t3", root=str(root), changes=changes, approved=True)
    result = apply_plan(root, plan, approved=True)
    assert result.status == "FAILED"
    assert result.rolled_back is True
    assert (root / "pkg" / "math.py").read_text(encoding="utf-8") == before_math
    assert (root / "pkg" / "utils.py").read_text(encoding="utf-8") == before_utils


def test_unapproved_plan_is_not_applied(tmp_path):
    root = _make_project(tmp_path)
    changes = [FileChange(path="pkg/math.py", operation="modify", content="def add(a,b): return a+b\n", reason="x")]
    plan = ChangePlan(task_id="t4", root=str(root), changes=changes, approved=False)
    result = apply_plan(root, plan, approved=False)
    assert result.status == "BLOCKED"


def test_applying_without_reviewer_is_blocked(tmp_path):
    root = _make_project(tmp_path)
    before = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    provider = DeterministicTestProvider([
        FileChange(path="pkg/math.py", operation="modify", content="def add(a,b): return a+b\n", reason="x")
    ])
    task = create_task("no-reviewer", "change add", root)
    with pytest.raises(SecurityError, match="explicit reviewer approval"):
        run_engineering_task(task, provider, apply=True)
    assert (root / "pkg" / "math.py").read_text(encoding="utf-8") == before


def test_reviewer_rejection_does_not_apply(tmp_path):
    root = _make_project(tmp_path)
    before = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    provider = DeterministicTestProvider([
        FileChange(path="pkg/math.py", operation="modify", content="def add(a,b): return a+b\n", reason="x")
    ])
    task = create_task("rejected", "change add", root)
    task = run_engineering_task(task, provider, apply=True, reviewer=lambda plan: False)
    assert task.state.value == "PLANNED"
    assert (root / "pkg" / "math.py").read_text(encoding="utf-8") == before


def test_validation_failure_rolls_back_applied_change(tmp_path):
    root = _make_project(tmp_path)
    before = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    provider = DeterministicTestProvider([
        FileChange(
            path="pkg/math.py",
            operation="modify",
            content="def add(a, b)\n    return a + b\n",
            reason="intentionally invalid",
        )
    ])
    task = create_task("rollback", "break add", root)
    task = run_engineering_task(
        task, provider, apply=True, validate=True, reviewer=lambda plan: True
    )
    assert task.state.value == "FAILED"
    assert task.validation["status"] == "FAIL"
    assert task.validation["rolled_back"] is True
    assert (root / "pkg" / "math.py").read_text(encoding="utf-8") == before


def test_invalid_patch_rejected(tmp_path):
    root = _make_project(tmp_path)
    changes = [FileChange(path="../escape.py", operation="create", content="x=1\n", reason="escape")]
    plan = ChangePlan(task_id="t5", root=str(root), changes=changes, approved=True)
    result = apply_plan(root, plan, approved=True)
    assert result.status == "FAILED"
    assert "traversal" in (result.error or "").lower() or "escape" in (result.error or "").lower()


def test_validation_and_test_execution(tmp_path):
    root = _make_project(tmp_path)
    source = (root / "pkg" / "math.py").read_text(encoding="utf-8")
    result = validate_source(source)
    assert result["status"] == "PASS"
    validation = validate_project(root)
    assert validation["status"] == "PASS"
    tests = run_tests(root)
    assert tests["status"] == "PASS"
    assert "test_math.py" in " ".join(tests["tests"])


def test_bounded_repair_loop(tmp_path):
    root = _make_project(tmp_path)
    # Start with a broken function.
    (root / "pkg" / "math.py").write_text("def add(a, b)\n    return a + b\n", encoding="utf-8")

    def repair_provider(task, plan, validation):
        return ChangePlan(
            task_id=task.task_id,
            root=task.root,
            changes=[FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b\n", reason="repair syntax")],
            approved=True,
        )

    from taksh.engineering import run_repair_loop, apply_plan as _apply, validate_project as _validate

    plan = ChangePlan(task_id="repair", root=str(root), changes=[], approved=True)
    result = run_repair_loop(
        create_task("repair", "fix syntax", root),
        plan,
        lambda p: _apply(root, p, approved=True),
        lambda: _validate(root, ["pkg/math.py"]),
        repair_provider,
        max_iterations=2,
    )
    assert result["status"] == "PASS"
    assert (root / "pkg" / "math.py").read_text(encoding="utf-8").startswith("def add")


def test_provider_unavailable_fails_clearly(tmp_path):
    root = _make_project(tmp_path)
    task = create_task("t6", "do something", root)
    task, plan = plan_task(task, native_change_provider())
    assert plan is None
    assert task.state.value == "BLOCKED"


def test_end_to_end_multi_file_engineering(tmp_path):
    root = _make_project(tmp_path)
    readme_before = _sha256(root / "README.md")
    utils_before = _sha256(root / "pkg" / "utils.py")

    changes = [
        FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n", reason="add multiply"),
        FileChange(path="pkg/new_module.py", operation="create", content="from pkg.math import multiply\n\n\ndef triple(x):\n    return multiply(x, 3)\n", reason="add new module"),
        FileChange(path="tests/test_new.py", operation="create", content="from pkg.new_module import triple\n\n\ndef test_triple():\n    assert triple(4) == 12\n", reason="add test"),
    ]
    provider = DeterministicTestProvider(changes, summary="deterministic multi-file change")
    task = create_task("e2e", "add multiply and triple", root, max_repair_iterations=2)
    task = run_engineering_task(
        task, provider, apply=True, validate=True, run_project_tests=True,
        reviewer=lambda plan: True,
    )

    assert task.state.value == "COMPLETED"
    assert (root / "pkg" / "new_module.py").is_file()
    assert "multiply" in (root / "pkg" / "math.py").read_text(encoding="utf-8")
    assert (root / "tests" / "test_new.py").is_file()
    assert _sha256(root / "README.md") == readme_before
    assert _sha256(root / "pkg" / "utils.py") == utils_before
    assert task.validation["status"] == "PASS"
    assert task.test_result["status"] == "PASS"
    assert task.capability_claim is None


def test_specialist_routing_and_plan(tmp_path):
    root = _make_project(tmp_path)
    changes = [FileChange(path="pkg/math.py", operation="modify", content="def add(a,b): return a+b\n", reason="x")]
    provider = DeterministicTestProvider(changes)
    task = create_task("spec", "debug and test the add function", root)
    task, plan = plan_task(task, provider)
    assert plan is not None
    assert "test_failure_diagnosis" in (task.context.get("specialists") or {})
    assert plan.specialist is not None


def test_engineering_report_does_not_claim_capability(tmp_path):
    from taksh.engineering import engineering_report, create_task

    task = create_task("r", "objective", tmp_path)
    report = engineering_report(task)
    assert report["capability_claim"] is None
    assert report["native_model_capability"] == "NOT_EVALUATED"
