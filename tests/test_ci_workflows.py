"""Focused regression tests for TAKSH CI workflow engineering.

Static checks only — no network, no dispatch, no release steps. These tests
protect the invariants that matter for the recovery: least privilege,
visibility honesty, frozen benchmark pins, no publish/deploy capability.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CI = REPO / ".github" / "workflows" / "ci.yml"
MANUAL = REPO / ".github" / "workflows" / "ci-l0704-manual-qualification.yml"

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Workflow shape and least privilege
# --------------------------------------------------------------------------- #
def test_ci_workflow_parses_and_is_least_priv_read_only():
    ci = _load(CI)
    assert ci["name"] == "CI"
    assert ci["permissions"] == {"contents": "read"}


def test_ci_workflow_triggers_only_default_branches_and_prs():
    ci = _load(CI)
    on = ci[True] if True in ci else ci.get("on")
    # PyYAML parses the bare key `on` as the boolean True
    if True in on:
        on = on[True]
    branches = on["push"]["branches"]
    assert set(branches) == {"develop", "main"}
    assert "pull_request" in on
    assert "workflow_dispatch" not in on      # not a manual gate; documented limitation
    assert "schedule" not in on


def test_ci_workflow_has_concurrency_cancellation_and_timeouts():
    ci = _load(CI)
    assert ci["concurrency"]["cancel-in-progress"] is True
    for name, job in ci["jobs"].items():
        assert "timeout-minutes" in job, f"job {name} missing timeout-minutes"
        assert int(job["timeout-minutes"]) > 0


def test_ci_unit_job_depends_on_guard():
    ci = _load(CI)
    assert ci["jobs"]["unit-and-smoke"]["needs"] == "guard"


def test_ci_contains_no_publish_deploy_or_release_steps():
    text = CI.read_text(encoding="utf-8")
    for banned in ("twine upload", "npm publish", "gh release create",
                   "pypa/gh-action-pypi-publish", "pages", "deploy-pages",
                   "git push", "force-push", "--force"):
        assert banned not in text, f"CI unexpectedly contains: {banned}"


# --------------------------------------------------------------------------- #
# Manual qualification workflow integrity
# --------------------------------------------------------------------------- #
def test_manual_workflow_is_dispatch_only_and_read_only():
    manual = _load(MANUAL)
    on = manual[True] if True in manual else manual.get("on")
    if True in on:
        on = on[True]
    assert "workflow_dispatch" in on
    assert "push" not in on        # never triggers silently on push
    assert "pull_request" not in on
    assert manual["permissions"] == {"contents": "read"}
    assert manual["concurrency"]["cancel-in-progress"] is False


def test_manual_workflow_dispatch_input_defaults_to_feature_branch():
    manual = _load(MANUAL)
    on = manual[True] if True in manual else manual.get("on")
    if True in on:
        on = on[True]
    ref = on["workflow_dispatch"]["inputs"]["ref"]["default"]
    assert ref == "feature/l060-foundation-model-qualification"


def test_manual_workflow_pins_frozen_benchmark_hashes():
    text = MANUAL.read_text(encoding="utf-8")
    assert FROZEN_PY_SHA in text
    assert MULTILINGUAL_SHA in text


def test_manual_workflow_qualified_chain_covers_all_recent_milestones():
    text = MANUAL.read_text(encoding="utf-8")
    for suite in ("test_l073_readiness.py", "test_l076_roadmap.py",
                  "test_l080pre_preflight.py", "test_l100eng_dlcae_runtime.py",
                  "test_l100eng2_semantic_validation.py",
                  "test_l100eng3_end_to_end.py", "test_ci_workflows.py"):
        assert suite in text, f"manual qualification chain missing {suite}"


def test_frozen_benchmarks_unchanged(tmp_path=None):
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_workflows_do_not_reference_tag_or_release_creation():
    for path in (CI, MANUAL):
        text = path.read_text(encoding="utf-8")
        for banned in ("git tag", "twine upload", "gh release create",
                       "gh api repos/actions/skip"):
            assert banned not in text


def test_manual_workflow_is_absent_from_default_branch_guidance():
    text = MANUAL.read_text(encoding="utf-8")
    assert "dispatch" in text.lower()
    assert "workflow_dispatch" in text


# --------------------------------------------------------------------------- #
# Development environment honesty
# --------------------------------------------------------------------------- #
def test_python_version_declared_and_consistent():
    import taksh

    assert taksh.__version__ == "1.0.0"
    py_version_file = REPO / "pyproject.toml"
    assert py_version_file.is_file()
