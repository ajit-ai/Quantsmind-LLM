"""Release metadata stays consistent with the final M7 v1.0.0 Apache release."""

import subprocess
import sys
import tomllib
from pathlib import Path

from serving import openai_server
from taksh import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs" / "source"


def test_package_version_is_final():
    assert __version__ == "1.0.0"


def test_pyproject_version_comes_from_package():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert project["name"] == "taksh"
    assert "version" in project["dynamic"]
    assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "taksh.__version__"


def test_sphinx_version_comes_from_package():
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert "from taksh import __version__" in conf
    assert "release = __version__" in conf
    assert 'release = "0.7.0"' not in conf


def test_serving_reports_package_version():
    assert openai_server.app.version == __version__


def test_current_release_docs_use_v1_0_0():
    assert f"v{__version__}" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for path in sorted(DOCS.glob("roadmap/M7_*.rst")):
        text = path.read_text(encoding="utf-8")
        assert f"v{__version__}" in text, path
        assert "Target version: v0.7.0" not in text, path


def test_no_extra_release_milestone():
    roadmap = (REPO_ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "M8 | **Release" not in roadmap


def test_release_readiness_report_records_authorized_publication():
    report = DOCS / "releases" / "M7_RELEASE_READINESS.rst"
    text = report.read_text(encoding="utf-8")

    assert "PUBLICATION AUTHORIZED" in text
    assert f"v{__version__}" in text
    assert "0/40" in text
    assert "Full test suite" in text


def test_version_consistency_check_passes():
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_version_consistency.py")],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Version consistency: PASS" in completed.stdout
    assert f"taksh.__version__ = {__version__}" in completed.stdout


def test_changelog_discloses_negative_capability_and_licence():
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "0/40 locked syntax-valid generations" in text
    assert "0/8 functional tasks" in text
    assert "Apache-2.0" in text
    assert "NOTICE" in text
