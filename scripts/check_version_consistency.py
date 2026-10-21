"""Verify that every public release declaration matches taksh.__version__.

The authoritative version is ``taksh/__init__.py::__version__``. This script
fails when packaging metadata, runtime declarations, documentation or historical
release records disagree with it.

It is read-only: it never builds, tags, commits, publishes or uploads anything.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
CURRENT_M7_DOCS = sorted((REPO_ROOT / "docs" / "source" / "roadmap").glob("M7_*.rst"))
HISTORICAL_REPORTS = (
    REPO_ROOT / "reports" / "M7_BASELINE_REPORT.rst",
    REPO_ROOT / "reports" / "M7_CANDIDATE_REPORT.rst",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _require(errors: list[str], root: Path, relative: str, needles: tuple[str, ...]) -> None:
    path = root / relative
    if not path.is_file():
        errors.append(f"{relative}: missing")
        return
    text = _read(path)
    for needle in needles:
        if needle not in text:
            errors.append(f"{relative}: missing {needle!r}")


def _forbid(errors: list[str], root: Path, relative: str, needles: tuple[str, ...]) -> None:
    path = root / relative
    if not path.is_file():
        return
    text = _read(path)
    for needle in needles:
        if needle in text:
            errors.append(f"{relative}: forbidden stale declaration {needle!r}")


def _parse_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - release tooling needs Python 3.11+
        raise SystemExit("version checks require Python 3.11+ (tomllib)")
    return tomllib.loads(_read(path))


def _declared_version(init_text: str) -> tuple[str, str]:
    tree = ast.parse(init_text)
    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    values[target.id] = node.value.value
    try:
        return values["__version__"], values["__release_status__"]
    except KeyError as exc:
        raise SystemExit(f"taksh/__init__.py is missing {exc.args[0]}") from exc


def check(root: Path = REPO_ROOT, *, check_cli: bool = True) -> tuple[list[str], str, str]:
    root = root.resolve()
    errors: list[str] = []

    init_path = root / "taksh" / "__init__.py"
    if not init_path.is_file():
        return [f"{init_path}: missing authoritative version module"], "", ""
    version, release_status = _declared_version(_read(init_path))
    if not SEMVER.fullmatch(version):
        errors.append(f"taksh/__init__.py: __version__ {version!r} is not MAJOR.MINOR.PATCH")
    if not release_status:
        errors.append("taksh/__init__.py: __release_status__ is empty")

    pyproject = _parse_toml(root / "pyproject.toml")
    project = pyproject.get("project", {})
    if project.get("name") != "taksh":
        errors.append(f"pyproject.toml: project.name is {project.get('name')!r}, expected 'taksh'")
    if "version" in project:
        errors.append("pyproject.toml: static project.version must not duplicate taksh.__version__")
    if "version" not in project.get("dynamic", []):
        errors.append("pyproject.toml: project.dynamic must contain 'version'")
    attr = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get(
        "version", {}
    ).get("attr")
    if attr != "taksh.__version__":
        errors.append(f"pyproject.toml: dynamic version attr is {attr!r}, expected 'taksh.__version__'")
    if project.get("license") not in ("Apache-2.0", {"text": "Apache-2.0"}):
        errors.append("pyproject.toml: license must be Apache-2.0")
    if "license-files" not in project:
        errors.append("pyproject.toml: license-files is missing")
    else:
        license_files = project.get("license-files", [])
        if "LICENSE" not in license_files or "NOTICE" not in license_files:
            errors.append("pyproject.toml: license-files must include LICENSE and NOTICE")
    _require(errors, root, "NOTICE", ("Copyright 2026 Ajit Kumar", "Apache"))
    _require(errors, root, "LICENSE", ("Apache License", "Version 2.0"))

    _require(
        errors,
        root,
        "docs/source/conf.py",
        ("from taksh import __version__", "version = __version__", "release = __version__"),
    )
    _require(
        errors,
        root,
        "serving/openai_server.py",
        ("from taksh import __version__", "version=__version__", '"version": __version__'),
    )
    _require(errors, root, "README.md", (f"v{version}", "taksh --version", "pip install ."))
    _require(
        errors,
        root,
        "docs/RELEASING.md",
        (f"current release is `{version}`", "status `final`", "twine upload"),
    )
    _require(errors, root, "MANIFEST.in", ("include CHANGELOG.md", "include NOTICE", "recursive-include configs"))
    _require(
        errors,
        root,
        "docs/VERSIONING.md",
        (f"| `{version}` | repository/package |", "Published only after"),
    )
    _require(errors, root, "docs/source/releases/index.rst", (f"`v{version}`",))
    _require(
        errors,
        root,
        "docs/source/releases/M7_RELEASE_READINESS.rst",
        (f"`v{version}`", f"`taksh=={version}`", "PUBLICATION AUTHORIZED"),
    )
    _require(errors, root, "docs/ROADMAP.md", (f"at v{version}", "PUBLICATION AUTHORIZED"))
    _forbid(errors, root, "docs/ROADMAP.md", ("M8 | **Release",))
    _require(errors, root, "docs/STATUS.md", (f"**v{version}**",))
    _require(errors, root, "CHANGELOG.md", (f"## [{version}]",))

    for path in CURRENT_M7_DOCS:
        relative = path.relative_to(root).as_posix()
        _require(errors, root, relative, (f"Release target: v{version}",))
        _forbid(errors, root, relative, ("Target version: v0.7.0",))

    for path in HISTORICAL_REPORTS:
        relative = path.relative_to(root).as_posix()
        _require(
            errors,
            root,
            relative,
            ("Target version: v0.7.0", "never released", f"Current release target: v{version}"),
        )

    if check_cli:
        expected = f"Taksh {version} ({release_status})"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from taksh import main; main(['--version'])",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            errors.append(f"CLI version check exited {completed.returncode}: {combined}")
        elif expected not in combined:
            errors.append(f"CLI version output {combined!r} does not contain {expected!r}")

    return errors, version, release_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--no-cli", action="store_true", help="skip the CLI invocation")
    args = parser.parse_args(argv)

    errors, version, status = check(args.root, check_cli=not args.no_cli)
    if errors:
        print("Version consistency: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Version consistency: PASS")
    print(f"  authoritative: taksh.__version__ = {version}")
    print(f"  release status: {status}")
    print(f"  checked root: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
