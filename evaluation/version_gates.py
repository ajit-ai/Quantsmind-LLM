"""Validate the five-version program labels against release gates.

The program labels (1.1.0 ... 2.0.0) are execution labels. They are not
package versions and they do not renumber the master roadmap. This module
checks that distinction, validates the status vocabulary, and reports whether
any release gate is actually open.

It never trains, creates a checkpoint, uploads, tags, releases or merges.

Run:

    python -m evaluation.version_gates
    python -m evaluation.version_gates --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROGRAM_CONFIG = "configs/roadmap/five_version_program.yaml"
MASTER_ROADMAP = "docs/roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md"
DEFAULT_OUT = "reports/taksh_version_gates.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")
REQUIRED_VERSION_FIELDS = (
    "id",
    "label",
    "kind",
    "status",
    "scope",
    "artifact",
    "package_version",
    "release_version",
    "exit_gate",
    "blocked_by",
)
AUTHORIZATION_KEYS = (
    "training",
    "compute",
    "publication",
    "merge_to_develop_or_main",
    "tags_or_releases",
)


class VersionGateError(RuntimeError):
    """The program configuration or version declarations are inconsistent."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_program(path: str | Path = PROGRAM_CONFIG) -> dict[str, Any]:
    import yaml

    config_path = _path(path)
    if not config_path.is_file():
        raise VersionGateError(f"program config not found: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise VersionGateError(f"program config is not a mapping: {config_path}")
    return document


def _validate(document: dict[str, Any], current_version: str) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != "taksh.five_version_program/1":
        errors.append(
            f"unexpected schema: {document.get('schema')!r}"
        )

    package = document.get("package") or {}
    if package.get("current_version") != current_version:
        errors.append(
            "package.current_version "
            f"{package.get('current_version')!r} != taksh.__version__ {current_version!r}"
        )
    if not isinstance(package.get("release_authorized"), bool):
        errors.append("package.release_authorized must be a boolean")

    namespaces = document.get("namespaces") or {}
    for key in ("package_version", "master_roadmap", "program_label"):
        if key not in namespaces:
            errors.append(f"namespaces is missing {key!r}")

    allowed = list(document.get("allowed_statuses") or [])
    if allowed != list(ALLOWED_STATUSES):
        errors.append(
            f"allowed_statuses {allowed!r} != {list(ALLOWED_STATUSES)!r}"
        )

    versions = document.get("versions") or []
    if not versions:
        errors.append("versions must not be empty")
    seen: set[str] = set()
    for index, version in enumerate(versions):
        if not isinstance(version, dict):
            errors.append(f"versions[{index}] is not a mapping")
            continue
        missing = [field for field in REQUIRED_VERSION_FIELDS if field not in version]
        if missing:
            errors.append(
                f"versions[{index}] ({version.get('id', '?')}) missing: "
                + ", ".join(missing)
            )
            continue
        version_id = str(version["id"])
        if version_id in seen:
            errors.append(f"duplicate version id: {version_id}")
        seen.add(version_id)
        if version["status"] not in ALLOWED_STATUSES:
            errors.append(
                f"version {version_id} has invalid status {version['status']!r}"
            )
        if version["kind"] != "program_label":
            errors.append(
                f"version {version_id} kind must be 'program_label', got "
                f"{version['kind']!r}"
            )
        if version["package_version"] != current_version:
            errors.append(
                f"version {version_id} package_version {version['package_version']!r} "
                f"!= current package version {current_version!r}"
            )
        if version["release_version"] is not None:
            errors.append(
                f"version {version_id} claims release_version "
                f"{version['release_version']!r} without authorization"
            )
        if not isinstance(version["blocked_by"], list):
            errors.append(f"version {version_id} blocked_by must be a list")

    authorization = document.get("authorization") or {}
    for key in AUTHORIZATION_KEYS:
        if not isinstance(authorization.get(key), bool):
            errors.append(f"authorization.{key} must be a boolean")

    master = _path(namespaces.get("master_roadmap", {}).get("path", MASTER_ROADMAP))
    if not master.is_file():
        errors.append(f"master roadmap not found: {master}")

    return errors


def evaluate(
    path: str | Path = PROGRAM_CONFIG,
    *,
    current_version: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic gate report; never writes a release artifact."""
    if current_version is None:
        from taksh import __version__

        current_version = __version__

    document = load_program(path)
    errors = _validate(document, current_version)
    if errors:
        raise VersionGateError("; ".join(errors))

    versions = document["versions"]
    status_counts = {status: 0 for status in ALLOWED_STATUSES}
    for version in versions:
        status_counts[version["status"]] += 1

    blocked_versions = [
        {
            "id": version["id"],
            "label": version["label"],
            "status": version["status"],
            "blocked_by": list(version["blocked_by"]),
        }
        for version in versions
        if version["status"] != "PASS"
    ]
    authorization = document["authorization"]
    closed_gates = [
        f"authorization.{key} is false"
        for key in AUTHORIZATION_KEYS
        if authorization.get(key) is not True
    ]
    if document["package"].get("release_authorized") is not True:
        closed_gates.append("package.release_authorized is false")

    release_ready = not blocked_versions and not closed_gates
    return {
        "schema": "taksh.version_gates/1",
        "generated_by": "evaluation/version_gates.py",
        "program": document["program"],
        "branch": document["branch"],
        "package": {
            "authoritative_source": document["package"]["authoritative_source"],
            "current_version": current_version,
            "release_status": document["package"]["release_status"],
            "release_authorized": bool(
                document["package"].get("release_authorized")
            ),
            "bump_policy": document["package"]["bump_policy"],
        },
        "namespaces": document["namespaces"],
        "versions": versions,
        "status_counts": status_counts,
        "blocked_versions": blocked_versions,
        "release_ready": release_ready,
        "closed_gates": closed_gates,
        "verdict": "RELEASE_GATES_OPEN" if release_ready else "NO_RELEASE_AUTHORIZED",
        "policy": {
            "program_label_is_not_package_version": True,
            "master_roadmap_remains_canonical": True,
            "training_authorized": bool(authorization.get("training")),
            "compute_authorized": bool(authorization.get("compute")),
            "publication_authorized": bool(authorization.get("publication")),
            "merge_authorized": bool(
                authorization.get("merge_to_develop_or_main")
            ),
            "tags_or_releases_authorized": bool(
                authorization.get("tags_or_releases")
            ),
        },
        "master_roadmap": {
            "path": _relative(
                _path(document["namespaces"]["master_roadmap"]["path"])
            ),
            "status": "CANONICAL",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=PROGRAM_CONFIG,
        help=f"program gate config (default: {PROGRAM_CONFIG})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = evaluate(args.config)
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"package: {report['package']['current_version']}")
        print(f"release ready: {report['release_ready']}")
        print("versions:")
        for version in report["versions"]:
            print(f"  {version['status']:<12} {version['id']} {version['label']}")
        print(f"closed gates: {len(report['closed_gates'])}")
        print(f"report: {out_path}")
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
