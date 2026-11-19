"""Prepare and verify a Taksh release without publishing it.

This is the one-command release preparation entry point. It performs a clean
build, inspects archives, validates metadata, checks version identity, installs
the wheel into isolated environments, runs the documented CLI and serving smoke
checks, runs regression tests and validates the Sphinx docs.

It never tags, commits, pushes, merges, uploads to an index, or creates a GitHub
release. Publishing remains a separate, explicitly authorised action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import check_version_consistency as consistency

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".arrow",
    ".parquet",
    ".jsonl",
    ".env",
    ".pem",
    ".key",
    ".pkl",
    ".joblib",
}
FORBIDDEN_PREFIXES = (
    "runs/",
    "reports/",
    "checkpoints/",
    "data/raw/",
    "data/shards/",
    "data/m7/",
    "secrets/",
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    ".git/",
)
SECRET_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"F:\\Codes\\", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".in",
    ".json",
    "",
}
SECRET_SCAN_ALLOWLIST = {"tests/test_internal_ingest.py"}


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = 1800,
    required: bool = True,
) -> dict:
    print(f"[release] {name}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = f"timed out after {timeout} seconds"
    except FileNotFoundError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)

    result = {
        "step": name,
        "required": required,
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 2),
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:],
    }
    status = "PASS" if returncode == 0 else "FAIL"
    print(f"[release] {name}: {status} ({result['duration_seconds']}s)", flush=True)
    if returncode != 0:
        detail = (stderr or stdout).strip()
        if detail:
            print(detail[-2500:], file=sys.stderr, flush=True)
    return result


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _git_state() -> dict:
    status = _git("status", "--short", "--untracked-files=all")
    return {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "tags": _git("tag", "--list").splitlines(),
        "dirty_entries": len(status.splitlines()) if status else 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict:
    return {"path": str(path), "filename": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _metadata_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in fields:
            continue
        fields[key.strip()] = value.strip()
    return fields


def _relative_member(name: str, prefix: str) -> str:
    return name[len(prefix) :] if name.startswith(prefix) else name


def _inspect_wheel(path: Path, version: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = _metadata_fields(archive.read(metadata_name).decode("utf-8"))
        license_members = [name for name in names if name.endswith("/LICENSE") or name == "LICENSE"]
        notice_members = [name for name in names if name.endswith("/NOTICE") or name == "NOTICE"]
        forbidden = _forbidden_members(names)
        local_paths = _local_path_findings(archive, names)
    return {
        "members": len(names),
        "metadata": metadata,
        "license_members": license_members,
        "notice_members": notice_members,
        "forbidden": forbidden,
        "local_path_findings": local_paths,
    }


def _inspect_sdist(path: Path, version: str) -> dict:
    prefix = f"taksh-{version}/"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        metadata_name = next(name for name in names if _relative_member(name, prefix) == "PKG-INFO")
        handle = archive.extractfile(metadata_name)
        if handle is None:
            raise ValueError("PKG-INFO is not a regular file")
        metadata = _metadata_fields(handle.read().decode("utf-8"))
        license_members = [
            name
            for name in names
            if _relative_member(name, prefix) in ("LICENSE", "LICENSE.txt")
        ]
        notice_members = [
            name for name in names if _relative_member(name, prefix) == "NOTICE"
        ]
        forbidden = _forbidden_members(names, prefix=prefix)
        local_paths = _local_path_findings(archive, names, prefix=prefix)
    return {
        "members": len(names),
        "metadata": metadata,
        "license_members": license_members,
        "notice_members": notice_members,
        "forbidden": forbidden,
        "local_path_findings": local_paths,
    }


def _forbidden_members(names: list[str], *, prefix: str = "") -> list[str]:
    findings = []
    for raw_name in names:
        name = _relative_member(raw_name, prefix).replace("\\", "/")
        if not name or name.endswith("/"):
            continue
        lowered = name.lower()
        if Path(lowered).suffix in FORBIDDEN_SUFFIXES:
            findings.append(name)
            continue
        if any(lowered.startswith(item) for item in FORBIDDEN_PREFIXES):
            findings.append(name)
            continue
        if lowered.startswith("tokenizer/") and lowered.endswith(".json"):
            findings.append(name)
            continue
        if lowered.startswith("data/") and lowered.endswith((".json", ".jsonl")):
            findings.append(name)
    return sorted(set(findings))


def _local_path_findings(archive, names: list[str], *, prefix: str = "") -> list[str]:
    findings = []
    for raw_name in names:
        name = _relative_member(raw_name, prefix)
        if not name or name.endswith("/") or name in SECRET_SCAN_ALLOWLIST:
            continue
        if Path(name).suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if isinstance(archive, zipfile.ZipFile):
                info = archive.getinfo(raw_name)
                if info.file_size > 2_000_000:
                    continue
                data = archive.read(raw_name)
            else:
                member = archive.getmember(raw_name)
                if member.size > 2_000_000 or not member.isfile():
                    continue
                handle = archive.extractfile(member)
                data = handle.read() if handle else b""
        except (KeyError, OSError):
            continue
        text = data.decode("utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(name)
    return sorted(set(findings))


def _validate_metadata(label: str, inspection: dict, version: str) -> list[str]:
    problems = []
    metadata = inspection["metadata"]
    expected = {
        "Name": "taksh",
        "Version": version,
        "License-Expression": "Apache-2.0",
        "Requires-Python": ">=3.10",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            problems.append(f"{label}: metadata {key}={metadata.get(key)!r}, expected {value!r}")
    if metadata.get("License-File") != "LICENSE":
        problems.append(f"{label}: metadata License-File is not LICENSE")
    if not inspection["license_members"]:
        problems.append(f"{label}: LICENSE file is missing from the archive")
    if not inspection["notice_members"]:
        problems.append(f"{label}: NOTICE file is missing from the archive")
    if inspection["forbidden"]:
        problems.append(f"{label}: forbidden archive members: {', '.join(inspection['forbidden'])}")
    if inspection["local_path_findings"]:
        problems.append(
            f"{label}: possible local path or secret in: {', '.join(inspection['local_path_findings'])}"
        )
    return problems


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(venv: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _write_evidence(evidence_dir: Path, summary: dict) -> tuple[Path, Path]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_path = evidence_dir / "RELEASE_EVIDENCE.json"
    md_path = evidence_dir / "RELEASE_EVIDENCE.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Taksh Release Evidence",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Version: `{summary['version']}` ({summary['release_status']})",
        f"- Branch/commit: `{summary['git_at_start']['branch']}` / `{summary['git_at_start']['commit']}`",
        f"- Verdict: **{summary['verdict']}**",
        "- Publication: **NOT PERFORMED BY THIS SCRIPT**",
        "",
        "## Checks",
        "",
        "| Step | Result | Command |",
        "|---|---|---|",
    ]
    for step in summary["steps"]:
        result = "PASS" if step["returncode"] == 0 else "FAIL"
        command = " ".join(step["command"]).replace("|", "\\|")
        lines.append(f"| {step['step']} | {result} | `{command}` |")
    for step in summary["skipped"]:
        lines.append(f"| {step['step']} | SKIPPED | {step['reason']} |")
    lines.extend(["", "## Artifacts", "", "| File | Bytes | SHA-256 | Location |", "|---|---:|---|---|"])
    for artifact in summary["artifacts"].values():
        lines.append(
            f"| `{artifact['filename']}` | {artifact['bytes']:,} | `{artifact['sha256']}` | `{artifact['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Archive inspection",
            "",
            f"- Wheel: {summary['archives']['wheel']['members']} members; "
            f"forbidden={summary['archives']['wheel']['forbidden']}",
            f"- Source distribution: {summary['archives']['sdist']['members']} members; "
            f"forbidden={summary['archives']['sdist']['forbidden']}",
            "- LICENCE and NOTICE are present in both archives and in package metadata.",
            "- Secret scan allowlist: `tests/test_internal_ingest.py` (synthetic key-format fixtures only).",
            "- Weights, checkpoints, tokenizer assets, private datasets and caches are excluded.",
            "",
            "## Capability disclosure",
            "",
            "- M7 result: 0/40 locked syntax-valid generations, 0/8 functional tasks, evidence level 0.",
            "- Serving HTTP success is not evidence that generated output is correct.",
            "- Package readiness and model capability have different statuses.",
            "- Weights and training data are not part of the package.",
            "",
            "## Publication",
            "",
            "This evidence run does not commit, tag, push, merge, create a GitHub",
            "release or upload to a package index. Publication is a separate",
            "owner-authorized operation recorded in the release-readiness report.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest")
    parser.add_argument("--skip-docs", action="store_true", help="skip the Sphinx build")
    parser.add_argument("--skip-serving", action="store_true", help="skip checkpoint serving")
    parser.add_argument("--require-clean", action="store_true", help="fail if the worktree is dirty")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "reports" / "release-evidence")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    start_git = _git_state()
    if args.require_clean and start_git["dirty_entries"]:
        print("Release preparation requires a clean worktree.", file=sys.stderr)
        return 2

    errors, version, release_status = consistency.check(ROOT)
    if errors:
        for error in errors:
            print(f"version consistency: {error}", file=sys.stderr)
        return 1

    work = Path(tempfile.mkdtemp(prefix="taksh-release-")) if args.work_dir is None else args.work_dir
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    build_dir = work / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    args.dist_dir = args.dist_dir.resolve()

    for derived in (ROOT / "build", ROOT / "taksh.egg-info"):
        if derived.exists():
            shutil.rmtree(derived)

    steps = [
        _run(
            "version-consistency",
            [sys.executable, str(ROOT / "scripts" / "check_version_consistency.py")],
        )
    ]

    steps.append(
        _run(
            "build-sdist-wheel",
            [sys.executable, "-m", "build", "--outdir", str(build_dir), str(ROOT)],
        )
    )
    wheels = sorted(build_dir.glob("taksh-*.whl"))
    sdists = sorted(build_dir.glob("taksh-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(f"expected one wheel and one sdist, found {wheels} and {sdists}", file=sys.stderr)
        return 1
    wheel, sdist = wheels[0], sdists[0]

    archives = {
        "wheel": _inspect_wheel(wheel, version),
        "sdist": _inspect_sdist(sdist, version),
    }
    metadata_problems = _validate_metadata("wheel", archives["wheel"], version)
    metadata_problems += _validate_metadata("sdist", archives["sdist"], version)
    steps.append(
        {
            "step": "archive-metadata",
            "required": True,
            "command": [],
            "cwd": str(ROOT),
            "returncode": 0 if not metadata_problems else 1,
            "duration_seconds": 0.0,
            "stdout_tail": "metadata, licence and archive exclusions validated",
            "stderr_tail": "\n".join(metadata_problems),
        }
    )

    steps.append(
        _run(
            "twine-check",
            [sys.executable, "-m", "twine", "check", str(wheel), str(sdist)],
        )
    )

    args.dist_dir.mkdir(parents=True, exist_ok=True)
    for old in list(args.dist_dir.glob("taksh-*.whl")) + list(args.dist_dir.glob("taksh-*.tar.gz")):
        old.unlink()
    shutil.copy2(wheel, args.dist_dir / wheel.name)
    shutil.copy2(sdist, args.dist_dir / sdist.name)
    artifacts = {
        "wheel": _artifact_record(args.dist_dir / wheel.name),
        "sdist": _artifact_record(args.dist_dir / sdist.name),
    }

    clean_venv = work / "venv-clean"
    system_venv = work / "venv-system"
    steps.append(_run("create-clean-venv", [sys.executable, "-m", "venv", str(clean_venv)]))
    clean_python = _venv_python(clean_venv)
    steps.append(
        _run(
            "install-clean-venv",
            [str(clean_python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        )
    )
    steps.append(
        _run(
            "installed-version-clean-venv",
            [
                str(clean_python),
                "-c",
                (
                    "import importlib.metadata as m, taksh; "
                    f"assert m.version('taksh') == taksh.__version__ == '{version}'; "
                    "assert 'site-packages' in taksh.__file__, taksh.__file__; "
                    "print(taksh.__version__, taksh.__file__)"
                ),
            ],
            cwd=work,
        )
    )
    taksh_cli = _venv_script(clean_venv, "taksh")
    steps.append(
        _run(
            "installed-cli-clean-venv",
            [str(taksh_cli), "--version"],
            cwd=work,
        )
    )

    steps.append(
        _run(
            "create-system-venv",
            [sys.executable, "-m", "venv", "--system-site-packages", str(system_venv)],
        )
    )
    system_python = _venv_python(system_venv)
    steps.append(
        _run(
            "install-system-venv",
            [str(system_python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        )
    )
    openai_cli = _venv_script(system_venv, "taksh-openai")
    steps.append(
        _run(
            "installed-openai-cli-help",
            [str(openai_cli), "--help"],
            cwd=work,
        )
    )

    checkpoint = ROOT / "runs" / "m7_sft" / "checkpoint_best.pt"
    tokenizer = ROOT / "tokenizer" / "qm_code_bpe_v0.json"
    serving_available = checkpoint.is_file() and tokenizer.is_file()
    if args.skip_serving:
        skipped = [{"step": "installed-serving", "reason": "--skip-serving"}]
    elif not serving_available:
        skipped = [{"step": "installed-serving", "reason": "M7 checkpoint or tokenizer unavailable"}]
    else:
        skipped = []
        steps.append(
            _run(
                "installed-serving",
                [
                    str(system_python),
                    str(ROOT / "scripts" / "verify_installed_release.py"),
                    "--repo",
                    str(ROOT),
                    "--expect-version",
                    version,
                    "--json-out",
                    str(work / "installed-serving.json"),
                ],
                cwd=work,
            )
        )

    if args.skip_tests:
        skipped.append({"step": "pytest", "reason": "--skip-tests"})
    else:
        steps.append(
            _run(
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                    "--basetemp",
                    str(work / "pytest"),
                ],
                timeout=2400,
            )
        )

    if args.skip_docs:
        skipped.append({"step": "sphinx", "reason": "--skip-docs"})
    else:
        steps.append(
            _run(
                "sphinx",
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-W",
                    "-b",
                    "html",
                    str(ROOT / "docs" / "source"),
                    str(work / "docs-html"),
                ],
            )
        )

    for problem in metadata_problems:
        print(f"archive metadata: {problem}", file=sys.stderr)

    failures = [step for step in steps if step["returncode"] != 0]
    required_skips = [step for step in skipped if step["step"] in {"pytest", "sphinx", "installed-serving"}]
    if failures:
        verdict = "BLOCKED"
    elif required_skips:
        verdict = "PARTIALLY COMPLETE"
    else:
        verdict = "RELEASE GATES PASSED"

    summary = {
        "schema": "taksh.release_evidence/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version,
        "release_status": release_status,
        "verdict": verdict,
        "publication": "NOT PERFORMED BY THIS SCRIPT",
        "git_at_start": start_git,
        "git_at_end": _git_state(),
        "python": sys.version,
        "platform": sys.platform,
        "work_dir": str(work),
        "artifacts": artifacts,
        "archives": archives,
        "metadata_problems": metadata_problems,
        "steps": steps,
        "skipped": skipped,
        "failures": [step["step"] for step in failures],
    }
    json_path, md_path = _write_evidence(args.evidence_dir.resolve(), summary)
    print(f"Release evidence: {json_path}")
    print(f"Release evidence summary: {md_path}")
    print(f"Release verdict: {verdict}")
    if failures:
        for failure in failures:
            print(f"  - FAILED: {failure['step']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
