"""Bounded local toolchain support for C# evaluation fixtures and candidates.

This module compiles and, only with an explicit opt-in, runs code. It is not a
security boundary. Do not point it at model-generated code unless the execution
policy for that evaluation permits it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.csharp.diagnostics import (
    CompilerDiagnostic,
    category_for,
    detect_async_misuse,
    parse_roslyn_diagnostics,
    summarize_diagnostics,
)

_TARGET_FRAMEWORK = re.compile(r"^net[0-9]+\.0$")
_DEFAULT_TIMEOUT = 90


@dataclass(frozen=True)
class CSharpEnvironment:
    dotnet_path: str
    sdk_version: str
    runtime_version: str | None
    target_framework: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CSharpRunResult:
    outcome: str
    environment: CSharpEnvironment
    diagnostics: list[dict[str, Any]]
    source_diagnostics: list[dict[str, Any]]
    stdout: str
    stderr: str
    exit_code: int | None
    duration_seconds: float
    artifact_dir: str | None
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "environment": self.environment.as_dict(),
            "diagnostics": self.diagnostics,
            "source_diagnostics": self.source_diagnostics,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 4),
            "artifact_dir": self.artifact_dir,
            "summary": self.summary,
        }


def validate_target_framework(target_framework: str) -> str:
    if not _TARGET_FRAMEWORK.fullmatch(target_framework):
        raise ValueError("target_framework must look like 'net10.0'")
    return target_framework


def detect_csharp_environment(target_framework: str = "net10.0") -> CSharpEnvironment:
    validate_target_framework(target_framework)
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise RuntimeError("dotnet is not installed or is not on PATH")
    try:
        sdk = subprocess.run(
            [dotnet, "--version"], capture_output=True, text=True,
            timeout=10, check=False,
        )
        runtimes = subprocess.run(
            [dotnet, "--list-runtimes"], capture_output=True, text=True,
            timeout=10, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("dotnet metadata query timed out") from exc
    if sdk.returncode != 0:
        raise RuntimeError("dotnet --version failed: " + (sdk.stderr or sdk.stdout).strip())
    runtime_version = None
    for line in runtimes.stdout.splitlines():
        if line.startswith("Microsoft.NETCore.App "):
            runtime_version = line.split()[1]
            break
    return CSharpEnvironment(
        dotnet_path=dotnet,
        sdk_version=sdk.stdout.strip(),
        runtime_version=runtime_version,
        target_framework=target_framework,
    )


def _candidate_env(temp_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_HOME": str(temp_dir),
        "NUGET_PACKAGES": str(temp_dir / "nuget-cache"),
    })
    return env


def _write_project(root: Path, source: str, target_framework: str) -> tuple[Path, Path]:
    project = root / "TakshEvaluation.csproj"
    program = root / "Program.cs"
    project.write_text(
        "<Project Sdk=\"Microsoft.NET.Sdk\">\n"
        "  <PropertyGroup>\n"
        "    <OutputType>Exe</OutputType>\n"
        f"    <TargetFramework>{target_framework}</TargetFramework>\n"
        "    <ImplicitUsings>enable</ImplicitUsings>\n"
        "    <Nullable>enable</Nullable>\n"
        "    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>\n"
        "  </PropertyGroup>\n"
        "</Project>\n",
        encoding="utf-8",
    )
    program.write_text(source, encoding="utf-8")
    return project, program


def _run_command(command, *, cwd: Path, env: dict[str, str], timeout: int):
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def compile_csharp(
    source: str,
    *,
    target_framework: str = "net10.0",
    timeout: int = _DEFAULT_TIMEOUT,
    allow_execution: bool = False,
    stdin_text: str = "",
    keep_artifacts: bool = False,
) -> CSharpRunResult:
    if not source.strip():
        raise ValueError("source must not be empty")
    started = time.perf_counter()
    environment = detect_csharp_environment(target_framework)
    source_diagnostics = [item.as_dict() for item in detect_async_misuse(source)]
    if keep_artifacts:
        root = Path(tempfile.mkdtemp(prefix="taksh_csharp_keep_"))
        artifact_dir = str(root)
    else:
        root = Path(tempfile.mkdtemp(prefix="taksh_csharp_"))
        artifact_dir = None
    try:
        project, _program = _write_project(root, source, target_framework)
        env = _candidate_env(root)
        try:
            build = _run_command(
                [environment.dotnet_path, "build", str(project),
                 "--nologo", "--verbosity", "minimal", "--disable-build-servers"],
                cwd=root,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _result(
                "timeout", environment, source_diagnostics,
                "", f"build timed out after {timeout}s",
                None, started, artifact_dir,
            )
        compiler_output = (build.stdout or "") + "\n" + (build.stderr or "")
        diagnostics = [item.as_dict() for item in parse_roslyn_diagnostics(compiler_output)]
        if build.returncode != 0:
            return _result(
                "compile_failed", environment, source_diagnostics,
                build.stdout, build.stderr, build.returncode,
                started, artifact_dir, diagnostics=diagnostics,
            )
        if not allow_execution:
            return _result(
                "compiled", environment, source_diagnostics,
                build.stdout, build.stderr, build.returncode,
                started, artifact_dir, diagnostics=diagnostics,
            )
        dll = _find_built_dll(root, target_framework)
        if dll is None:
            return _result(
                "error", environment, source_diagnostics,
                build.stdout, (build.stderr or "") + "\nbuilt assembly not found",
                None, started, artifact_dir, diagnostics=diagnostics,
            )
        try:
            run = subprocess.run(
                [environment.dotnet_path, str(dll)],
                cwd=str(root),
                env=env,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _result(
                "timeout", environment, source_diagnostics,
                build.stdout, (build.stderr or "") + f"\nexecution timed out after {timeout}s",
                None, started, artifact_dir, diagnostics=diagnostics,
            )
        return _result(
            "executed", environment, source_diagnostics,
            (build.stdout or "") + "\n" + (run.stdout or ""),
            (build.stderr or "") + "\n" + (run.stderr or ""),
            run.returncode, started, artifact_dir, diagnostics=diagnostics,
        )
    finally:
        if not keep_artifacts:
            shutil.rmtree(root, ignore_errors=True)


def _find_built_dll(root: Path, target_framework: str) -> Path | None:
    candidates = sorted((root / "bin").glob(f"*/{target_framework}/*.dll"))
    for candidate in candidates:
        if candidate.stat().st_size > 0:
            return candidate
    return None


def _as_diagnostics(items: list[dict[str, Any]]) -> list[CompilerDiagnostic]:
    converted = []
    for item in items:
        code = str(item.get("code", "TAK-UNKNOWN")).upper()
        converted.append(CompilerDiagnostic(
            severity=str(item.get("severity", "warning")).lower(),
            code=code,
            line=item.get("line"),
            column=item.get("column"),
            message=str(item.get("message", "")),
            category=str(item.get("category") or category_for(code)),
        ))
    return converted


def _result(
    outcome: str,
    environment: CSharpEnvironment,
    source_diagnostics: list[dict[str, Any]],
    stdout: str,
    stderr: str,
    exit_code: int | None,
    started: float,
    artifact_dir: str | None,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> CSharpRunResult:
    diagnostic_items = list(diagnostics or [])
    summary = summarize_diagnostics([
        *_as_diagnostics(diagnostic_items),
        *_as_diagnostics(source_diagnostics),
    ])
    return CSharpRunResult(
        outcome=outcome,
        environment=environment,
        diagnostics=diagnostic_items,
        source_diagnostics=source_diagnostics,
        stdout=stdout or "",
        stderr=stderr or "",
        exit_code=exit_code,
        duration_seconds=time.perf_counter() - started,
        artifact_dir=artifact_dir,
        summary=summary,
    )
