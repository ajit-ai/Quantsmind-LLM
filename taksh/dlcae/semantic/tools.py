"""Pluggable local toolchain adapter framework (offline).

Discovery, version inspection, availability checks, syntax validation,
semantic/static analysis, compilation/interpretation where safe, execution
where safe, structured diagnostics, timeout and resource controls,
cancellation, environment fingerprint, explicit unavailable states.

No network access, no package installs, no dependency downloads. Every
execution runs in an isolated temporary workspace with bounded output.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class ToolchainResult:
    tool: str
    tool_version: str                      # "unavailable" if absent
    available: bool
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    duration_ms: float = 0.0
    argv: list[str] = field(default_factory=list)
    execution_context: str = "isolated-tmp-workspace"
    evidence_classification: str = "TOOL_VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.toolchain_result/1",
            "tool": self.tool, "tool_version": self.tool_version,
            "available": self.available, "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:MAX_OUTPUT_BYTES],
            "stderr": self.stderr[:MAX_OUTPUT_BYTES],
            "timed_out": self.timed_out, "cancelled": self.cancelled,
            "duration_ms": self.duration_ms,
            "argv": self.argv, "execution_context": self.execution_context,
            "evidence_classification": self.evidence_classification,
        }


class ToolchainUnavailable(Exception):
    """Raised when a requested operation needs an absent toolchain."""


class Sandbox:
    """Isolated temporary workspace; cleans only its own tree."""

    def __init__(self, prefix: str = "l100eng2_tools_") -> None:
        self.root = Path(tempfile.mkdtemp(prefix=prefix))

    def write(self, name: str, content: str | bytes) -> Path:
        path = self.root / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def cleanup(self) -> None:
        if self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def _clean_env() -> dict[str, str]:
    keys = ("PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE",
            "PATHEXT", "LANG", "LC_ALL", "PYTHONIOENCODING", "JAVA_HOME", "DOTNET_CLI_HOME")
    return {k: v for k in keys if (v := os.environ.get(k)) is not None} | {
        "PYTHONIOENCODING": "utf-8",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }


def _run(argv: list[str], *, timeout_s: float, cancel_event, cwd: Path,
         max_output: int = MAX_OUTPUT_BYTES) -> ToolchainResult:
    tool = Path(argv[0]).name
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=str(cwd), env=_clean_env())
    except FileNotFoundError:
        return ToolchainResult(tool=tool, tool_version="unavailable", available=False,
                               ok=False, exit_code=None, stdout="", stderr="toolchain not found",
                               duration_ms=0.0, argv=argv)
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_b, stderr_b = proc.communicate()
        timed_out = True
    cancelled = False
    if cancel_event is not None and cancel_event.is_set():
        cancelled = True
        try:
            proc.kill()
        except OSError:
            pass
        timed_out = True
    duration_ms = (time.perf_counter() - t0) * 1000
    return ToolchainResult(
        tool=tool,
        tool_version="detected",
        available=True,
        ok=(proc.returncode == 0 and not timed_out and not cancelled),
        exit_code=proc.returncode,
        stdout=(stdout_b or b"")[:max_output].decode("utf-8", errors="replace"),
        stderr=(stderr_b or b"")[:max_output].decode("utf-8", errors="replace"),
        timed_out=timed_out, cancelled=cancelled,
        duration_ms=round(duration_ms, 3), argv=argv)


class ToolchainAdapter(ABC):
    """Common interface for local toolchain adapters."""

    language_ids: tuple[str, ...] = ()
    name: str = "abstract"
    version: str = "0"

    @abstractmethod
    def discover(self) -> bool: ...

    @abstractmethod
    def tool_version(self) -> str | None: ...

    def available(self) -> bool:
        try:
            return bool(self.discover())
        except Exception:  # noqa: BLE001
            return False

    @abstractmethod
    def syntax_check(self, source: str, *, filename: str,
                     timeout_s: float, cancel_event=None) -> ToolchainResult: ...

    def semantic_check(self, source: str, *, filename: str,
                       timeout_s: float, cancel_event=None) -> ToolchainResult:
        return ToolchainResult(
            tool=self.name, tool_version="n/a", available=self.available(),
            ok=False, exit_code=None, stdout="",
            stderr="TOOLCHAIN_UNAVAILABLE: no semantic/static analysis adapter",
            evidence_classification="UNVERIFIED")

    def execute(self, source: str, *, filename: str,
                timeout_s: float, cancel_event=None) -> ToolchainResult:
        return ToolchainResult(
            tool=self.name, tool_version="n/a", available=self.available(),
            ok=False, exit_code=None, stdout="",
            stderr="EXECUTION_BLOCKED: execution is not authorized for this toolchain",
            evidence_classification="UNVERIFIED")

    def env_fingerprint(self) -> str:
        env = {"tool": self.name, "version": self.tool_version(),
               "platform": sys.platform}
        return hashlib.sha256(repr(sorted(env.items())).encode()).hexdigest()


def _python_exe() -> str:
    return sys.executable


class PythonToolchain(ToolchainAdapter):
    language_ids = ("python",)
    name = "python"
    version = "stdlib-offline"

    def discover(self) -> bool:
        return Path(_python_exe()).is_file()

    def tool_version(self) -> str | None:
        return sys.version.split()[0]

    def env_fingerprint(self) -> str:
        import platform

        return hashlib.sha256(
            repr(sorted({"tool": self.name, "version": self.tool_version(),
                         "platform": platform.platform(),
                         "python_exe": _python_exe()}.items())).encode()).hexdigest()

    def syntax_check(self, source, *, filename="fixture.py", timeout_s=DEFAULT_TIMEOUT_SECONDS,
                     cancel_event=None) -> ToolchainResult:
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([_python_exe(), "-I", "-m", "py_compile", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool_version = self.tool_version() or "unknown"
            result.evidence_classification = "TOOL_VERIFIED"
            return result

    def semantic_check(self, source, *, filename="fixture.py",
                       timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        exe = _python_exe()
        mypy = None
        try:
            proc = subprocess.run([exe, "-m", "mypy", "--version"],
                                  capture_output=True, timeout=10)
            if proc.returncode == 0:
                mypy = proc.stdout.decode().strip()
        except Exception:  # noqa: BLE001
            mypy = None
        if mypy is None:
            return ToolchainResult(tool=self.name, tool_version=self.tool_version() or "unknown",
                                   available=True, ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE: mypy not available; "
                                          "types not verified",
                                   evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([exe, "-m", "mypy", "--ignore-missing-imports",
                           "--no-error-summary", "--show-error-codes", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool = "python-mypy"
            result.tool_version = mypy
            result.evidence_classification = "TOOL_VERIFIED"
            return result

    def lint_check(self, source, *, filename="fixture.py",
                   timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        exe = _python_exe()
        ruff = None
        try:
            proc = subprocess.run([exe, "-m", "ruff", "--version"],
                                  capture_output=True, timeout=10)
            if proc.returncode == 0:
                ruff = proc.stdout.decode().strip()
        except Exception:  # noqa: BLE001
            ruff = None
        if ruff is None:
            return ToolchainResult(tool="python-ruff", tool_version="unavailable",
                                   available=False, ok=False, exit_code=None,
                                   stdout="", stderr="TOOLCHAIN_UNAVAILABLE: ruff not available",
                                   evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([exe, "-m", "ruff", "check", "--output-format=concise", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool = "python-ruff"
            result.tool_version = ruff
            result.evidence_classification = "TOOL_VERIFIED"
            return result

    def execute(self, source, *, filename="fixture.py",
                timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([_python_exe(), "-I", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool = "python-exec"
            result.tool_version = self.tool_version() or "unknown"
            result.evidence_classification = "TOOL_VERIFIED"
            return result


class NodeToolchain(ToolchainAdapter):
    language_ids = ("javascript",)
    name = "node"
    version = "discovered"

    @staticmethod
    def _node() -> str | None:
        return shutil.which("node")

    def discover(self) -> bool:
        return self._node() is not None

    def tool_version(self) -> str | None:
        exe = self._node()
        if not exe:
            return None
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, timeout=10)
            return proc.stdout.decode().strip() if proc.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def syntax_check(self, source, *, filename="fixture.js",
                     timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        exe = self._node()
        if exe is None:
            return ToolchainResult(tool="node", tool_version="unavailable", available=False,
                                   ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE", evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([exe, "--check", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool_version = self.tool_version() or "unknown"
            return result


class BashToolchain(ToolchainAdapter):
    language_ids = ("bash", "shell", "sh")
    name = "bash"
    version = "discovered"

    @staticmethod
    def _bash() -> str | None:
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", "bash"):
            resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
            if resolved:
                return resolved
        return None

    def discover(self) -> bool:
        return self._bash() is not None

    def tool_version(self) -> str | None:
        exe = self._bash()
        if not exe:
            return None
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, timeout=10)
            return proc.stdout.decode().splitlines()[0] if proc.stdout else None
        except Exception:  # noqa: BLE001
            return None

    def syntax_check(self, source, *, filename="fixture.sh",
                     timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        exe = self._bash()
        if exe is None:
            return ToolchainResult(tool="bash", tool_version="unavailable", available=False,
                                   ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE", evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([exe, "-n", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool_version = self.tool_version() or "unknown"
            return result


class GoToolchain(ToolchainAdapter):
    language_ids = ("go",)
    name = "go"
    version = "discovered"

    @staticmethod
    def _go() -> str | None:
        return shutil.which("go")

    def discover(self) -> bool:
        return self._go() is not None

    def tool_version(self) -> str | None:
        exe = self._go()
        if not exe:
            return None
        try:
            proc = subprocess.run([exe, "version"], capture_output=True, timeout=10)
            return proc.stdout.decode().strip() if proc.stdout else None
        except Exception:  # noqa: BLE001
            return None

    def syntax_check(self, source, *, filename="fixture.go",
                     timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        exe = self._go()
        if exe is None:
            return ToolchainResult(tool="go", tool_version="unavailable", available=False,
                                   ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE", evidence_classification="UNVERIFIED")
        gofmt = (Path(exe).parent / "gofmt.exe") if Path(exe).parent.joinpath("gofmt.exe").is_file() else shutil.which("gofmt")
        if not gofmt:
            return ToolchainResult(tool="go", tool_version=self.tool_version() or "unknown",
                                   available=True, ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE: gofmt not available",
                                   evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([str(gofmt), "-l", str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            # gofmt -l lists files that WOULD change; parse errors exit non-zero
            result.ok = result.exit_code == 0 and not result.timed_out
            result.tool_version = self.tool_version() or "unknown"
            return result


class JavaToolchain(ToolchainAdapter):
    language_ids = ("java",)
    name = "java-jdk21"
    version = "discovered"

    @staticmethod
    def _jdk_home() -> Path | None:
        candidate = Path(r"C:\Program Files\Java\jdk-21")
        if (candidate / "bin" / "java.exe").is_file():
            return candidate
        return None

    def discover(self) -> bool:
        return self._jdk_home() is not None

    def tool_version(self) -> str | None:
        home = self._jdk_home()
        if not home:
            return None
        try:
            proc = subprocess.run([str(home / "bin" / "javap.exe"), "-version"],
                                  capture_output=True, timeout=10)
            text = (proc.stderr or proc.stdout).decode("utf-8", errors="replace").strip()
            return text.splitlines()[0] if text else "jdk-21"
        except Exception:  # noqa: BLE001
            return "jdk-21"

    def syntax_check(self, source, *, filename="Fixture.java",
                     timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        home = self._jdk_home()
        if not home:
            return ToolchainResult(tool="javac", tool_version="unavailable", available=False,
                                   ok=False, exit_code=None, stdout="",
                                   stderr="TOOLCHAIN_UNAVAILABLE", evidence_classification="UNVERIFIED")
        with Sandbox() as sandbox:
            path = sandbox.write(filename, source)
            result = _run([str(home / "bin" / "javac.exe"), "-d", str(sandbox.root), str(path)],
                          timeout_s=timeout_s, cancel_event=cancel_event, cwd=sandbox.root)
            result.tool_version = self.tool_version() or "jdk-21"
            result.evidence_classification = "TOOL_VERIFIED"
            return result


class DotnetToolchain(ToolchainAdapter):
    language_ids = ("csharp", "fsharp")
    name = "dotnet"
    version = "discovered"

    @staticmethod
    def _dotnet() -> str | None:
        return shutil.which("dotnet")

    def discover(self) -> bool:
        return self._dotnet() is not None

    def tool_version(self) -> str | None:
        exe = self._dotnet()
        if not exe:
            return None
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, timeout=30)
            return proc.stdout.decode().strip() if proc.stdout else None
        except Exception:  # noqa: BLE001
            return None

    def syntax_check(self, source, *, filename="fixture.cs",
                     timeout_s=DEFAULT_TIMEOUT_SECONDS, cancel_event=None) -> ToolchainResult:
        return ToolchainResult(
            tool="dotnet",
            tool_version=self.tool_version() or "unavailable",
            available=self.discover(), ok=False, exit_code=None, stdout="",
            stderr="NOT_APPLICABLE: no C# project creation is authorized in this "
                   "milestone; dotnet discovered for version reporting only",
            evidence_classification="UNVERIFIED")


class ToolchainRegistry:
    """Discovery and routing for all local toolchains."""

    def __init__(self) -> None:
        self.adapters: dict[str, ToolchainAdapter] = {
            "python": PythonToolchain(),
            "javascript": NodeToolchain(),
            "bash": BashToolchain(),
            "go": GoToolchain(),
            "java": JavaToolchain(),
            "csharp": DotnetToolchain(),
        }

    def for_language(self, language_id: str) -> ToolchainAdapter | None:
        return self.adapters.get(language_id.lower())

    def discover_all(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for language, adapter in sorted(self.adapters.items()):
            available = adapter.available()
            report[language] = {
                "tool": adapter.name,
                "available": available,
                "version": adapter.tool_version() if available else None,
                "env_fingerprint": adapter.env_fingerprint() if available else None,
                "status": "AVAILABLE" if available else "TOOLCHAIN_UNAVAILABLE",
            }
        return report
