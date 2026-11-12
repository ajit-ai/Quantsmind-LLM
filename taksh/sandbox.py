"""Controlled execution sandbox for TAKSH engineering."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SandboxPolicy:
    workspace: str
    allowed_commands: list[str]
    timeout_seconds: int = 30
    max_output_bytes: int = 65536
    env_allowlist: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Sandbox:
    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy

    def run(self, command: list[str], *, cwd: str | None = None, input_text: str | None = None) -> ExecutionResult:
        if not command:
            return ExecutionResult("rejected", None, "", "empty command", 0.0)
        executable = Path(command[0]).name
        if executable not in self.policy.allowed_commands:
            return ExecutionResult("rejected", None, "", f"command not allowlisted: {executable}", 0.0)

        workspace = Path(self.policy.workspace).resolve()
        workdir = Path(cwd).resolve() if cwd else workspace
        if not workdir.is_relative_to(workspace):
            return ExecutionResult("rejected", None, "", "working directory outside workspace", 0.0)

        env = {}
        allow = self.policy.env_allowlist or ["PATH", "SYSTEMROOT", "TEMP", "TMP"]
        for key in allow:
            if key in os.environ:
                env[key] = os.environ[key]

        start = time.time()
        timed_out = False
        truncated = False
        try:
            proc = subprocess.run(
                command,
                cwd=workdir,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=self.policy.timeout_seconds,
                env=env,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
            status = "PASS" if returncode == 0 else "FAIL"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            returncode = None
            status = "TIMEOUT"
        duration = time.time() - start

        if len(stdout) > self.policy.max_output_bytes:
            stdout = stdout[: self.policy.max_output_bytes]
            truncated = True
        if len(stderr) > self.policy.max_output_bytes:
            stderr = stderr[: self.policy.max_output_bytes]
            truncated = True

        if timed_out:
            # Best-effort termination is handled by subprocess timeout.
            pass

        return ExecutionResult(status, returncode, stdout, stderr, duration, timed_out, truncated)
