"""Guarded execution of generated code (Taksh v1.4.0).

Generated code is never executed inside the evaluator's own process, and it
is not executed by a plain subprocess that is described as "a sandbox". A
plain subprocess inherits the host's filesystem, network and credentials, so
running one would be a false claim.

What this module does instead:

* It **detects** whether a strong isolation mechanism is configured. By
  default there is none, and functional execution is **blocked**: the
  evaluator reports syntax-only results and says why.
* If the operator configures an external isolation command (a container
  runtime, a jailed interpreter, a microVM), the executor runs the generated
  program **inside that command**, with a process timeout, an output-size cap
  and a temporary workspace that is removed afterwards.

The capabilities that an external command must provide are stated in the
policy, so an operator cannot configure a bare interpreter and call it
isolated: the policy records the required guarantees and the report states
which of them the configured mechanism is trusted to provide.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Guarantees an external isolation mechanism must provide. They are recorded
#: in the report; the executor does not pretend to provide them itself.
REQUIRED_GUARANTEES = (
    "no_network",
    "restricted_filesystem",
    "no_credentials",
    "process_timeout",
    "memory_limit",
    "output_size_limit",
    "temporary_workspace",
)


@dataclass
class SandboxPolicy:
    """How (or whether) generated code may be executed."""

    mode: str = "blocked"
    external_command: list = field(default_factory=list)
    timeout_seconds: int = 10
    memory_mb: int = 256
    output_bytes: int = 65536
    source_path: str = "generated.py"
    guarantees: tuple = REQUIRED_GUARANTEES

    @property
    def is_blocked(self) -> bool:
        return self.mode != "external" or not self.external_command

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "external_command": list(self.external_command),
            "timeout_seconds": self.timeout_seconds,
            "memory_mb": self.memory_mb,
            "output_bytes": self.output_bytes,
            "required_guarantees": list(self.guarantees),
            "blocked": self.is_blocked,
        }


@dataclass
class ExecutionResult:
    """Outcome of an attempted execution."""

    executed: bool
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked: bool = False
    reason: str = ""
    duration_seconds: float = 0.0


def default_policy() -> SandboxPolicy:
    """The default policy: functional execution is blocked."""
    return SandboxPolicy()


def capability_report(policy: SandboxPolicy) -> dict:
    """Describe what the configured mechanism can and cannot promise."""
    return {
        "functional_execution": "blocked" if policy.is_blocked else "enabled",
        "mechanism": (
            "none-configured"
            if policy.is_blocked
            else "external:" + " ".join(policy.external_command)
        ),
        "required_guarantees": list(policy.guarantees),
        "provided_by_operator": [] if policy.is_blocked else list(policy.guarantees),
        "not_guaranteed_by_this_process": [
            "network isolation",
            "filesystem isolation",
            "credential isolation",
        ],
        "note": (
            "This module never executes generated code in the evaluator's own "
            "process and never describes a plain subprocess as a sandbox."
        ),
    }


def execute_python(source: str, policy: SandboxPolicy | None = None) -> ExecutionResult:
    """Execute generated Python under the policy, or refuse to.

    When the policy blocks execution, the result says so explicitly; it never
    falls back to running the code locally.
    """
    policy = policy or default_policy()
    if policy.is_blocked:
        return ExecutionResult(
            executed=False,
            ok=False,
            blocked=True,
            reason=(
                "functional execution is blocked: no isolation mechanism is "
                "configured, and generated code must not run in the evaluator's "
                "host process"
            ),
        )

    workspace = Path(tempfile.mkdtemp(prefix="taksh-eval-"))
    try:
        target = workspace / policy.source_path
        target.write_text(source, encoding="utf-8")
        command = list(policy.external_command) + [str(target)]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "HOME": str(workspace),
            "TMPDIR": str(workspace),
        }
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                env=environment,
                capture_output=True,
                text=True,
                timeout=policy.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                executed=True,
                ok=False,
                timed_out=True,
                reason=f"execution exceeded {policy.timeout_seconds}s",
            )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        truncated = False
        if len(stdout) > policy.output_bytes:
            stdout = stdout[: policy.output_bytes]
            truncated = True
        if len(stderr) > policy.output_bytes:
            stderr = stderr[: policy.output_bytes]
            truncated = True
        return ExecutionResult(
            executed=True,
            ok=completed.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            reason=("output truncated" if truncated else ""),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
