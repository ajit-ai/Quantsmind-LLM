"""Optional isolation adapter boundary for TAKSH tool execution.

This module does not provide OS-level isolation by itself. It defines the
adapter boundary and fails closed when strict isolation is requested but no
adapter is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from taksh.sandbox import ExecutionResult


class IsolationAdapter(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def run(self, command: list[str], policy: Any) -> ExecutionResult: ...


@dataclass
class UnavailableIsolationAdapter:
    name: str = "unavailable"
    reason: str = "no isolation adapter is available on this platform"

    def is_available(self) -> bool:
        return False

    def run(self, command: list[str], policy: Any) -> ExecutionResult:
        return ExecutionResult("rejected", None, "", self.reason, 0.0)


class StrictIsolationUnavailable(RuntimeError):
    pass


def default_isolation_adapter():
    """Return the best available isolation adapter.

Today no OS-level adapter is implemented for the target environment. The
strict path therefore fails closed rather than downgrading to application mode.
"""
    return UnavailableIsolationAdapter()


def resolve_isolation(policy) -> IsolationAdapter:
    if policy.isolation == "strict":
        adapter = default_isolation_adapter()
        if adapter.is_available():
            return adapter
        return UnavailableIsolationAdapter(name="strict-required-unavailable",
                                           reason="strict OS isolation requested but no adapter is available; failing closed")
    return UnavailableIsolationAdapter(name="application", reason="application-level safeguards active")
