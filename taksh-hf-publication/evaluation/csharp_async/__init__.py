"""Async/await probe helpers for the C# evaluation surface."""

from evaluation.csharp.diagnostics import detect_async_misuse
from evaluation.csharp.suite import CSHARP_PROBES

ASYNC_PROBES = tuple(probe for probe in CSHARP_PROBES if probe.category == "async")

__all__ = ["ASYNC_PROBES", "detect_async_misuse"]
