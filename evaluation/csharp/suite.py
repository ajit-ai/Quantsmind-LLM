"""Small, hand-written C# probe set for toolchain and diagnostic checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.csharp.runner import compile_csharp, CSharpRunResult


@dataclass(frozen=True)
class CSharpProbe:
    case_id: str
    category: str
    description: str
    source: str
    expected_outcome: str
    expected_codes: tuple[str, ...] = ()


CSHARP_PROBES = (
    CSharpProbe(
        case_id="csharp-linq-001",
        category="linq",
        description="LINQ filter and projection runs and prints the expected sequence.",
        source=(
            "using System;\n"
            "using System.Linq;\n"
            "var values = new[] { 1, 2, 3, 4, 5, 6 };\n"
            "var result = values.Where(value => value % 2 == 0).Select(value => value * 10);\n"
            "Console.WriteLine(string.Join(\",\", result));\n"
        ),
        expected_outcome="compiled",
    ),
    CSharpProbe(
        case_id="csharp-nullable-001",
        category="nullable",
        description="Dereferencing a possibly null reference yields a compiler warning.",
        source=(
            "string? name = null;\n"
            "System.Console.WriteLine(name.Length);\n"
        ),
        expected_outcome="compiled",
        expected_codes=("CS8602",),
    ),
    CSharpProbe(
        case_id="csharp-async-001",
        category="async",
        description="Blocking on an async task is reported by the async heuristic.",
        source=(
            "using System;\n"
            "using System.Threading.Tasks;\n"
            "var work = Task.Delay(1);\n"
            "var ignored = work.Result;\n"
            "Console.WriteLine(\"done\");\n"
        ),
        expected_outcome="compiled",
        expected_codes=("TAK-ASYNC001",),
    ),
    CSharpProbe(
        case_id="csharp-async-002",
        category="async",
        description="An async method with no await produces CS1998.",
        source=(
            "using System;\n"
            "using System.Threading.Tasks;\n"
            "await RunAsync();\n"
            "async Task RunAsync()\n"
            "{\n"
            "    Task.Delay(1);\n"
            "}\n"
        ),
        expected_outcome="compiled",
        expected_codes=("CS1998", "TAK-ASYNC003"),
    ),
    CSharpProbe(
        case_id="csharp-unit-001",
        category="unit",
        description="A deterministic assertion-based probe passes at runtime.",
        source=(
            "using System;\n"
            "var actual = 2 + 2;\n"
            "if (actual != 4) throw new InvalidOperationException(\"assertion failed\");\n"
            "Console.WriteLine(\"PASS\");\n"
        ),
        expected_outcome="executed",
    ),
)


def probes_by_category() -> dict[str, list[CSharpProbe]]:
    grouped: dict[str, list[CSharpProbe]] = {}
    for probe in CSHARP_PROBES:
        grouped.setdefault(probe.category, []).append(probe)
    return grouped


def probe_fingerprint(probe: CSharpProbe) -> dict[str, Any]:
    return {
        "case_id": probe.case_id,
        "category": probe.category,
        "expected_outcome": probe.expected_outcome,
        "expected_codes": list(probe.expected_codes),
    }


def evaluate_probe(
    probe: CSharpProbe,
    *,
    target_framework: str = "net10.0",
    allow_execution: bool = False,
    timeout: int = 90,
) -> dict[str, Any]:
    result: CSharpRunResult = compile_csharp(
        probe.source,
        target_framework=target_framework,
        timeout=timeout,
        allow_execution=allow_execution,
    )
    observed_codes = {item["code"] for item in result.diagnostics}
    observed_codes.update(item["code"] for item in result.source_diagnostics)
    matched_codes = set(probe.expected_codes).issubset(observed_codes)
    return {
        "probe": probe_fingerprint(probe),
        "result": result.as_dict(),
        "matched_codes": matched_codes,
        "execution_authorized": allow_execution,
    }
