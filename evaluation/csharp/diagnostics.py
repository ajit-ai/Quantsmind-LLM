"""Static and compiler-output diagnostics for C# evaluation artifacts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_ROSLYN_DIAGNOSTIC = re.compile(
    r"(?P<file>[^\s:()]+)\((?P<line>\d+),(?P<column>\d+)\):\s*"
    r"(?P<severity>warning|error)\s+(?P<code>CS\d{4})\s*:\s*(?P<message>.+)",
    re.IGNORECASE,
)
_ASYNC_WAIT_RESULT = re.compile(r"\.(?:Result\b|Wait\s*\()")
_ASYNC_VOID = re.compile(r"\basync\s+void\b")
_ASYNC_METHOD = re.compile(r"\basync\s+(?:static\s+)?(?:Task|ValueTask|void)\s+\w+\s*\(")
_AWAIT = re.compile(r"\bawait\b")


@dataclass(frozen=True)
class CompilerDiagnostic:
    severity: str
    code: str
    line: int | None
    column: int | None
    message: str
    category: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def category_for(code: str) -> str:
    normalized = code.upper()
    if normalized in {"CS1983", "CS1998", "CS4014"}:
        return "async"
    if normalized.startswith("CS8"):
        return "nullable"
    digits = normalized[2:]
    number = int(digits) if digits.isdigit() else -1
    # Roslyn error numbers are zero-padded (CS0029 -> 29, CS1002 -> 1002), so the
    # numeric value -- not the string prefix -- separates parse errors (CS1xxx)
    # from compiler errors such as type mismatches (CS0xxx/CS2xxx and friends).
    if 1000 <= number < 2000:
        return "syntax"
    if number > 0:
        return "semantic"
    return "other"


def parse_roslyn_diagnostics(output: str) -> list[CompilerDiagnostic]:
    diagnostics = []
    seen = set()
    for line in output.splitlines():
        match = _ROSLYN_DIAGNOSTIC.search(line.strip())
        if not match:
            continue
        code = match.group("code").upper()
        diagnostic = CompilerDiagnostic(
            severity=match.group("severity").lower(),
            code=code,
            line=int(match.group("line")),
            column=int(match.group("column")),
            message=match.group("message").strip(),
            category=category_for(code),
        )
        fingerprint = (
            diagnostic.severity, diagnostic.code, diagnostic.line,
            diagnostic.column, diagnostic.message, diagnostic.category,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        diagnostics.append(diagnostic)
    return diagnostics


def detect_async_misuse(source: str) -> list[CompilerDiagnostic]:
    diagnostics = []
    for line_number, line in enumerate(source.splitlines(), 1):
        if _ASYNC_WAIT_RESULT.search(line):
            diagnostics.append(
                CompilerDiagnostic(
                    severity="warning",
                    code="TAK-ASYNC001",
                    line=line_number,
                    column=None,
                    message="synchronous blocking on Task (.Result/.Wait); review for deadlock risk",
                    category="async",
                )
            )
        if _ASYNC_VOID.search(line):
            diagnostics.append(
                CompilerDiagnostic(
                    severity="warning",
                    code="TAK-ASYNC002",
                    line=line_number,
                    column=None,
                    message="async void does not compose with exception or completion handling",
                    category="async",
                )
            )
    for match in _ASYNC_METHOD.finditer(source):
        block_start = source.rfind("\n", 0, match.start()) + 1
        next_brace = source.find("{", match.end())
        if next_brace == -1:
            continue
        depth = 0
        block_end = len(source)
        for index in range(next_brace, len(source)):
            char = source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    block_end = index + 1
                    break
        body = source[next_brace:block_end]
        if not _AWAIT.search(body):
            line = source[: match.start()].count("\n") + 1
            diagnostics.append(
                CompilerDiagnostic(
                    severity="warning",
                    code="TAK-ASYNC003",
                    line=line,
                    column=None,
                    message="async method body contains no await; it completes synchronously",
                    category="async",
                )
            )
    return diagnostics


def analyse_csharp_source(source: str) -> dict[str, Any]:
    diagnostics = detect_async_misuse(source)
    return {
        "diagnostics": [item.as_dict() for item in diagnostics],
        "summary": summarize_diagnostics(diagnostics),
    }


def summarize_diagnostics(diagnostics) -> dict[str, Any]:
    items = list(diagnostics)
    by_category: dict[str, int] = {}
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1
    warnings = sum(1 for item in items if item.severity == "warning")
    errors = sum(1 for item in items if item.severity == "error")
    return {
        "total": len(items),
        "warnings": warnings,
        "errors": errors,
        "nullable": by_category.get("nullable", 0),
        "async": by_category.get("async", 0),
        "warning_free": warnings == 0 and errors == 0,
        "by_category": dict(sorted(by_category.items())),
    }
