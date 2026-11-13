"""TAKSH 1.3.0 — Python specialist implementations.

Each specialist exposes a stable contract: input, output, and testability.
Specialists perform local work (parsing, formatting, suggestion) — they do not
generate model weights, nor do they claim correctness.
"""

from __future__ import annotations

import ast
import io
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class SpecialistResult:
    specialist: str
    status: str               # OK | FAIL | UNVALIDATED
    output: str
    findings: list[str] = None

    def __post_init__(self):
        if self.findings is None:
            self.findings = []


class Specialist(Protocol):
    name: str

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult: ...


class PythonGenerationSpecialist:
    name = "python_generation"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        # Synthesize a small starter function from a declarative brief
        if not target:
            return SpecialistResult(self.name, "FAIL", source, ["target required"])
        body = textwrap.dedent(source).strip()
        return SpecialistResult(self.name, "OK", body, ["generation is a template; no model used"])


class PythonCompletionSpecialist:
    name = "python_completion"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        # Indentation-aware line completion from prior context
        lines = source.splitlines()
        last = lines[-1] if lines else ""
        indent = len(last) - len(last.lstrip())
        suggestion = (" " * indent) + "pass"
        return SpecialistResult(self.name, "OK", suggestion, ["indentation inferred"])


class PythonExplanationSpecialist:
    name = "python_explanation"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return SpecialistResult(self.name, "FAIL", str(exc), ["unparseable"])
        names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return SpecialistResult(self.name, "OK", f"defines: {', '.join(names) or 'module-level statements'}", names)


class PythonDebuggingSpecialist:
    name = "python_debugging"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        issues = []
        if not source.strip():
            issues.append("empty source")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            issues.append(f"syntax error: {exc}")
        return SpecialistResult(self.name, "OK" if not issues else "FAIL",
                                "debugging findings attached", issues)


class PythonRefactoringSpecialist:
    name = "python_refactoring"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return SpecialistResult(self.name, "FAIL", source, ["unparseable"])
        # simple, safe refactor: suggestion only, no destructive rewrite
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        return SpecialistResult(self.name, "OK", f"candidate function(s): {', '.join(names)}", names)


class TestGenerationSpecialist:
    name = "test_generation"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return SpecialistResult(self.name, "FAIL", "unparseable", [])
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        stubs = [f"def test_{fn}_placeholder():\n    assert True\n" for fn in funcs]
        code = "\n".join(stubs) if stubs else "def test_module_placeholder():\n    assert True\n"
        return SpecialistResult(self.name, "OK", code, funcs)


class TestFailureDiagnosisSpecialist:
    name = "test_failure_diagnosis"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        # Expects target={"stderr": "...", "returncode": N}
        if not isinstance(target, dict):
            return SpecialistResult(self.name, "FAIL", "target must be a dict with stderr", [])
        stderr = target.get("stderr", "") or target.get("stdout", "")
        markers = []
        if "AssertionError" in stderr:
            markers.append("assertion failure")
        if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
            markers.append("import error")
        if "SyntaxError" in stderr:
            markers.append("syntax error")
        return SpecialistResult(self.name, "OK", stderr[:1000], markers or ["unknown failure"])


class CodeReviewSpecialist:
    name = "code_review"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        findings = []
        if len(source.splitlines()) > 400:
            findings.append("large module; consider splitting")
        if source.count("eval("):
            findings.append("eval/exec detected")
        if "except:" in source or re.search(r"except\s*:", source):
            findings.append("bare except clause")
        return SpecialistResult(self.name, "OK", "review findings", findings)


class DependencyAnalysisSpecialist:
    name = "dependency_analysis"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return SpecialistResult(self.name, "FAIL", "unparseable", [])
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return SpecialistResult(self.name, "OK", "\n".join(sorted(set(imports))), list(sorted(set(imports))))


class DocumentationGenerationSpecialist:
    name = "documentation_generation"

    def run(self, source: str, *, target: str | None = None) -> SpecialistResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return SpecialistResult(self.name, "FAIL", "unparseable", [])
        docs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node) or ""
                docs.append(f"{node.name}: {doc or 'no docstring'}")
        return SpecialistResult(self.name, "OK", "\n".join(docs) or "no public functions", docs)


SPECIALISTS = {
    "python_generation": PythonGenerationSpecialist(),
    "python_completion": PythonCompletionSpecialist(),
    "python_explanation": PythonExplanationSpecialist(),
    "python_debugging": PythonDebuggingSpecialist(),
    "python_refactoring": PythonRefactoringSpecialist(),
    "test_generation": TestGenerationSpecialist(),
    "test_failure_diagnosis": TestFailureDiagnosisSpecialist(),
    "code_review": CodeReviewSpecialist(),
    "dependency_analysis": DependencyAnalysisSpecialist(),
    "documentation_generation": DocumentationGenerationSpecialist(),
}
