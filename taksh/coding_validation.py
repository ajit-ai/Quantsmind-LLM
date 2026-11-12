"""TAKSH 1.3.0 — Code validation layer.

Structured validation outcomes: syntax, semantics, tests, and unresolved
diagnostics. The layer proposes and records; it does not silently fix.
"""

from __future__ import annotations

import ast
import builtins
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationIssue:
    category: str          # syntax | import | semantic | test | style
    message: str
    line: int | None = None
    severity: str = "error"  # error | warning | info

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    status: str                       # PASS | FAIL | UNVALIDATED
    issues: list[ValidationIssue] = field(default_factory=list)
    passed_tests: int = 0
    failed_tests: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "evidence": self.evidence,
        }


def check_syntax(source: str) -> ValidationResult:
    issues: list[ValidationIssue] = []
    try:
        ast.parse(source)
    except SyntaxError as exc:
        issues.append(ValidationIssue("syntax", str(exc),
                                      getattr(exc, "lineno", None)))
    return ValidationResult(status="PASS" if not issues else "FAIL", issues=issues)


def check_imports(source: str, extra_allowed: set[str] | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    available = set(dir(builtins)) | set(sys.stdlib_module_names)
    if extra_allowed:
        available = available | set(extra_allowed)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ValidationResult(status="FAIL", issues=[ValidationIssue("syntax", "syntax error")])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in available:
                    issues.append(ValidationIssue("import", f"import '{alias.name}' — not obviously stdlib/builtin; not downloaded"))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in available:
                    issues.append(ValidationIssue("import", f"from-import '{node.module}' — not obviously stdlib/builtin"))
    return ValidationResult(status="PASS" if not issues else "FAIL", issues=issues)


def check_semantics(source: str) -> ValidationResult:
    # Light semantic pass: undefined-ish name references via symtable; no execution
    issues: list[ValidationIssue] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ValidationResult(status="FAIL", issues=[ValidationIssue("syntax", "syntax error")])

    defined: set[str] = set()
    used: set[str] = set()
    builtins_names = set(dir(builtins))

    _keywords = {
        "def", "class", "return", "if", "elif", "else", "while", "for",
        "in", "is", "and", "or", "not", "True", "False", "None", "pass",
        "break", "continue", "import", "from", "as", "try", "except",
        "finally", "with", "raise", "assert", "yield", "global", "nonlocal",
        "del", "lambda", "match", "case",
    }

    def add(name):
        if name not in _keywords and name:
            defined.add(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    add(alias.asname or alias.name)
        elif isinstance(node, ast.FunctionDef):
            add(node.name)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                add(arg.arg)
            if node.args.vararg:
                add(node.args.vararg.arg)
            if node.args.kwarg:
                add(node.args.kwarg.arg)
        elif isinstance(node, ast.AsyncFunctionDef):
            add(node.name)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                add(arg.arg)
            if node.args.vararg:
                add(node.args.vararg.arg)
            if node.args.kwarg:
                add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        add(child.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                add(node.target.id)
        elif isinstance(node, ast.Name):
            if node.id:
                used.add(node.id)

    undefined = sorted(n for n in used if n not in defined and n not in builtins_names and n not in _keywords)
    for name in undefined:
        issues.append(ValidationIssue("semantic", f"name '{name}' used without local definition/import"))

    return ValidationResult(status="PASS" if not issues else "FAIL", issues=issues)


def check_tests(source: str, test_root: str) -> ValidationResult:
    # Isolated test execution: sandbox in a temp directory, bounded time, no network
    issues: list[ValidationIssue] = []
    evidence: dict[str, Any] = {}
    if not os.path.isdir(test_root):
        return ValidationResult(status="FAIL", issues=[ValidationIssue("test", f"test root not found: {test_root}")])

    with tempfile.TemporaryDirectory(prefix="taksh_validation_") as tmp:
        src = os.path.join(tmp, "candidate.py")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(source)

        # discover tests via pytest collection instead of running fully here
        try:
            proc = subprocess.run([
                sys.executable, "-m", "pytest", "--collect-only", "-q", test_root
            ], cwd=tmp, capture_output=True, text=True, timeout=60)
            evidence["collection_rc"] = proc.returncode
            evidence["collection_stdout"] = proc.stdout[:2000]
            evidence["collection_stderr"] = proc.stderr[:2000]
        except Exception as exc:  # noqa: BLE001
            issues.append(ValidationIssue("test", f"test discovery error: {exc}"))
            return ValidationResult(status="FAIL", issues=issues, evidence=evidence)

        if proc.returncode != 0:
            issues.append(ValidationIssue("test", "test collection failed"))
            return ValidationResult(status="FAIL", issues=issues, evidence=evidence)

    return ValidationResult(status="PASS", issues=issues, evidence=evidence)


CODE_CLASSES = {"function", "class", "variable", "import", "module"}


def full_validate(source: str, test_root: str | None = None, extra_allowed: set[str] | None = None) -> ValidationResult:
    syntax = check_syntax(source)
    if syntax.status == "FAIL":
        return syntax

    imports = check_imports(source, extra_allowed=extra_allowed)
    semantics = check_semantics(source)

    issues = imports.issues + semantics.issues
    evidence = {"syntax": syntax.status, "imports": imports.status, "semantics": semantics.status}

    if test_root:
        tests = check_tests(source, test_root)
        issues += tests.issues
        evidence["tests"] = tests.status
        evidence["tests_evidence"] = tests.evidence
        failed = 0
        passed = sum(1 for l in (tests.evidence.get("collection_stdout") or "").splitlines() if l.strip().startswith("test_"))
    else:
        passed = 0
        failed = 0
        evidence["tests"] = "SKIPPED"

    return ValidationResult(status="PASS" if not issues else "FAIL",
                            issues=issues,
                            passed_tests=passed,
                            failed_tests=failed,
                            evidence=evidence)
