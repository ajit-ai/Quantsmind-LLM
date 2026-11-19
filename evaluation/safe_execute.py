"""Restricted execution of explicitly approved pure Python function fixtures.

This is a gate for the binding-repair qualification suite, not a sandbox for
arbitrary user code or model output from untrusted prompts.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.If, ast.For, ast.Pass,
    ast.Name, ast.Load, ast.Store, ast.Del, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.Call, ast.keyword, ast.Attribute, ast.Subscript,
    ast.Slice, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp,
    ast.GeneratorExp, ast.comprehension, ast.IfExp, ast.JoinedStr, ast.FormattedValue,
)
_ALLOWED_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.MatMult,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
)
_ALLOWED_BUILTINS = frozenset({
    "len", "max", "min", "sum", "abs", "sorted", "range", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "isinstance", "enumerate",
    "zip", "any", "all", "reversed", "round",
})
_FORBIDDEN_NAMES = frozenset({
    "__builtins__", "__import__", "breakpoint", "compile", "eval", "exec",
    "exit", "globals", "locals", "open", "input", "print", "vars", "dir",
    "getattr", "setattr", "delattr", "super", "type", "object",
})
_ALLOWED_ATTRIBUTES = frozenset({
    "append", "extend", "get", "keys", "values", "items", "upper", "lower",
    "strip", "startswith", "endswith", "split", "join", "index", "count",
})
_MAX_NODES = 240


@dataclass(frozen=True)
class SafeExecutionResult:
    passed: bool
    reason: str


def validate_source(source: str) -> tuple[ast.Module | None, str]:
    if len(source) > 4000:
        return None, "source too long"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return None, f"syntax error: {exc}"
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1:
        return None, "expected exactly one top-level function"
    function = functions[0]
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_NODES:
        return None, "source exceeds node limit"
    allowed = _ALLOWED_NODES + _ALLOWED_OPERATORS
    function_name = function.name
    for node in nodes:
        if not isinstance(node, allowed):
            return None, f"disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _ALLOWED_ATTRIBUTES:
                return None, f"disallowed attribute: {node.attr}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _ALLOWED_BUILTINS:
                    return None, f"disallowed call: {func.id}"
            elif isinstance(func, ast.Attribute):
                if func.attr not in _ALLOWED_ATTRIBUTES:
                    return None, f"disallowed method: {func.attr}"
            else:
                return None, "disallowed call form"
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                return None, f"disallowed name: {node.id}"
    if any(isinstance(node, ast.Call)
           and isinstance(node.func, ast.Name)
           and node.func.id == function_name
           for node in nodes):
        return None, "recursive calls are not approved for this gate"
    return tree, "ok"


def execute_tests(source: str, function_name: str, tests: list[dict[str, Any]]) -> SafeExecutionResult:
    _tree, reason = validate_source(source)
    if _tree is None:
        return SafeExecutionResult(False, reason)
    script = (
        source
        + "\n\nif __name__ == '__main__':\n"
        + f"    function = globals()[{function_name!r}]\n"
        + f"    tests = {tests!r}\n"
        + "    for test in tests:\n"
        + "        actual = function(*test['args'], **test['kwargs'])\n"
        + "        assert actual == test['expected'], (actual, test['expected'])\n"
        + "    print('PASS')\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            input="",
            capture_output=True,
            text=True,
            # v1.4.1: the previous 2s budget flaked under full-suite load
            # (subprocess spawn dominates). Fixture functions run in
            # milliseconds; 10s preserves determinism without retries.
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SafeExecutionResult(False, "execution timed out")
    if completed.returncode == 0:
        return SafeExecutionResult(True, "pass")
    detail = (completed.stderr or completed.stdout or "non-zero exit").strip().splitlines()
    return SafeExecutionResult(False, detail[-1] if detail else "non-zero exit")
