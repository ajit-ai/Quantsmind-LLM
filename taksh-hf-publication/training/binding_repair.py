"""Conservative repair of free parameter names in generated completions.

Repairs are applied only when there is exactly one unbound name and exactly
one unused requested parameter. Any other shape is left unchanged. This module
is a deterministic source transform, never executes code, and abstains on
nested scopes, unknown calls, unsafe attributes, or ambiguous mappings.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import asdict, dataclass

_SAFE_CALLS = frozenset({
    "len", "max", "min", "sum", "abs", "sorted", "range", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "isinstance", "enumerate",
    "zip", "any", "all", "reversed", "round",
})
_SAFE_ATTRIBUTES = frozenset({
    "append", "extend", "get", "keys", "values", "items", "upper", "lower",
    "strip", "startswith", "endswith", "split", "join", "index", "count",
})
_SIGNATURE = re.compile(r"(?m)^(?P<signature>def\s+[A-Za-z_]\w*\([^\n]*\):)\s*$")


@dataclass(frozen=True)
class RepairInfo:
    completion: str
    status: str
    reason: str
    changed: bool
    mapping: dict[str, str]
    unbound: list[str]
    unused_params: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_signature(prompt: str) -> str | None:
    matches = list(_SIGNATURE.finditer(prompt or ""))
    if not matches:
        matches = [item for item in re.finditer(r"def\s+[A-Za-z_]\w*\([^\n]*\):", prompt or "")]
    if not matches:
        return None
    return matches[-1].groupdict().get("signature") or matches[-1].group(0)


def _indent_body(completion: str) -> str:
    lines = [line.rstrip() for line in (completion or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def _named_arguments(node: ast.FunctionDef) -> list[str]:
    args = node.args
    names = [arg.arg for arg in args.posonlyargs]
    names += [arg.arg for arg in args.args]
    names += [arg.arg for arg in args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _assigned_names(func: ast.FunctionDef) -> set[str]:
    assigned: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            assigned.update(_target_names(node.target))
        elif isinstance(node, ast.comprehension):
            assigned.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    assigned.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(node.name)
    return assigned


def _target_names(target: ast.AST | None) -> set[str]:
    if target is None:
        return set()
    names = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Starred):
            names.update(_target_names(node.value))
    return names


def _loaded_names_by_position(func: ast.FunctionDef) -> list[str]:
    nodes = [node for node in ast.walk(func)
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]
    nodes.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    seen = set()
    ordered = []
    for node in nodes:
        if node.id not in seen:
            seen.add(node.id)
            ordered.append(node.id)
    return ordered


def _abstain(status: str, reason: str, completion: str,
             unbound: list[str] | None = None,
             unused_params: list[str] | None = None) -> RepairInfo:
    return RepairInfo(
        completion=completion, status=status, reason=reason, changed=False,
        mapping={}, unbound=list(unbound or []), unused_params=list(unused_params or []),
    )


def _unsafe_construct(tree: ast.Module, outer: ast.FunctionDef) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not outer:
            return "nested scope or lambda"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import statement"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global/nonlocal declaration"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _SAFE_ATTRIBUTES:
                return f"unsafe attribute: {node.attr}"
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                if function.id not in _SAFE_CALLS:
                    return f"unsafe call: {function.id}"
            elif isinstance(function, ast.Attribute):
                if function.attr not in _SAFE_ATTRIBUTES:
                    return f"unsafe method: {function.attr}"
            else:
                return "unsafe call form"
    return None


def repair_completion_info(signature: str | None, completion: str) -> RepairInfo:
    if not signature:
        return _abstain("abstained_no_signature", "no signature available", completion)
    if not completion.strip():
        return _abstain("abstained_empty_source", "empty completion", completion)
    signature = signature.rstrip("\n")
    stripped = completion.lstrip()
    if stripped.startswith("def "):
        return _abstain("abstained_full_function", "whole-function completions are left unchanged", completion)

    body = _indent_body(completion)
    source = signature + "\n" + body
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _abstain("abstained_parse_error", f"syntax error: {exc}", completion)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return _abstain("abstained_structure", "expected one function definition", completion)
    func = functions[0]

    unsafe = _unsafe_construct(tree, func)
    if unsafe is not None:
        return _abstain("abstained_unsafe", unsafe, completion)
    if not any(isinstance(node, ast.Return) for node in ast.walk(func)):
        return _abstain("abstained_no_return", "completion has no return statement", completion)

    params = _named_arguments(func)
    assigned = _assigned_names(func)
    loaded = _loaded_names_by_position(func)
    unbound = [name for name in loaded
               if name not in params and name not in assigned and name not in _SAFE_CALLS]
    unused_params = [name for name in params if name not in loaded]

    if not unbound:
        return _abstain("unchanged_no_candidates", "no unbound replacement candidates", completion,
                        [], unused_params)
    if not unused_params:
        return _abstain("abstained_no_unused_params", "no unused requested parameter exists", completion,
                        unbound, unused_params)
    if len(unbound) != 1 or len(unused_params) != 1:
        return _abstain(
            "abstained_ambiguous",
            f"expected one unbound name and one unused parameter; got {len(unbound)} and {len(unused_params)}",
            completion,
            unbound,
            unused_params,
        )

    mapping = {unbound[0]: unused_params[0]}
    repaired = _rename_names(source, mapping)
    lines = repaired.splitlines()
    body_out = "\n".join(lines[1:])
    if completion.endswith("\n"):
        body_out += "\n"
    return RepairInfo(
        completion=body_out, status="repaired", reason="unique one-to-one mapping",
        changed=True, mapping=mapping, unbound=unbound, unused_params=unused_params,
    )


def repair_completion(signature: str | None, completion: str) -> str:
    return repair_completion_info(signature, completion).completion


def repair_from_prompt(prompt: str, completion: str) -> str:
    return repair_completion(extract_signature(prompt), completion)


def repair_from_prompt_info(prompt: str, completion: str) -> RepairInfo:
    return repair_completion_info(extract_signature(prompt), completion)


def _rename_names(source: str, mapping: dict[str, str]) -> str:
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for index, token in enumerate(tokens):
        if token.type == tokenize.NAME and token.string in mapping:
            tokens[index] = token._replace(string=mapping[token.string])
    return tokenize.untokenize(tokens)
