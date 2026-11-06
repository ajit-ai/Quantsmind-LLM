"""Static composition-aware analysis of generated functions (Taksh v2.2.0).

No program is executed. Parameter names are normalised by position, operator
order is preserved, and control-flow shape is recorded.
"""

from __future__ import annotations

import ast
import re

__all__ = ["analyse"]

_BUILTINS = frozenset(
    {"len", "max", "min", "sum", "abs", "sorted", "range", "list", "dict",
     "set", "tuple", "str", "int", "float", "bool"}
)


def _params(node):
    return {arg.arg: f"p{index}" for index, arg in enumerate(node.args.args)}


def _names(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for item in node.elts:
            out.extend(_names(item))
        return out
    return []


def _e(node, params, local):
    if node is None:
        return ("none",)
    if isinstance(node, ast.Name):
        if node.id in params:
            return ("p", params[node.id])
        if node.id in local:
            return local[node.id]
        return ("local",)
    if isinstance(node, ast.Constant):
        return ("c", repr(node.value))
    if isinstance(node, ast.BinOp):
        op = type(node.op).__name__
        left = _e(node.left, params, local)
        right = _e(node.right, params, local)
        if op in ("Add", "Mult") and repr(left) > repr(right):
            left, right = right, left
        return ("bin", op, left, right)
    if isinstance(node, ast.UnaryOp):
        return ("un", type(node.op).__name__, _e(node.operand, params, local))
    if isinstance(node, ast.BoolOp):
        return ("bool", type(node.op).__name__,
                tuple(_e(v, params, local) for v in node.values))
    if isinstance(node, ast.Compare):
        return ("cmp", _e(node.left, params, local),
                tuple(type(op).__name__ for op in node.ops),
                tuple(_e(v, params, local) for v in node.comparators))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func = ("name", node.func.id)
        elif isinstance(node.func, ast.Attribute):
            func = ("attr", _e(node.func.value, params, local), node.func.attr)
        else:
            func = ("callable", _e(node.func, params, local))
        return ("call", func, tuple(_e(a, params, local) for a in node.args))
    if isinstance(node, ast.Attribute):
        return ("attr", _e(node.value, params, local), node.attr)
    if isinstance(node, ast.Subscript):
        return ("sub", _e(node.value, params, local), _e(node.slice, params, local))
    if isinstance(node, ast.IfExp):
        return ("ifexp", _e(node.test, params, local), _e(node.body, params, local),
                _e(node.orelse, params, local))
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        inner = dict(local)
        gens = []
        for gen in node.generators:
            for name in _names(gen.target):
                inner[name] = ("local",)
            gens.append((_e(gen.iter, params, local),
                         tuple(_e(c, params, inner) for c in gen.ifs)))
        return ("comp", type(node).__name__, _e(node.elt, params, inner), tuple(gens))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return ("seq", tuple(_e(item, params, local) for item in node.elts))
    if isinstance(node, ast.Dict):
        return ("dict", tuple(_e(v, params, local) for v in node.values))
    if isinstance(node, ast.JoinedStr):
        return ("fstr", tuple(_e(v, params, local) for v in node.values))
    return ("other", type(node).__name__)

def _stmt(node, params, local):
    if isinstance(node, ast.Return):
        return ("return", _e(node.value, params, local))
    if isinstance(node, ast.If):
        return ("if", _e(node.test, params, local),
                tuple(_stmt(x, params, local) for x in node.body),
                tuple(_stmt(x, params, local) for x in node.orelse))
    if isinstance(node, ast.For):
        inner = dict(local)
        for name in _names(node.target):
            inner[name] = ("local",)
        return ("for", _e(node.iter, params, local),
                tuple(_stmt(x, params, inner) for x in node.body))
    if isinstance(node, ast.Assign):
        return ("assign", _e(node.value, params, local))
    if isinstance(node, ast.AugAssign):
        return ("aug", type(node.op).__name__, _e(node.value, params, local))
    if isinstance(node, ast.Expr):
        return ("expr", _e(node.value, params, local))
    return ("stmt", type(node).__name__)


def _shape(statements):
    out = []
    for node in statements:
        if isinstance(node, ast.Return):
            out.append("return")
        elif isinstance(node, ast.If):
            item = ["if", _shape(node.body)]
            if node.orelse:
                item.extend(["else", _shape(node.orelse)])
            out.append(tuple(item))
        elif isinstance(node, ast.For):
            out.append(("for", _shape(node.body)))
        elif isinstance(node, ast.AugAssign):
            out.append(("aug", type(node.op).__name__))
        elif isinstance(node, ast.Assign):
            out.append("assign")
        else:
            out.append(type(node).__name__)
    return tuple(out)

class _Ops(ast.NodeVisitor):
    def __init__(self):
        self.found = set()

    def visit_BinOp(self, node):
        self.found.add({"Add": "add", "Sub": "subtract", "Mult": "multiply"}.get(
            type(node.op).__name__, type(node.op).__name__.lower()))
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == "len":
                self.found.add("length")
            elif node.func.id == "max":
                self.found.add("max_builtin")
            elif node.func.id == "min":
                self.found.add("min_builtin")
        elif isinstance(node.func, ast.Attribute):
            self.found.add({"upper": "upper", "lower": "lower",
                            "get": "mapping_get", "keys": "mapping_keys"}.get(
                                node.func.attr, "attr_" + node.func.attr))
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            name = type(op).__name__
            if name == "In":
                self.found.add("mapping_contains")
            elif name == "Gt":
                self.found.add("comparison_gt")
            elif name == "Lt":
                self.found.add("comparison_lt")
        self.generic_visit(node)

    def visit_If(self, node):
        self.found.add("if_selection")
        self.generic_visit(node)

    def visit_For(self, node):
        self.found.add("iteration")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.value, ast.Constant) and node.value.value == 1:
            self.found.add("count_items")
        elif isinstance(node.value, ast.Name):
            self.found.add("sum_items")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.found.add("comprehension")
        for cond in node.generators[0].ifs:
            if isinstance(cond, ast.Compare) and cond.ops:
                if isinstance(cond.ops[0], ast.Gt):
                    self.found.add("filter_positive")
                elif isinstance(cond.ops[0], ast.Lt):
                    self.found.add("filter_negative")
        self.generic_visit(node)


def _free_names(tree, params):
    assigned = set(params)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, (ast.For, ast.comprehension)):
            assigned.update(_names(node.target))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assigned.update(_names(target))
        elif isinstance(node, ast.AugAssign):
            assigned.update(_names(node.target))
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - assigned - _BUILTINS)

def analyse(signature: str, body: str) -> dict:
    """Return JSON-safe static facts about one function."""
    source = signature + "\n" + body
    try:
        tree = ast.parse(source)
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef):
            raise ValueError("not a function definition")
    except (SyntaxError, ValueError, IndexError) as exc:
        return {"parses": False, "error": f"{type(exc).__name__}: {exc}",
                "body_dump": None, "program_signature": None, "shape": None,
                "operations": [], "params_used": [], "undefined_names": [],
                "key": None}
    params = _params(function)
    key = tuple(_stmt(item, params, {}) for item in function.body)
    normalized = body
    for old, new in params.items():
        normalized = re.sub(r"\b" + re.escape(old) + r"\b", new, normalized)
    normalized_source = "def f(" + ", ".join(params.values()) + "):\n" + normalized
    normalized_tree = ast.parse(normalized_source)
    normalized_body = normalized_tree.body[0].body
    visitor = _Ops()
    for item in function.body:
        visitor.visit(item)
    used = {n.id for n in ast.walk(function)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return {
        "parses": True,
        "error": None,
        "body_dump": ast.dump(ast.Module(body=normalized_body, type_ignores=[]),
                              annotate_fields=False, include_attributes=False),
        "program_signature": repr(key),
        "shape": repr(_shape(function.body)),
        "operations": sorted(visitor.found),
        "params_used": sorted(params[name] for name in used if name in params),
        "undefined_names": _free_names(function, params),
        "key": key,
    }

