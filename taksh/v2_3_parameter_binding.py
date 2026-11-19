"""Taksh v2.3.0 — Parameter-binding diagnostics.

Extends existing evaluation to test parameter binding robustness.
Work Package 2 of Taksh v2.3.0.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Set


def analyse_parameter_binding(signature: str, body: str) -> Dict[str, Any]:
    """Analyse parameter binding in a function using static analysis."""
    try:
        tree = ast.parse(signature + "\n" + body)
    except SyntaxError:
        return {"parses": False, "error": "syntax error", "params": [], "params_used": [], "undefined_names": [], "consistent_references": False}

    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return {"parses": False, "error": "not a function definition", "params": [], "params_used": [], "undefined_names": [], "consistent_references": False}

    param_names = [arg.arg for arg in func.args.args]
    param_set = set(param_names)

    used_names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)

    assigned_names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                assigned_names.add(node.target.id)

    builtins = {"len", "max", "min", "sum", "abs", "sorted", "range", "list",
                "dict", "set", "tuple", "str", "int", "float", "bool", "print",
                "True", "False", "None"}
    undefined = sorted(used_names - param_set - assigned_names - builtins)
    params_used = sorted(param_set & used_names)
    consistent = len(undefined) == 0

    return {
        "parses": True,
        "error": None,
        "params": param_names,
        "params_used": params_used,
        "undefined_names": undefined,
        "consistent_references": consistent,
        "all_params_used": len(params_used) == len(param_names),
    }

def check_renamed_params(signature, body, new_param_names):
    original = analyse_parameter_binding(signature, body)
    try:
        tree = ast.parse(signature + "\n" + body)
    except SyntaxError:
        return {"error": "syntax error", "binding_preserved": False}
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return {"error": "not a function", "binding_preserved": False}
    original_params = [arg.arg for arg in func.args.args]
    if len(new_param_names) != len(original_params):
        return {"error": "param count mismatch", "binding_preserved": False}
    import re
    renamed_sig = signature
    renamed_body = body
    for old, new in zip(original_params, new_param_names):
        renamed_sig = re.sub(r"\b" + old + r"\b", new, renamed_sig)
        renamed_body = re.sub(r"\b" + old + r"\b", new, renamed_body)
    renamed = analyse_parameter_binding(renamed_sig, renamed_body)
    return {
        "original_params": original_params,
        "new_params": new_param_names,
        "original_binding": original,
        "renamed_binding": renamed,
        "binding_preserved": (
            renamed.get("parses", False)
            and renamed.get("consistent_references", False)
            and len(renamed.get("undefined_names", [])) == 0
        ),
    }


def check_reordered_params(signature, body, new_order):
    try:
        tree = ast.parse(signature + "\n" + body)
    except SyntaxError:
        return {"error": "syntax error", "binding_preserved": False}
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return {"error": "not a function", "binding_preserved": False}
    original_params = [arg.arg for arg in func.args.args]
    if sorted(new_order) != list(range(len(original_params))):
        return {"error": "invalid reorder", "binding_preserved": False}
    reordered_params = [original_params[i] for i in new_order]
    return {
        "original_params": original_params,
        "reordered_params": reordered_params,
        "new_order": new_order,
        "binding_preserved": True,
    }


def check_unseen_variable_names(signature, body, unseen_names):
    results = {}
    for name in unseen_names:
        test_body = body + "\n    _ = " + name
        analysis = analyse_parameter_binding(signature, test_body)
        results[name] = {
            "defined": name not in analysis.get("undefined_names", []),
            "analysis": analysis,
        }
    return {
        "unseen_names": unseen_names,
        "results": results,
        "all_undefined": all(
            name in r["analysis"].get("undefined_names", [])
            for name, r in results.items()
        ),
    }


def compare_binding(baseline, candidate):
    return {
        "baseline_consistent": baseline.get("consistent_references", False),
        "candidate_consistent": candidate.get("consistent_references", False),
        "baseline_undefined": baseline.get("undefined_names", []),
        "candidate_undefined": candidate.get("undefined_names", []),
        "improved": (
            candidate.get("consistent_references", False)
            and not baseline.get("consistent_references", False)
        ),
        "regressed": (
            baseline.get("consistent_references", False)
            and not candidate.get("consistent_references", False)
        ),
    }

