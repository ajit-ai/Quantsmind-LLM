"""Taksh v2.3.0 — Composition generalization diagnostics.

Evaluates composition generalization capabilities.
Work Package 3 of Taksh v2.3.0.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Set


KNOWN_OPERATIONS = frozenset({
    "add", "subtract", "multiply", "max_builtin", "min_builtin",
    "comparison_gt", "comparison_lt", "if_selection", "filter_positive",
    "filter_negative", "comprehension", "length", "upper", "lower",
    "mapping_get", "mapping_keys", "mapping_contains", "iteration",
    "count_items", "sum_items", "attr_get",
})


def analyse_composition(signature: str, body: str) -> Dict[str, Any]:
    """Analyse composition structure of a function."""
    try:
        tree = ast.parse(signature + "\n" + body)
    except SyntaxError:
        return {"parses": False, "error": "syntax error", "operations": [], "composition_key": None, "depth": 0}

    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return {"parses": False, "error": "not a function", "operations": [], "composition_key": None, "depth": 0}

    # Extract operations
    ops = set()
    for node in ast.walk(func):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                ops.add("add")
            elif isinstance(node.op, ast.Sub):
                ops.add("subtract")
            elif isinstance(node.op, ast.Mult):
                ops.add("multiply")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                ops.add(node.func.id)
        elif isinstance(node, ast.If):
            ops.add("if_selection")
        elif isinstance(node, ast.ListComp):
            ops.add("comprehension")

    # Composition key: sorted operations + structure
    comp_key = tuple(sorted(ops))

    # Depth of composition
    depth = _compute_depth(func.body)

    return {
        "parses": True,
        "error": None,
        "operations": sorted(ops),
        "composition_key": comp_key,
        "depth": depth,
        "seen_operations": ops & KNOWN_OPERATIONS,
        "novel_operations": ops - KNOWN_OPERATIONS,
    }


def _compute_depth(nodes, current=0):
    if not nodes:
        return current
    max_d = current
    for node in nodes:
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            d = _compute_depth(getattr(node, "body", []), current + 1)
            max_d = max(max_d, d)
            if hasattr(node, "orelse") and node.orelse:
                d = _compute_depth(node.orelse, current + 1)
                max_d = max(max_d, d)
        elif isinstance(node, ast.FunctionDef):
            d = _compute_depth(node.body, current + 1)
            max_d = max(max_d, d)
        else:
            max_d = max(max_d, current)
    return max_d


def classify_composition(case_type: str, seen: bool, ops: List[str]) -> Dict[str, Any]:
    """Classify a composition case into categories."""
    categories = {
        "seen_composition": seen and case_type == "seen",
        "unseen_combination": not seen and case_type == "unseen",
        "reordered_operations": case_type == "reordered",
        "nested_composition": case_type == "nested",
        "equivalent_syntax": case_type == "equivalent",
        "novel_structure": case_type == "novel",
    }
    return categories


def compare_composition(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compare composition results between baseline and candidate."""
    baseline_ops = set(baseline.get("operations", []))
    candidate_ops = set(candidate.get("operations", []))
    return {
        "baseline_parses": baseline.get("parses", False),
        "candidate_parses": candidate.get("parses", False),
        "operations_preserved": baseline_ops.issubset(candidate_ops),
        "baseline_depth": baseline.get("depth", 0),
        "candidate_depth": candidate.get("depth", 0),
        "depth_changed": baseline.get("depth", 0) != candidate.get("depth", 0),
        "composition_key_match": baseline.get("composition_key") == candidate.get("composition_key"),
    }


def test_equivalent_syntax(signature: str, bodies: List[str]) -> Dict[str, Any]:
    """Test if different syntax implementations produce same composition analysis."""
    results = []
    for body in bodies:
        analysis = analyse_composition(signature, body)
        results.append(analysis)
    # Check if all produce the same composition key
    keys = [r.get("composition_key") for r in results if r.get("parses")]
    all_same = len(set(keys)) == 1 if keys else False
    return {
        "implementations": len(bodies),
        "all_parse": all(r.get("parses", False) for r in results),
        "composition_keys": keys,
        "all_same_composition": all_same,
        "results": results,
    }