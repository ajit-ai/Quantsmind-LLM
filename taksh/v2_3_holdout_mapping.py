"""Taksh v2.3.0 — Held-out task mapping diagnostics.

Measures whether the model can map unfamiliar instructions to code structures.
Work Package 4 of Taksh v2.3.0.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Set


def fingerprint_instruction(instruction: str) -> str:
    """Create a content fingerprint for an instruction to detect contamination."""
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]


def build_mapping_case(
    case_id: str,
    instruction: str,
    expected_structure: Dict[str, Any],
    is_negative: bool = False,
    paraphrase_of: str = None,
) -> Dict[str, Any]:
    """Build a held-out mapping case.

    Args:
        case_id: unique identifier
        instruction: the instruction text
        expected_structure: expected code structure signature
        is_negative: if True, this is a negative control
        paraphrase_of: case_id this paraphrases (for contamination tracking)
    """
    return {
        "case_id": case_id,
        "instruction": instruction,
        "instruction_fingerprint": fingerprint_instruction(instruction),
        "expected_structure": expected_structure,
        "is_negative": is_negative,
        "paraphrase_of": paraphrase_of,
    }


def evaluate_mapping_case(case: Dict[str, Any], completion: str) -> Dict[str, Any]:
    """Evaluate a single mapping case against a completion."""
    import ast

    expected = case.get("expected_structure", {})
    result: Dict[str, Any] = {
        "case_id": case["case_id"],
        "instruction": case["instruction"],
        "is_negative": case.get("is_negative", False),
        "paraphrase_of": case.get("paraphrase_of"),
        "parses": False,
        "structure_match": False,
        "expected_ops": expected.get("operations", []),
        "found_ops": [],
        "expected_params": expected.get("params", []),
        "found_params": [],
    }

    try:
        tree = ast.parse(completion)
    except SyntaxError:
        result["parse_error"] = "syntax error"
        return result

    result["parses"] = True

    # Find function definitions
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        result["error"] = "no function definition found"
        return result

    func = funcs[0]
    result["found_params"] = [arg.arg for arg in func.args.args]

    # Extract operations
    found_ops = set()
    for node in ast.walk(func):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                found_ops.add("add")
            elif isinstance(node.op, ast.Sub):
                found_ops.add("subtract")
            elif isinstance(node.op, ast.Mult):
                found_ops.add("multiply")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                found_ops.add(node.func.id)
    result["found_ops"] = sorted(found_ops)

    # Check structure match
    expected_ops = set(expected.get("operations", []))
    expected_params = set(expected.get("params", []))

    ops_match = expected_ops.issubset(found_ops)
    params_match = expected_params.issubset(set(result["found_params"]))
    result["structure_match"] = ops_match and params_match

    return result


def summarize_mapping(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize mapping evaluation results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "no results"}

    parse_rate = sum(1 for r in results if r.get("parses")) / total
    match_rate = sum(1 for r in results if r.get("structure_match")) / total

    negatives = [r for r in results if r.get("is_negative")]
    positive = [r for r in results if not r.get("is_negative")]

    neg_match_rate = (
        sum(1 for r in negatives if r.get("structure_match")) / len(negatives)
        if negatives else 0.0
    )

    return {
        "total": total,
        "parse_rate": round(parse_rate, 4),
        "structure_match_rate": round(match_rate, 4),
        "negative_control_match_rate": round(neg_match_rate, 4),
        "negative_control_count": len(negatives),
        "positive_count": len(positive),
        "recommendation": "PASS" if match_rate >= 0.7 and neg_match_rate <= 0.3 else "REVIEW",
    }