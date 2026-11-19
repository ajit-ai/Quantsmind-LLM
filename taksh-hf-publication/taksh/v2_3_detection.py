"""Taksh v2.3.0 — Generation degeneration detection.

Implements diagnostics for generation-side failures that ordinary training
loss and syntax-validity metrics cannot detect.

Work Package 1 of Taksh v2.3.0.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DetectionWarning:
    """A single detection warning with severity and evidence."""

    type: str
    severity: str  # "warning" or "failure"
    message: str
    evidence: Optional[Dict[str, Any]] = None
    location: Optional[str] = None


@dataclass
class DegenerationResult:
    """Complete degeneration detection result for one generation."""

    generation_id: str
    warnings: List[DetectionWarning]
    classification: str  # "pass", "warning", "fail"
    total_tokens: int
    unique_tokens: int
    repeat_rate: float
    repeated_token_sequences: List[Tuple[str, int]]
    repeated_operator_sequences: List[Dict[str, Any]]
    repeated_ast_subtrees: List[Dict[str, Any]]
    premature_eos: bool
    excessive_repetition: bool
    degenerate_semantics: bool




def detect_repeated_tokens(completion, min_repeat=3):
    if not completion.strip():
        return []
    tokens = re.findall(r"\S+|[^\s\S]", completion)
    if not tokens:
        return []
    meaningful_tokens = [t for t in tokens if t.strip() or t in "\n"]
    repeats = []
    i = 0
    while i < len(meaningful_tokens):
        token = meaningful_tokens[i]
        count = 1
        j = i + 1
        while j < len(meaningful_tokens) and meaningful_tokens[j] == token:
            count += 1
            j += 1
        if count >= min_repeat:
            repeats.append((token, count))
        i = j
    seen_tokens = set()
    deduped = []
    for token, count in repeats:
        if token not in seen_tokens:
            seen_tokens.add(token)
            deduped.append((token, count))
    return deduped


def detect_repeated_operators(completion):
    if not completion.strip():
        return []
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        return _detect_repeated_operators_regex(completion)
    operator_counts = Counter()
    def extract_operators(node):
        if isinstance(node, ast.BoolOp):
            operator_counts[type(node.op).__name__] += 1
            for value in node.values:
                extract_operators(value)
        elif isinstance(node, ast.BinOp):
            operator_counts[type(node.op).__name__] += 1
            extract_operators(node.left)
            extract_operators(node.right)
        elif isinstance(node, ast.UnaryOp):
            operator_counts[type(node.op).__name__] += 1
            extract_operators(node.operand)
        elif isinstance(node, ast.Compare):
            if node.ops:
                operator_counts[type(node.ops[0]).__name__] += 1
            extract_operators(node.left)
            for comparator in node.comparators:
                extract_operators(comparator)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                operator_counts[node.func.id] += 1
            extract_operators(node.func)
            for arg in node.args:
                extract_operators(arg)
        elif isinstance(node, ast.IfExp):
            extract_operators(node.body)
            extract_operators(node.test)
            extract_operators(node.orelse)
    try:
        extract_operators(tree)
    except Exception:
        return _detect_repeated_operators_regex(completion)
    min_repeat = 2
    repeated = []
    for op, count in operator_counts.items():
        if count >= min_repeat:
            repeated.append({"operator": op, "count": count})
    repeated.sort(key=lambda x: x["count"], reverse=True)
    return repeated


def _detect_repeated_operators_regex(completion):
    import re as re_mod
    results = []
    for op_char in ["+", "-", "*", "/", "%", "|", "&", "^", "~"]:
        pattern = f"({re_mod.escape(op_char)})\1+"
        matches = list(re_mod.finditer(pattern, completion))
        for m in matches:
            op_str = m.group(1)
            count = m.group(0).count(op_str)
            if count >= 2:
                results.append({"operator": op_str, "count": count})
    for op in ["==", "!=", "<", ">", "<=", ">="]:
        count = len(re_mod.findall(re_mod.escape(op), completion))
        if count >= 2:
            results.append({"operator": op, "count": count})
    results.sort(key=lambda x: x["count"], reverse=True)
    return results


def detect_premature_eos(completion, expected_len_tokens=None):
    if not completion.strip():
        return {"premature_eos": False, "incomplete_function": False, "reason": "", "open_parens": 0, "open_braces": 0, "open_brackets": 0}
    result = {"premature_eos": False, "incomplete_function": False, "reason": "", "open_parens": 0, "open_braces": 0, "open_brackets": 0}
    result["open_parens"] = completion.count("(") - completion.count(")")
    result["open_braces"] = completion.count("{") - completion.count("}")
    result["open_brackets"] = completion.count("[") - completion.count("]")
    stripped = completion.rstrip()
    if not stripped:
        result["premature_eos"] = True
        result["reason"] = "empty completion"
        return result
    last_char = stripped[-1]
    if last_char in "([{":
        result["premature_eos"] = True
        result["reason"] = "completion ends with unclosed delimiter: %s" % last_char
        return result
    if expected_len_tokens is not None:
        actual_tokens = len(stripped.split())
        if actual_tokens < expected_len_tokens * 0.5 and expected_len_tokens > 0:
            result["premature_eos"] = True
            result["reason"] = "early termination: %d tokens vs expected ~%d" % (actual_tokens, expected_len_tokens)
    if result["open_parens"] != 0 or result["open_braces"] != 0 or result["open_brackets"] != 0:
        result["premature_eos"] = True
        parts = []
        if result["open_parens"] > 0:
            parts.append("%d open parens" % result["open_parens"])
        if result["open_braces"] > 0:
            parts.append("%d open braces" % result["open_braces"])
        if result["open_brackets"] > 0:
            parts.append("%d open brackets" % result["open_brackets"])
        result["reason"] = "; ".join(parts) + " - likely incomplete"
    if not result["premature_eos"] and not result["incomplete_function"]:
        token_count = len(stripped.split())
        if token_count <= 2:
            result["premature_eos"] = True
            result["reason"] = "very short completion: %d tokens" % token_count
    return result


def detect_excessive_repetition(completion, max_repeat_ratio=0.3):
    if not completion.strip():
        return {"excessive_repetition": False, "truncated": False, "repeat_ratio": 0.0, "unique_ratio": 0.0, "reason": ""}
    result = {"excessive_repetition": False, "truncated": False, "repeat_ratio": 0.0, "unique_ratio": 0.0, "reason": ""}
    tokens = completion.split()
    if len(tokens) < 3:
        result["truncated"] = True
        result["reason"] = "too short: %d tokens" % len(tokens)
        return result
    seen = set()
    repeat_count = 0
    for token in tokens:
        if token in seen:
            repeat_count += 1
        else:
            seen.add(token)
    unique_count = len(seen)
    repeat_ratio = repeat_count / len(tokens) if tokens else 0.0
    unique_ratio = unique_count / len(tokens) if tokens else 0.0
    result["repeat_ratio"] = round(repeat_ratio, 4)
    result["unique_ratio"] = round(unique_ratio, 4)
    if repeat_ratio >= max_repeat_ratio:
        result["excessive_repetition"] = True
        from collections import Counter
        token_freq = Counter(tokens)
        repeat_types = ["%s (%dx)" % (t, c) for t, c in token_freq.most_common(5) if c > 1]
        result["reason"] = "excessive repetition: %.1f%% of tokens are repeats (%s)" % (repeat_ratio * 100, ", ".join(repeat_types) if repeat_types else "unknown")
    stripped = completion.rstrip()
    if stripped:
        last_token = stripped.split()[-1] if stripped.split() else ""
        if "(" in last_token and ")" not in last_token:
            result["truncated"] = True
            if not result["excessive_repetition"]:
                result["reason"] = "output truncated mid-function-call"
            else:
                result["reason"] += "; output truncated mid-function-call"
    return result



def detect_degenerate_semantics(completion, used_ops=None):
    if not completion.strip():
        return {"degenerate": False, "reason": "empty completion", "has_body": False, "is_identity": False, "computation_depth": 0}
    result = {"degenerate": False, "reason": "", "has_body": False, "is_identity": False, "computation_depth": 0}
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        result["degenerate"] = False
        result["reason"] = "syntax error - not eligible for semantic degeneracy check"
        return result
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        func = tree.body[0]
        result["has_body"] = bool(func.body)
        if not func.body:
            result["degenerate"] = True
            result["reason"] = "empty function body"
            return result
        if len(func.args.args) >= 1:
            first_param = func.args.args[0].arg
            # Check if this is a simple identity: return the first param unchanged
            # Only flag as identity if the body is JUST a return of the param,
            # and there are no other operations or complex expressions
            if _is_simple_identity(func, first_param):
                # Simple identity - not degenerate, just a valid function
                result["is_identity"] = True
                # Do NOT set result["degenerate"] = True for simple identities
                result["reason"] = "simple identity function - returns input unchanged"
                return result
            # For non-identity functions, continue depth check
        depth = _compute_ast_depth(func.body)
        result["computation_depth"] = depth
        if depth <= 1 and not result.get("is_identity", False):
            result["degenerate"] = True
            if not result["reason"]:
                result["reason"] = "minimal computation depth: %d" % depth
            else:
                result["reason"] += " - minimal computation depth: %d" % depth
        elif depth <= 1 and result.get("is_identity", False):
            # Identity with depth 1 is not degenerate
            result["degenerate"] = False
            result["reason"] = "identity function with minimal depth"
        elif depth > 1:
            # Has meaningful computation - not degenerate
            result["degenerate"] = False
            result["reason"] = "meaningful computation depth: %d" % depth
    else:
        depth = _compute_ast_depth(tree.body)
        result["computation_depth"] = depth
        if depth <= 1:
            # Only flag as degenerate if NOT a simple identity
            if not result.get("is_identity", False):
                result["degenerate"] = True
                result["reason"] = "trivial expression: minimal computation"
            else:
                result["degenerate"] = False
                result["reason"] = "identity function - valid"
        else:
            result["degenerate"] = False
            result["reason"] = "meaningful computation depth: %d" % depth
    return result


def _is_simple_identity(func, first_param):
    """Check if a function is a simple identity or constant-return function.
    
    A "simple" function is one that:
    - Returns the first parameter unchanged, OR
    - Returns a constant value
    - Has no other operations or complex expressions
    
    Such functions are valid Python and should NOT be classified as degenerate.
    """
    if len(func.body) != 1:
        return False
    ret = func.body[0]
    if not isinstance(ret, ast.Return):
        return False
    if ret.value is None:
        return False  # 'return' with no value - not simple
    # Check for: return first_param (identity)
    if isinstance(ret.value, ast.Name) and ret.value.id == first_param:
        # Additional check: no other statements, no complex expressions
        # The function body should just be 'return param'
        return True
    # Check for: return constant (e.g., 'return 5')
    if isinstance(ret.value, ast.Constant):
        # Constant return - also considered "simple" and not degenerate
        return True
    # Check for: return simple arithmetic of the parameter (e.g., 'return x + 1')
    # These are also valid and not degenerate
    if isinstance(ret.value, ast.BinOp) and _is_simple_binop(ret.value, first_param):
        return True
    return False


def _is_simple_binop(node, first_param):
    """Check if a binary operation is a simple operation involving the first param."""
    if not isinstance(node.left, ast.Name) or node.left.id != first_param:
        return False
    if not isinstance(node.right, (ast.Constant, ast.Name)):
        return False
    # Allow: return x + 1, return x * 2, return x + 0, etc.
    if isinstance(node.right, ast.Constant):
        # Any constant is fine
        return True
    if isinstance(node.right, ast.Name) and node.right.id != first_param:
        # Different variable - still simple
        return True
    return False


def _compute_ast_depth(nodes):
    if not nodes:
        return 0
    max_depth = 0
    for node in nodes:
        depth = _node_depth(node)
        max_depth = max(max_depth, depth)
    return max_depth


def _node_depth(node):
    if node is None:
        return 0
    depth = 1
    if isinstance(node, ast.FunctionDef):
        child_depths = [_node_depth(child) for child in node.body]
        depth += max(child_depths) if child_depths else 0
    elif isinstance(node, ast.If):
        child_depths = [_node_depth(node.test), _node_depth(node.body) if node.body else 0, _node_depth(node.orelse) if node.orelse else 0]
        depth += max(child_depths)
    elif isinstance(node, ast.BinOp):
        child_depths = [_node_depth(node.left), _node_depth(node.right)]
        depth += max(child_depths)
    elif isinstance(node, ast.UnaryOp):
        depth += _node_depth(node.operand)
    elif isinstance(node, ast.Compare):
        child_depths = [_node_depth(node.left)] + [_node_depth(c) for c in node.comparators]
        depth += max(child_depths)
    elif isinstance(node, ast.Call):
        depth += _node_depth(node.func)
        for arg in node.args:
            depth = max(depth, _node_depth(arg))
        for kw in node.keywords:
            depth = max(depth, _node_depth(kw.value))
    elif isinstance(node, ast.Return):
        if node.value is not None:
            depth += _node_depth(node.value)
        return depth
    elif isinstance(node, ast.Expr):
        if node.value is not None:
            depth += _node_depth(node.value)
        return depth
    elif isinstance(node, (ast.Assign, ast.AugAssign)):
        if hasattr(node, "value") and node.value is not None:
            depth += _node_depth(node.value)
        return depth
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        child_depths = [_node_depth(item) for item in node.elts]
        depth += max(child_depths) if child_depths else 0
    return depth


def _compute_expr_depth(node):
    if node is None:
        return 0
    depth = 1
    if isinstance(node, ast.BinOp):
        depth += max(_compute_expr_depth(node.left), _compute_expr_depth(node.right))
    elif isinstance(node, ast.UnaryOp):
        depth += _compute_expr_depth(node.operand)
    elif isinstance(node, ast.Compare):
        depth += max(_compute_expr_depth(node.left), *[_compute_expr_depth(c) for c in node.comparators])
    elif isinstance(node, ast.Call):
        depth += max([_compute_expr_depth(a) for a in node.args], default=0)
        depth += max([_compute_expr_depth(kw.value) for kw in node.keywords], default=0)
    return depth



def detect_degeneration(
    completion,
    generation_id="unknown",
    expected_len_tokens=None,
    known_ops=None,
):
    warnings = []
    repeated_tokens = detect_repeated_tokens(completion, min_repeat=3)
    if repeated_tokens:
        top_repeats = repeated_tokens[:5]
        evidence = {"repeated_sequences": [{"token": t, "count": c} for t, c in top_repeats]}
        warnings.append(DetectionWarning(type="repeated_tokens", severity="warning" if len(repeated_tokens) < 5 else "failure", message="Repeated token sequences detected (%d total)" % len(repeated_tokens), evidence=evidence))
    repeated_ops = detect_repeated_operators(completion)
    if repeated_ops:
        top_ops = repeated_ops[:3]
        evidence = {"repeated_operators": [{"op": r["operator"], "count": r["count"]} for r in top_ops]}
        sev = "failure" if any(r["count"] > 10 for r in repeated_ops) else "warning"
        warnings.append(DetectionWarning(type="repeated_operators", severity=sev, message="Repeated operators detected (%d types)" % len(repeated_ops), evidence=evidence))
    repeated_ast = detect_repeated_ast_subtrees(completion)
    if repeated_ast:
        top_ast = repeated_ast[:3]
        evidence = {"repeated_ast_subtrees": [{"type": r["subtree_type"], "occurrences": r["occurrences"]} for r in top_ast]}
        warnings.append(DetectionWarning(type="repeated_ast_subtrees", severity="warning" if len(repeated_ast) < 5 else "failure", message="Repeated AST subtrees detected (%d types)" % len(repeated_ast), evidence=evidence))
    eos_result = detect_premature_eos(completion, expected_len_tokens)
    if eos_result["premature_eos"]:
        sev = "failure" if eos_result.get("reason", "").startswith("completion ends") else "warning"
        warnings.append(DetectionWarning(type="premature_eos", severity=sev, message=eos_result.get("reason", "premature EOS detected"), evidence={"open_parens": eos_result["open_parens"], "open_braces": eos_result["open_braces"], "open_brackets": eos_result["open_brackets"], "reason": eos_result["reason"]}))
    repeat_result = detect_excessive_repetition(completion)
    if repeat_result["excessive_repetition"]:
        warnings.append(DetectionWarning(type="excessive_repetition", severity="failure", message="Excessive repetition: %s" % repeat_result["reason"], evidence={"repeat_ratio": repeat_result["repeat_ratio"], "unique_ratio": repeat_result["unique_ratio"]}))
    deg_result = detect_degenerate_semantics(completion, known_ops)
    if deg_result.get("degenerate", False):
        sev = "failure" if deg_result.get("is_identity", False) else "warning"
        warnings.append(DetectionWarning(type="degenerate_semantics", severity=sev, message=deg_result.get("reason", "degenerate semantics detected"), evidence={"is_identity": deg_result.get("is_identity", False), "computation_depth": deg_result.get("computation_depth", 0), "has_body": deg_result.get("has_body", False)}))
    failure_types = {"premature_eos", "excessive_repetition", "degenerate_semantics"}
    warning_types = {"repeated_tokens", "repeated_operators", "repeated_ast_subtrees"}
    has_failure = any(w.type in failure_types for w in warnings)
    has_warning = any(w.type in warning_types for w in warnings) and not has_failure
    if has_failure:
        classification = "fail"
    elif has_warning:
        classification = "warning"
    else:
        classification = "pass"
    total_tokens = len(completion.split()) if completion.strip() else 0
    unique_tokens = len(set(completion.split())) if completion.strip() else 0
    repeat_rate = (total_tokens - unique_tokens) / total_tokens if total_tokens > 0 else 0.0
    premature_eos = any(w.type == "premature_eos" for w in warnings)
    excessive_repetition = any(w.type == "excessive_repetition" for w in warnings)
    degenerate_semantics = any(w.type == "degenerate_semantics" for w in warnings)
    return DegenerationResult(
        generation_id=generation_id, warnings=warnings, classification=classification,
        total_tokens=total_tokens, unique_tokens=unique_tokens, repeat_rate=round(repeat_rate, 4),
        repeated_token_sequences=repeated_tokens,
        repeated_operator_sequences=repeated_ops,
        repeated_ast_subtrees=repeated_ast,
        premature_eos=premature_eos,
        excessive_repetition=excessive_repetition,
        degenerate_semantics=degenerate_semantics,
    )


def classify_warning(result):
    categories = {"repetition": "warning", "structure": "warning", "syntax": "warning", "semantics": "warning"}
    for w in result.warnings:
        if w.type == "repeated_tokens":
            categories["repetition"] = w.severity
        elif w.type == "repeated_operators":
            categories["structure"] = w.severity
        elif w.type == "repeated_ast_subtrees":
            categories["structure"] = w.severity
        elif w.type == "premature_eos":
            categories["syntax"] = w.severity
        elif w.type == "excessive_repetition":
            categories["repetition"] = w.severity
        elif w.type == "degenerate_semantics":
            categories["semantics"] = w.severity
    return categories


def detect_repeated_ast_subtrees(completion, min_occurrences=2):
    if not completion.strip():
        return []
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        return []
    subtree_serializations = Counter()
    visited = set()
    def serialize_subtree(node):
        if node is None:
            return "None"
        node_id = id(node)
        if node_id in visited:
            return None
        visited.add(node_id)
        type_name = type(node).__name__
        fields = []
        if isinstance(node, ast.FunctionDef):
            fields = [node.name, node.args.args, node.returns, node.body, node.decorator_list]
        elif isinstance(node, ast.If):
            fields = [node.test, node.body, node.orelse]
        elif isinstance(node, ast.BinOp):
            fields = [type(node.op).__name__, node.left, node.right]
        elif isinstance(node, ast.UnaryOp):
            fields = [type(node.op).__name__, node.operand]
        elif isinstance(node, ast.Compare):
            fields = [node.left, type(node.ops[0]).__name__ if node.ops else None, node.comparators]
        elif isinstance(node, ast.Call):
            fields = [node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__, node.args]
        elif isinstance(node, ast.Name):
            fields = [node.id]
        elif isinstance(node, ast.Constant):
            fields = [repr(node.value)]
        serialization = (type_name, str(fields))
        subtree_serializations[serialization] += 1
        if isinstance(node, ast.FunctionDef):
            for child in node.args.args:
                serialize_subtree(child)
            for child in node.body:
                serialize_subtree(child)
        elif isinstance(node, ast.If):
            serialize_subtree(node.test)
            for child in node.body:
                serialize_subtree(child)
            for child in node.orelse:
                serialize_subtree(child)
        elif isinstance(node, ast.BinOp):
            serialize_subtree(node.left)
            serialize_subtree(node.right)
        elif isinstance(node, ast.UnaryOp):
            serialize_subtree(node.operand)
        elif isinstance(node, ast.Compare):
            serialize_subtree(node.left)
            for comp in node.comparators:
                serialize_subtree(comp)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                serialize_subtree(node.func)
            for child in node.args:
                serialize_subtree(child)
            for child in node.keywords:
                serialize_subtree(child)
        else:
            for field_key in ["left", "right", "test", "body", "orelse", "args"]:
                if hasattr(node, field_key):
                    child = getattr(node, field_key)
                    if isinstance(child, list):
                        for c in child:
                            if isinstance(c, ast.AST):
                                serialize_subtree(c)
                    elif isinstance(child, ast.AST):
                        serialize_subtree(child)
        return serialization
    for node in ast.walk(tree):
        serialize_subtree(node)
    # Only report non-trivial subtrees (skip leaf nodes and simple expressions)
    # Filter out subtrees that are just a single node type or too small
    trivial_types = {"Name", "Constant", "Num", "Str", "Load", "Store", "Del"}
    filtered_repeated = []
    for serialization, count in subtree_serializations.items():
        type_name, fields = serialization
        if type_name in trivial_types:
            continue
        # Also skip if the serialized fields are too short (trivial subtree)
        if len(fields) < 5:
            continue
        if count >= min_occurrences:
            type_name, fields = serialization
            filtered_repeated.append({"subtree_type": type_name, "occurrences": count})
    filtered_repeated.sort(key=lambda x: x["occurrences"], reverse=True)
    return filtered_repeated
