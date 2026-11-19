content = '''

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
'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append 4 done")