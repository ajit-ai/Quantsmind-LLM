content = '''

def test_renamed_params(signature, body, new_param_names):
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
        renamed_sig = re.sub(r"\\b" + old + r"\\b", new, renamed_sig)
        renamed_body = re.sub(r"\\b" + old + r"\\b", new, renamed_body)
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


def test_reordered_params(signature, body, new_order):
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


def test_unseen_variable_names(signature, body, unseen_names):
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
'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_parameter_binding.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append PB done")