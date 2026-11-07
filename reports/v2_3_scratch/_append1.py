content = '''

def detect_repeated_tokens(completion, min_repeat=3):
    if not completion.strip():
        return []
    tokens = re.findall(r"\\S+|[^\\s\\S]", completion)
    if not tokens:
        return []
    meaningful_tokens = [t for t in tokens if t.strip() or t in "\\n"]
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
        pattern = f"({re_mod.escape(op_char)})\\1+"
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
'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append 1 done")