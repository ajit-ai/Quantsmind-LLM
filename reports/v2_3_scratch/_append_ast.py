content = '''

def detect_repeated_ast_subtrees(completion, min_occurrences=2):
    if not completion.strip():
        return []
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        return []
    subtree_serializations = Counter()
    def serialize_subtree(node):
        if node is None:
            return "None"
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
    repeated = []
    for serialization, count in subtree_serializations.items():
        if count >= min_occurrences:
            type_name, fields = serialization
            repeated.append({"subtree_type": type_name, "occurrences": count})
    repeated.sort(key=lambda x: x["occurrences"], reverse=True)
    return repeated
'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append AST done")