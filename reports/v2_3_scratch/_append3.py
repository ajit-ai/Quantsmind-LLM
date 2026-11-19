content = '''

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
            if len(func.body) == 1 and isinstance(func.body[0], ast.Return):
                if isinstance(func.body[0].value, ast.Name) and func.body[0].value.id == first_param:
                    result["is_identity"] = True
                    result["degenerate"] = True
                    result["reason"] = "identity function - returns input unchanged"
                    return result
        depth = _compute_ast_depth(func.body)
        result["computation_depth"] = depth
        if depth <= 1:
            result["degenerate"] = True
            if not result["reason"]:
                result["reason"] = "minimal computation depth: %d" % depth
            else:
                result["reason"] += " - minimal computation depth: %d" % depth
    else:
        depth = _compute_ast_depth(tree.body)
        result["computation_depth"] = depth
        if depth <= 1:
            if len(tree.body) == 1:
                first_stmt = tree.body[0]
                if isinstance(first_stmt, ast.Expr):
                    expr_depth = _compute_expr_depth(first_stmt.value)
                    result["computation_depth"] = expr_depth
                    if expr_depth <= 1:
                        result["degenerate"] = True
                        result["reason"] = "trivial expression: %s" % str(first_stmt.value)[:50]
    return result


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

'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append 3a done")