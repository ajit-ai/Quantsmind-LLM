"""Curriculum diversity and mode-collapse safeguards (Taksh v1.2.0 Increment 2).

These checks are structural (AST based) rather than raw-text matching so
that superficial renamings do not count as diversity. They can be applied
equally to a curriculum (list of supervised examples) and to evaluation
output records (dicts with ``completion`` and ``category``).

Threshold rationale
-------------------
The defaults are deliberately generous: they must reject the Increment 1
failure shape (one template per operation, a dominant comprehension) while
accepting legitimate Python variation.

* ``max_operation_share = 0.12``: with 13 operations the uniform share is
  0.077. 0.12 allows about 55% over-representation but rejects a 2x
  dominance. A single-template corpus still passes this check, so it is
  not sufficient on its own; the per-operation template checks below are
  the decisive ones.
* ``min_templates_per_operation = 3``: Increment 1 §10 recommends 3-4
  surface forms per operation. A corpus with fewer than three distinct
  template families for any operation gives the model a single attractor
  per operation and is rejected.
* ``max_variant_share_within_operation = 0.5``: no single normalised
  shape may exceed half of its operation's examples. Increment 1 scored
  1.0 on every operation.
* ``max_template_share = 0.20``: no single global template family may
  exceed 20% of the corpus. With ~50 templates in a diversified corpus,
  20% is roughly ten times the uniform expectation, so only a strongly
  skewed corpus fails.
* ``max_listcomp_share = 0.10``: list comprehensions are one valid form
  among many. Increment 1 scored 24/156 = 0.154 and collapsed to this
  form on unrelated prompts; a diversified corpus should score well under
  0.10. Legitimate comprehension use is not banned, only dominance.
* ``collapse_threshold = 0.30`` with ``collapse_min_cases = 5`` and
  ``collapse_min_categories = 3`` (evaluation side): one template
  appearing on at least 30% of held-out outputs, at least five cases, and
  at least three different categories is treated as mode collapse. The
  category requirement prevents a single-category legitimation (for
  example, five genuinely list-shaped tasks) from being flagged.
* Degenerate comprehension text (``for item in in ...``) is flagged at a
  count of 3 or more regardless of share: it is never legitimate output.

All thresholds are configurable via :class:`QualityThresholds`; violations
raise :class:`CurriculumQualityError` with a message naming the check and
the observed value.
"""

from __future__ import annotations

import ast
import copy
import re
from collections import Counter
from dataclasses import dataclass

from training.program_analysis import analyse

__all__ = [
    "CurriculumQualityError",
    "QualityThresholds",
    "template_family",
    "normalised_shape",
    "check_curriculum",
    "diversity_report",
    "detect_mode_collapse",
]

_DEGENERATE_LISTCOMP = re.compile(r"for\s+\w+\s+in\s+in\b")


class CurriculumQualityError(ValueError):
    """Raised when a curriculum or output set violates a quality threshold."""


@dataclass(frozen=True)
class QualityThresholds:
    max_operation_share: float = 0.12
    max_template_share: float = 0.20
    max_listcomp_share: float = 0.10
    min_templates_per_operation: int = 3
    max_variant_share_within_operation: float = 0.5
    collapse_threshold: float = 0.30
    collapse_min_cases: int = 5
    collapse_min_categories: int = 3
    degenerate_min_count: int = 3


def _parse_function(signature: str, body: str):
    try:
        tree = ast.parse(signature + "\n" + body)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    return tree.body[0]


def _return_kind(node) -> str:
    if node is None:
        return "no_return"
    value = node.value
    if value is None:
        return "bare_return"
    if isinstance(value, ast.Name):
        return "name_return"
    if isinstance(value, ast.BinOp):
        return "binop_return"
    if isinstance(value, ast.UnaryOp):
        return "unary_return"
    if isinstance(value, ast.IfExp):
        return "ternary_return"
    if isinstance(value, ast.ListComp):
        return "listcomp_return"
    if isinstance(value, ast.GeneratorExp):
        return "genexp_return"
    if isinstance(value, ast.JoinedStr):
        return "fstring_return"
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name):
            if func.id == "len":
                return "len_call_return"
            if func.id == "sum":
                return "sum_call_return"
            if func.id == "list":
                return "list_call_return"
            return f"{func.id}_call_return"
        if isinstance(func, ast.Attribute):
            return f"method_{func.attr}_return"
        return "call_return"
    if isinstance(value, ast.Attribute):
        return "attribute_return"
    return type(value).__name__.lower() + "_return"


def template_family(signature: str, body: str) -> str:
    """Coarse AST template family of a function body, ignoring identifiers."""
    function = _parse_function(signature, body)
    if function is None:
        return "unparseable"
    statements = function.body
    prefix = ""
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
        and len(statements) > 1
    ):
        prefix = "docstring_"
        statements = statements[1:]
    kinds = [type(stmt).__name__ for stmt in statements]
    last_return = None
    for stmt in statements:
        if isinstance(stmt, ast.Return):
            last_return = stmt
    return_kind = _return_kind(last_return)
    if "While" in kinds:
        shape = "while_loop"
    elif "For" in kinds:
        shape = "for_loop"
    elif "If" in kinds and "Assign" in kinds:
        shape = "assign_if"
    elif "If" in kinds:
        shape = "if"
    elif "AugAssign" in kinds:
        shape = "augassign"
    elif "Assign" in kinds:
        shape = "assign"
    else:
        shape = "direct"
    return f"{prefix}{shape}_{return_kind}"


def normalised_shape(signature: str, body: str) -> str:
    """Fine-grained AST shape with all identifiers and constants masked.

    Parameter names become ``p0``, ``p1``, ... by position, local names
    become ``L``, the function name becomes ``f``, string constants become
    ``S`` and numeric constants become ``N``. Two examples that differ
    only in names or literal values share a shape; the Increment 2
    variants of one operation do not.
    """
    tree = ast.parse(signature + "\n" + body)
    function = tree.body[0]
    params = [arg.arg for arg in function.args.args]
    assigned: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    assigned.add(target.id)

    def mask(name: str) -> str:
        if name in params:
            return f"p{params.index(name)}"
        if name in assigned:
            return "L"
        if name in {"len", "sum", "list", "str"}:
            return name
        return "G"

    class Masker(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mask(node.id)
            return node

        def visit_arg(self, node):
            node.arg = mask(node.arg)
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = "S"
            elif isinstance(node.value, (int, float)):
                node.value = "N"
            return node

    masked = copy.deepcopy(function)
    masked.name = "f"
    Masker().visit(masked)
    return ast.dump(masked)


def _params_of(signature: str) -> list[str]:
    tree = ast.parse(signature + "\n    pass")
    return [arg.arg for arg in tree.body[0].args.args]


def _is_degenerate_text(completion: str) -> bool:
    return bool(_DEGENERATE_LISTCOMP.search(completion or ""))


def check_curriculum(train, val=None, *, thresholds: QualityThresholds = QualityThresholds(), heldout_pairs=None) -> dict:
    """Validate a curriculum and return a statistics report.

    Raises :class:`CurriculumQualityError` on the first violated check.
    ``train`` and ``val`` are sequences of supervised examples with
    ``operation``, ``instruction``, ``signature`` and ``body`` attributes.
    ``heldout_pairs`` is an optional set of ``(instruction, signature)``
    pairs that training data must not overlap.
    """
    thresholds = thresholds or QualityThresholds()
    train = list(train)
    val = list(val) if val is not None else []
    if not train:
        raise CurriculumQualityError("curriculum check failed: training set is empty")
    n = len(train)

    op_counts = Counter(example.operation for example in train)
    template_counts = Counter(template_family(e.signature, e.body) for e in train)
    shape_counts = Counter((e.operation, normalised_shape(e.signature, e.body)) for e in train)
    listcomp_count = sum(count for name, count in template_counts.items() if "listcomp" in name)

    # 1. Operation balance.
    for operation, count in op_counts.items():
        share = count / n
        if share > thresholds.max_operation_share:
            raise CurriculumQualityError(
                f"operation balance check failed: operation {operation!r} is {count}/{n} "
                f"({share:.1%}), above max_operation_share {thresholds.max_operation_share:.0%}"
            )

    # 2. Global template balance.
    dominant_template, dominant_count = template_counts.most_common(1)[0]
    if dominant_count / n > thresholds.max_template_share:
        raise CurriculumQualityError(
            f"template balance check failed: template {dominant_template!r} is "
            f"{dominant_count}/{n} ({dominant_count / n:.1%}), above "
            f"max_template_share {thresholds.max_template_share:.0%}"
        )

    # 3. Per-operation template diversity and 4. variant dominance.
    templates_per_operation: dict[str, set[str]] = {}
    for example in train:
        templates_per_operation.setdefault(example.operation, set()).add(
            template_family(example.signature, example.body)
        )
    for operation, templates in templates_per_operation.items():
        if len(templates) < thresholds.min_templates_per_operation:
            raise CurriculumQualityError(
                f"template diversity check failed: operation {operation!r} has "
                f"{len(templates)} template families {sorted(templates)}, fewer than "
                f"min_templates_per_operation {thresholds.min_templates_per_operation}"
            )
    for (operation, shape), count in shape_counts.items():
        share = count / op_counts[operation]
        if share > thresholds.max_variant_share_within_operation:
            raise CurriculumQualityError(
                f"variant dominance check failed: one normalised shape for operation "
                f"{operation!r} covers {count}/{op_counts[operation]} ({share:.1%}) of "
                f"that operation, above max_variant_share_within_operation "
                f"{thresholds.max_variant_share_within_operation:.0%}"
            )

    # 5. List-comprehension dominance.
    if listcomp_count / n > thresholds.max_listcomp_share:
        raise CurriculumQualityError(
            f"list-comprehension dominance check failed: list-comprehension templates are "
            f"{listcomp_count}/{n} ({listcomp_count / n:.1%}), above "
            f"max_listcomp_share {thresholds.max_listcomp_share:.0%}"
        )

    # 6. Exact duplicates.
    seen: dict[tuple, str] = {}
    for example in train:
        key = (example.instruction, example.signature, example.body)
        if key in seen:
            raise CurriculumQualityError(
                f"duplicate check failed: {example.example_id!r} duplicates {seen[key]!r} "
                "(identical instruction, signature and body)"
            )
        seen[key] = example.example_id
    ids = [example.example_id for example in train]
    if len(ids) != len(set(ids)):
        raise CurriculumQualityError("duplicate check failed: example_id values are not unique in the training set")

    # 7. Invalid Python and binding (undefined names, parameter coverage).
    for example in train + val:
        result = analyse(example.signature, example.body)
        if not result["parses"]:
            raise CurriculumQualityError(
                f"AST validity check failed: {example.example_id!r} does not parse: {result['error']}"
            )
        if result["undefined_names"]:
            raise CurriculumQualityError(
                f"binding check failed: {example.example_id!r} uses undefined names "
                f"{result['undefined_names']} (parameter names were replaced or invented)"
            )
        params = _params_of(example.signature)
        # analyse() reports params_used positionally ("p0", "p1", ...).
        used = set(result["params_used"])
        expected_all = {f"p{index}" for index in range(len(params))}
        if example.operation == "choose_first":
            if used != {"p0"}:
                raise CurriculumQualityError(
                    f"binding check failed: {example.example_id!r} must use only the first "
                    f"parameter {params[0]!r}, used {sorted(used)}"
                )
        elif example.operation == "choose_second":
            if used != {"p1"}:
                raise CurriculumQualityError(
                    f"binding check failed: {example.example_id!r} must use only the second "
                    f"parameter {params[1]!r}, used {sorted(used)}"
                )
        elif used != expected_all:
            raise CurriculumQualityError(
                f"binding check failed: {example.example_id!r} must use every parameter "
                f"{params}, used {sorted(used)}"
            )

    # 8. Train/validation leakage.
    if val:
        train_pairs = {(e.instruction, e.signature) for e in train}
        val_pairs = {(e.instruction, e.signature) for e in val}
        overlap = train_pairs & val_pairs
        if overlap:
            raise CurriculumQualityError(
                f"leakage check failed: {len(overlap)} (instruction, signature) pairs appear "
                f"in both train and validation, first: {sorted(overlap)[:1]}"
            )
        train_bodies = {(e.signature, e.body) for e in train}
        val_bodies = {(e.signature, e.body) for e in val}
        body_overlap = train_bodies & val_bodies
        if body_overlap:
            raise CurriculumQualityError(
                f"leakage check failed: {len(body_overlap)} (signature, body) pairs appear "
                f"in both train and validation, first: {sorted(body_overlap)[:1]}"
            )

    # 9. Held-out overlap.
    if heldout_pairs:
        overlap = [
            e.example_id for e in train
            if (e.instruction, e.signature) in heldout_pairs
        ]
        if overlap:
            raise CurriculumQualityError(
                f"held-out leakage check failed: training examples {overlap[:5]} overlap "
                "frozen evaluation cases"
            )

    family_counts = Counter(example.family for example in train)
    return {
        "train_examples": n,
        "validation_examples": len(val),
        "operations": dict(sorted(op_counts.items())),
        "families": dict(sorted(family_counts.items())),
        "template_families": dict(sorted(template_counts.items())),
        "distinct_template_families": len(template_counts),
        "distinct_normalised_shapes": len(shape_counts),
        "templates_per_operation": {op: sorted(t) for op, t in sorted(templates_per_operation.items())},
        "listcomp_examples": listcomp_count,
        "listcomp_share": round(listcomp_count / n, 4),
        "max_template_share": round(dominant_count / n, 4),
        "exact_duplicates": 0,
        "invalid_examples": 0,
        "train_validation_leakage": 0,
        "heldout_overlap": 0,
    }


def _record_template(record: dict) -> str:
    completion = record.get("completion") or ""
    signature = record.get("signature") or "def probe(value):"
    family = template_family(signature, completion)
    if _is_degenerate_text(completion):
        return "degenerate_listcomp_text"
    if family == "unparseable" and "[" in completion and "for " in completion:
        return "degenerate_listcomp_text"
    return family


def diversity_report(records) -> dict:
    """Template-diversity statistics for a set of evaluation records."""
    records = list(records)
    templates = Counter(_record_template(r) for r in records)
    completions = Counter((r.get("completion") or "").strip() for r in records)
    degenerate = sum(1 for r in records if _is_degenerate_text(r.get("completion") or ""))
    n = len(records)
    top_template, top_count = templates.most_common(1)[0] if templates else (None, 0)
    return {
        "records": n,
        "distinct_templates": len(templates),
        "template_counts": dict(sorted(templates.items())),
        "dominant_template": top_template,
        "dominant_count": top_count,
        "dominant_share": round(top_count / n, 4) if n else 0.0,
        "exact_repeated_completions": sum(1 for text, count in completions.items() if text and count > 1),
        "degenerate_listcomp_text": degenerate,
    }


def detect_mode_collapse(records, *, thresholds: QualityThresholds = QualityThresholds()) -> dict:
    """Detect the Increment 1 failure mode in evaluation output records.

    Collapsed when one template family covers at least
    ``collapse_threshold`` of the records, at least
    ``collapse_min_cases`` records, and at least
    ``collapse_min_categories`` distinct categories; or when degenerate
    comprehension text (``for item in in ...``) appears at least
    ``degenerate_min_count`` times.
    """
    thresholds = thresholds or QualityThresholds()
    records = list(records)
    report = diversity_report(records)
    n = report["records"]
    dominant = report["dominant_template"]
    dominant_count = report["dominant_count"]
    dominant_categories = sorted({
        r.get("category") for r in records
        if _record_template(r) == dominant
    }) if dominant else []
    degenerate = report["degenerate_listcomp_text"]
    share = report["dominant_share"]
    collapsed = bool(
        n
        and dominant_count >= thresholds.collapse_min_cases
        and share >= thresholds.collapse_threshold
        and len(dominant_categories) >= thresholds.collapse_min_categories
    )
    degenerate_collapsed = degenerate >= thresholds.degenerate_min_count
    reasons = []
    if collapsed:
        reasons.append(
            f"template {dominant!r} on {dominant_count}/{n} ({share:.1%}) across "
            f"{len(dominant_categories)} categories {dominant_categories}"
        )
    if degenerate_collapsed:
        reasons.append(f"degenerate comprehension text on {degenerate} records")
    return {
        "collapsed": collapsed or degenerate_collapsed,
        "dominant_template": dominant,
        "dominant_count": dominant_count,
        "dominant_share": share,
        "dominant_categories": dominant_categories,
        "degenerate_listcomp_text": degenerate,
        "template_counts": report["template_counts"],
        "reasons": reasons,
    }
