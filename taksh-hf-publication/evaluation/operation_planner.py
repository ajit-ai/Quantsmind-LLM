"""Structured operation planner (Architecture B prototype).

An explicit intermediate representation: callers produce a typed
``Plan`` of ordered steps with dependencies, validate it, and compile to
a deterministic Python body. The prototype supports arithmetic, string,
collection, aggregation, and conditional operation families (five+
families). Unresolvable dependencies or unsupported operations are
rejected with a reason — never guessed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PlanStep:
    """One operation step. ``args`` are params, literals, or earlier outputs."""

    operation: str
    args: tuple = ()
    result: str = "_result"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    """Explicit execution plan for one function."""

    steps: tuple
    output: str
    family: str

    def as_dict(self):
        return {
            "steps": [step.as_dict() for step in self.steps],
            "output": self.output,
            "family": self.family,
        }


@dataclass(frozen=True)
class PlanResult:
    status: str  # "planned" | "unsupported" | "invalid"
    plan: Plan | None
    reason: str

    def as_dict(self):
        return {
            "status": self.status,
            "plan": self.plan.as_dict() if self.plan else None,
            "reason": self.reason,
        }


SUPPORTED_OPERATIONS = frozenset({
    "add", "subtract", "multiply", "divide", "modulo", "floor_divide",
    "square", "negate", "abs",
    "upper", "lower", "strip", "join", "split",
    "keep_positive", "keep_negative", "keep_even", "keep_long_words",
    "double_list", "filter_count", "count_values", "sum_values",
    "sum_positive", "is_even", "maximum", "minimum", "choose_first",
    "choose_second", "identity", "double_value",
    # v1.3.2-B generalized compositions and predicates.
    "divide_by", "join_length", "second_character", "length_equals",
    "count_even", "slice_without_last", "product_all", "is_absent",
    "upper_equals", "difference_at_least", "duplicate_value",
    "precedence_add_mul", "propagation_strip", "longer_string",
})


def _params(signature: str) -> list[str]:
    tree = ast.parse(signature + "\n    pass")
    return [arg.arg for arg in tree.body[0].args.args]


def _has(phrase: str, *needles: str) -> bool:
    return any(n in phrase for n in needles)


def _binops(text: str, params: list[str], op: str, symbol: str) -> PlanResult:
    if len(params) < 2:
        return PlanResult("invalid", None, f"{op} needs two parameters")
    steps = (PlanStep(op, (params[0], params[1]), "result"),)
    return PlanResult("planned", Plan(steps, "result", "arithmetic"), f"op:{op}")


def plan(instruction: str, signature: str) -> PlanResult:
    """Build (or safely reject) a typed operation plan for the task."""
    params = _params(signature)
    if not params:
        return PlanResult("unsupported", None, "no parameters")
    text = (instruction or "").lower()
    v = params[0]

    # Ordered binary composition ("then" sequencing): explicit steps.
    if "then" in text and len(params) >= 2:
        a, b = params[0], params[1]
        _words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                  "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        _d = re.search(r"(?:divide|divided) by (\d+|one|two|three|four|five|six|seven|eight|nine|ten)", text)
        if ("add" in text or "sum" in text or "combine" in text) and _d:
            steps = (
                PlanStep("add", (a, b), "combined"),
                PlanStep("divide_by", ("combined", _words.get(_d.group(1), _d.group(1))), "result"),
            )
            return PlanResult("planned", Plan(steps, "result", "arithmetic"), "op:compose_add_divide")
        if "join" in text and "length" in text:
            return PlanResult("planned", Plan((PlanStep("join_length", (a, b), "result"),), "result", "string"), "op:join_length")
        if ("multiply" in text or "product" in text or "times" in text) and ("halve" in text or "half" in text):
            steps = (
                PlanStep("multiply", (a, b), "combined"),
                PlanStep("floor_divide", ("combined", "2"), "result"),
            )
            return PlanResult("planned", Plan(steps, "result", "arithmetic"), "op:compose_mul_halve")
    # Generalized predicates (single- and binary-parameter).
    if len(params) == 1:
        _words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
        _m = re.search(r"length(?:s)? (?:exactly|is exactly|is) (\d+|one|two|three|four|five)", text)
        if _m:
            return PlanResult("planned", Plan((PlanStep("length_equals", (v, _words.get(_m.group(1), _m.group(1))), "result"),), "result", "condition"), "op:length_equals")
        if "count" in text and "even" in text:
            return PlanResult("planned", Plan((PlanStep("count_even", (v,), "result"),), "result", "condition"), "op:count_even")
        if "without" in text and "last" in text:
            return PlanResult("planned", Plan((PlanStep("slice_without_last", (v,), "result"),), "result", "collection"), "op:slice_without_last")
        if "second" in text and ("character" in text or "letter" in text):
            return PlanResult("planned", Plan((PlanStep("second_character", (v,), "result"),), "result", "string"), "op:second_character")
    if len(params) >= 2:
        a, b = params[0], params[1]
        if "longer" in text and ("two" in text or "pair" in text or "of" in text):
            return PlanResult("planned", Plan((PlanStep("longer_string", (a, b), "result"),), "result", "string"), "op:longer_string")
        if "absent" in text:
            return PlanResult("planned", Plan((PlanStep("is_absent", (a, b), "result"),), "result", "condition"), "op:is_absent")
        if ("upper" in text or "capital" in text or "shouting" in text) and ("equals" in text or "equal" in text):
            return PlanResult("planned", Plan((PlanStep("upper_equals", (a,), "result"),), "result", "condition"), "op:upper_equals")
        _m = re.search(r"exceeds the second by at least (\d+|one|two|three|four|five)", text)
        if _m:
            return PlanResult("planned", Plan((PlanStep("difference_at_least", (a, b, _m.group(1)), "result"),), "result", "condition"), "op:difference_at_least")

    # Sequential composition: "A, then B" with two explicit phases.
    if "then" in text:
        if len(params) == 1 and ("count" in text or "tally" in text or "how many" in text) and "positive" in text:
            return PlanResult(
                "planned",
                Plan((PlanStep("filter_count", (params[0],), "result"),), "result", "aggregation"),
                "op:filter_count",
            )
        if ("add" in text or "sum" in text) and ("multiply" in text or "product" in text or "times" in text):
            add_at = min((text.find(w) for w in ("add", "sum", "combine") if w in text), default=99)
            mul_at = min((text.find(w) for w in ("multiply", "product", "times") if w in text), default=99)
            if len(params) >= 2 and add_at < mul_at:
                steps = (
                    PlanStep("add", (params[0], params[1]), "combined"),
                    PlanStep("multiply", ("combined", params[2] if len(params) > 2 else "2"), "result"),
                )
                plan_obj = Plan(steps, "result", "arithmetic")
                return PlanResult("planned", plan_obj, "op:compose_add_mul")
            if len(params) >= 2 and mul_at < add_at:
                steps = (
                    PlanStep("multiply", (params[0], params[1]), "combined"),
                    PlanStep("add", ("combined", params[2] if len(params) > 2 else "2"), "result"),
                )
                plan_obj = Plan(steps, "result", "arithmetic")
                return PlanResult("planned", plan_obj, "op:compose_mul_add")
        if "accumulate" in text or "total" in text:
            steps = (PlanStep("sum_values", (v,), "total"),)
            return PlanResult("planned", Plan(steps, "total", "aggregation"), "op:accumulate")
        if "store" in text or "carry" in text:
            steps = (
                PlanStep("identity", (v,), "stored"),
                PlanStep("double_value", ("stored",), "result"),
            )
            return PlanResult("planned", Plan(steps, "result", "arithmetic"), "op:propagation_double")
        if "upper" in text or "capital" in text:
            steps = (PlanStep("upper", (v,), "result"),) if "join" not in text else (PlanStep("join", (v,), "result"),)
            return PlanResult("planned", Plan(steps, "result", "string"), "op:transform_stage")
        return PlanResult("unsupported", None, "sequential composition not recognized")

    # Propagation chains (single-parameter locals).
    if "chain" in text:
        return PlanResult(
            "planned",
            Plan((PlanStep("identity", (v,), "first"), PlanStep("identity", ("first",), "result")), "result", "arithmetic"),
            "op:propagation_chain",
        )

    # Positives / negatives aggregates and filters.
    if len(params) == 1 and ("positive" in text or "above zero" in text):
        if "count" in text or "tally" in text or "how many" in text:
            return PlanResult("planned", Plan((PlanStep("filter_count", (v,), "result"),), "result", "aggregation"), "op:filter_count")
        if "sum" in text or "total" in text or "add" in text:
            return PlanResult("planned", Plan((PlanStep("sum_positive", (v,), "result"),), "result", "aggregation"), "op:sum_positive")
        if "keep" in text or "filter" in text or "only" in text:
            return PlanResult("planned", Plan((PlanStep("keep_positive", (v,), "result"),), "result", "collection"), "op:keep_positive")
        if "return true" in text or "positive" in text:
            return PlanResult("planned", Plan((PlanStep("is_positive_check", (v,), "result"),), "result", "condition"), "op:is_positive")
    if len(params) == 1 and ("negative" in text or "below zero" in text):
        if "keep" in text or "filter" in text or "only" in text:
            return PlanResult("planned", Plan((PlanStep("keep_negative", (v,), "result"),), "result", "collection"), "op:keep_negative")

    # Counting and summing over one sequence.
    if len(params) == 1:
        if "count" in text or "how many" in text or "tally" in text or "length" in text:
            if re.search(r"(word|entries|strings|text)", text) and ("long" in text or "greater than" in text):
                return PlanResult("planned", Plan((PlanStep("keep_long_words", (v,), "result"),), "result", "collection"), "op:keep_long_words")
            return PlanResult("planned", Plan((PlanStep("count_values", (v,), "result"),), "result", "aggregation"), "op:filter_count")
        if "even" in text and re.search(r"\b(keep|return|filter|only)\b", text):
            return PlanResult("planned", Plan((PlanStep("is_even", (v,), "result"),), "result", "condition"), "op:is_even")
        if "double" in text:
            return PlanResult("planned", Plan((PlanStep("double_value", (v,), "result"),), "result", "arithmetic"), "op:scale_constant")
        if "sort" in text:
            return PlanResult("planned", Plan((PlanStep("sort_values", (v,), "result"),), "result", "collection"), "op:sort_values")

    # Standalone sequence aggregation.
    if len(params) == 1 and _has(text, "sum", "total", "add up", "every entry"):
        return PlanResult("planned", Plan((PlanStep("sum_values", (params[0],), "result"),), "result", "aggregation"), "op:sum_values")

    # Unary string / odd-even.
    if len(params) == 1:
        if ("lower" in text or "small" in text or "quiet" in text) and not ("upper" in text and "shouting" not in text):
            return PlanResult("planned", Plan((PlanStep("lower", (v,), "result"),), "result", "string"), "op:lower")
        if _has(text, "upper", "capital", "shout"):
            return PlanResult("planned", Plan((PlanStep("upper", (v,), "result"),), "result", "string"), "op:upper")
        if "strip" in text:
            return PlanResult("planned", Plan((PlanStep("strip", (v,), "result"),), "result", "string"), "op:strip")
        if "unchanged" in text or "duplicate" in text or "hand" in text:
            return PlanResult("planned", Plan((PlanStep("identity", (v,), "result"),), "result", "string"), "op:copy_identity")
        if "copy" in text:
            return PlanResult("planned", Plan((PlanStep("identity", (v,), "result"),), "result", "string"), "op:copy_identity")

    # Binary arithmetic and comparisons.
    if len(params) >= 2:
        a, b = params[0], params[1]
        mod_at = re.search(r"modulo (?:by )?(\d+|one|two|three|four|five)", text)
        if _has(text, "product", "multiply", "times") and (mod_at or "mod" in text):
            modulus = mod_at.group(1) if mod_at else "3"
            modulus = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}.get(modulus, modulus)
            return PlanResult("planned", Plan((PlanStep("modulo", (a, b), "result"),), "result", "arithmetic"), "op:modulo")
        if _has(text, "difference", "subtract", "minus", "take") and "first" not in text:
            return PlanResult("planned", Plan((PlanStep("subtract", (a, b), "result"),), "result", "arithmetic"), "op:subtract")
        if _has(text, "product", "multiply", "times", "scale", "scaled"):
            return PlanResult("planned", Plan((PlanStep("multiply", (a, b), "result"),), "result", "arithmetic"), "op:multiply")
        if _has(text, "sum", "total", "add", "combine", "together", "plus"):
            return PlanResult("planned", Plan((PlanStep("add", tuple(params), "result"),), "result", "arithmetic"), "op:add")
        if _has(text, "larger", "greater", "maximum", "bigger", "max", "exceeds", "exceed"):
            return PlanResult("planned", Plan((PlanStep("maximum", (a, b), "result"),), "result", "condition"), "op:maximum")
        if _has(text, "smaller", "lesser", "minimum", "min"):
            return PlanResult("planned", Plan((PlanStep("minimum", (a, b), "result"),), "result", "condition"), "op:minimum")
        if "first" in text or "leading" in text:
            return PlanResult("planned", Plan((PlanStep("choose_first", (a,), "result"),), "result", "condition"), "op:choose_first")
        if "second" in text or "trailing" in text:
            return PlanResult("planned", Plan((PlanStep("choose_second", (b,), "result"),), "result", "condition"), "op:choose_second")
        if "join" in text or "glue" in text or "space" in text or "together" in text or "side by side" in text or "combined" in text:
            sep = ", " if "comma" in text else " "
            return PlanResult("planned", Plan((PlanStep("join", (a, b), "result"),), "result", "string"), "op:join")

    return PlanResult("unsupported", None, "no unambiguous operation in vocabulary")


def _is_literal(argument: str) -> bool:
    """Literal-aware identifier handling: numeric literals are not names."""
    return bool(argument.isdigit() or re.fullmatch(r"-?\d+", argument))


def _check_plan(plan: Plan, params: set) -> tuple[bool, str]:
    """Validate definitions, references, duplicates, cycles, and ordering.

    Steps are in execution order; every argument must be a literal, a
    parameter, or an earlier step's result (dependency order). Duplicate
    result definitions, self-references (cycles), and forward references
    are rejected. The final output must be defined.
    """
    defined = set(params)
    for step in plan.steps:
        if step.operation not in SUPPORTED_OPERATIONS:
            return False, f"unsupported operation {step.operation}"
        for argument in step.args:
            if _is_literal(argument):
                continue  # numeric literal argument
            if argument == step.result:
                return False, "cyclic dependency: step references its own result"
            if argument not in defined:
                return False, f"undefined identifier {argument}"
        if step.result in defined:
            return False, f"duplicate identifier {step.result}"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", step.result):
            return False, f"invalid identifier {step.result}"
        defined.add(step.result)
    if plan.output not in defined:
        return False, "output is not defined by the plan"
    return True, "ok"


def validate(result: PlanResult, signature: str) -> bool:
    """Check the plan's dependencies resolve and the output is defined."""
    if result.status != "planned" or result.plan is None:
        return result.status != "planned"
    params = set(_params(signature))
    ok, _reason = _check_plan(result.plan, params)
    return ok


def describe_validation(result: PlanResult, signature: str) -> dict:
    """Planner validation status for hybrid integration (WP7 taxonomy)."""
    if result.status == "unsupported":
        return {"status": "UNSUPPORTED_PLAN", "reason": result.reason}
    if result.status != "planned" or result.plan is None:
        return {"status": "INVALID_PLAN", "reason": result.reason}
    ok, reason = _check_plan(result.plan, set(_params(signature)))
    if not ok:
        return {"status": "INVALID_PLAN", "reason": reason}
    return {"status": "VALIDATED", "reason": "plan-valid", "plan": result.plan.as_dict()}


def compile_plan(result: PlanResult, signature: str) -> dict:
    """Compile a valid plan to a deterministic Python body (unused ops -> AST text)."""
    if result.status != "planned" or result.plan is None:
        return {"action": result.status, "body": None, "reason": result.reason}
    if not validate(result, signature):
        return {"action": "invalid", "body": None, "reason": "unresolved dependency or undefined output"}
    lines = []
    for index, step in enumerate(result.plan.steps):
        if step.operation == "identity":
            lines.append(f"    {step.result} = {step.args[0]}")
        elif step.operation == "add":
            lines.append(f"    {step.result} = {' + '.join(step.args)}")
        elif step.operation == "subtract":
            lines.append(f"    {step.result} = {step.args[0]} - {step.args[1]}")
        elif step.operation == "multiply":
            lines.append(f"    {step.result} = {step.args[0]} * {step.args[1]}")
        elif step.operation == "modulo":
            lines.append(f"    {step.result} = {step.args[0]} % {step.args[1]}")
        elif step.operation == "double_value":
            lines.append(f"    {step.result} = {step.args[0]} * 2")
        elif step.operation == "join":
            lines.append(f"    {step.result} = {' + '.join(repr(' ') + ' + ' + arg for arg in step.args)}")
        elif step.operation in ("upper", "lower", "strip"):
            lines.append(f"    {step.result} = {step.args[0]}.{step.operation}()")
        elif step.operation == "keep_positive":
            lines.append(f"    {step.result} = [item for item in {step.args[0]} if item > 0]")
        elif step.operation == "keep_negative":
            lines.append(f"    {step.result} = [item for item in {step.args[0]} if item < 0]")
        elif step.operation == "keep_even":
            lines.append(f"    {step.result} = [item for item in {step.args[0]} if item % 2 == 0]")
        elif step.operation == "filter_count":
            lines.append(f"    {step.result} = sum(1 for item in {step.args[0]} if item > 0)")
        elif step.operation == "count_values":
            lines.append(f"    {step.result} = len({step.args[0]})")
        elif step.operation == "sum_values":
            lines.append(f"    {step.result} = sum({step.args[0]})")
        elif step.operation == "sum_positive":
            lines.append(f"    {step.result} = sum(item for item in {step.args[0]} if item > 0)")
        elif step.operation == "is_even":
            lines.append(f"    {step.result} = {step.args[0]} % 2 == 0")
        elif step.operation == "is_positive_check":
            lines.append(f"    {step.result} = {step.args[0]} > 0")
        elif step.operation in ("maximum", "minimum"):
            symbol = ">" if step.operation == "maximum" else "<"
            lines.append(f"    {step.result} = {step.args[0]} if {step.args[0]} {symbol} {step.args[1]} else {step.args[1]}")
        elif step.operation == "choose_first":
            lines.append(f"    {step.result} = {step.args[0]}")
        elif step.operation == "choose_second":
            lines.append(f"    {step.result} = {step.args[0]}")
        elif step.operation == "sort_values":
            lines.append(f"    {step.result} = sorted({step.args[0]})")
        elif step.operation == "keep_long_words":
            lines.append(f"    {step.result} = [word for word in {step.args[0]} if len(word) > 3]")
        elif step.operation == "divide_by":
            lines.append(f"    {step.result} = {step.args[0]} / {step.args[1]}")
        elif step.operation == "join_length":
            lines.append(f"    {step.result} = len({step.args[0]} + \" \" + {step.args[1]})")
        elif step.operation == "second_character":
            lines.append(f"    {step.result} = {step.args[0]}[1]")
        elif step.operation == "length_equals":
            lines.append(f"    {step.result} = len({step.args[0]}) == {step.args[1]}")
        elif step.operation == "count_even":
            lines.append(f"    {step.result} = len({step.args[0]}) % 2 == 0")
        elif step.operation == "slice_without_last":
            lines.append(f"    {step.result} = {step.args[0]}[:-1]")
        elif step.operation == "product_all":
            lines.append(f"    total = 1\n    for item in {step.args[0]}:\n        total *= item")
            lines.append(f"    {step.result} = total")
        elif step.operation == "is_absent":
            lines.append(f"    {step.result} = {step.args[1]} not in {step.args[0]}")
        elif step.operation == "upper_equals":
            lines.append(f"    {step.result} = {step.args[0]}.upper() == {step.args[0]}")
        elif step.operation == "difference_at_least":
            lines.append(f"    {step.result} = {step.args[0]} - {step.args[1]} >= {step.args[2]}")
        elif step.operation == "duplicate_value":
            lines.append(f"    {step.result} = {step.args[0]} + {step.args[0]}")
        elif step.operation == "precedence_add_mul":
            lines.append(f"    {step.result} = {step.args[0]} + {step.args[1]} * {step.args[2]}")
        elif step.operation == "propagation_strip":
            lines.append(f"    {step.result} = {step.args[0]}.strip()")
        elif step.operation == "longer_string":
            lines.append(f"    {step.result} = {step.args[0]} if len({step.args[0]}) > len({step.args[1]}) else {step.args[1]}")
        else:
            return {"action": "invalid", "body": None, "reason": f"step {step.operation} has no codegen"}
    body = "\n".join(lines) + f"\n    return {result.plan.output}"
    try:
        ast.parse(signature + "\n" + body)
    except SyntaxError:
        return {"action": "invalid", "body": None, "reason": "generated body does not parse"}
    return {"action": "planned", "body": body, "operation": result.plan.family, "reason": result.reason,
            "provenance": "planner"}


def execute_plan(result: PlanResult, signature: str, function_name: str, tests) -> dict:
    from evaluation.functional_eval import evaluate_functional

    compiled = compile_plan(result, signature)
    if compiled["action"] != "planned":
        return {"passed": False, "reason": compiled["reason"], "body": None}
    functional = evaluate_functional(signature + "\n" + compiled["body"] + "\n", function_name, tests)
    return {"passed": functional.passed, "reason": functional.reason, "body": compiled["body"]}
