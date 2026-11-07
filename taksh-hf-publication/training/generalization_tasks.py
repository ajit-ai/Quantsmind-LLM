"""Frozen v2.2.0 evaluation suite and controlled training cells."""

from __future__ import annotations

import re

from training.experiment_artifacts import canonical_digest
from training.operation_tasks import OPERATIONS, PARAPHRASES

TRAINED_OPS = tuple(op for op, value in OPERATIONS.items() if value[0] != "mapping")
MAPPING_OPS = ("lookup", "count_keys", "lookup_default", "has_key")

COMPOSITIONS = {
    "add_then_multiply": ("composition",
        "Write a function that adds two numbers and multiplies the result by a third.",
        "def add_then_multiply(a, b, c):",
        "    return (a + b) * c"),
    "subtract_then_multiply": ("composition",
        "Write a function that subtracts one number from another and multiplies by a third.",
        "def subtract_then_multiply(a, b, c):",
        "    return (a - b) * c"),
    "filter_positive_count": ("composition",
        "Write a function that counts how many positive numbers a list holds.",
        "def count_positive(values):",
        "    return len([value for value in values if value > 0])"),
    "upper_then_length": ("composition",
        "Write a function that returns the length of a string after upper-casing it.",
        "def upper_length(text):",
        "    return len(text.upper())"),
    "multiply_then_add": ("composition",
        "Write a function that multiplies two numbers and then adds a third.",
        "def multiply_then_add(a, b, c):",
        "    return a * b + c"),
    "add_then_subtract": ("composition",
        "Write a function that adds two numbers and then subtracts a third.",
        "def add_then_subtract(a, b, c):",
        "    return a + b - c"),
    "max_then_add": ("composition",
        "Write a function that adds a third number to the larger of two numbers.",
        "def add_to_maximum(a, b, c):",
        "    if a > b:\n        return a + c\n    return b + c"),
    "filter_negative_sum": ("composition",
        "Write a function that sums only the negative numbers in a list.",
        "def sum_negatives(values):",
        "    total = 0\n    for value in values:\n"
        "        if value < 0:\n            total += value\n    return total"),
}
TRAINED_COMPOSITIONS = (
    "add_then_multiply", "subtract_then_multiply",
    "filter_positive_count", "upper_then_length",
)

E1_NAMES = {
    "arith": (("compute_total", ("x", "y")), ("sum_together", ("first", "second")),
              ("add_numbers", ("p", "q"))),
    "binary": (("pick_one", ("x", "y")), ("select_value", ("first", "second")),
               ("choose_between", ("p", "q"))),
    "unary_list": (("select_items", ("data",)), ("filter_numbers", ("entries",)),
                   ("pick_values", ("seq",))),
    "unary_text": (("change_case", ("data",)), ("transform_string", ("entry",)),
                   ("convert_text", ("value",))),
    "aggregate": (("tally", ("data",)), ("compute_total", ("entries",)),
                  ("calculate", ("values",))),
}
E1_KIND = {
    "add": "arith", "subtract": "arith", "multiply": "arith",
    "maximum": "binary", "minimum": "binary",
    "keep_positive": "unary_list", "keep_negative": "unary_list",
    "count_values": "aggregate", "sum_values": "aggregate",
    "upper": "unary_text", "lower": "unary_text",
}

EXTRA_PARAPHRASES = {
    "add": "Return the result of adding the two inputs.",
    "subtract": "Return the first input minus the second input.",
    "multiply": "Return the product of the two inputs.",
    "maximum": "Return whichever input is greater.",
    "minimum": "Return whichever input is smaller.",
    "keep_positive": "Keep only values that are greater than zero.",
    "keep_negative": "Keep only values that are less than zero.",
    "upper": "Return the input converted to capital letters.",
    "lower": "Return the input converted to small letters.",
    "count_values": "Count the elements contained in the list.",
    "sum_values": "Add up the elements contained in the list.",
}
PARAPHRASE_VARIANTS = {
    op: (PARAPHRASES[op], EXTRA_PARAPHRASES[op]) for op in TRAINED_OPS
}
E2_INSTRUCTIONS = {
    op: (PARAPHRASES[op], EXTRA_PARAPHRASES[op],
         "Produce a function that " + EXTRA_PARAPHRASES[op].split(" ", 2)[2].lower())
    for op in TRAINED_OPS
}

MAPPING = {
    "lookup": ("mapping",
        "Write a function that returns the value stored under a key.",
        "def lookup_value(mapping, key):",
        "    return mapping.get(key, 0)"),
    "count_keys": ("mapping",
        "Write a function that returns how many keys a mapping holds.",
        "def count_keys_mapping(mapping, key):",
        "    return len(mapping)"),
    "lookup_default": ("mapping",
        "Write a function that returns a stored value or a default when absent.",
        "def lookup_or_default(mapping, key, default):",
        "    return mapping.get(key, default)"),
    "has_key": ("mapping",
        "Write a function that reports whether a mapping contains a key.",
        "def contains_key(mapping, key):",
        "    return key in mapping"),
}

def _rename(signature, body, function_name, parameter_names):
    match = re.match(r"def\s+([A-Za-z_]\w*)\(([^)]*)\):", signature)
    if not match:
        raise ValueError("invalid signature: " + signature)
    old_params = [part.strip() for part in match.group(2).split(",") if part.strip()]
    if len(old_params) != len(parameter_names):
        raise ValueError("parameter count mismatch for " + signature)
    mapping = dict(zip(old_params, parameter_names))
    new_signature = "def " + function_name + "(" + ", ".join(parameter_names) + "):"
    new_body = body
    for old, new in mapping.items():
        new_body = re.sub(r"\b" + re.escape(old) + r"\b", new, new_body)
    return new_signature, new_body


def _base(op):
    family, instruction, signature, body, _markers = OPERATIONS[op]
    return family, instruction, signature, body


def _case(case_id, partition, operation, instruction, signature, body,
          family, *, composition=None, negative=None, metadata=None):
    return {
        "case_id": case_id,
        "partition": partition,
        "operation": operation,
        "instruction": instruction,
        "signature": signature,
        "body": body,
        "family": family,
        "composition": composition,
        "negative": negative,
        "metadata": dict(metadata or {}),
    }


def e1_cases():
    cases = []
    for op in TRAINED_OPS:
        family, instruction, signature, body = _base(op)
        kind = E1_KIND[op]
        for index, (name, params) in enumerate(E1_NAMES[kind]):
            sig, bod = _rename(signature, body, name, params)
            cases.append(_case(f"E1-{op}-{index:03d}", "E1", op, instruction,
                               sig, bod, family, metadata={"variant": "name"}))
    return cases


def e2_cases():
    cases = []
    for op in TRAINED_OPS:
        family, _instruction, signature, body = _base(op)
        for index, text in enumerate(E2_INSTRUCTIONS[op]):
            cases.append(_case(f"E2-{op}-{index:03d}", "E2", op, text,
                               signature, body, family,
                               metadata={"variant": "paraphrase"}))
    return cases


CONTRAST_PAIRS = (
    ("add", "subtract"), ("maximum", "minimum"),
    ("keep_positive", "keep_negative"), ("upper", "lower"),
    ("count_values", "sum_values"), ("add", "multiply"),
)


def e3_cases():
    cases = []
    for pair_index, (left, right) in enumerate(CONTRAST_PAIRS):
        for side, op in (("left", left), ("right", right)):
            family, instruction, signature, body = _base(op)
            for variant in range(2):
                if variant == 0:
                    sig, bod = signature, body
                    text = instruction
                else:
                    match = re.match(r"def\s+([A-Za-z_]\w*)\(([^)]*)\):", signature)
                    old = [p.strip() for p in match.group(2).split(",") if p.strip()]
                    kind = E1_KIND[op]
                    name, params = E1_NAMES[kind][0]
                    sig, bod = _rename(signature, body, name + "_contrast", params)
                    text = PARAPHRASES[op]
                cases.append(_case(
                    f"E3-{pair_index}-{side}-{variant}", "E3", op, text,
                    sig, bod, family, metadata={"pair": pair_index, "side": side}))
    return cases

E4_PARAPHRASES = {
    "add_then_multiply": "Add the first two numbers, then multiply that total by the third.",
    "subtract_then_multiply": "Subtract the second number from the first, then multiply by the third.",
    "filter_positive_count": "Count the values in a list that are above zero.",
    "upper_then_length": "Upper-case a string and report how many characters it has.",
    "multiply_then_add": "Multiply the first two numbers, then add the third.",
    "add_then_subtract": "Add the first two numbers, then take away the third.",
    "max_then_add": "Take the larger of the first two numbers and add the third.",
    "filter_negative_sum": "Add together only the values below zero.",
}


def _comp_variant(name, params):
    family, instruction, signature, body = COMPOSITIONS[name]
    sig, bod = _rename(signature, body, name + "_variant", params)
    return sig, bod


def e4_cases():
    cases = []
    param_sets = {
        2: ("x", "y"), 3: ("x", "y", "z"),
    }
    for name in COMPOSITIONS:
        family, instruction, signature, body = COMPOSITIONS[name]
        count = signature.count(",") + 1 - signature.split("(")[1].count("=")
        params = ("data",) if count == 1 else param_sets[count]
        for variant in range(3):
            if variant == 0:
                sig, bod, text = signature, body, instruction
            elif variant == 1:
                sig, bod = _comp_variant(name, params)
                text = instruction
            else:
                sig, bod = _comp_variant(name, tuple(reversed(params)) if count > 1 else params)
                text = E4_PARAPHRASES[name]
            cases.append(_case(
                f"E4-{name}-{variant}", "E4", name, text, sig, bod, family,
                composition=name,
                metadata={"seen_in_training": name in TRAINED_COMPOSITIONS}))
    return cases


MAPPING_VARIANTS = {
    "lookup": ("lookup_value", "read_mapping", "retrieve_entry"),
    "count_keys": ("count_keys_mapping", "how_many_keys", "count_entries"),
    "lookup_default": ("lookup_or_default", "read_with_default", "fetch_or_default"),
    "has_key": ("contains_key", "key_present", "mapping_has_key"),
}
MAPPING_PARAPHRASES = {
    "lookup": ("Return mapping[key] when present, otherwise zero.",
               "Find the value for a key, using zero as the fallback."),
    "count_keys": ("Return the number of entries in a mapping.",
                   "Report how many keys the mapping contains."),
    "lookup_default": ("Return the stored value for a key, or the supplied default.",
                       "Use a fallback when the key is missing."),
    "has_key": ("Return True when the key is present in the mapping.",
                "Check whether the mapping contains the key."),
}


def e5_cases():
    cases = []
    for op in MAPPING_OPS:
        family, instruction, signature, body = MAPPING[op]
        names = MAPPING_VARIANTS[op]
        for index in range(6):
            name = names[index % 3]
            text = instruction if index < 3 else MAPPING_PARAPHRASES[op][index - 4]
            sig = re.sub(r"def\s+[A-Za-z_]\w*\(", "def " + name + "(", signature)
            cases.append(_case(
                f"E5-{op}-{index:03d}", "E5", op, text, sig, body, family,
                metadata={"held_out_operation": True}))
    return cases


NEGATIVES = {
    "syntax_error": (
        ("def broken(a, b):", "    return a +"),
        ("def incomplete(x):", "    return (x"),
        ("def missing_paren(a):", "    return a * (a + 1"),
        ("def bad_indent(a):", "return a"),
    ),
    "unsupported_operator": (
        ("def divide(a, b):", "    return a / b"),
        ("def remainder(a, b):", "    return a % b"),
        ("def power(a, b):", "    return a ** b"),
        ("def floor_divide(a, b):", "    return a // b"),
    ),
    "undefined_name": (
        ("def use_missing(a):", "    return missing + a"),
        ("def add_unknown(a, b):", "    return a + unknown"),
        ("def call_nothing(a):", "    return nothing(a)"),
        ("def return_other(a):", "    return other_value"),
    ),
    "side_effect": (
        ("def show(a):", "    print(a)\n    return a"),
        ("def emit(a):", "    print('value')\n    return a"),
        ("def write_out(a):", "    print(a)\n    return 0"),
        ("def report(a):", "    print('done')\n    return a"),
    ),
    "empty_body": (
        ("def empty(a):", "    pass"),
        ("def nothing(a):", "    pass"),
        ("def noop(a):", "    pass"),
        ("def blank(a):", "    pass"),
    ),
    "unsupported_call": (
        ("def read_file(path):", "    return open(path).read()"),
        ("def ask(a):", "    return input(a)"),
        ("def load(path):", "    return open(path)"),
        ("def read_bytes(path):", "    return open(path, 'rb')"),
    ),
}


def e6_cases():
    cases = []
    for category, variants in NEGATIVES.items():
        for index, (signature, body) in enumerate(variants):
            cases.append(_case(
                f"E6-{category}-{index:03d}", "E6", "negative",
                "Write a function for the requested behavior.",
                signature, body, "negative", negative=category,
                metadata={"expected": "not_credited"}))
    return cases

def build_eval_cases():
    cases = e1_cases() + e2_cases() + e3_cases() + e4_cases() + e5_cases() + e6_cases()
    counts = {partition: sum(1 for case in cases if case["partition"] == partition)
              for partition in ("E1", "E2", "E3", "E4", "E5", "E6")}
    expected = {"E1": 33, "E2": 33, "E3": 24, "E4": 24, "E5": 24, "E6": 24}
    if counts != expected:
        raise ValueError(f"evaluation counts {counts} != {expected}")
    if len({case["case_id"] for case in cases}) != 162:
        raise ValueError("case identifiers are not unique")
    return cases


def eval_digest(cases):
    return canonical_digest({"version": "2.2.0", "cases": cases})


B_PARAMS = {
    1: (("alpha",), ("x",), ("first",), ("m",)),
    2: (("alpha", "beta"), ("x", "y"), ("first", "second"), ("m", "n")),
    3: (("alpha", "beta", "gamma"), ("x", "y", "z"),
        ("first", "second", "third"), ("m", "n", "o")),
}


def _record(example_id, op, instruction, signature, body, family, split,
            composition=None):
    return {"example_id": example_id, "operation": op, "instruction": instruction,
            "signature": signature, "body": body, "family": family,
            "split": split, "composition": composition}


def _t_records():
    from training.operation_tasks import build_operation_curriculum
    partitions = build_operation_curriculum()
    return [_record(e.example_id, e.operation, e.instruction, e.signature,
                    e.body, e.family, "train")
            for e in partitions["T"]]


def _v_records():
    from training.operation_tasks import build_operation_curriculum
    partitions = build_operation_curriculum()
    return [_record(e.example_id, e.operation, e.instruction, e.signature,
                    e.body, e.family, "validation")
            for e in partitions["V"]]


def cell_a():
    return _t_records()


def cell_b():
    records = _t_records()
    for op in TRAINED_OPS:
        family, instruction, signature, body = _base(op)
        kind = E1_KIND[op]
        arity = {"arith": 2, "binary": 2, "unary_list": 1,
                 "unary_text": 1, "aggregate": 1}[kind]
        texts = (instruction, PARAPHRASES[op], EXTRA_PARAPHRASES[op], instruction)
        for index in range(4):
            name = f"cell_b_{op}_{index}"
            params = B_PARAMS[arity][index]
            sig, bod = _rename(signature, body, name, params)
            records.append(_record(f"B-{op}-{index:03d}", op, texts[index],
                                   sig, bod, family, "train"))
    if len(records) != 66:
        raise ValueError(f"cell B has {len(records)} records, expected 66")
    return records


COMP_PARAMS = {
    1: (("data",), ("values",), ("items",), ("seq",), ("entries",)),
    2: (("x", "y"), ("left", "right"), ("first", "second"), ("p", "q"), ("a", "b")),
    3: (("x", "y", "z"), ("a", "b", "c"), ("first", "second", "third"),
        ("p", "q", "r"), ("left", "middle", "right")),
}


def _composition_records():
    records = []
    for name in TRAINED_COMPOSITIONS:
        family, instruction, signature, body = COMPOSITIONS[name]
        arity = signature.split("(")[1].count(",") + 1
        for index in range(5):
            params = COMP_PARAMS[arity][index]
            sig, bod = _rename(signature, body, f"cell_c_{name}_{index}", params)
            text = instruction if index % 2 == 0 else E4_PARAPHRASES[name]
            records.append(_record(f"C-{name}-{index:03d}", name, text, sig, bod,
                                   family, "train", composition=name))
    return records


def cell_c():
    records = cell_b() + _composition_records()
    if len(records) != 86:
        raise ValueError(f"cell C has {len(records)} records, expected 86")
    return records


def validation_cases():
    return _v_records()


def build_cells():
    return {"A": cell_a(), "B": cell_b(), "C": cell_c()}

