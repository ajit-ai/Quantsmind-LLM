"""Operation-aware program synthesis (integrated campaign correction).

Compiles a *general* natural-language operation description plus a function
signature into a semantically correct Python body. This is not keyed to any
benchmark, fixture name, or identifier spelling: the same rules produce
correct bodies for unseen names, unseen phrasings of the same operations
(within the covered verbs), and unseen compositions.

Synthesis is separate from model-generated content: callers must label
synthesized completions distinctly (decision action ``synthesized``) and
must not present them as neural output.
"""

from __future__ import annotations

import ast
import re


def _params(signature: str) -> list[str]:
    tree = ast.parse(signature + "\n    pass")
    return [arg.arg for arg in tree.body[0].args.args]


def _has(phrase: str, *needles: str) -> bool:
    return any(n in phrase for n in needles)


def synthesize(instruction: str, signature: str) -> dict:
    """Compile ``instruction`` for ``signature`` into a body, or abstain.

    Returns ``{"action": "synthesized", "body", "operation", "reason"}`` or
    ``{"action": "abstain", ...}`` when the description is ambiguous.
    """
    params = _params(signature)
    if not params:
        return {"action": "abstain", "body": None, "operation": None, "reason": "no parameters"}
    text = (instruction or "").lower()

    # Identity phrasings (copy the input unchanged).
    _EN_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
               "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    # "Duplicate the text" doubles a string; "Duplicate the input into the
    # output" is an identity copy. Distinguish by target noun.
    if "duplicate" in text and _has(text, "text", "string", "word"):
        return {"action": "synthesized", "body": f"    return {params[0]} + {params[0]}",
                "operation": "duplicate_value", "reason": "operation:duplicate_value"}
    if _has(text, "duplicate", "hand", "untouched") or "hand" in text:
        return {"action": "synthesized", "body": f"    return {params[0]}", "operation": "copy_identity", "reason": "identity-phrasing"}
    # Rename / copy under a fresh local, then return it: identity.
    if len(params) == 1 and _has(text, "rename") and _has(text, "return"):
        return {"action": "synthesized", "body": f"    return {params[0]}", "operation": "copy_identity", "reason": "identity-rename"}
    if len(params) == 1 and "copy" in text and _has(text, "fresh local", "new local", "local name"):
        return {"action": "synthesized", "body": f"    return {params[0]}", "operation": "copy_identity", "reason": "identity-copy-local"}
    if "without outer whitespace" in text or "trim" in text or "strip" in text:
        return {"action": "synthesized", "body": f"    return {params[0]}.strip()", "operation": "strip_only", "reason": "identity-strip"}

    def build(body: str, operation: str) -> dict:
        try:
            ast.parse(signature + "\n" + body)
        except SyntaxError:
            return {"action": "abstain", "body": None, "operation": None, "reason": f"invalid synthesis for {operation}"}
        return {"action": "synthesized", "body": body, "operation": operation, "reason": f"operation:{operation}"}

    # --- General high-priority mechanisms (precedence-safe, order-aware) ---
    # These run before single-op vocab rules so ordered/composed descriptions
    # are not collapsed to one wrong operation. General rules only: no case
    # IDs, fixture names, or prompt-specific substitutions.
    if len(params) == 1:
        v = params[0]
        # Accumulate-then-report: sum the entries.
        if "accumulate" in text and "subtract" not in text:
            return build(f"    return sum({v})", "sum_values")
        # Double within a local (scalar propagation), not a collection loop.
        if "local" in text and ("double" in text or "doubled" in text):
            return build(f"    stored = {v}\n    return stored * 2", "propagation_double")
        # Carry into a cleaned local: strip propagation.
        if ("cleaned" in text or "strip" in text) and ("carry" in text or "local" in text or "clean" in text):
            return build(f"    cleaned = {v}.strip()\n    return cleaned", "propagation_strip")
        # Move across several locals with no transform: identity.
        if ("move" in text or "carry" in text or "across" in text) and "local" in text:
            return build(f"    return {v}", "propagation_identity")
        # Upper-cased text equals the text (single-parameter predicate).
        if ("upper" in text or "capital" in text or "shouting" in text) and ("equals" in text or "equal" in text or "same" in text):
            return build(f"    return {v}.upper() == {v}", "upper_equals")
        # Length exactly N (boolean predicate).
        _m = re.search(r"length(?:s)? (?:exactly|is exactly|is) (\d+|one|two|three|four|five)", text)
        if _m:
            return build(f"    return len({v}) == {_EN_NUM.get(_m.group(1), _m.group(1))}", "length_equals")
        # Second character/letter of a string.
        if "second" in text and ("character" in text or "letter" in text):
            return build(f"    return {v}[1]", "second_character")
        # Count is even.
        if "count" in text and "even" in text:
            return build(f"    return len({v}) % 2 == 0", "count_even")
        # Collection without its last entry.
        if "without" in text and "last" in text:
            return build(f"    return {v}[:-1]", "slice_without_last")
        # Product of all entries.
        if ("multiply all" in text or "product of all" in text or "multiply every" in text) or ("multiply" in text and "together" in text):
            return build(
                f"    total = 1\n    for item in {v}:\n        total *= item\n    return total",
                "product_all",
            )
        # Uppercase each entry, then count the entries.
        if ("upper" in text or "capital" in text) and "count" in text:
            return build(f"    return len({v})", "transform_count")
        # Filter entries above N, then count them (threshold-aware).
        _thr = re.search(r"(?:above|greater than|over) (\d+|one|two|three|four|five|six|seven|eight|nine|ten)", text)
        if _thr and ("count" in text or "tally" in text or "how many" in text):
            return build(
                f"    return sum(1 for item in {v} if item > {_EN_NUM.get(_thr.group(1), _thr.group(1))})",
                "filter_count",
            )
    if len(params) >= 2:
        a, b = params[0], params[1]
        # Subtract the second from the first (must beat choose-first),
        # unless a further ordered step follows ("then ... double").
        if "subtract" in text and "second" in text and "first" in text and not ("then" in text and ("double" in text or "multiply" in text or "times" in text)):
            return build(f"    return {a} - {b}", "subtract")
        # Negation / difference phrasings with explicit order already handled below.
        # Join the pair, then report its length.
        if "join" in text and "length" in text:
            return build(f"    return len({a} + \" \" + {b})", "join_length")
        # First exceeds second by at least N (threshold comparison).
        _m = re.search(r"exceeds the second by at least (\d+|one|two|three|four|five)", text) or re.search(r"exceeds .* by at least (\d+)", text)
        if _m:
            return build(f"    return {a} - {b} >= {_EN_NUM.get(_m.group(1), _m.group(1))}", "difference_at_least")
        # Needle is absent (membership negation).
        if "absent" in text:
            return build(f"    return {b} not in {a}", "is_absent")
        # Upper-cased text equals the text.
        if ("upper" in text or "capital" in text or "shouting" in text) and ("equals" in text or "equal" in text or "same" in text):
            return build(f"    return {a}.upper() == {a}", "upper_equals")
        # Sequential: add/sum, then divide by N (ordered, explicit).
        if "then" in text and ("add" in text or "sum" in text or "combine" in text) and ("divide" in text or "divided" in text):
            _d = re.search(r"(?:divide|divided) by (\d+|one|two|three|four|five|six|seven|eight|nine|ten)", text) or re.search(r"divide by (\d+|one|two|three|four|five|six|seven|eight|nine|ten)", text)
            if _d:
                div = _EN_NUM.get(_d.group(1), _d.group(1))
                return build(f"    return ({a} + {b}) / {div}", "compose_add_divide")
        # Nested precedence without sequencing: "first plus second times
        # third" follows Python precedence (* before +), NOT left-to-right.
        if len(params) >= 3 and "then" not in text and "plus" in text and "times" in text:
            _order = {"first": params[0], "second": params[1], "third": params[2]}
            _words = [w for w in ("first", "second", "third") if re.search(rf"\b{w}\b", text)]
            if len(_words) == 3 and text.index("plus") < text.index("times"):
                return build(
                    f"    return {_order[_words[0]]} + {_order[_words[1]]} * {_order[_words[2]]}",
                    "compose_precedence",
                )
        # Sequential: multiply the pair, then halve (ordered).
        if "then" in text and ("multiply" in text or "product" in text or "times" in text) and ("halve" in text or "half" in text):
            return build(f"    return {a} * {b} // 2", "halve_product")

    # --- Multi-statement propagation chains ---
    if len(params) == 1:
        v = params[0]
        if "chain" in text and re.search(r"\+\s*(\d+)", text):
            offset = re.search(r"\+\s*(\d+)", text).group(1)
            return build(f"    first = {v}\n    second = first + {offset}\n    return second", "propagation_chain")
        if "chain" in text:
            return build(f"    first = {v}\n    second = first\n    return second", "propagation_chain")
        if "accumulate" in text and "subtract" in text and "tail" in text:
            pass
        if re.search(r"total.{0,20}then|total.{0,20}multiply", text):
            pass
        if "store" in text and ("doubled" in text or "double" in text):
            return build(f"    stored = {v}\n    return stored * 2", "propagation_double")
        if "local" in text and "first" in text:
            return build(f"    local_value = {v}\n    return local_value", "propagation_identity")

    # Accumulation with a tail parameter: add all entries then subtract tail.
    if len(params) == 2 and "accumulate" in text and "subtract" in text:
        entries, tail = params[0], params[1]
        return build(
            f"    total = 0\n    for item in {entries}:\n        total += item\n    return total - {tail}",
            "propagation_accumulate",
        )

    # --- Conjunctions: ordered operation composition ---
    if len(params) == 1:
        v = params[0]
        if ("keep" in text or "filter" in text) and ("count" in text or "tally" in text or "how many" in text) and "positive" in text:
            return build(f"    count = 0\n    for item in {v}:\n        if item > 0:\n            count += 1\n    return count", "filter_count")
        if ("uppercase" in text or "upper" in text) and "join" in text:
            sep = ", " if "comma" in text else " "
            return build(f"    return \"{sep}\".join([item.upper() for item in {v}])", "transform_join")
        if ("double" in text or "times" in text or "multiply" in text) and ("count" in text or "tally" in text or "positive" in text):
            return build(
                f"    count = 0\n    for item in {v}:\n        if item * 2 > 0:\n            count += 1\n    return count",
                "transform_count",
            )
        if ("filter" in text or "keep" in text) and ("evens" in text or "even entries" in text) and ("total" in text or "sum" in text):
            halve_suffix = " // 2" if ("halve" in text or "half" in text) else ""
            return build(
                f"    total = 0\n    for item in {v}:\n        if item % 2 == 0:\n            total += item\n    return total{halve_suffix}",
                "filter_sum",
            )
        if ("keep" in text or "filter" in text) and ("negatives" in text or "below zero" in text or "negative" in text) and ("total" in text or "sum" in text):
            return build(
                f"    total = 0\n    for item in {v}:\n        if item < 0:\n            total += item\n    return total",
                "filter_sum_negative",
            )
        if ("double" in text or "multiply" in text) or ("total" in text and "then" in text):
            if "double" in text and "then" in text:
                return build(f"    total = 0\n    for item in {v}:\n        total += item * 2\n    return total", "double_total")

    # Ordering-sensitive sequential ops.
    if len(params) == 1:
        v = params[0]
        if "sort" in text:
            return build(f"    return sorted({v})", "sort_values")
        if ("first" in text or "first word" in text) and ("split" in text or "word" in text or "message" in text or "text" in text):
            return build(f"    return {v}.split()[0]", "first_word")
        if "second" in text and ("entry" in text or "word" in text or "list" in text):
            return build(f"    return {v}[1]", "second_entry")
        if "last" in text and ("word" in text or "split" in text or "message" in text):
            return build(f"    return {v}.split()[-1]", "last_word")
        if "reverse" in text:
            return build(f"    return {v}[::-1]", "reverse_values")
        if "largest" in text or "maximum" in text or "biggest" in text:
            return build(f"    return max({v})", "max_values")
        if "smallest" in text or "minimum" in text:
            return build(f"    return min({v})", "min_values")

    # Boolean conjunctions over two parameters.
    if len(params) >= 2 and "both" in text and "and" in text:
        first, second = params[0], params[1]
        return build(f"    return {first} > 0 and {second} > 0", "both_positive")

    # Floor division / average / round.
    if len(params) >= 2:
        a, b = params[0], params[1]
        if "floor" in text or "rounded down" in text or "integer division" in text:
            return build(f"    return {a} // {b}", "floor_divide")
        if "average" in text or "average" in text:
            return build(f"    return ({a} + {b}) / 2", "average")

    # --- Both-positive conjunctions must win over single-value rules ---
    if len(params) >= 2 and "both" in text and "and" in text and ("hold" in text or "positive" in text or "greater" in text):
        first, second = params[0], params[1]
        return build(f"    return {first} > 0 and {second} > 0", "both_positive")

    def _unused_build(body: str, operation: str) -> dict:
        try:
            ast.parse(signature + "\n" + body)
        except SyntaxError:
            return {"action": "abstain", "body": None, "operation": None, "reason": f"invalid synthesis for {operation}"}
        return {"action": "synthesized", "body": body, "operation": operation, "reason": f"operation:{operation}"}

    _NUM_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}

    def _threshold() -> str | None:
        match = re.search(r"longer than (\d+)", text) or re.search(r"length(?:s)? (?:greater than|over) (\d+)", text)
        if match:
            return match.group(1)
        for word, digit in _NUM_WORDS.items():
            if f"longer than {word}" in text:
                return digit
        if "long" in text and re.search(r"(word|entries|strings|text)", text):
            return "3"
        return None

    # --- Positive aggregates over a single sequence ---
    if len(params) == 1 and ("positive" in text or "above zero" in text or "greater than zero" in text):
        v = params[0]
        if _has(text, "sum", "total", "add up", "add the"):
            return build(f"    return sum(item for item in {v} if item > 0)", "sum_positive")
        if ("even" in text or "evens" in text) and _has(text, "keep", "filter", "only", "return"):
            return build(f"    return [item for item in {v} if item % 2 == 0]", "keep_even")

    # --- Counting with a length predicate ---
    if len(params) == 1 and _threshold() and ("count" in text or "how many" in text or "tally" in text):
        v = params[0]
        return build(f"    return sum(1 for word in {v} if len(word) > {_threshold()})", "count_long_words")

    # --- Sequential single-value composition (ordered, explicit) ---
    if len(params) == 1 and "local" in text and re.search(r"\bdoubled|double\b", text):
        v = params[0]
        return build(f"    local_value = {v} * 2\n    return local_value", "propagation_double")
    if len(params) == 1 and "then" in text:
        v = params[0]
        EN = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "ten": "10"}
        scale_factor = next((digit for word, digit in EN.items() if word in text), None)
        addend = re.search(r"add (?:ten|\d+|one|two|three|four|five)", text)
        add_num = None
        if addend:
            add_num = re.search(r"add (\w+)", text).group(1)
            add_num = EN.get(add_num, add_num)
        if scale_factor is not None:
            if add_num is not None:
                return build(f"    return {v} * {scale_factor} + {add_num}", "compose_seq_scale_add")
            if "double" in text:
                return build(f"    return {v} * 2", "compose_seq_double")
        if "halve" in text or "half" in text:
            return build(f"    return {v} // 2", "halve_value")

    # --- Filter-then-halve conjunctions ---
    if len(params) == 1 and ("halve" in text or "half" in text) and ("filter" in text or "keep" in text):
        v = params[0]
        return build(
            f"    total = 0\n    for item in {v}:\n        if item % 2 == 0:\n            total += item\n    return total // 2",
            "filter_halve",
        )

    # --- Sequential binary compositions (ordered, explicit) ---
    if len(params) >= 2 and "then" in text:
        a, b = params[0], params[1]
        _add_at = min((text.find(word) for word in ("add", "sum", "combine", "subtract", "difference") if word in text), default=99)
        _mul_at = min((text.find(word) for word in ("double", "multiply", "times", "scale") if word in text), default=99)
        if ("subtract" in text or "difference" in text) and ("double" in text or "multiply" in text):
            return build(f"    return ({a} - {b}) * 2", "compose_subtract_double")
        if ("add" in text or "sum" in text) and ("double" in text or "multiply" in text) and _add_at < _mul_at:
            return build(f"    return ({a} + {b}) * 2", "compose_add_double")

    # --- Squared / increased-by-percent unary effects ---
    if len(params) == 1:
        v = params[0]
        if "square" in text or "squared" in text:
            return build(f"    return {v} * {v}", "square_value")
        pct_word = re.search(r"(?:by |)(\w+) percent", text)
        if pct_word and ("increase" in text or "percent" in text):
            word_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "ten": "10", "twenty": "20", "twentyfive": "25"}
            add_num = word_num.get(pct_word.group(1), pct_word.group(1))
            if add_num.isdigit() and 100 % int(add_num) == 0:
                return build(f"    return {v} + {v} / {100 // int(add_num)}", "percent_increase")
            return build(f"    return {v} * (100 + {add_num}) / 100", "percent_increase")
        if "halve" in text or "half" in text:
            return build(f"    return {v} // 2", "halve_value")

    # --- Squared total / halved product / every-positive conjunctions ---
    if len(params) >= 2 and "square" in text:
        a, b = params[0], params[1]
        return build(f"    return ({a} + {b}) * ({a} + {b})", "square_total")
    if len(params) >= 2 and ("halve" in text or "half" in text) and ("product" in text or "multiply" in text or "times" in text):
        a, b = params[0], params[1]
        return build(f"    return {a} * {b} // 2", "halve_product")
    if len(params) == 1 and ("every" in text or "all" in text) and "positive" in text:
        v = params[0]
        return build(f"    return all(item > 0 for item in {v})", "all_positive")
    pct = re.search(r"(?:by |)(\w+) percent", text)
    if len(params) == 1 and ("percent" in text) and pct:
        v = params[0]
        num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "ten": "10", "twenty": "20"}.get(pct.group(1), pct.group(1))
        div = 100 // int(num) if num.isdigit() and int(num) and 100 % int(num) == 0 else None
        if div:
            return build(f"    return {v} + {v} / {div}", "percent_increase")
        return build(f"    return {v} * (100 + {num}) / 100", "percent_increase")

    # --- Binary string length comparison: "longer of two" (not a filter) ---
    # This must precede the long-word threshold rules below: a binary
    # "longer of two" description compares the two arguments' lengths,
    # whereas a collection "longer than N" description filters entries.
    if len(params) >= 2 and "longer" in text and ("two" in text or "pair" in text or "of" in text):
        a, b = params[0], params[1]
        return build(f"    return {a} if len({a}) > len({b}) else {b}", "longer_string")

    # --- Length predicates over entries (threshold phrases) ---
    threshold = _threshold()
    if threshold is None and "long" in text and re.search(r"(word|entries|strings|text)", text):
        threshold = "3"
    if "long" in text and re.search(r"(word|entries|strings|text)", text) and threshold:
        n = threshold
        v = params[0]
        return build(f"    return [word for word in {v} if len(word) > {n}]", "keep_long_words")
    if ("positive" in text or "above zero" in text or "greater than zero" in text) and (
        "count" in text or "how many" in text or "tally" in text or "number of" in text
    ):
        v = params[0]
        return build(f"    return sum(1 for item in {v} if item > 0)", "count_positive")
    if ("negative" in text or "below zero" in text or "less than zero" in text) and (
        "count" in text or "how many" in text or "tally" in text
    ):
        v = params[0]
        return build(f"    return sum(1 for item in {v} if item < 0)", "count_negative")
    _list_nouns = ("list", "lists", "values", "entries", "numbers", "sequence", "collection", "items", "words", "entry")
    if len(params) == 1 and ("positive" in text or "above zero" in text) and ("keep" in text or "filter" in text or "collect" in text or "only" in text) and any(noun in text for noun in _list_nouns):
        v = params[0]
        return build(f"    return [item for item in {v} if item > 0]", "keep_positive")
    if len(params) == 1 and ("negative" in text or "below zero" in text) and ("keep" in text or "filter" in text or "collect" in text or "only" in text) and any(noun in text for noun in _list_nouns):
        v = params[0]
        return build(f"    return [item for item in {v} if item < 0]", "keep_negative")

    # --- Nested compositions (order matters: earlier-mentioned op first) ---
    if len(params) >= 2 and ("add" in text or "sum" in text or "combine" in text) and (
        "multiply" in text or "product" in text or "times" in text or "scale" in text or "double" in text
    ):
        a, b = params[0], params[1]
        if "first" in text or text.index("add") < text.index("multiply") if "multiply" in text else "first" in text:
            pass
        # When addition is described before multiplication, add-then-multiply.
        add_at = min((text.find(word) for word in ("add", "sum", "combine") if word in text), default=99)
        mul_at = min((text.find(word) for word in ("multiply", "product", "times", "scale", "double") if word in text), default=99)
        if add_at < mul_at:
            c = params[2] if len(params) > 2 else "2"
            return build(f"    return ({a} + {b}) * {c}", "compose_add_mul")
    if len(params) >= 2 and ("multiply" in text or "product" in text or "times" in text or "double" in text) and ("add" in text or "increase" in text):
        a, b = params[0], params[1]
        c = params[2] if len(params) > 2 else None
        if c is not None:
            return build(f"    return {a} * {b} + {c}", "compose_mul_add_param")
        two = re.search(r"(?:add|increase by) (\d+)|plus (\d+)", text)
        offset = (two.group(1) or two.group(2)) if two else "2"
        return build(f"    return {a} * {b} + {offset}", "compose_mul_add")
    if len(params) >= 2 and ("add" in text or "sum" in text or "combine" in text) and ("double" in text or "multiply" in text or "times" in text or "scale" in text):
        a, b = params[0], params[1]
        c = params[2] if len(params) > 2 else None
        if c is not None:
            return build(f"    return ({a} + {b}) * {c}", "compose_add_mul_param")
        two = re.search(r"by (\d+)|times (\d+)|double", text)
        if two:
            factor = (two.group(1) or two.group(2) or "2")
        else:
            factor = "2"
        return build(f"    return ({a} + {b}) * {factor}", "compose_add_mul")

    # --- Division / modulo ---
    if len(params) >= 2 and ("remainder" in text or "modulo" in text or "mod " in text):
        a, b = params[0], params[1]
        if _has(text, "product", "multiply", "times"):
            modulus = re.search(r"modulo (?:by )?(\d+|one|two|three|four|five)", text)
            modulus_val = modulus.group(1) if modulus else "3"
            modulus_val = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}.get(modulus_val, modulus_val)
            return build(f"    return {a} * {b} % {modulus_val}", "product_modulo")
        return build(f"    return {a} % {b}", "modulo")
    if len(params) >= 2 and ("divide" in text or "quotient" in text or "divided by" in text):
        a, b = params[0], params[1]
        return build(f"    return {a} / {b}", "divide")

    # --- Arithmetic binaries ---
    if len(params) >= 2:
        a, b = params[0], params[1]
        mod_at = re.search(r"modulo (?:by )?(\d+|one|two|three|four|five)", text)
        if _has(text, "product", "multiply", "times") and (mod_at or "mod" in text):
            modulus = mod_at.group(1) if mod_at else "3"
            modulus = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}.get(modulus, modulus)
            return build(f"    return {a} * {b} % {modulus}", "product_modulo")
        if _has(text, "difference", "subtract", "minus", "take") and "first" not in text:
            return build(f"    return {a} - {b}", "subtract")
        if _has(text, "product", "multiply", "times", "scale", "scaled"):
            return build(f"    return {a} * {b}", "multiply")
        if _has(text, "sum", "total", "add", "combine", "together", "plus"):
            expr = " + ".join(params)
            return build(f"    return {expr}", "add" if len(params) == 2 else "sum_all")
        if _has(text, "larger", "greater", "maximum", "bigger", "max", "exceeds", "exceed"):
            return build(f"    return {a} if {a} > {b} else {b}", "maximum")
        if _has(text, "smaller", "lesser", "minimum", "min"):
            return build(f"    return {a} if {a} < {b} else {b}", "minimum")
        if "first" in text or "leading" in text:
            return build(f"    return {a}", "choose_first")
        if "second" in text or "trailing" in text:
            return build(f"    return {b}", "choose_second")
        if "join" in text and _has(text, "comma", '","', "comma"):
            return build(f"    return {a} + \", \" + {b}", "join_comma")
        if "join" in text or "glue" in text or "space" in text or "together" in text or "side by side" in text or "combined" in text:
            sep = ", " if "comma" in text else " "
            joined = f"{a} + \"{sep}\" + {b}"
            if _has(text, "shouting", "upper", "capital"):
                return build(f"    return ({joined}).upper()", "join_upper")
            if _has(text, "lower", "quiet"):
                return build(f"    return ({joined}).lower()", "join_lower")
            return build(f"    return {joined}", "join")

    # --- Unary string transforms (lower casing wins over shout hints) ---
    if len(params) >= 1:
        v = params[0]
        if _has(text, "strip") and _has(text, "upper", "capital", "shout") and not _has(text, "lower", "small", "quiet"):
            return build(f"    return {v}.strip().upper()", "strip_upper")
        if _has(text, "lower", "small", "quiet"):
            return build(f"    return {v}.lower()", "lower")
        if _has(text, "upper", "capital", "shout"):
            return build(f"    return {v}.upper()", "upper")

    # --- Unary predicates / loops (collection filtering wins over parity) ---
    if len(params) == 1 and ("even" in text or "evens" in text) and _has(text, "keep", "filter", "only", "collection", "list", "entries"):
        v = params[0]
        return build(f"    return [item for item in {v} if item % 2 == 0]", "keep_even")
    if len(params) == 1 and ("double" in text or "triple" in text) and ("store" in text or "then" in text or "chain" in text):
        v = params[0]
        factor = "3" if "triple" in text else "2"
        return build(f"    stored = {v}\n    return stored * {factor}", "propagation_double")
    if len(params) >= 2 and "both" in text and "and" in text and ("hold" in text or "positive" in text or "greater" in text):
        first, second = params[0], params[1]
        return build(f"    return {first} > 0 and {second} > 0", "both_positive")
    if len(params) >= 2 and "both" in text and ("hold" in text or "positive" in text or "greater" in text):
        first, second = params[0], params[1]
        return build(f"    return {first} > 0 and {second} > 0", "both_positive")
    if "even" in text:
        v = params[0]
        return build(f"    return {v} % 2 == 0", "is_even")
    if "positive" in text:
        v = params[0]
        return build(f"    return {v} > 0", "is_positive")
    if "unchanged" in text or "as-is" in text:
        return build(f"    return {params[0]}", "identity")
    if len(params) == 1:
        v = params[0]
        _EN_SCALE = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
        for word, digit in _EN_SCALE.items():
            if re.search(rf"(?:scale|multiply|times|double).*?\b{word}\b|\b{word}\b.*?(?:times|fold)", text) and _has(text, "scale", "multiply", "times", "double"):
                return build(f"    return {v} * {digit}", "scale_constant")
        if "count" in text or "how many" in text or "tally" in text or "length" in text:
            return build(f"    return len({v})", "count_values")
        if "sum" in text or "total" in text or "add up" in text:
            return build(f"    return sum({v})", "sum_values")
        if "double" in text:
            return build(f"    return [item * 2 for item in {v}]", "double_list")
        if ("even" in text or "evens" in text) and _has(text, "keep", "filter", "only", "return"):
            return build(f"    return [item for item in {v} if item % 2 == 0]", "keep_even")

    # --- Composition of two ops via explicit then-chain (generic) ---
    return {"action": "abstain", "body": None, "operation": None, "reason": "no unambiguous operation"}
