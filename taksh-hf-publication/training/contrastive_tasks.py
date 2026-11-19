"""Contrastive task pairs for Taksh v2.0.0.

Each pair is two tasks that differ by exactly one meaningful requirement, so
the only way to answer both correctly is to read the instruction. A model
that emits one template for everything fails the pair; a model that changes
only the name fails it as well.

Pairs: arithmetic add/subtract, conditionals max/min, list keep-positive /
keep-negative, string upper/lower, dict lookup/count-keys, loop sum/count.

Partitions: ``train`` (both variants of four families), ``A`` (same templates,
unseen parameters and names), ``B`` (composed tasks), ``C`` (two families
never trained), ``D`` (the contrastive holdout: matched pairs of the trained
families with unseen names and parameters).
"""

from __future__ import annotations

import random

from training.synthetic_curriculum import Case, _hash

#: (family, variant) -> (function name, arguments, docstring, body, operation)
TASKS = {
    ("arithmetic", "add"): (
        "combine_values", "left, right", "Return the sum of the two inputs.",
        "    return left + right", "add"),
    ("arithmetic", "subtract"): (
        "separate_values", "left, right", "Return the difference of the two inputs.",
        "    return left - right", "subtract"),
    ("conditionals", "maximum"): (
        "choose_larger", "left, right", "Return the larger of the two values.",
        "    if left > right:\n        return left\n    return right", "maximum"),
    ("conditionals", "minimum"): (
        "choose_smaller", "left, right", "Return the smaller of the two values.",
        "    if left < right:\n        return left\n    return right", "minimum"),
    ("list_ops", "keep_positive"): (
        "select_positive", "values", "Return only the values above zero.",
        "    return [value for value in values if value > 0]", "keep_positive"),
    ("list_ops", "keep_negative"): (
        "select_negative", "values", "Return only the values below zero.",
        "    return [value for value in values if value < 0]", "keep_negative"),
    ("string_ops", "upper"): (
        "shout_text", "text", "Return the text in upper case.",
        "    return text.upper()", "upper"),
    ("string_ops", "lower"): (
        "whisper_text", "text", "Return the text in lower case.",
        "    return text.lower()", "lower"),
    ("dict_ops", "lookup"): (
        "read_value", "mapping, key", "Return the value for a key or a default.",
        "    return mapping.get(key, 0)", "lookup"),
    ("dict_ops", "count_keys"): (
        "count_entries", "mapping, key", "Return how many keys the mapping has.",
        "    return len(mapping)", "count_keys"),
    ("loop_agg", "sum_values"): (
        "total_values", "values", "Accumulate the values with a loop.",
        "    total = 0\n    for value in values:\n        total += value\n    return total",
        "sum_values"),
    ("loop_agg", "count_values"): (
        "count_entries", "values", "Count the values with a loop.",
        "    count = 0\n    for value in values:\n        count += 1\n    return count",
        "count_values"),
}

#: Markers that indicate the requested operation in a generated program.
OPERATION_MARKERS = {
    "add": [" + "],
    "subtract": [" - "],
    "maximum": [">", "max"],
    "minimum": ["<", "min"],
    "keep_positive": ["> 0", ">0"],
    "keep_negative": ["< 0", "<0"],
    "upper": [".upper("],
    "lower": [".lower("],
    "lookup": [".get("],
    "count_keys": ["len("],
    "sum_values": ["+=", "sum"],
    "count_values": ["+= 1", "count"],
}

TRAINED_FAMILIES = ("arithmetic", "conditionals", "list_ops", "string_ops")
WITHHELD_FAMILIES = ("dict_ops", "loop_agg")

#: Unseen function names used by the holdout partitions.
HOLDOUT_NAMES = {
    "arithmetic": ("merge_numbers", "split_numbers"),
    "conditionals": ("pick_bigger", "pick_smaller"),
    "list_ops": ("select_above_zero", "select_below_zero"),
    "string_ops": ("capitalise_text", "lowercase_text"),
    "dict_ops": ("read_entry", "size_of_mapping"),
    "loop_agg": ("running_total", "running_count"),
}

#: Partition D uses different names again, so a matched pair differs in the
#: operation *and* in the requested name.
CONTRASTIVE_NAMES = {
    "arithmetic": ("add_up_numbers", "take_away_numbers"),
    "conditionals": ("bigger_of_two", "smaller_of_two"),
    "list_ops": ("keep_above_zero", "keep_below_zero"),
    "string_ops": ("shout_it", "whisper_it"),
    "dict_ops": ("entry_value", "entry_count"),
    "loop_agg": ("total_it", "count_it"),
}


def _case(case_id, family, variant, name, partition, salt):
    function, arguments, docstring, body, operation = TASKS[(family, variant)]
    prompt = (
        f"def {name}({arguments}):\n"
        f"    \"\"\"{docstring} [case {salt}]\"\"\"\n"
    )
    reference = f"{prompt}{body}\n"
    return Case(
        case_id=case_id,
        family=family,
        partition=partition,
        template_id=f"{family}:{variant}",
        prompt=prompt,
        reference=reference,
        content_sha256=_hash(reference),
        params={
            "operation": operation,
            "variant": variant,
            "pair": family,
            "salt": salt,
        },
    )


def _variants(family):
    return [variant for (fam, variant) in TASKS if fam == family]


def build_contrastive_curriculum(seed: int = 20261021) -> dict:
    """Build the frozen contrastive curriculum (train, A, B, C, D)."""
    partitions = {"train": [], "A": [], "B": [], "C": [], "D": []}
    rng = random.Random(f"{seed}:contrastive")

    for index in range(80):
        family = TRAINED_FAMILIES[index % len(TRAINED_FAMILIES)]
        variants = _variants(family)
        variant = variants[index % len(variants)]
        name = TASKS[(family, variant)][0]
        partitions["train"].append(
            _case(f"train-{family}-{variant}-{index:03d}", family, variant, name,
                  "train", index)
        )

    for index in range(40):
        family = TRAINED_FAMILIES[index % len(TRAINED_FAMILIES)]
        variants = _variants(family)
        variant = variants[index % len(variants)]
        name = HOLDOUT_NAMES[family][0 if variant == variants[0] else 1]
        partitions["A"].append(
            _case(f"A-{family}-{variant}-{index:03d}", family, variant, name, "A", index)
        )

    for index in range(24):
        family = WITHHELD_FAMILIES[index % len(WITHHELD_FAMILIES)]
        variants = _variants(family)
        variant = variants[index % len(variants)]
        name = HOLDOUT_NAMES[family][0 if variant == variants[0] else 1]
        partitions["C"].append(
            _case(f"C-{family}-{variant}-{index:03d}", family, variant, name, "C", index)
        )

    # Partition D: matched pairs of the trained families, both variants, unseen
    # names. A model that ignores the instruction answers both members of a
    # pair the same way; a model that only copies the name still fails it.
    for index in range(24):
        family = TRAINED_FAMILIES[index % len(TRAINED_FAMILIES)]
        variants = _variants(family)
        pair_id = f"{family}-{index // len(TRAINED_FAMILIES):02d}"
        for position, variant in enumerate(variants):
            name = CONTRASTIVE_NAMES[family][position]
            case = _case(
                f"D-{pair_id}-{variant}", family, variant, name, "D", index
            )
            case.params["pair_id"] = pair_id
            partitions["D"].append(case)

    # Partition B: composed tasks (filter then sum) built from trained ops.
    for index in range(16):
        threshold = 1 + index
        prompt = (
            f"def filter_then_total_{index:02d}(values):\n"
            f'    """Keep the values above a threshold, then sum them."""\n'
        )
        reference = (
            f"{prompt}"
            f"    kept = [value for value in values if value > {threshold}]\n"
            f"    total = 0\n"
            f"    for value in kept:\n"
            f"        total += value\n"
            f"    return total\n"
        )
        partitions["B"].append(
            Case(
                case_id=f"B-composed-{index:03d}",
                family="composed",
                partition="B",
                template_id="composed:filter_then_total",
                prompt=prompt,
                reference=reference,
                content_sha256=_hash(reference),
                params={"operation": "filter_then_total", "pair": "composed",
                        "salt": index},
            )
        )
    return partitions


def curriculum_digest(partitions: dict) -> str:
    """A single hash over the whole contrastive curriculum."""
    payload = "\n".join(
        f"{case.partition}:{case.case_id}:{case.content_sha256}"
        for partition in sorted(partitions)
        for case in partitions[partition]
    )
    return _hash(payload)


def contamination_report(partitions: dict) -> dict:
    """Cross-partition contamination for the contrastive curriculum."""
    from training.synthetic_curriculum import _normalized_hash

    violations = []
    exact = {}
    normalized = {}
    for partition in sorted(partitions):
        for case in partitions[partition]:
            exact.setdefault(case.content_sha256, {}).setdefault(partition, []).append(
                case.case_id
            )
            normalized.setdefault(
                _normalized_hash(case.reference), {}
            ).setdefault(partition, []).append(case.case_id)
    for key, group in exact.items():
        if len(group) > 1:
            violations.append(f"identical target across partitions {key[:12]}: {group}")
    # Parameter-erased (template-level) sharing between train, A and D is
    # deliberate in this curriculum: D asks for the *same* operations with
    # unseen names and constants, which is exactly what makes it a test of
    # instruction following rather than of template recall. Only a withheld
    # family (C) appearing in training is a leak.
    train_families = {case.family for case in partitions["train"]}
    for case in partitions["C"]:
        if case.family in train_families:
            violations.append(
                f"{case.case_id}: withheld family {case.family} appears in training"
            )
    return {
        "schema": "quantsmind.synthetic_contamination/1",
        "violations": violations,
        "violation_count": len(violations),
        "contaminated": bool(violations),
        "verdict": "SYNTHETIC_EVALUATION_CONTAMINATED" if violations else "CLEAN",
        "partition_sizes": {
            partition: len(cases) for partition, cases in sorted(partitions.items())
        },
        "note": (
            "Template-level sharing between train, A and D is deliberate: the "
            "contrastive holdout asks for the same operations with unseen names "
            "and constants."
        ),
    }
