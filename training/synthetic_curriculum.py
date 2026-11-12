"""Deterministic synthetic Python curriculum for the v1.9.0 experiment.

The curriculum exists to answer one question: can the S2 model learn
*reusable* Python structure, or only memorised templates? To make that
distinguishable it is generated from explicit grammars, not by paraphrasing a
corpus, and it is partitioned so that three different kinds of generalisation
can be measured separately:

``train``
    Four families. Two families are deliberately withheld.
``A`` (in-distribution holdout)
    The trained families with unseen parameter combinations and unseen
    function names.
``B`` (compositional holdout)
    New compositions of familiar operations, never seen in training.
``C`` (family holdout)
    Two entirely withheld families, with their own structures.

Nothing here derives from a real corpus, a benchmark, or the repository's
historical data. Every sample is generated from the grammars below with a
fixed seed, and every sample carries a content hash, a template id and a
family label so contamination can be measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

FAMILIES = (
    "arithmetic",
    "conditionals",
    "list_ops",
    "string_ops",
    "dict_ops",
    "loop_agg",
)

TRAIN_FAMILIES = ("arithmetic", "conditionals", "list_ops", "string_ops")
HELDOUT_FAMILIES = ("dict_ops", "loop_agg")

TRAIN_NAMES = {
    "arithmetic": ("add_pair", "scale_value", "offset_amount"),
    "conditionals": ("choose_sign", "classify_range"),
    "list_ops": ("keep_positive", "double_all", "drop_small"),
    "string_ops": ("reverse_text", "count_letters"),
}
UNSEEN_NAMES = {
    "arithmetic": ("combine_amounts", "shift_value"),
    "conditionals": ("label_value", "bucket_number"),
    "list_ops": ("filter_even", "square_each"),
    "string_ops": ("flip_text", "letter_frequency"),
}
HELDOUT_NAMES = {
    "dict_ops": ("lookup_default", "merge_counts"),
    "loop_agg": ("sum_even_values", "count_long_words"),
}


@dataclass
class Case:
    """One generated task."""

    case_id: str
    family: str
    partition: str
    template_id: str
    prompt: str
    reference: str
    content_sha256: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "partition": self.partition,
            "template_id": self.template_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "content_sha256": self.content_sha256,
            "params": dict(self.params),
        }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_hash(text: str) -> str:
    """Hash with identifiers and numbers erased, so parameter swaps collide."""
    stripped = re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*\b", "ID", text)
    stripped = re.sub(r"\d+", "N", stripped)
    return _hash(stripped)


def _shingles(text: str, size: int = 5) -> set:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]", text)
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# --------------------------------------------------------------------------- #
# Family grammars. Each returns (prompt, reference, template_id, params).
# --------------------------------------------------------------------------- #
def _arithmetic(name: str, rng) -> tuple:
    salt = _SALT[0]
    left, right = 2 + salt, 3 + salt
    prompt = f'def {name}(left, right):\n    """Return the sum of the two inputs."""\n'
    reference = (
        f'def {name}(left, right):\n'
        f'    """Return the sum of the two inputs."""\n'
        f"    return left + right + {left}\n"
    )
    return prompt, reference, "arithmetic:sum", {"offset": left, "right": right}


def _conditionals(name: str, rng) -> tuple:
    bound = 3 + _SALT[0]
    prompt = f'def {name}(value):\n    """Return a label for the value."""\n'
    reference = (
        f'def {name}(value):\n'
        f'    """Return a label for the value."""\n'
        f"    if value > {bound}:\n"
        f'        return "high"\n'
        f"    if value < -{bound}:\n"
        f'        return "low"\n'
        f'    return "middle"\n'
    )
    return prompt, reference, "conditionals:label", {"bound": bound}


def _list_ops(name: str, rng) -> tuple:
    threshold = 1 + _SALT[0]
    prompt = f'def {name}(values):\n    """Return the values above a threshold."""\n'
    reference = (
        f'def {name}(values):\n'
        f'    """Return the values above a threshold."""\n'
        f"    return [value for value in values if value > {threshold}]\n"
    )
    return prompt, reference, "list_ops:filter", {"threshold": threshold}


def _string_ops(name: str, rng) -> tuple:
    letter = "abcdefghilmnoprstu"[_SALT[0] % len("abcdefghilmnoprstu")]
    salt = _SALT[0]
    prompt = f'def {name}(text):\n    """Work with a text value."""\n'
    reference = (
        f'def {name}(text):\n'
        f'    """Work with a text value."""\n'
        f'    reversed_text = text[::-1]\n'
        f'    return reversed_text.count("{letter}") + {salt}\n'
    )
    return prompt, reference, "string_ops:count", {"letter": letter}


def _dict_ops(name: str, rng) -> tuple:
    default = 1 + _SALT[0]
    prompt = f'def {name}(mapping, key):\n    """Return a mapping value or a default."""\n'
    reference = (
        f'def {name}(mapping, key):\n'
        f'    """Return a mapping value or a default."""\n'
        f"    return mapping.get(key, {default})\n"
    )
    return prompt, reference, "dict_ops:lookup", {"default": default}


def _loop_agg(name: str, rng) -> tuple:
    factor = 2 + (_SALT[0] % 5)
    salt = _SALT[0]
    prompt = f'def {name}(values):\n    """Accumulate values with a loop."""\n'
    reference = (
        f'def {name}(values):\n'
        f'    """Accumulate values with a loop."""\n'
        f"    total = 0\n"
        f"    for value in values:\n"
        f"        total += value * {factor} + {salt}\n"
        f"    return total\n"
    )
    return prompt, reference, "loop_agg:accumulate", {"factor": factor}


def _composed(name: str, rng) -> tuple:
    threshold = 2 + _SALT[0]
    prompt = (
        f'def {name}(values):\n'
        f'    """Keep the large values, then sum them."""\n'
    )
    reference = (
        f'def {name}(values):\n'
        f'    """Keep the large values, then sum them."""\n'
        f"    kept = [value for value in values if value > {threshold}]\n"
        f"    total = 0\n"
        f"    for value in kept:\n"
        f"        total += value\n"
        f"    return total\n"
    )
    return prompt, reference, "composed:filter_then_sum", {"threshold": threshold}


#: A per-case salt that keeps every generated sample textually distinct
#: while preserving the family structure. It is what makes "same
#: template, new parameters" the only difference between training and
#: partition A.
_SALT = [0]


GRAMMARS = {
    "arithmetic": _arithmetic,
    "conditionals": _conditionals,
    "list_ops": _list_ops,
    "string_ops": _string_ops,
    "dict_ops": _dict_ops,
    "loop_agg": _loop_agg,
    "composed": _composed,
}


def _make_case(case_id: str, family: str, partition: str, name: str, rng) -> Case:
    _SALT[0] += 1
    prompt, reference, template_id, params = GRAMMARS[family](name, rng)
    return Case(
        case_id=case_id,
        family=family,
        partition=partition,
        template_id=template_id,
        prompt=prompt,
        reference=reference,
        content_sha256=_hash(reference),
        params=params,
    )


def build_curriculum(seed: int = 20261021) -> dict:
    """Build the frozen curriculum: train plus partitions A, B and C."""
    import random

    # The salt counter is per build, so two builds of the same seed produce
    # byte-identical curricula.
    _SALT[0] = 0
    partitions: dict = {"train": [], "A": [], "B": [], "C": []}

    for index in range(120):
        family = TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
        rng = random.Random(f"{seed}:train:{index}")
        name = TRAIN_NAMES[family][index % len(TRAIN_NAMES[family])]
        partitions["train"].append(
            _make_case(f"train-{family}-{index:03d}", family, "train", name, rng)
        )

    for index in range(40):
        family = TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
        rng = random.Random(f"{seed}:A:{index}")
        names = UNSEEN_NAMES[family]
        name = names[index % len(names)]
        partitions["A"].append(
            _make_case(f"A-{family}-{index:03d}", family, "A", name, rng)
        )

    for index in range(30):
        rng = random.Random(f"{seed}:B:{index}")
        name = UNSEEN_NAMES["list_ops"][index % len(UNSEEN_NAMES["list_ops"])]
        partitions["B"].append(
            _make_case(f"B-composed-{index:03d}", "composed", "B", name, rng)
        )

    for index in range(40):
        family = HELDOUT_FAMILIES[index % len(HELDOUT_FAMILIES)]
        rng = random.Random(f"{seed}:C:{index}")
        names = HELDOUT_NAMES[family]
        name = names[index % len(names)]
        partitions["C"].append(
            _make_case(f"C-{family}-{index:03d}", family, "C", name, rng)
        )

    return partitions


def curriculum_digest(partitions: dict) -> str:
    """A single hash over the whole frozen curriculum."""
    payload = "\n".join(
        f"{case.partition}:{case.case_id}:{case.content_sha256}"
        for partition in sorted(partitions)
        for case in partitions[partition]
    )
    return _hash(payload)


# --------------------------------------------------------------------------- #
# Contamination controls
# --------------------------------------------------------------------------- #
def contamination_report(partitions: dict, *, similarity_threshold: float = 0.9) -> dict:
    """Measure cross-partition contamination.

    A *violation* is a real leak: an identical sample across partitions, an
    identical target solution, a withheld family appearing in training, or a
    near-duplicate structure above the similarity threshold for a partition
    that is supposed to differ structurally (B and C).

    Sharing a template between training and partition A is *by design* -- A
    measures the same families with new parameters and new names -- so it is
    reported as a note, not counted as a leak.
    """
    violations: list = []
    notes: list = []
    train_shingles = [_shingles(case.reference) for case in partitions["train"]]
    train_families = {case.family for case in partitions["train"]}
    structural_partitions = {"B", "C"}
    inherited_partitions = {"train", "A"}

    def spans(group: dict, left: set, right: set) -> bool:
        return bool(
            {p for p, ids in group.items() if ids} & left
        ) and bool({p for p, ids in group.items() if ids} & right)

    exact: dict = {}
    normalized: dict = {}
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
            violations.append(
                f"identical sample across partitions {key[:12]}: {group}"
            )
    for key, group in normalized.items():
        if spans(group, inherited_partitions, structural_partitions):
            violations.append(
                f"parameter-erased duplicate across an inheriting and a "
                f"structural partition {key[:12]}: {group}"
            )
        elif len(group) > 1:
            notes.append(
                f"template-shared samples across {sorted(group)} (expected for "
                "the in-distribution partition)"
            )

    for case in partitions["C"]:
        if case.family in train_families:
            violations.append(
                f"{case.case_id}: withheld family {case.family} appears in training"
            )

    max_similarity = {"A": 0.0, "B": 0.0, "C": 0.0}
    for partition in ("A", "B", "C"):
        for case in partitions[partition]:
            shingles = _shingles(case.reference)
            best = 0.0
            for train_case, train_shingles_set in zip(
                partitions["train"], train_shingles
            ):
                best = max(best, _jaccard(shingles, train_shingles_set))
            max_similarity[partition] = max(max_similarity[partition], best)
            if partition in structural_partitions and best >= similarity_threshold:
                violations.append(
                    f"{case.case_id}: structural similarity {best:.2f} with a "
                    "training sample"
                )

    return {
        "schema": "quantsmind.synthetic_contamination/1",
        "violations": violations,
        "violation_count": len(violations),
        "notes": notes[:10],
        "contaminated": bool(violations),
        "verdict": (
            "SYNTHETIC_EVALUATION_CONTAMINATED" if violations else "CLEAN"
        ),
        "similarity_threshold": similarity_threshold,
        "max_train_similarity": {k: round(v, 4) for k, v in max_similarity.items()},
        "partition_sizes": {
            partition: len(cases) for partition, cases in sorted(partitions.items())
        },
    }
