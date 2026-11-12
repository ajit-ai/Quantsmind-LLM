"""Source-group and duplicate-group aware dataset splitting (v1.2.0).

The unit of assignment is a *component*: the connected component of the
graph whose nodes are samples and whose edges are (a) shared source group and
(b) shared exact or near-duplicate group. Assigning whole components to splits
gives both integrity properties at once:

* no source group straddles train/val/test, and
* no known duplicate (exact or near) is separated across splits.

Assignment is deterministic: components are ordered by a stable hash of their
identity, and the largest-remainder method fills the configured proportions.
The result does not depend on input order, on dictionary ordering, or on the
number of samples seen so far.

Proportions are configuration, not a constant. When the corpus is too small
to support the requested proportions -- too few source groups for three
non-empty splits, for instance -- the split is produced but reported as
**not ready**, with the reason. No statistical claim is made from a small
holdout.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

SPLIT_NAMES = ("train", "val", "test")


def _order_key(component_id: str) -> str:
    return hashlib.sha256(("split-order:" + component_id).encode("utf-8")).hexdigest()


def assign_splits(components: dict, config: dict | None = None) -> dict:
    """Assign components to splits.

    ``components`` maps a component id to a list of record indexes. Returns a
    dict with the per-component assignment, the resulting counts and the
    readiness assessment.
    """
    config = config or {}
    proportions = config.get("proportions") or {"train": 0.8, "val": 0.1, "test": 0.1}
    total = sum(float(proportions.get(name, 0.0)) for name in SPLIT_NAMES) or 1.0
    fractions = {name: float(proportions.get(name, 0.0)) / total for name in SPLIT_NAMES}

    ordered = sorted(components, key=_order_key)
    targets = {name: fractions[name] * len(ordered) for name in SPLIT_NAMES}
    assigned_counts = {name: 0 for name in SPLIT_NAMES}
    assignment: dict = {}
    for component_id in ordered:
        best = max(
            SPLIT_NAMES,
            key=lambda name: (
                targets[name] - assigned_counts[name],
                -SPLIT_NAMES.index(name),
            ),
        )
        assignment[component_id] = best
        assigned_counts[best] += 1
    return {"assignment": assignment, "component_counts": assigned_counts}


def _split_for(record_index: int, component_of: dict, assignment: dict) -> str:
    return assignment[component_of[record_index]]


def build_splits(records: list, component_of: dict, config: dict | None = None) -> dict:
    """Produce the split assignment and the leakage audit.

    ``component_of`` maps a record index to its component id. Components are
    the connected units of source group and duplicate group, so assigning
    whole components satisfies both integrity properties.

    Returns a dict with ``split_of_record``, per-split record lists, the
    component map and an integrity report. The report is explicit about what
    was checked and what could not be.
    """
    config = config or {}
    components: dict = {}
    for index, component_id in component_of.items():
        components.setdefault(component_id, []).append(index)
    result = assign_splits(components, config)
    assignment = result["assignment"]

    split_of_record = {
        index: _split_for(index, component_of, assignment)
        for index in range(len(records))
    }
    per_split: dict = {name: [] for name in SPLIT_NAMES}
    for index, name in split_of_record.items():
        per_split[name].append(index)

    # Leakage audit ---------------------------------------------------------
    source_splits: dict = defaultdict(set)
    for index, record in enumerate(records):
        source_splits[record.get("source_id", "")].add(split_of_record[index])
    straddling_sources = sorted(s for s, names in source_splits.items() if len(names) > 1)

    digest_splits: dict = defaultdict(set)
    for index, record in enumerate(records):
        digest = record.get("content_sha256")
        if digest:
            digest_splits[digest].add(split_of_record[index])
    straddling_digests = sorted(d for d, names in digest_splits.items() if len(names) > 1)

    component_splits: dict = defaultdict(set)
    for index, component_id in component_of.items():
        component_splits[component_id].add(split_of_record[index])
    straddling_components = sorted(
        c for c, names in component_splits.items() if len(names) > 1
    )

    counts = {name: len(per_split[name]) for name in SPLIT_NAMES}
    min_groups = int(config.get("min_groups_per_split", 1))
    min_records = int(config.get("min_records_per_split", 1))
    insufficient = []
    for name in SPLIT_NAMES:
        if result["component_counts"].get(name, 0) < min_groups:
            insufficient.append(
                f"{name}: {result['component_counts'].get(name, 0)} groups "
                f"< {min_groups} required"
            )
        if counts[name] < min_records:
            insufficient.append(
                f"{name}: {counts[name]} records < {min_records} required"
            )

    ready = not insufficient and not straddling_sources and not straddling_digests
    return {
        "split_of_record": split_of_record,
        "per_split": per_split,
        "assignment": assignment,
        "counts": counts,
        "component_counts": result["component_counts"],
        "components": components,
        "split_ready": ready,
        "insufficient": insufficient,
        "integrity": {
            "sources_straddling_splits": straddling_sources,
            "duplicate_digests_straddling_splits": straddling_digests,
            "components_straddling_splits": straddling_components,
            "checked": (
                "source groups, exact-duplicate digests and duplicate components"
            ),
        },
        "proportions": config.get("proportions")
        or {"train": 0.8, "val": 0.1, "test": 0.1},
        "seed": config.get("seed", 120),
    }
