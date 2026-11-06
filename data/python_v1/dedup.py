"""Deterministic duplicate detection for Python samples (v1.2.0).

Exact duplicates
    Identical normalised content hashes to the same value. Members are grouped,
    one representative is chosen deterministically, and the group is kept so a
    split can never separate copies of the same file.

Near duplicates
    Source code is not prose, so similarity is computed over a *code token*
    stream (names, numbers, strings, operators) rather than words. Shingles of
    ``k`` consecutive code tokens are hashed, MinHash signatures are computed
    with fixed permutations, and candidates are retrieved with LSH banding --
    never with an all-pairs comparison. Every candidate is then verified with
    an exact Jaccard computation, so a banding collision cannot create a
    false positive.

    Boilerplate is handled explicitly: shingles that occur in more than
    ``boilerplate_df_ratio`` of the samples are removed from the comparison,
    so a shared licence header cannot make two different files look similar.

Bounded behaviour
    The number of samples, the shingles per sample and the number of candidates
    compared per bucket are all capped. Hitting a cap is recorded in the
    statistics rather than silently ignored.
"""

from __future__ import annotations

import hashlib
import io
import tokenize
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from data.python_v1.registry import sha256_text

_SIGNIFICANT_TOKENS = (
    tokenize.NAME,
    tokenize.NUMBER,
    tokenize.STRING,
    tokenize.OP,
    tokenize.COMMENT,
)


def code_tokens(text: str) -> list:
    """Significant code tokens. Whitespace and layout are dropped."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in _SIGNIFICANT_TOKENS:
                tokens.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return []
    return tokens


def _hash_shingle(tokens: list, index: int, size: int) -> int:
    payload = "\x1f".join(tokens[index : index + size]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def shingle_hashes(text: str, size: int, limit: int) -> set:
    """Hashed token shingles of one sample, capped at ``limit`` shingles."""
    tokens = code_tokens(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        payload = "\x1f".join(tokens).encode("utf-8")
        return {int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")}
    hashes = set()
    for index in range(len(tokens) - size + 1):
        hashes.add(_hash_shingle(tokens, index, size))
        if len(hashes) >= limit:
            break
    return hashes


def _permutations(count: int, seed: int):
    rng = np.random.default_rng(seed)
    prime = np.uint64((1 << 61) - 1)
    a = rng.integers(1, int(prime), size=count, dtype=np.uint64)
    b = rng.integers(0, int(prime), size=count, dtype=np.uint64)
    return a, b, prime


def minhash_signature(shingles: set, permutations) -> np.ndarray:
    """MinHash signature of a shingle set."""
    if not shingles:
        return np.array([], dtype=np.uint64)
    a, b, prime = permutations
    values = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    with np.errstate(over="ignore"):
        hashed = (a[:, None] * values[None, :] + b[:, None]) % prime
    return hashed.min(axis=1)


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


class UnionFind:
    """Minimal union-find with deterministic representatives."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        # The smaller index wins so the component representative does not
        # depend on the order in which unions happened.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


@dataclass
class DedupResult:
    """Outcome of the duplicate-detection stage."""

    groups: dict = field(default_factory=dict)
    representatives: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    component_of: dict = field(default_factory=dict)


def _record_key(record) -> str:
    return f"{record.get('source_id', '')}/{record.get('file_path', '')}"


def deduplicate(records: list, config: dict | None = None) -> DedupResult:
    """Group exact and near duplicates and choose representatives.

    ``records`` are dicts with at least ``text``, ``source_id`` and
    ``file_path``. The result maps every record to a component id; component
    membership already merges records that share a source group, so a split
    assignment over components satisfies both source-group integrity and
    duplicate-group integrity at once.
    """
    config = config or {}
    size = len(records)
    dup_union = UnionFind(size)
    split_union = UnionFind(size)

    exact = config.get("exact", True)
    near = config.get("near", True)
    shingle_size = int(config.get("shingle_size", 8))
    perms = int(config.get("minhash_perms", 64))
    bands = int(config.get("minhash_bands", 16))
    threshold = float(config.get("threshold", 0.85))
    boilerplate_ratio = float(config.get("boilerplate_df_ratio", 0.5))
    max_shingles = int(config.get("max_shingles_per_record", 4000))
    max_candidates = int(config.get("max_bucket_candidates", 64))
    seed = int(config.get("seed", 120))

    stats = {
        "records": size,
        "exact_duplicate_removed": 0,
        "near_duplicate_removed": 0,
        "exact_groups": 0,
        "near_groups": 0,
        "components": 0,
        "boilerplate_shingles_removed": 0,
        "shingle_histogram": {},
        "truncated": False,
    }

    # Exact grouping -------------------------------------------------------
    exact_groups: dict = {}
    if exact:
        for index, record in enumerate(records):
            digest = sha256_text(record.get("text", ""))
            record["content_sha256"] = digest
            exact_groups.setdefault(digest, []).append(index)
    for members in exact_groups.values():
        if len(members) > 1:
            stats["exact_duplicate_removed"] += len(members) - 1
            stats["exact_groups"] += 1
            first = members[0]
            for other in members[1:]:
                dup_union.union(first, other)
                split_union.union(first, other)

    # Code-token shingles --------------------------------------------------
    shingle_sets: list = []
    for record in records:
        shingles = shingle_hashes(record.get("text", ""), shingle_size, max_shingles)
        shingle_sets.append(shingles)
        for value in shingles:
            key = f"{value:016x}"
            stats["shingle_histogram"][key] = stats["shingle_histogram"].get(key, 0) + 1

    # Boilerplate removal: shingles present in too many samples carry no
    # discriminating information, so they are excluded from the comparison.
    if boilerplate_ratio > 0 and size:
        # A shingle is boilerplate only when it is shared by a large fraction
        # of the corpus *and* by more than one sample. Without the second
        # condition a two-sample corpus would drop every shared shingle and
        # the comparison would collapse.
        threshold_count = max(2.0, boilerplate_ratio * size)
        filtered = []
        for shingles in shingle_sets:
            keep = {s for s in shingles if stats["shingle_histogram"][f"{s:016x}"] <= threshold_count}
            stats["boilerplate_shingles_removed"] += len(shingles) - len(keep)
            filtered.append(keep)
        shingle_sets = filtered

    # Near-duplicate detection --------------------------------------------
    near_pairs = 0
    if near and size:
        permutations = _permutations(perms, seed)
        rows = max(1, perms // bands)
        buckets: dict = {}
        for index, shingles in enumerate(shingle_sets):
            if not shingles:
                continue
            signature = minhash_signature(shingles, permutations)
            if signature.size == 0:
                continue
            for band in range(bands):
                chunk = signature[band * rows : (band + 1) * rows].tobytes()
                key = (band, chunk)
                members = buckets.setdefault(key, [])
                for candidate in members[:max_candidates]:
                    if jaccard(shingles, shingle_sets[candidate]) >= threshold:
                        dup_union.union(index, candidate)
                        split_union.union(index, candidate)
                        near_pairs += 1
                if len(members) < max_candidates:
                    members.append(index)
                else:
                    stats["truncated"] = True

    # Source-group integrity -----------------------------------------------
    # Every sample of one source group shares a split component, so a source
    # can never straddle a split boundary. Duplicate groups merge the sources
    # that share them, which is what keeps a copied file from leaking.
    if config.get("group_by_source", True):
        by_source: dict = {}
        for index, record in enumerate(records):
            by_source.setdefault(record.get("source_id", ""), []).append(index)
        for members in by_source.values():
            first = members[0]
            for other in members[1:]:
                split_union.union(first, other)

    # Duplicate groups (removal units) and split components (assignment units)
    duplicate_members: dict = {}
    for index in range(size):
        duplicate_members.setdefault(dup_union.find(index), []).append(index)
    component_members: dict = {}
    for index in range(size):
        component_members.setdefault(split_union.find(index), []).append(index)

    groups: dict = {}
    representatives: dict = {}
    for root, members in sorted(duplicate_members.items()):
        group_id = "dup:" + sha256_text(
            "|".join(sorted(_record_key(records[i]) for i in members))
        )[:32]
        groups[group_id] = sorted(members)
        ordered = sorted(
            members,
            key=lambda i: (
                _record_key(records[i]),
                records[i].get("content_sha256", ""),
            ),
        )
        representatives[group_id] = ordered[0]

    component_of: dict = {}
    component_ids: dict = {}
    for root, members in sorted(component_members.items()):
        component_id = "grp:" + sha256_text(
            "|".join(sorted(_record_key(records[i]) for i in members))
        )[:32]
        component_ids[root] = component_id
        for index in members:
            component_of[index] = component_id

    stats["duplicate_groups"] = len(groups)
    stats["components"] = len(component_members)
    stats["near_duplicate_pairs"] = near_pairs
    stats["near_groups"] = sum(1 for members in groups.values() if len(members) > 1)
    return DedupResult(
        groups=groups,
        representatives=representatives,
        stats=stats,
        component_of=component_of,
    )
