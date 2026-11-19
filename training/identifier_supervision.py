"""Identifier-aware supervision (Taksh v1.2.0 Increment 9, WP1 + WP2).

An AST-based classifier labels every identifier occurrence in a training
target as copied / propagated / substituted / new / builtin, and a
configurable weighting turns those labels into per-token loss weights.
The existing :func:`training.supervised.supervise` output is never
modified: callers compute a sidecar weight vector aligned with its
``labels`` array. Nothing here trains a model.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from training.objective import IGNORE_INDEX

_BUILTINS = frozenset({
    "len", "max", "min", "sum", "abs", "sorted", "range", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "enumerate", "filter",
})


@dataclass(frozen=True)
class IdentifierSupervisionConfig:
    """Per-token loss-weight configuration for identifier targets.

    ``copy_weight`` multiplies the loss on tokens that copy a prompt
    identifier; ``propagation_weight`` multiplies tokens for locally
    propagated names; repeated occurrences decay geometrically
    (``decay`` per repeat) and are floored at 1.0 so repeats never fall
    below ordinary supervision; ``max_weight`` caps any single token.
    Weights are normalized to mean 1.0 over supervised positions inside
    :func:`loss_weights`, so the global loss scale is unchanged.
    """

    copy_weight: float = 2.0
    propagation_weight: float = 1.5
    max_weight: float = 4.0
    decay: float = 0.85

    def __post_init__(self):
        if self.copy_weight < 1.0 or self.propagation_weight < 1.0:
            raise ValueError("identifier weights must be at least 1.0 (never demote)")
        if self.max_weight < 1.0:
            raise ValueError("max_weight must be at least 1.0")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")


def _parse_params(signature: str) -> list[str]:
    tree = ast.parse(signature + "\n    pass")
    return [arg.arg for arg in tree.body[0].args.args]


def _assigned_names(function) -> set[str]:
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
    return assigned


def classify_identifiers(signature: str, body: str) -> list[dict]:
    """Classify every identifier occurrence in ``body`` by provenance.

    Categories: ``copied`` (a prompt parameter loaded in the target),
    ``substituted`` (a prompt parameter stored under a different name,
    i.e. renamed binding), ``propagated`` (a locally assigned name),
    ``builtin``, and ``new`` (anything else — invented identifiers).
    Returns one dict per occurrence with its statement index.
    """
    params = _parse_params(signature)
    function = ast.parse(signature + "\n" + body).body[0]
    assigned = _assigned_names(function)
    # Param names stored under any name (renaming) vs. stored in place.
    renamed = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            value_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if value_names & set(params):
                for target in node.targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name) and sub.id not in params:
                            renamed.add(sub.id)
    occurrences = []
    for statement_index, statement in enumerate(function.body):
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name):
                continue
            name = node.id
            if isinstance(node.ctx, ast.Load):
                if name in params:
                    category = "copied"
                elif name in _BUILTINS:
                    category = "builtin"
                elif name in assigned:
                    category = "propagated"
                else:
                    category = "new"
            else:
                if name in renamed:
                    category = "substituted"
                elif name in assigned:
                    category = "propagated"
                elif name in _BUILTINS:
                    category = "builtin"
                else:
                    category = "new"
            occurrences.append(
                {"name": name, "category": category, "statement": statement_index}
            )
    return occurrences


def _occurrence_spans(body: str, occurrences: list[dict]) -> list[tuple[int, int, str]]:
    """Character spans of each occurrence's name inside ``body``."""
    spans = []
    consumed: dict[str, int] = {}
    for occurrence in occurrences:
        name = occurrence["name"]
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        offset = consumed.get(name, 0)
        match = None
        for candidate in pattern.finditer(body, offset):
            match = candidate
            break
        if match is None:
            continue
        consumed[name] = match.end()
        spans.append((match.start(), match.end(), occurrence["category"]))
    return spans


def loss_weights(
    supervision: dict,
    occurrences: list[dict],
    *,
    tokenizer,
    config: IdentifierSupervisionConfig | None = None,
) -> dict:
    """Per-label loss weights for a :func:`supervised.supervise` result.

    The returned vector has one entry per label (``len(ids) - 1``): 1.0 for
    ordinary syntax and for masked prompt positions (which carry
    ``IGNORE_INDEX`` and contribute no loss), the configured decaying
    weight for identifier tokens, renormalized to mean 1.0 across
    supervised positions.
    """
    config = config or IdentifierSupervisionConfig()
    labels = supervision["labels"]
    prompt_len = supervision["prompt_len"]
    target_text = supervision["kept_target_text"]
    # Map body character spans to target-token indices via offsets.
    spans = _occurrence_spans(target_text, occurrences)
    encoded = tokenizer.encode(target_text, add_special_tokens=False)
    offsets = encoded.offsets
    # Labels index i predicts ids[i+1]; target token j sits at label
    # prompt_len - 1 + j. eos label index: len(labels)-1.
    weights = [1.0] * len(labels)
    seen: dict[str, int] = {}
    token_weights: dict[int, float] = {}
    for start, end, category in spans:
        if category not in ("copied", "propagated", "substituted"):
            continue
        base = config.copy_weight if category == "copied" else config.propagation_weight
        for token_index, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            occurrence_key = f"{category}:{token_index}:{start}"
            repeat = seen.get(occurrence_key, 0)
            seen[occurrence_key] = repeat + 1
            decayed = max(1.0, base * (config.decay ** repeat))
            token_weights[token_index] = min(config.max_weight, decayed)
    for token_index, weight in token_weights.items():
        label_index = (prompt_len - 1) + token_index
        if 0 <= label_index < len(weights):
            weights[label_index] = weight
    supervised_indices = [
        index for index, label in enumerate(labels) if label != IGNORE_INDEX
    ]
    mean = (
        sum(weights[index] for index in supervised_indices) / len(supervised_indices)
        if supervised_indices
        else 1.0
    )
    if mean > 0:
        weights = [weight / mean for weight in weights]
    counts: dict[str, int] = {}
    for occurrence in occurrences:
        counts[occurrence["category"]] = counts.get(occurrence["category"], 0) + 1
    return {
        "weights": weights,
        "occurrence_counts": counts,
        "weighted_tokens": len(token_weights),
        "mean_weight": round(mean, 4) if mean else 1.0,
    }


def weighted_token_mean_loss(logits, labels, weights):
    """Token-mean cross-entropy with per-token weights (test helper).

    Weights renormalize to mean 1.0 over supervised positions so the loss
    scale matches the unweighted objective when weights are uniform.
    """
    import torch
    import torch.nn.functional as F

    from training.objective import IGNORE_INDEX as _IGNORE

    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_labels = labels.reshape(-1)
    flat_weights = weights.reshape(-1).float().to(logits.device)
    mask = flat_labels != _IGNORE
    count = int(mask.sum().item())
    if count == 0:
        return torch.zeros((), dtype=torch.float32, device=logits.device), 0
    per_token = F.cross_entropy(
        flat_logits, flat_labels, ignore_index=_IGNORE, reduction="none"
    )
    weighted = per_token[mask] * flat_weights[mask]
    return weighted.sum() / flat_weights[mask].sum(), count
