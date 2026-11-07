"""Conditional-learning diagnostics (Increment 12).

Pure helpers for paired-prompt teacher-forced probes. Inference-only:
nothing here trains, modifies weights, or touches frozen fixtures.
"""

from __future__ import annotations

import math

# (pair_id, operation, instruction, signature_a, signature_b, target_prefix, identifier_a, identifier_b)
PAIRED_CASES = (
    ("P1-add", "add", "Return the sum of two numbers.",
     "def add_values(left_value, right_value):", "def add_values(source_amount, target_amount):",
     "    return ", "left_value", "source_amount"),
    ("P2-sub", "subtract", "Return the difference between two numbers.",
     "def diff_values(left_value, right_value):", "def diff_values(source_amount, target_amount):",
     "    return ", "left_value", "source_amount"),
    ("P3-first", "first", "Return the first parameter.",
     "def pick_first(alpha_value, beta_value):", "def pick_first(probe_value, segment_value):",
     "    return ", "alpha_value", "probe_value"),
    ("P4-upper", "upper", "Return the text in upper case.",
     "def shout_text(raw_text):", "def shout_text(message_body):",
     "    return ", "raw_text", "message_body"),
    ("P5-mul", "multiply", "Return the product of two numbers.",
     "def prod_values(left_value, right_value):", "def prod_values(customer_total, invoice_total):",
     "    return ", "left_value", "customer_total"),
)


def token_rank(probabilities: list[float], token_id: int) -> int:
    """1-based rank of ``token_id`` in the predicted distribution."""
    if token_id < 0 or token_id >= len(probabilities):
        return -1
    correct = probabilities[token_id]
    return 1 + sum(1 for p in probabilities if p > correct)


def token_probability(probabilities: list[float], token_id: int) -> float:
    if token_id < 0 or token_id >= len(probabilities):
        return 0.0
    return float(probabilities[token_id])


def token_cross_entropy(probabilities: list[float], token_id: int) -> float:
    probability = token_probability(probabilities, token_id)
    if probability <= 0:
        return float("inf")
    return -math.log(probability)


def distribution_divergence(probs_a: list[float], probs_b: list[float]) -> dict:
    """Total-variation distance and top-token overlap between two casts."""
    if len(probs_a) != len(probs_b):
        return {"tv_distance": None, "top_token_agree": False}
    tv = 0.5 * sum(abs(a - b) for a, b in zip(probs_a, probs_b))
    top_a = max(range(len(probs_a)), key=lambda i: probs_a[i])
    top_b = max(range(len(probs_b)), key=lambda i: probs_b[i])
    return {"tv_distance": round(tv, 6), "top_token_agree": top_a == top_b, "top_a": top_a, "top_b": top_b}


def swap_case(signature: str, old: str, new: str) -> str:
    """Replace one identifier occurrence in a signature."""
    return signature.replace(old, new, 1)


def first_identifier_tokens(tokenizer, identifier: str) -> list[int]:
    return list(tokenizer.encode(identifier, add_special_tokens=False).ids)
