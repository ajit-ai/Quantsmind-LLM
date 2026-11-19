"""Deterministic diagnostics for Increment 8 (no training).

Helpers for tokenizer/identifier analysis, supervision alignment checks,
and checkpoint consistency. All functions are pure reads: they never
modify a tokenizer, model, checkpoint, or benchmark file.
"""

from __future__ import annotations

import hashlib

FAMILIAR_IDENTIFIERS = (
    "left_value", "right_value", "source_value", "customer_total",
    "invoice_total", "raw_text", "message_body", "text", "value",
    "alpha_value", "beta_value", "item_count", "factor",
)

UNFAMILIAR_IDENTIFIERS = (
    "probe_00", "segment_00", "vertex_27", "slot_19", "zebra_count",
    "quixotic_label", "jumbotron_qty", "x", "y", "qq", "customer",
    "invoice_segment_total", "node_41", "cell_08",
)

# Synthetic copy/transform probes: clearly separate from frozen AD-1/C3
# and the Increment 6 supplemental suite (novel names + wording).
COPY_CASES = (
    # (case_id, kind, instruction, signature, expected_body_fragment)
    ("COPY-novel-add-001", "add", "Return the combined total of the two inputs.",
     "def zz_combined(qq_item, rr_item):", "qq_item + rr_item"),
    ("COPY-novel-add-002", "add", "Add these quantities together.",
     "def probe_add_total(qq_val, rr_val):", "qq_val + rr_val"),
    ("COPY-familiar-add-001", "add", "Return the sum of two numbers.",
     "def adds_pair(left_value, right_value):", "left_value + right_value"),
    ("COPY-familiar-add-002", "add", "Compute a plus b.",
     "def sums_two(first_number, second_number):", "first_number + second_number"),
    ("COPY-novel-first-001", "first", "Return the first parameter.",
     "def zz_head(qq_item, rr_item):", "qq_item"),
    ("COPY-novel-first-002", "first", "Hand back the leading value.",
     "def probe_head(qq_val, rr_val):", "qq_val"),
    ("COPY-familiar-first-001", "first", "Pick the first input.",
     "def returns_first(alpha_value, beta_value):", "alpha_value"),
    ("COPY-novel-upper-001", "upper", "Return the text in upper case.",
     "def zz_shout(qq_text):", "qq_text.upper()"),
    ("COPY-familiar-upper-001", "upper", "Shout the input text.",
     "def shouts_text(raw_text):", "raw_text.upper()"),
    ("COPY-novel-sub-001", "subtract", "Return the difference between two numbers.",
     "def zz_diff(qq_item, rr_item):", "qq_item - rr_item"),
    ("COPY-familiar-sub-001", "subtract", "Compute a minus b.",
     "def subtracts_pair(left_value, right_value):", "left_value - right_value"),
    ("COPY-novel-mul-001", "multiply", "Return the product of two numbers.",
     "def zz_prod(qq_val, rr_val):", "qq_val * rr_val"),
    ("COPY-familiar-mul-001", "multiply", "Multiply two inputs together.",
     "def multiplies_pair(left_value, right_value):", "left_value * right_value"),
)


def tokenizer_stats(tokenizer) -> dict:
    """Fragmentation and round-trip statistics for identifier sets."""
    stats = {}
    for label, names in (("familiar", FAMILIAR_IDENTIFIERS), ("unfamiliar", UNFAMILIAR_IDENTIFIERS)):
        per_name = []
        for name in names:
            encoded = tokenizer.encode(name)
            per_name.append({
                "identifier": name,
                "tokens": len(encoded.ids),
                "pieces": list(encoded.tokens),
                "round_trip": tokenizer.decode(encoded.ids) == name,
            })
        stats[label] = {
            "count": len(names),
            "mean_fragments": round(sum(item["tokens"] for item in per_name) / len(per_name), 3),
            "max_fragments": max(item["tokens"] for item in per_name),
            "all_round_trip": all(item["round_trip"] for item in per_name),
            "per_name": per_name,
        }
    return stats


def checkpoint_param_hash(path) -> str:
    """SHA-256 over the sorted state-dict entries (load consistency)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def architectures_consistent(paths) -> dict:
    """Confirm every checkpoint carries the same architecture config."""
    import torch

    configs = {}
    for label, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        configs[label] = payload["config"]["model"]
        keys = sorted(payload["model"].keys())
        configs[label]["_tensors"] = len(keys)
    unique = {json_tuple(cfg) for cfg in configs.values()}
    return {"configs": configs, "all_same_architecture": len(unique) == 1}


def json_tuple(value):
    import json

    return json.dumps(value, sort_keys=True, default=str)


def supervision_alignment(example, tokenizer) -> dict:
    """Check prompt/target alignment and loss masking for one example.

    Returns facts: prompt ids are masked, target ids are supervised, the
    target body names every example parameter, and the EOS token is
    present in the supervised tail.
    """
    from training.objective import IGNORE_INDEX
    from training.supervised import supervise

    supervised = supervise(tokenizer, example, max_length=256)
    labels = supervised["labels"]
    prompt_len = supervised["prompt_len"]
    target_labels = labels[prompt_len - 1:] if prompt_len else labels
    import re

    params = [arg.strip() for arg in example.signature.split("(", 1)[1].split(")", 1)[0].split(",") if arg.strip()]
    return {
        "example_id": example.example_id,
        "prompt_len": prompt_len,
        "target_len": supervised["target_len"],
        "prompt_all_masked": all(label == IGNORE_INDEX for label in labels[: max(0, prompt_len - 1)]),
        "target_all_supervised": all(label != IGNORE_INDEX for label in target_labels) if target_labels else False,
        "params_in_target": {param: (param in example.body) for param in params},
        "all_params_in_target": all(param in example.body for param in params),
        "eos_in_target": supervised["ids"][-1] == tokenizer.token_to_id("<eos>"),
    }


def copy_case_bindings(completion: str, signature: str) -> dict:
    """Does the completion use exactly the requested parameter tokens?

    Method/attribute names (``.upper()``, ``.append()``) are not invented
    identifiers — they are operations on values — and are excluded.
    """
    import ast as _ast

    params = [arg.strip() for arg in signature.split("(", 1)[1].split(")", 1)[0].split(",") if arg.strip()]
    used = {param: (param in completion) for param in params}
    import re

    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", completion)) - {"return"}
    method_names: set[str] = set()
    try:
        tree = _ast.parse(signature + "\n" + completion)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Attribute):
                method_names.add(node.attr)
    except SyntaxError:
        pass
    invented = sorted(identifiers - set(params) - method_names)
    return {
        "used": used,
        "exact_copy": all(used.values()) and not invented,
        "invented_identifiers": invented,
        "parses": _parses(signature, completion),
    }


def _parses(signature: str, completion: str) -> bool:
    import ast

    lines = [line.rstrip() for line in (completion or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return False
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    try:
        ast.parse(signature + "\n" + "\n".join(lines))
        return True
    except SyntaxError:
        return False
