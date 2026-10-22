"""Evaluation protocol for the v1.9.0 synthetic curriculum.

The protocol is frozen: the same cases, the same prompts, the same token
budget, the same decoding settings and the same validators are used before
and after training. Nothing here executes generated code; every metric is
static.

Metrics
-------
``syntax_valid``          the assembled program parses
``complete_function``     the assembled program defines the requested function
``exact_target``          the assembled program matches the reference
``structural_similarity`` token-shingle similarity against the reference
``degenerate``            the completion collapses into repetition
``eos_terminated``        generation stopped on the end-of-document token
``truncated``             generation stopped on the budget
``tokens``                tokens generated
"""

from __future__ import annotations

import ast
import re
import time
from collections import Counter, defaultdict

from training.synthetic_curriculum import _jaccard, _shingles


def requested_function(prompt: str) -> str:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z_0-9]*)", prompt)
    return match.group(1) if match else ""


def _defines_function(program: str, name: str) -> bool:
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
    return False


def _exact(program: str, reference: str) -> bool:
    def norm(text: str) -> str:
        text = re.sub(r"#.*", "", text)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        return "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())

    return norm(program) == norm(reference)


def evaluate_cases(model, tokenizer, cases, *, budget: int = 96,
                   temperature: float = 0.0, seed: int | None = None,
                   device: str = "cpu") -> dict:
    """Evaluate the frozen cases with one checkpoint."""
    from serving.local_inference import generate_ids

    records = []
    for case in cases:
        started = time.time()
        result = generate_ids(
            model,
            tokenizer,
            case.prompt,
            max_new_tokens=budget,
            temperature=temperature,
            seed=seed if temperature and temperature > 0 else None,
            device=device,
        )
        text = result.get("text", "")
        program = case.prompt + text
        name = requested_function(case.prompt)
        syntax_valid = False
        try:
            ast.parse(program)
            syntax_valid = True
        except (SyntaxError, ValueError):
            syntax_valid = False
        tokens = result.get("token_ids") or []
        shingles = _shingles(text)
        reference_shingles = _shingles(case.reference)
        records.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "partition": case.partition,
                "syntax_valid": syntax_valid,
                "complete_function": _defines_function(program, name) if name else False,
                "exact_target": _exact(program, case.reference),
                "structural_similarity": round(_jaccard(shingles, reference_shingles), 4),
                "degenerate": _is_degenerate(text, tokens),
                "eos_terminated": result.get("finish_reason") == "stop",
                "truncated": result.get("finish_reason") == "length",
                "tokens": len(tokens),
                "latency_seconds": round(time.time() - started, 3),
            }
        )
    return {"records": records, "aggregate": aggregate_records(records)}


def _is_degenerate(text: str, tokens) -> bool:
    if not tokens:
        return True
    longest = 1
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    if longest >= 8:
        return True
    if len(text.strip().splitlines()) >= 2:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and len(set(lines)) / len(lines) < 0.5:
            return True
    return False


def aggregate_records(records) -> dict:
    """Aggregate case records into the reported metrics, with denominators."""
    total = len(records)
    if not total:
        return {"cases": 0}
    syntax = sum(1 for record in records if record["syntax_valid"])
    complete = sum(1 for record in records if record["complete_function"])
    exact = sum(1 for record in records if record["exact_target"])
    degenerate = sum(1 for record in records if record["degenerate"])
    eos = sum(1 for record in records if record["eos_terminated"])
    truncated = sum(1 for record in records if record["truncated"])
    similarity = [record["structural_similarity"] for record in records]
    per_family = defaultdict(lambda: {"total": 0, "syntax_valid": 0, "complete_function": 0})
    for record in records:
        entry = per_family[record["family"]]
        entry["total"] += 1
        entry["syntax_valid"] += int(record["syntax_valid"])
        entry["complete_function"] += int(record["complete_function"])
    return {
        "cases": total,
        "syntax_valid": syntax,
        "syntax_valid_rate": syntax / total,
        "complete_function": complete,
        "complete_function_rate": complete / total,
        "exact_target": exact,
        "exact_target_rate": exact / total,
        "degenerate": degenerate,
        "degenerate_rate": degenerate / total,
        "eos_terminated": eos,
        "truncated": truncated,
        "mean_structural_similarity": round(sum(similarity) / total, 4),
        "mean_tokens": round(
            sum(record["tokens"] for record in records) / total, 2
        ),
        "per_family": {name: dict(counts) for name, counts in sorted(per_family.items())},
    }


#: The outcome vocabulary of the experiment. Only measured evidence selects
#: one of these; a lower training loss never does.
CLASSIFICATIONS = (
    "SYNTHETIC_GENERALIZATION_DEMONSTRATED",
    "SYNTHETIC_PATTERN_LEARNING_ONLY",
    "LOSS_REDUCTION_WITHOUT_CODING_GAIN",
    "NO_MEASURABLE_LEARNING",
    "EXPERIMENT_BLOCKED",
)


def classify(baseline: dict, trained: dict, *, loss_delta: float | None = None) -> dict:
    """Classify the experiment from measured evidence only.

    Syntax validity alone is not a learning signal: a model can emit a
    parseable, function-shaped but degenerate completion. Generalisation is
    therefore only claimed when the held-out syntax rate improves *and* the
    degenerate rate falls *and* the outputs move toward the targets (exact
    agreement or structural similarity). Otherwise the run is reported as
    pattern learning, which is what it is.
    """

    def rate(block, key, partition):
        return block.get(partition, {}).get(key, 0.0)

    def mean(block, key):
        values = [
            block.get(partition, {}).get(key, 0.0)
            for partition in ("A", "B", "C")
            if block.get(partition)
        ]
        return sum(values) / len(values) if values else 0.0

    held_out_gain = 0.0
    for partition in ("B", "C"):
        held_out_gain += rate(trained, "syntax_valid_rate", partition) - rate(
            baseline, "syntax_valid_rate", partition
        )
    in_distribution_gain = rate(
        trained, "syntax_valid_rate", "A"
    ) - rate(baseline, "syntax_valid_rate", "A")
    degenerate_change = rate(trained, "degenerate_rate", "A") - rate(
        baseline, "degenerate_rate", "A"
    )
    exact_gain = mean(trained, "exact_target_rate") - mean(baseline, "exact_target_rate")
    similarity_gain = mean(
        trained, "mean_structural_similarity"
    ) - mean(baseline, "mean_structural_similarity")
    loss_fell = bool(loss_delta is not None and loss_delta < 0)

    if (
        held_out_gain > 0
        and degenerate_change < 0
        and (exact_gain > 0 or similarity_gain > 0.05)
    ):
        verdict = "SYNTHETIC_GENERALIZATION_DEMONSTRATED"
        reason = (
            f"held-out syntax validity improved ({held_out_gain:+.3f}), the "
            f"degenerate rate fell ({degenerate_change:+.3f}) and the outputs "
            f"moved toward the targets (exact {exact_gain:+.3f}, structural "
            f"{similarity_gain:+.3f})"
        )
    elif in_distribution_gain > 0 or held_out_gain > 0:
        verdict = "SYNTHETIC_PATTERN_LEARNING_ONLY"
        reason = (
            f"completion shape improved (in-distribution "
            f"{in_distribution_gain:+.3f}, held-out {held_out_gain:+.3f}) but the "
            f"output did not become semantically correct (exact {exact_gain:+.3f}, "
            f"structural {similarity_gain:+.3f})"
            + (
                f" and degeneracy rose ({degenerate_change:+.3f})"
                if degenerate_change >= 0
                else ""
            )
        )
    elif loss_fell:
        verdict = "LOSS_REDUCTION_WITHOUT_CODING_GAIN"
        reason = "training loss fell but no completion-quality metric improved"
    else:
        verdict = "NO_MEASURABLE_LEARNING"
        reason = "neither loss nor completion quality improved measurably"

    return {
        "verdict": verdict,
        "reason": reason,
        "in_distribution_gain": round(in_distribution_gain, 4),
        "held_out_gain": round(held_out_gain, 4),
        "degenerate_change": round(degenerate_change, 4),
        "exact_target_gain": round(exact_gain, 4),
        "structural_similarity_gain": round(similarity_gain, 4),
        "loss_delta": loss_delta,
        "note": (
            "This classification describes behaviour on a synthetic "
            "curriculum. It is not a claim about Python programming ability, "
            "and the experimental checkpoint is not promoted."
        ),
    }
