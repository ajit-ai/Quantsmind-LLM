"""Bounded decoding diagnostics (Taksh v1.8.0).

The v1.7.0 budget study showed the M7 checkpoint filling every budget with
degenerate repetition. This harness compares decoding strategies under a
fixed checkpoint, tokenizer, prompt set and token budget, so that only the
decoding strategy varies.

Only strategies the existing sampler actually supports are compared.
Repetition penalty is **not** supported by the shared sampler
(:func:`serving.local_inference.generate_ids`); it is reported as unavailable
rather than silently emulated with a new serving behaviour.

Diversity is reported alongside correctness on purpose: a strategy that
produces more varied text but less correct code is not an improvement.
"""

from __future__ import annotations

import time

from evaluation.python_eval.budget import (
    classify_completion,
    classify_termination,
    repetition_metrics,
    repeated_line_ratio,
)
from evaluation.python_eval.syntax import validate_syntax

#: name -> sampler kwargs. ``repetition_penalty`` is intentionally absent.
STRATEGIES = {
    "greedy": {"temperature": 0.0},
    "temperature_0.8": {"temperature": 0.8, "top_k": 0, "top_p": 1.0},
    "top_k_50": {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
    "top_p_0.95": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
}

UNSUPPORTED_STRATEGIES = {
    "repetition_penalty": (
        "the shared sampler has no repetition-penalty parameter; adding one "
        "would change serving behaviour and is out of scope for this milestone"
    ),
}


def available_strategies() -> dict:
    return dict(STRATEGIES)


def run_strategy(model, tokenizer, cases, *, strategy: str, budget: int,
                 seed: int, device: str = "cpu", max_new_tokens: int | None = None) -> dict:
    """Run one decoding strategy over the case set and aggregate the results."""
    from serving.local_inference import generate_ids

    if strategy not in STRATEGIES:
        raise ValueError(
            f"unsupported strategy {strategy!r}; available: {sorted(STRATEGIES)}"
        )
    kwargs = dict(STRATEGIES[strategy])
    records = []
    started = time.time()
    for case in cases:
        case_started = time.time()
        result = generate_ids(
            model,
            tokenizer,
            case["prompt"],
            max_new_tokens=max_new_tokens or budget,
            device=device,
            seed=seed if kwargs.get("temperature", 0) > 0 else None,
            **kwargs,
        )
        text = result.get("text", "")
        ids = result.get("token_ids") or []
        program = case["prompt"] + text
        parseable, _, _ = validate_syntax(program, case.get("language", "python"))
        complete = False
        if parseable:
            try:
                compile(program, "<completion>", "exec")
                complete = True
            except (SyntaxError, ValueError):
                complete = False
        repetition = repetition_metrics(ids)
        repetition["repeated_line_ratio"] = repeated_line_ratio(text)
        records.append(
            {
                "case_id": case.get("case_id"),
                "syntax_valid": bool(parseable),
                "complete_program": complete,
                "quality": classify_completion(
                    text=text, token_ids=ids, parseable=bool(parseable),
                    complete_program=complete,
                ),
                "termination": classify_termination(
                    finish_reason=result.get("finish_reason"),
                    generated_tokens=len(ids),
                    budget=max_new_tokens or budget,
                    empty=not text.strip(),
                ),
                "tokens_generated": len(ids),
                "repetition": repetition,
                "latency_seconds": round(time.time() - case_started, 3),
            }
        )
    syntax_valid = sum(1 for record in records if record["syntax_valid"])
    complete = sum(1 for record in records if record["complete_program"])
    degenerate = sum(
        1 for record in records if record["quality"] == "degenerate_repetition"
    )
    diversity = [
        record["repetition"]["unique_token_ratio"] for record in records
    ]
    return {
        "strategy": strategy,
        "parameters": kwargs,
        "seed": seed if kwargs.get("temperature", 0) > 0 else None,
        "budget": budget,
        "cases": len(records),
        "syntax_valid": syntax_valid,
        "complete_program": complete,
        "degenerate_repetition": degenerate,
        "mean_unique_token_ratio": (
            round(sum(diversity) / len(diversity), 4) if diversity else None
        ),
        "mean_latency_seconds": round(
            sum(record["latency_seconds"] for record in records) / max(1, len(records)), 3
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "records": records,
    }
