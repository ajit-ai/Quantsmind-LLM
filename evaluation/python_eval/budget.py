"""Generation-budget validation, termination and repetition diagnostics.

The v1.6.0 baseline showed that every completion in the corrected suite hit
the 64-token generation limit, so the measurement was budget-limited rather
than capability-limited. This module makes that dimension explicit and
measurable:

* a budget is validated against the model's context window *before* the
  request is issued, and an invalid combination is rejected rather than
  silently changed;
* the reason generation stopped is classified (end-of-document, budget,
  context, empty);
* repetition is measured with several transparent indicators rather than a
  single arbitrary heuristic;
* completion quality separates "parses" from "is a complete program" from
  "is degenerate repetition".

Nothing here executes generated code.
"""

from __future__ import annotations

from collections import Counter

VALID_BUDGETS = (32, 64, 128, 256)


class BudgetError(ValueError):
    """The requested budget cannot be honoured."""


def validate_budget(prompt_tokens: int, budget: int, context_length: int) -> dict:
    """Check that a generation budget fits the context window.

    Raises :class:`BudgetError` for an impossible combination; the caller
    must not silently substitute a different budget.
    """
    if budget <= 0:
        raise BudgetError(f"budget must be positive, got {budget}")
    if prompt_tokens < 0:
        raise BudgetError(f"prompt_tokens must be >= 0, got {prompt_tokens}")
    if prompt_tokens >= context_length:
        raise BudgetError(
            f"prompt of {prompt_tokens} tokens does not fit the context "
            f"window of {context_length}"
        )
    if prompt_tokens + budget > context_length:
        raise BudgetError(
            f"prompt {prompt_tokens} + budget {budget} exceeds the context "
            f"window of {context_length}"
        )
    return {
        "prompt_tokens": prompt_tokens,
        "budget": budget,
        "context_length": context_length,
        "remaining_context": context_length - prompt_tokens,
    }


def classify_termination(*, finish_reason: str | None, generated_tokens: int,
                         budget: int, empty: bool) -> str:
    """Classify why generation stopped."""
    if empty or generated_tokens == 0:
        return "empty"
    if finish_reason == "stop":
        return "eos"
    if finish_reason == "context":
        return "context"
    if generated_tokens >= budget:
        return "budget"
    return "other"


def repetition_metrics(token_ids) -> dict:
    """Transparent repetition indicators for a generated token sequence."""
    ids = list(token_ids)
    if not ids:
        return {
            "tokens": 0,
            "unique_token_ratio": 0.0,
            "max_repeated_run": 0,
            "repeated_line_ratio": 0.0,
            "repeated_bigram_ratio": 0.0,
        }
    counts = Counter(ids)
    runs = 1
    longest = 1
    for previous, current in zip(ids, ids[1:]):
        runs = runs + 1 if current == previous else 1
        longest = max(longest, runs)
    bigrams = list(zip(ids, ids[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(count - 1 for count in bigram_counts.values())
    return {
        "tokens": len(ids),
        "unique_token_ratio": len(counts) / len(ids),
        "max_repeated_run": longest,
        "repeated_bigram_ratio": (repeated_bigrams / len(bigrams)) if bigrams else 0.0,
    }


def repeated_line_ratio(text: str) -> float:
    """Share of non-empty lines that are exact repeats of an earlier line."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values())
    return repeated / len(lines)


def classify_completion(*, text: str, token_ids, parseable: bool,
                         complete_program: bool) -> str:
    """Classify a completion's quality.

    ``degenerate_repetition`` is reported separately from "parses" so a
    repetition that happens to be valid Python is not counted as useful
    capability.
    """
    metrics = repetition_metrics(token_ids)
    lines = repeated_line_ratio(text)
    degenerate = (
        metrics["max_repeated_run"] >= 8
        or metrics["repeated_bigram_ratio"] >= 0.5
        or lines >= 0.5
    )
    if degenerate:
        return "degenerate_repetition"
    if complete_program:
        return "complete_program"
    if parseable:
        return "parseable"
    return "invalid"
