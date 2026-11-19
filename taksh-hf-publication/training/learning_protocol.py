"""Root-cause register and measurable learning protocol (Taksh v1.8.0).

Two things live here:

* the **hypothesis register** for the repetition problem, with the evidence
  that supports or fails to support each hypothesis;
* the **learning protocol**: the evaluation dimensions, the experimental
  controls, the success criteria and the stop conditions that a future
  authorized run must satisfy, so that a lower training loss cannot be
  mistaken for learning.

Nothing here executes a training run. The protocol's status is
``NOT_AUTHORIZED_FOR_EXECUTION``.
"""

from __future__ import annotations

EXPERIMENT_STATUS = "NOT_AUTHORIZED_FOR_EXECUTION"

#: The repetition hypotheses, each with the evidence class the current
#: measurements support. Verdicts are updated by evidence, not by opinion.
HYPOTHESES = (
    {
        "hypothesis": "insufficient training data diversity",
        "verdict": "PARTIALLY_SUPPORTED",
        "evidence": (
            "the M7 corpus is 256 records drawn from a handful of upstream "
            "repositories; the candidate metadata shows narrow repository "
            "coverage, and the evaluation shows 8/8 completions degenerate"
        ),
    },
    {
        "hypothesis": "synthetic-family overfitting",
        "verdict": "PARTIALLY_SUPPORTED",
        "evidence": (
            "the v1.5.0 synthetic diagnostic reached 30.3% next-token accuracy "
            "on its own validation split but 0/16 on held-out template families"
        ),
    },
    {
        "hypothesis": "tokenizer fragmentation",
        "verdict": "PARTIALLY_SUPPORTED",
        "evidence": (
            "2.85-2.97 characters per token and 32-43% identifier "
            "fragmentation inflate prompts and consume context, but they do "
            "not explain a collapse into repeated tokens on its own"
        ),
    },
    {
        "hypothesis": "model capacity limitation",
        "verdict": "UNTESTED",
        "evidence": (
            "no larger model has been trained or evaluated with the same "
            "protocol, so capacity cannot be separated from data"
        ),
    },
    {
        "hypothesis": "greedy decoding collapse",
        "verdict": "SUPPORTED",
        "evidence": (
            "greedy decoding is 8/8 degenerate with a unique-token ratio of "
            "0.20; sampled decoding is 0/8 degenerate with ratios of "
            "0.67-0.78, so the collapse lives in the model's distribution "
            "rather than in the decoder alone"
        ),
    },
    {
        "hypothesis": "EOS or termination behaviour",
        "verdict": "NOT_SUPPORTED_BY_CURRENT_EVIDENCE",
        "evidence": (
            "every completion terminated on the budget rather than on EOS, and "
            "the EOS id resolves correctly; the terminator is not the cause"
        ),
    },
    {
        "hypothesis": "repetition in the training examples",
        "verdict": "UNTESTED",
        "evidence": (
            "requires inspecting the content of the historical corpus, which "
            "is not authorized for source-content review"
        ),
    },
    {
        "hypothesis": "training objective or checkpoint quality",
        "verdict": "PARTIALLY_SUPPORTED",
        "evidence": (
            "the objective, masks and resume behaviour are verified by test, "
            "but the historical run's own metrics were not recorded in a form "
            "that can be re-checked"
        ),
    },
    {
        "hypothesis": "prompt distribution mismatch",
        "verdict": "PARTIALLY_SUPPORTED",
        "evidence": (
            "the corrected measurement uses code prefixes, and the model still "
            "repeats; a base model fed prose will repeat prose"
        ),
    },
    {
        "hypothesis": "evaluation task difficulty",
        "verdict": "NOT_SUPPORTED_BY_CURRENT_EVIDENCE",
        "evidence": (
            "the same eight cases give 1/8 for greedy and 0/8 for every "
            "sampled strategy, so the tasks are not the discriminating "
            "factor"
        ),
    },
)

#: The evaluation dimensions a future run must report.
EVALUATION_DIMENSIONS = (
    "next_token_accuracy",
    "validation_loss",
    "syntax_valid_rate",
    "complete_program_rate",
    "degenerate_repetition_rate",
    "held_out_family_rate",
    "novel_identifier_handling",
    "unseen_composition_generalisation",
    "memorisation_indicators",
    "context_limit_failures",
    "eos_behaviour",
)

#: The controls that make two runs comparable.
EXPERIMENTAL_CONTROLS = (
    "same_checkpoint_family",
    "same_tokenizer",
    "same_frozen_evaluation_cases",
    "same_generation_configuration",
    "same_evaluation_implementation",
    "explicit_train_validation_test_separation",
    "fixed_seeds",
    "recorded_artifact_hashes",
    "independent_held_out_task_families",
)

#: Success criteria. Every one of them must hold; a lower training loss alone
#: is explicitly not sufficient.
SUCCESS_CRITERIA = (
    "held_out_completion_rate_improves_over_the_baseline_checkpoint",
    "degenerate_repetition_rate_decreases_against_the_baseline",
    "no_evaluation_case_appears_in_the_training_corpus",
    "held_out_family_rate_improves_without_held_out_family_memorisation",
    "the_run_reproduces_with_the_same_seed",
)

#: Stop or review conditions.
STOP_CONDITIONS = (
    "validation_loss_divergence",
    "train_validation_gap_above_half_a_nat",
    "more_than_half_the_completions_degenerate",
    "no_held_out_improvement",
    "memorisation_of_held_out_families",
    "contamination_detected",
    "resource_exhaustion",
    "checkpoint_corruption",
    "repeated_output_collapse",
)


def build_protocol(*, baseline: dict | None = None) -> dict:
    """The frozen protocol a future authorized run must follow."""
    return {
        "schema": "quantsmind.learning_protocol/1",
        "status": EXPERIMENT_STATUS,
        "evaluation_dimensions": list(EVALUATION_DIMENSIONS),
        "experimental_controls": list(EXPERIMENTAL_CONTROLS),
        "success_criteria": list(SUCCESS_CRITERIA),
        "stop_conditions": list(STOP_CONDITIONS),
        "baseline": baseline or {},
        "statistical_limitations": (
            "the eight-case Track A suite is a smoke test: it can show a total "
            "failure but cannot support a capability claim, so a larger, "
            "independently controlled suite with held-out task families is "
            "required before any improvement is claimed"
        ),
        "authorization_gates": [
            "source rights approved",
            "provenance verified",
            "contamination controls pass",
            "corpus revision frozen",
            "compute feasibility accepted",
            "evaluation protocol frozen",
            "the owner explicitly authorizes the specific training run",
        ],
        "note": (
            "Generating this protocol authorizes nothing. The real-data "
            "experiment remains NOT_AUTHORIZED_FOR_EXECUTION until every gate "
            "above is satisfied."
        ),
    }


def hypothesis_register() -> list:
    """The hypothesis table with its evidence classes."""
    return [dict(entry) for entry in HYPOTHESES]


def verdict_counts() -> dict:
    counts: dict = {}
    for entry in HYPOTHESES:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    return counts
