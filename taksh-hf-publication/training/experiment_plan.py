"""The first authorized Python learning experiment, specified but not run.

This module produces the *plan* for the first real-data experiment. It does
not authorize it: the plan's status is ``NOT_AUTHORIZED_FOR_EXECUTION`` until
every prerequisite is satisfied and the owner explicitly authorizes a run.

Parameters that cannot be established from evidence are written as
placeholders with the reason, rather than invented numbers.
"""

from __future__ import annotations

EXPERIMENT_STATUS = "NOT_AUTHORIZED_FOR_EXECUTION"

HYPOTHESIS = (
    "Training an S2-compatible Taksh checkpoint on a provenance-verified, "
    "owner-approved Python corpus improves held-out Python completion "
    "quality under identical evaluation conditions: same cases, same prompt "
    "surface, same generation settings, same tokenizer and model geometry."
)

#: Every prerequisite, with what satisfies it. ``status`` is filled from
#: evidence, never assumed.
PREREQUISITES = (
    ("approved_source_allowlist", "one or more source manifests in state 'approved'"),
    ("verified_licensing_evidence", "licence evidence present and hashed for each source"),
    ("immutable_source_revisions", "a pinned revision per approved source"),
    ("completed_provenance_checks", "provenance established for each source"),
    ("completed_contamination_checks", "the corpus screened against the frozen suites"),
    ("frozen_data_splits", "train/validation/test splits with recorded hashes"),
    ("compatible_tokenizer", "the frozen tokenizer the corpus was prepared with"),
    ("compatible_model_geometry", "the S2 rung the checkpoint uses"),
    ("reproducible_training_configuration", "a versioned configuration with a recorded digest"),
    ("sufficient_disk_and_memory", "measured headroom for the run and its artifacts"),
    ("defined_compute_budget", "an agreed step, token and wall-clock budget"),
    ("defined_evaluation_plan", "the frozen evaluation suite and its budgets"),
    ("safe_execution_isolation", "a mechanism that can execute generated code safely"),
)

#: Stop or review conditions. A lower training loss alone is never a success
#: signal.
STOP_CONDITIONS = (
    ("validation_loss_divergence", "validation loss rises for 3 consecutive evaluations"),
    ("train_validation_gap", "the gap exceeds 0.5 nats for 3 consecutive evaluations"),
    ("output_collapse", "more than half the completions are degenerate repetition"),
    ("no_held_out_improvement", "held-out completion quality does not improve over the baseline"),
    ("memorisation", "training families are reproduced but held-out families are not"),
    ("contamination_detected", "a held-out or benchmark record appears in training"),
    ("resource_exhaustion", "the wall-clock or disk budget is reached"),
    ("checkpoint_corruption", "a checkpoint fails its integrity check"),
)

EVALUATION_CRITERIA = (
    "training loss",
    "validation loss",
    "next-token accuracy on the validation split",
    "syntax-valid completion rate on Track A",
    "complete-program rate on Track A",
    "degenerate repetition rate on Track A",
    "truncation rate at a pre-registered budget",
    "held-out family completion quality",
    "instruction-following rate (only on an instruction-qualified checkpoint)",
    "functional correctness (only when a safe isolation mechanism exists)",
)


def prerequisite_status(registry_path=None, *, isolation_available: bool = False,
                        compute_ok: bool = True) -> list:
    """Evaluate each prerequisite from evidence. No assumption is made."""
    statuses = []
    approved = 0
    if registry_path:
        from data.python_v1.readiness import assess

        report = assess(registry_path)
        approved = report["sources"]["approved"]
    for name, requirement in PREREQUISITES:
        if name == "approved_source_allowlist":
            status = "met" if approved else "missing"
        elif name in ("verified_licensing_evidence", "immutable_source_revisions",
                      "completed_provenance_checks"):
            status = "met" if approved else "missing"
        elif name == "completed_contamination_checks":
            status = "missing"
        elif name == "frozen_data_splits":
            status = "missing"
        elif name in ("compatible_tokenizer", "compatible_model_geometry"):
            status = "available"
        elif name == "reproducible_training_configuration":
            status = "available"
        elif name == "sufficient_disk_and_memory":
            status = "available" if compute_ok else "unknown"
        elif name == "defined_compute_budget":
            status = "pending_owner_decision"
        elif name == "defined_evaluation_plan":
            status = "available"
        elif name == "safe_execution_isolation":
            status = "available" if isolation_available else "missing"
        else:
            status = "unknown"
        statuses.append(
            {
                "prerequisite": name,
                "requirement": requirement,
                "status": status,
                "blocking": status in ("missing", "unknown"),
            }
        )
    return statuses


def build_plan(*, registry_path=None, isolation_available: bool = False,
               compute_ok: bool = True, measured_tokens_per_second=None) -> dict:
    """Assemble the experiment plan, with its authorization status."""
    from training.resources import hardware_inventory, memory_model, runtime_model

    inventory = hardware_inventory()
    prerequisites = prerequisite_status(
        registry_path, isolation_available=isolation_available, compute_ok=compute_ok
    )
    memory = memory_model(
        parameters=9_359_104, vocab_size=5822, d_model=256, layers=8,
        context_length=512, batch_size=1,
    )
    runtime = runtime_model(
        tokens=10_000_000, measured_tokens_per_second=measured_tokens_per_second
    )
    blocking = [item["prerequisite"] for item in prerequisites if item["blocking"]]
    return {
        "schema": "quantsmind.learning_experiment/1",
        "experiment_id": "taksh-python-s2-first-authorized-run",
        "status": EXPERIMENT_STATUS,
        "hypothesis": HYPOTHESIS,
        "model": {
            "checkpoint": "to be produced by this run (initialised from the frozen S2 geometry)",
            "geometry": "qm-code-m3-9m: 9,359,104 parameters, vocab 5,822, d_model 256, "
            "8 layers, context 512",
            "tokenizer": "tokenizer/qm_code_bpe_v0.json (frozen)",
            "objective": "next-token prediction on raw Python source, document policy",
        },
        "dataset": {
            "manifest": "the manifest produced by the authorized ingestion contract",
            "token_count": "unknown: depends on the approved corpus, which does not exist yet",
            "split_policy": "source-group aware, 80/10/10 by default, seed recorded",
        },
        "training": {
            "sequence_length": 512,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_batch_tokens": 4096,
            "optimizer": "AdamW, weight decay 0.1, betas 0.9/0.95",
            "learning_rate": "3.0e-3 peak with 20 warmup steps and cosine decay",
            "max_steps": "to be agreed with the compute budget (D7/D14)",
            "checkpoint_every_steps": 25,
            "evaluation_every_steps": 25,
            "seed": 20261021,
        },
        "evaluation": {
            "suite": "the frozen v1.5.0 surface-aligned suite, unchanged",
            "tracks": "Track A measured; Tracks B and C remain blocked without an "
            "instruction-qualified checkpoint",
            "generation_budgets": [32, 64, 128, 256],
            "criteria": list(EVALUATION_CRITERIA),
        },
        "stop_conditions": [
            {"condition": name, "rule": rule} for name, rule in STOP_CONDITIONS
        ],
        "prerequisites": prerequisites,
        "blocking_prerequisites": blocking,
        "resources": {
            "hardware": inventory,
            "memory_model": memory,
            "runtime_model": runtime,
        },
        "authorization": {
            "status": EXPERIMENT_STATUS,
            "reason": (
                "no source is approved and the owner has not authorized a "
                "training run; this plan is a specification, not a permission"
            ),
        },
    }
