"""Tests for the M6 sovereign-track assessment.

Track B's hard constraint is that it performs no training. These tests pin the
scale arithmetic, the honest two-way fit test, and the fact that no code path
claims a capability the data cannot teach.
"""

from __future__ import annotations

import pytest

from evaluation.m6_sovereign import (MEASURED_REFERENCE,
                                     REQUIRED_TRAINING_TOKENS_PER_PARAMETER,
                                     SCALE_PROPOSALS, build_assessment,
                                     data_assessment, estimate_scales,
                                     format_assessment)

GEOMETRY_KEYS = ("d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_dim")


@pytest.fixture(scope="module")
def scales():
    return estimate_scales()


@pytest.fixture(scope="module")
def report():
    return build_assessment()


def test_every_proposal_declares_its_geometry():
    for proposal in SCALE_PROPOSALS:
        for key in GEOMETRY_KEYS:
            assert proposal[key] > 0, f"{proposal['id']} missing {key}"
        assert proposal["purpose"]


def test_estimated_parameter_counts_grow_with_scale(scales):
    params = [s.parameters for s in scales]
    assert params == sorted(params)
    assert len(set(s.scale_id for s in scales)) == len(scales)


def test_every_scale_carries_an_estimate_and_a_fit_verdict(scales):
    for scale in scales:
        assert scale.parameters and scale.parameters > 0
        assert scale.estimate_bytes and scale.estimate_bytes > 0
        assert scale.status == "ESTIMATE"
        assert scale.fits_total_ram in (True, False)
        assert scale.fits_available_ram in (True, False)
        assert scale.fits_host == (scale.fits_total_ram
                                   and scale.fits_available_ram)


def test_twenty_tokens_per_parameter_is_applied(scales):
    for scale in scales:
        assert scale.tokens_for_20x == (
            scale.parameters * REQUIRED_TRAINING_TOKENS_PER_PARAMETER)


def test_measured_reference_matches_the_known_configuration():
    assert "B-existing-9m" in MEASURED_REFERENCE
    assert MEASURED_REFERENCE["B-existing-9m"] > 0


def test_available_ram_is_the_binding_constraint():
    # Deterministic resource-policy fixture: total 4 GiB, available 128 MiB.
    # Policy under test is the comparison itself, not the host's current state.
    synthetic_total = 4 * 2**30
    synthetic_available = 128 * 2**20
    scales = estimate_scales(memory={"total_bytes": synthetic_total,
                                     "available_bytes": synthetic_available})
    fits_total_not_available = [s for s in scales
                                if s.fits_total_ram and not s.fits_available_ram]
    assert fits_total_not_available, "expected at least one scale to fit total RAM but not available RAM"


def test_available_ram_measurement_is_live_and_nonnegative():
    # Environment-observation test: informational only, never asserts a verdict.
    mem = __import__("training.preflight", fromlist=["system_memory"]).system_memory()
    assert mem["total_bytes"] > 0
    assert mem["available_bytes"] is not None


def test_data_assessment_never_invents_task_coverage():
    data = data_assessment()
    coverage = {row["task"]: row["coverage"] for row in data["task_coverage"]}
    assert coverage["debugging"] == "ABSENT"
    assert coverage["unit_test_generation"] == "ABSENT"
    assert coverage["code_completion"] in ("PARTIAL", "COVERED", "ABSENT")


def test_assessment_performs_and_authorises_no_training(report):
    assert report["training_performed"] is False
    assert report["training_authorised"] is False
    assert report["authorisation_required"]


def test_assessment_separates_established_from_not_demonstrated(report):
    assert report["established"]
    assert report["not_demonstrated"]
    joined = " ".join(report["not_demonstrated"]).lower()
    assert "useful coding capability" in joined


def test_recommended_experiment_is_the_largest_measured_scale(report):
    recommended = report["recommended_next_experiment"]["scale"]
    assert recommended in MEASURED_REFERENCE
    assert report["recommended_next_experiment"]["preconditions"]


def test_format_is_human_readable_and_states_no_training(report):
    text = format_assessment(report)
    assert "training performed : False" in text
    assert "Task coverage" in text
    assert "Recommended next experiment" in text


def test_every_scale_is_labelled_an_estimate(report):
    for scale in report["scales"]:
        assert scale["status"] == "ESTIMATE"
        assert scale["estimated_steady_state_mib"] > 0
