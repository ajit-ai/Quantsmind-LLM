"""Tests for M6 resource preflight.

The module exists to refuse unsafe loads, so the tests check that it refuses,
that its KV-cache arithmetic is dimensionally right, and that coarse estimates
stay in the right order of magnitude instead of being off by 100x.
"""

from __future__ import annotations

from evaluation.m6_resources import (GIB, MIB, kv_cache_bytes, machine_facts,
                                     plan_for)


def test_kv_cache_matches_exact_geometry():
    size, basis, confidence = kv_cache_bytes(
        3_800_000_000, 512, layers=32, kv_heads=8, head_dim=128)
    per_token = 2 * 32 * 8 * 128 * 2
    assert size == per_token * 512
    assert confidence == "MEASURED_GEOMETRY"
    assert "32 layers" in basis
    assert "8 kv-heads" in basis


def test_grouped_query_attention_is_smaller_than_multi_head():
    gqa, _, _ = kv_cache_bytes(7_000_000_000, 1024, layers=28, kv_heads=4,
                               head_dim=128)
    mha, _, _ = kv_cache_bytes(7_000_000_000, 1024, layers=28, kv_heads=28,
                               head_dim=128)
    assert gqa * 7 == mha


def test_coarse_estimate_is_within_an_order_of_magnitude():
    # Phi-3-mini: 3.8B params, 32 layers, 32 kv-heads, head-dim 96.
    exact = 2 * 32 * 32 * 96 * 2 * 512
    coarse, basis, confidence = kv_cache_bytes(3_800_000_000, 512)
    assert confidence == "COARSE"
    assert "assumed 32 layers" in basis
    # The old 12*L*H formula was wrong by orders of magnitude; this guards it.
    assert 0.5 * exact <= coarse <= 2.0 * exact


def test_coarse_estimate_does_not_explode_for_a_tiny_model():
    # 2.85M-parameter checkpoint: hidden size is small, so the KV cache must be
    # small too. A 12*L*H mistake produced gigabytes here.
    size, _, _ = kv_cache_bytes(2_853_280, 512)
    assert 0 < size < 64 * MIB


def test_context_is_floored_at_one_token():
    size, _, _ = kv_cache_bytes(1_000_000, 0, layers=2, kv_heads=2,
                                head_dim=8)
    assert size == 2 * 2 * 2 * 8 * 2 * 1


def test_parameter_estimate_uses_bytes_per_dtype():
    plan = plan_for("x", 1_000_000, dtype="float32", context=1,
                    layers=1, kv_heads=1, head_dim=1)
    weights = plan.requirements[0]
    assert weights.component == "weights"
    assert weights.bytes == 4_000_000
    assert not weights.measured
    assert plan.estimate_confidence == "MEASURED_GEOMETRY"


def test_measured_artifact_is_marked_measured():
    plan = plan_for("x", None, weights_bytes=10 * MIB, artifact_bytes=10 * MIB)
    assert plan.requirements[0].measured is True
    assert "measured artifact size" in plan.requirements[0].basis


def test_disk_requirement_includes_headroom():
    plan = plan_for("x", 1_000_000, artifact_bytes=100 * MIB)
    assert plan.required_disk_bytes == 100 * MIB + 2 * GIB


def test_plan_refuses_ram_it_does_not_have():
    plan = plan_for("huge", 50_000_000_000, dtype="float32")
    plan.available_ram_bytes = 1 * GIB
    ok, reason = plan.decision()
    assert ok is False
    assert "insufficient RAM" in reason
    assert plan.ok is False


def test_plan_accepts_what_fits():
    plan = plan_for("small", 1_000_000, dtype="float16")
    plan.available_ram_bytes = 8 * GIB
    plan.available_disk_bytes = 100 * GIB
    ok, reason = plan.decision()
    assert ok is True
    assert "ESTIMATE" in reason


def test_missing_ram_measurement_does_not_block():
    plan = plan_for("unknown", 1_000_000)
    plan.available_ram_bytes = None
    assert plan.fits_ram is True


def test_serialised_plan_is_labelled_estimate():
    payload = plan_for("x", 1_000_000).to_dict()
    assert payload["estimate_kind"] == "ESTIMATE"
    assert payload["required_ram_bytes"] > 0
    assert all("basis" in requirement for requirement in payload["requirements"])


def test_machine_facts_are_marked_measured_and_positive():
    facts = machine_facts()
    assert facts["measured"] is True
    assert facts["total_ram_gib"] > 0
    assert facts["disk_free_gib"] > 0
