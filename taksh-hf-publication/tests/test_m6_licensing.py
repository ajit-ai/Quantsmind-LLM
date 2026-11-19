"""Tests for the M6 licensing registry.

A licence that was not fetched cannot be "assumed". These tests pin the
separation between a verified entry and an unresolved one, and check that the
registry keeps the evidence needed to repeat a verification.
"""

from __future__ import annotations

from evaluation.m6_licensing import REGISTRY, SOVEREIGN, lookup, unresolved_licensing

REQUIRED_FIELDS = ("publisher", "license_id", "license_verified",
                   "license_source", "license_evidence", "verified_on",
                   "task_head")


def test_every_registry_entry_carries_its_evidence():
    for model_id, record in REGISTRY.items():
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        assert not missing, f"{model_id} missing {missing}"
        assert record["license_source"].startswith("https://huggingface.co/")
        assert record["license_id"]


def test_every_watchlist_candidate_is_registered():
    from evaluation.m6_inventory import WATCHLIST
    for model_id in WATCHLIST:
        assert model_id in REGISTRY, f"{model_id} has no licence record"


def test_all_registered_licences_are_verified():
    assert unresolved_licensing() == []


def test_lookup_is_exact_not_fuzzy():
    assert lookup("microsoft/Phi-3-mini-4k-instruct")["license_id"] == "MIT"
    assert lookup("microsoft/phi-3-mini-4k-instruct") is None
    assert lookup("unknown/model") is None


def test_licence_and_task_head_are_separate_claims():
    bert = REGISTRY["google-bert/bert-base-uncased"]
    assert bert["license_id"] == "Apache-2.0"
    assert bert["task_head"] == "masked_lm"
    assert "cannot accept a free-form instruction" in bert["prompt_template"]


def test_phi3_geometry_is_recorded_for_kv_estimation():
    phi = REGISTRY["microsoft/Phi-3-mini-4k-instruct"]
    assert phi["layers"] > 0
    assert phi["kv_heads"] > 0
    assert phi["head_dim"] > 0
    assert phi["context"] > 0


def test_sovereign_checkpoints_are_project_owned():
    assert SOVEREIGN
    for path, note in SOVEREIGN.items():
        assert path.endswith(".pt")
        assert "quantsmind-owned" in note


def test_unresolved_licensing_reports_only_unverified_entries(monkeypatch):
    monkeypatch.setitem(REGISTRY, "test/unverified",
                        {"license_id": None, "license_verified": False})
    assert "test/unverified" in unresolved_licensing()
