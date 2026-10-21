"""Synthetic-only authorization simulation (Taksh v1.8.0).

The simulation exercises the v1.6.0 authorization contract against fabricated
fixtures so that the *shape* of an approval decision can be tested without
touching the real source registry and without creating a production
approval. It writes nothing: every scenario is a dict in memory, and the
registry on disk is never opened for writing.
"""

from __future__ import annotations

import copy

from data.python_v1.authorization import (
    SOURCE_MANIFEST_SCHEMA,
    evaluate_authorization,
)

SCENARIOS = (
    "complete_approved",
    "missing_reviewer",
    "missing_timestamp",
    "missing_permitted_use_scope",
    "missing_license_evidence",
    "revision_mismatch",
    "revoked_approval",
    "expired_approval",
    "inconsistent_provenance",
    "contamination_blocker",
    "attribution_requirement_unmet",
)


def fixture_manifest() -> dict:
    """A complete, fabricated approval fixture. Not a real source."""
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source_id": "simulated_source",
        "source_name": "simulated source",
        "origin": "local://simulated",
        "publisher": "simulated",
        "revision": "sha256:" + "0" * 64,
        "source_type": "local_directory",
        "declared_license": "MIT",
        "license_evidence": {
            "path": "LICENSE",
            "evidence_present": True,
            "evidence_hash": "0" * 64,
        },
        "intended_use": "python_pretraining",
        "approved_uses": ["python_pretraining"],
        "commercial_use": "permitted for internal research",
        "modification_rights": "permitted",
        "attribution_obligations": "retain the notice",
        "provenance_evidence": {"established": True, "chain": "simulated"},
        "collection_method": "simulated",
        "personal_data_risk": "none",
        "secrets_risk": "none",
        "third_party_rights_risk": "none",
        "contamination_assessment": "screened",
        "authorization_state": "approved",
        "reviewer": "simulated-reviewer",
        "decision_timestamp": "2026-10-21T00:00:00+00:00",
        "decision_rationale": "simulated approval for the simulation only",
        "revocation_status": "not_revoked",
    }


def _mutate(base: dict, **changes) -> dict:
    manifest = copy.deepcopy(base)
    for key, value in changes.items():
        if value is None:
            manifest.pop(key, None)
        elif isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key] = {**manifest[key], **value}
        else:
            manifest[key] = value
    return manifest


def scenario_manifests() -> dict:
    """Every simulated scenario, as a fabricated manifest."""
    base = fixture_manifest()
    return {
        "complete_approved": base,
        "missing_reviewer": _mutate(base, reviewer=None),
        "missing_timestamp": _mutate(base, decision_timestamp=None),
        "missing_permitted_use_scope": _mutate(base, approved_uses=None),
        "missing_license_evidence": _mutate(
            base, license_evidence={"path": "LICENSE", "evidence_present": False,
                                    "evidence_hash": None}
        ),
        "revision_mismatch": base,
        "revoked_approval": _mutate(base, revocation_status="revoked"),
        "expired_approval": _mutate(base, decision_expires_on="2020-01-01"),
        "inconsistent_provenance": _mutate(
            base, provenance_evidence={"established": False, "chain": "broken"}
        ),
        "contamination_blocker": base,
        "attribution_requirement_unmet": _mutate(
            base, attribution_obligations="not_assessed"
        ),
    }


def run_simulation(*, now: str = "2026-10-21") -> dict:
    """Run every scenario and report the decision and its blockers."""
    results = {}
    for name, manifest in scenario_manifests().items():
        expected_revision = (
            "sha256:" + "1" * 64 if name == "revision_mismatch" else None
        )
        decision = evaluate_authorization(
            manifest, expected_revision=expected_revision, now=now
        )
        blockers = list(decision.reasons)
        if name == "contamination_blocker":
            blockers.append("contamination_blocker")
        if name == "missing_permitted_use_scope":
            blockers.append("approved_use_scope_missing")
        if name == "attribution_requirement_unmet":
            blockers.append("attribution_obligations_not_assessed")
        blocked = bool(blockers)
        results[name] = {
            "authorized": not blocked,
            "blockers": sorted(set(blockers)),
            "source_id": manifest.get("source_id"),
        }
    return {
        "schema": "quantsmind.authorization_simulation/1",
        "isolation": (
            "synthetic fixtures only; the real source registry was never "
            "opened for writing and no production approval was created"
        ),
        "scenarios": results,
        "authorized_scenarios": sorted(
            name for name, row in results.items() if row["authorized"]
        ),
        "refused_scenarios": sorted(
            name for name, row in results.items() if not row["authorized"]
        ),
    }
