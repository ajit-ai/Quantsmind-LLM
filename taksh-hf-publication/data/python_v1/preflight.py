"""Training authorization preflight (Taksh v1.7.0).

A read-only gate that answers one question: *may a training run start with
this material?* It never approves anything, never mutates the registry, and
has no bypass flag. Its output is a structured decision with precise
blocker codes, so an owner can see exactly what is missing.

The preflight evaluates ten things:

1. candidate identity · 2. source manifest · 3. immutable revision ·
4. human approval evidence · 5. permitted training use · 6. commercial-use
scope · 7. attribution requirements · 8. provenance integrity ·
9. contamination status · 10. approved-source registry consistency
"""

from __future__ import annotations

import json
from pathlib import Path

from data.python_v1.authorization import (
    ELIGIBLE_STATE,
    evaluate_authorization,
    validate_source_manifest,
)

TRAINING_AUTHORIZED = "TRAINING_AUTHORIZED"
TRAINING_NOT_AUTHORIZED = "TRAINING_NOT_AUTHORIZED"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def preflight(
    manifest_path,
    *,
    intended_use: str = "python_pretraining",
    expected_revision: str | None = None,
    registry_path=None,
    contamination_status: str = "unknown",
    commercial_use_approved: bool | None = None,
    now: str | None = None,
) -> dict:
    """Evaluate whether training may start. Read-only; never approves."""
    checks: list = []
    blockers: list = []

    path = Path(manifest_path)
    if not path.is_file():
        checks.append(_check("source_manifest", False, f"missing: {path}"))
        return _decision(checks, blockers + ["source_manifest_missing"])

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.append(_check("source_manifest", False, str(exc)))
        return _decision(checks, blockers + ["source_manifest_unreadable"])

    problems = validate_source_manifest(manifest)
    checks.append(_check("source_manifest", not problems, "; ".join(problems)))
    if problems:
        blockers.append("source_manifest_malformed")

    identity = manifest.get("source_id")
    checks.append(_check("candidate_identity", bool(identity), str(identity)))
    if not identity:
        blockers.append("candidate_identity_missing")

    revision = manifest.get("revision")
    checks.append(
        _check(
            "immutable_revision",
            bool(revision) and not str(revision).lower().startswith("unknown"),
            str(revision),
        )
    )
    if not revision:
        blockers.append("immutable_revision_missing")

    decision = evaluate_authorization(
        manifest,
        expected_revision=expected_revision,
        intended_use=intended_use,
        now=now,
    )
    checks.append(
        _check("human_approval_evidence", decision.allowed, ", ".join(decision.reasons))
    )
    if not decision.allowed:
        blockers.extend(decision.reasons)

    if manifest.get("authorization_state") != ELIGIBLE_STATE:
        blockers.append("authorization_state_not_approved")
    if manifest.get("reviewer") in (None, "", "unknown"):
        blockers.append("reviewer_missing")
    if manifest.get("commercial_use") in ("not_assessed", "unknown", None):
        blockers.append("commercial_use_not_assessed")
    elif commercial_use_approved is False:
        blockers.append("commercial_use_out_of_scope")
    if manifest.get("attribution_obligations") in ("not_assessed", "unknown", None):
        blockers.append("attribution_obligations_not_assessed")

    provenance = manifest.get("provenance_evidence") or {}
    checks.append(
        _check("provenance_integrity", bool(provenance.get("established")),
               str(provenance.get("chain")))
    )
    if not provenance.get("established"):
        blockers.append("provenance_not_established")

    checks.append(_check("contamination_status", contamination_status != "unknown",
                          str(contamination_status)))
    if contamination_status in ("unknown", "contaminated"):
        blockers.append(f"contamination_status_{contamination_status}")

    registry_ok = True
    registry_detail = "no registry supplied"
    if registry_path:
        registry_file = Path(registry_path)
        if not registry_file.is_file():
            registry_ok = False
            registry_detail = f"missing: {registry_file}"
        else:
            try:
                registry = json.loads(registry_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                registry_ok = False
                registry_detail = str(exc)
            else:
                approved = {
                    source.get("source_id")
                    for source in registry.get("approved_sources", [])
                }
                registry_ok = identity in approved
                registry_detail = f"{identity} in {sorted(approved)}"
    checks.append(_check("approved_source_registry", registry_ok, registry_detail))
    if not registry_ok:
        blockers.append("source_not_in_approved_registry")

    return _decision(checks, blockers)


def _decision(checks: list, blockers: list) -> dict:
    unique = sorted(set(blockers))
    return {
        "decision": TRAINING_AUTHORIZED if not unique else TRAINING_NOT_AUTHORIZED,
        "blockers": unique,
        "checks": checks,
    }
