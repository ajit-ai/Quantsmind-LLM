"""Python training-source authorization framework (Taksh v1.6.0).

The v1.2.0 registry answers "may this source be ingested?". This module
answers the harder question first: "has a human, with evidence, decided that
this source may be used for this purpose?".

Rules that cannot be bent:

* Only ``approved`` is eligible. ``pending_review`` is a request, never a
  permission.
* Authorization is not inferred from a permissive-looking licence name. The
  declared licence, the evidence reference and the reviewer's decision are
  separate facts and all three are required.
* A missing, malformed or expired record is a refusal, not a warning.
* This module never makes a legal determination. It records what a reviewer
  decided and what evidence existed at that moment, and it says plainly when
  a fact is ``unknown`` or ``not_assessed``.

The manifest format is ``quantsmind.source_manifest/2``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SOURCE_MANIFEST_SCHEMA = "quantsmind.source_manifest/2"

AUTHORIZATION_STATES = (
    "pending_review",
    "approved",
    "rejected",
    "revoked",
    "expired",
    "insufficient_evidence",
)

#: The only state that may contribute production material.
ELIGIBLE_STATE = "approved"

#: Licence identifiers this framework can accept without further evidence.
PERMITTED_LICENCES = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC")

REQUIRED_FIELDS = (
    "source_id",
    "source_name",
    "origin",
    "publisher",
    "revision",
    "source_type",
    "declared_license",
    "license_evidence",
    "intended_use",
    "commercial_use",
    "modification_rights",
    "attribution_obligations",
    "provenance_evidence",
    "collection_method",
    "personal_data_risk",
    "secrets_risk",
    "third_party_rights_risk",
    "contamination_assessment",
    "authorization_state",
    "reviewer",
    "decision_timestamp",
    "decision_rationale",
    "revocation_status",
)

UNKNOWN_VALUES = ("unknown", "not_assessed")


class AuthorizationError(RuntimeError):
    """A source manifest is malformed."""


@dataclass
class AuthorizationDecision:
    """The outcome of an authorization check."""

    allowed: bool
    state: str
    reasons: list = field(default_factory=list)
    source_id: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "state": self.state,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "revision": self.revision,
        }


def load_source_manifest(path) -> dict:
    """Load a source manifest. A malformed manifest is an error, not a denial."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_source_manifest(data)
    if problems:
        raise AuthorizationError(
            f"{path}: malformed source manifest: " + "; ".join(problems)
        )
    return data


def validate_source_manifest(manifest: dict) -> list:
    """Return every problem with a source manifest."""
    problems: list = []
    if not isinstance(manifest, dict):
        return ["source manifest must be a mapping"]
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        problems.append(
            f"schema must be {SOURCE_MANIFEST_SCHEMA!r}, got "
            f"{manifest.get('schema')!r}"
        )
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            problems.append(f"missing '{field}'")
    state = manifest.get("authorization_state")
    if state is not None and state not in AUTHORIZATION_STATES:
        problems.append(f"authorization_state must be one of {list(AUTHORIZATION_STATES)}")
    licence = manifest.get("declared_license")
    if licence is not None and licence not in PERMITTED_LICENCES and licence not in UNKNOWN_VALUES:
        problems.append(
            f"declared_license {licence!r} is not a permitted identifier "
            f"{list(PERMITTED_LICENCES)}; an explicit decision is required"
        )
    for field in ("personal_data_risk", "secrets_risk", "third_party_rights_risk"):
        value = manifest.get(field)
        if value is not None and not isinstance(value, str):
            problems.append(f"{field} must be a string")
    return problems


def evaluate_authorization(
    manifest: dict,
    *,
    expected_revision: str | None = None,
    intended_use: str | None = None,
    now: str | None = None,
    permitted_licences=PERMITTED_LICENCES,
) -> AuthorizationDecision:
    """Decide whether a source may be used, and say why.

    Every refusal carries a stable reason code. The check is fail-closed: any
    missing fact is a refusal.
    """
    reasons: list = []
    problems = validate_source_manifest(manifest)
    state = manifest.get("authorization_state", "pending_review")
    decision = AuthorizationDecision(
        allowed=False, state=state, reasons=reasons,
        source_id=manifest.get("source_id"),
        revision=manifest.get("revision"),
    )
    if problems:
        reasons.append("malformed_source_manifest")
        reasons.extend(problems)
        return decision
    if state != ELIGIBLE_STATE:
        reasons.append(f"authorization_state_is_{state}")
    reviewer = manifest.get("reviewer")
    if not reviewer or reviewer in UNKNOWN_VALUES:
        reasons.append("missing_reviewer_authorization")
    timestamp = manifest.get("decision_timestamp")
    if not timestamp:
        reasons.append("missing_decision_timestamp")
    if manifest.get("revocation_status") not in (None, "", "not_revoked"):
        reasons.append("authorization_revoked")
    expires = manifest.get("decision_expires_on")
    reference = now or datetime.now(timezone.utc).date().isoformat()
    if expires and str(expires) < str(reference):
        reasons.append("authorization_expired")
    evidence = manifest.get("license_evidence") or {}
    if not evidence.get("evidence_present") or not evidence.get("evidence_hash"):
        reasons.append("missing_license_evidence")
    licence = manifest.get("declared_license")
    if licence in UNKNOWN_VALUES:
        reasons.append("license_not_assessed")
    elif licence not in permitted_licences:
        reasons.append("license_not_permitted")
    if expected_revision is not None and manifest.get("revision") != expected_revision:
        reasons.append("revision_mismatch")
    if intended_use is not None:
        approved = manifest.get("approved_uses")
        if approved is not None and intended_use not in approved:
            reasons.append("use_out_of_approved_scope")
    provenance = manifest.get("provenance_evidence") or {}
    if not provenance.get("established"):
        reasons.append("provenance_not_established")
    for field in ("personal_data_risk", "secrets_risk", "third_party_rights_risk"):
        if str(manifest.get(field, "")).lower() == "high":
            reasons.append(f"unmitigated_{field}")
    decision.allowed = not reasons
    return decision


def build_review_packet(manifest: dict) -> dict:
    """Build the packet a human reviewer needs to decide.

    A packet is a request, never an approval: the packet always states the
    decisions that are still required.
    """
    problems = validate_source_manifest(manifest)
    evidence = manifest.get("license_evidence") or {}
    provenance = manifest.get("provenance_evidence") or {}
    missing = []
    if not evidence.get("evidence_present"):
        missing.append("licence evidence file is not present")
    if not evidence.get("evidence_hash"):
        missing.append("licence evidence hash is not recorded")
    if not provenance.get("established"):
        missing.append("provenance is not established")
    if not manifest.get("reviewer"):
        missing.append("no reviewer is recorded")
    if manifest.get("declared_license") in UNKNOWN_VALUES:
        missing.append("declared licence is unknown")
    for field in ("personal_data_risk", "secrets_risk", "third_party_rights_risk"):
        if not manifest.get(field):
            missing.append(f"{field} has not been assessed")
    risks = {
        field: manifest.get(field, "not_assessed")
        for field in ("personal_data_risk", "secrets_risk", "third_party_rights_risk",
                      "contamination_assessment", "commercial_use",
                      "modification_rights", "attribution_obligations")
    }
    return {
        "schema": "quantsmind.source_review_packet/1",
        "source_id": manifest.get("source_id"),
        "source_name": manifest.get("source_name"),
        "origin": manifest.get("origin"),
        "publisher": manifest.get("publisher"),
        "revision": manifest.get("revision"),
        "declared_license": manifest.get("declared_license"),
        "proposed_intended_use": manifest.get("intended_use"),
        "evidence_collected": {
            "license_evidence": evidence,
            "provenance_evidence": provenance,
            "collection_method": manifest.get("collection_method"),
        },
        "missing_evidence": missing,
        "known_risks": risks,
        "required_human_decisions": [
            "accept or reject the declared licence and its evidence",
            "confirm the provenance chain and collection method",
            "accept or reject the intended use scope",
            "accept or reject the contamination assessment",
            "record the reviewer identity and the decision rationale",
        ],
        "explicit_blockers": missing or ["no decision has been taken yet"],
        "manifest_problems": problems,
        "current_state": manifest.get("authorization_state", "pending_review"),
        "note": (
            "A review packet is a request for a decision. It never implies "
            "approval, and the source stays ineligible until a human records "
            "an approval decision."
        ),
    }


def evidence_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
