"""Learning Feedback Engine — controlled handoff from DLCAE to the Learning Engine.

Hard boundaries under current owner decisions (A1.1-C / A2-E):

- training handoff is DISABLED. Candidates may be produced, validated and
  queued; they are never training-eligible automatically;
- eligibility requires: provenance complete, license allowlisted, validation
  verdict PASS, no duplicate, no conflict, and an explicit owner decision
  recorded in ``docs/L075_L080_OWNER_DECISION_RECORD.md`` selecting A1.1-A;
- everything else is reported, never promoted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from taksh.dlcae.types import Provenance, VerificationState, fingerprint, utcnow_iso

REPO = Path(__file__).resolve().parent.parent.parent
OWNER_DECISION_RECORD = REPO / "docs" / "L075_L080_OWNER_DECISION_RECORD.md"

LICENSE_ALLOWLIST = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0",
    "owner-internal-research",
}

CANDIDATE_STATES = ("NEW", "DUPLICATE", "CONFLICTED", "REJECTED",
                    "ELIGIBLE_CANDIDATE", "QUEUED_FOR_OWNER_REVIEW")


def read_owner_decision(path: Path | None = None) -> dict[str, Any]:
    """Parse the owner decision record fail-closed.

    Returns the set of checked decision ids; any parse ambiguity collapses to
    "nothing checked" — never to authorization.
    """
    record = path or OWNER_DECISION_RECORD
    try:
        text = record.read_text(encoding="utf-8")
    except OSError:
        return {"checked": [], "ok": False, "training_authorized": False,
                "reason": f"decision record unreadable: {record}"}
    checked = re.findall(r"- \[[xX]\] \*\*([A-Za-z0-9._-]+)", text)
    return {"checked": sorted(set(checked)), "ok": True,
            "training_authorized": "A1.1-A" in checked,
            "source": str(record)}


class LearningFeedbackEngine:
    """Turns verified comprehension output into learning candidates — gated."""

    def __init__(self) -> None:
        self.candidates: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def from_observation(self, observation: dict[str, Any], *,
                         language_id: str, source_provenance: Provenance,
                         content_ref: str) -> dict[str, Any]:
        """Build a learning candidate from one verified observation."""
        record = {
            "schema": "taksh.dlcae.learning_candidate/1",
            "candidate_id": fingerprint({"obs": observation.get("observation_id"),
                                         "src": source_provenance.record_id,
                                         "lang": language_id})[:16],
            "language_id": language_id,
            "content_ref": content_ref,
            "origin_observation": observation.get("observation_id"),
            "provenance": source_provenance.to_dict(),
            "verification_state": observation.get("state", "OBSERVED"),
            "created": utcnow_iso(),
        }
        return self.register(record)

    def register(self, candidate: dict[str, Any]) -> dict[str, Any]:
        cid = candidate["candidate_id"]
        if cid in self.candidates:
            existing = self.candidates[cid]
            return {**existing, "state": "DUPLICATE", "duplicate_of": cid}
        self.candidates[cid] = candidate
        candidate["state"] = "NEW"
        return candidate

    def eligibility_check(self, candidate: dict[str, Any], *,
                          validation_verdict: str | None) -> dict[str, Any]:
        """Fail-closed eligibility evaluation. Never upgrades state to training."""
        problems: list[str] = []
        prov = candidate.get("provenance", {})
        for field in ("record_id", "source", "license_spdx"):
            if not prov.get(field):
                problems.append(f"missing provenance: {field}")
        lic = str(prov.get("license_spdx", "")).lower()
        if lic not in LICENSE_ALLOWLIST:
            problems.append(f"license not allowlisted: {lic!r}")
        if str(candidate.get("verification_state")) != VerificationState.VERIFIED.value:
            problems.append("origin observation is not VERIFIED")
        if (validation_verdict or "UNVERIFIED") != "PASS":
            problems.append(f"validation verdict is not PASS (got {validation_verdict or 'none'})")
        candidate = dict(candidate)
        if problems:
            candidate["state"] = "REJECTED"
            candidate["rejection_reasons"] = problems
        else:
            candidate["state"] = "ELIGIBLE_CANDIDATE"
            candidate["note"] = ("eligible candidate only — never training data; "
                                 "owner authorization is a separate, explicit act")
        self.candidates[candidate["candidate_id"]] = candidate
        return {"candidate_id": candidate["candidate_id"],
                "state": candidate["state"], "problems": problems}

    def handoff_to_training(self, candidate_ids: list[str] | None = None) -> dict[str, Any]:
        """Refuse unless the owner decision record authorizes a run (A1.1-A).

        Under the current record (A1.1-C: preparation only) this always
        refuses — candidates are never released for training.
        """
        decision = read_owner_decision()
        checked = decision.get("checked", [])
        compute_authorized = any(m in checked for m in ("A2-A", "A2-B", "A2-D"))
        ids = candidate_ids or sorted(self.candidates)
        selected = [self.candidates[i] for i in ids if i in self.candidates]
        if not decision.get("training_authorized") or not compute_authorized:
            missing = []
            if not decision.get("training_authorized"):
                missing.append("A1.1-A (training authorization)")
            if not compute_authorized:
                missing.append("A2-A / A2-B / A2-D (compute authorization)")
            return {
                "schema": "taksh.dlcae.learning_handoff/1",
                "authorized": False,
                "owner_decision": decision,
                "candidates": [c["candidate_id"] for c in selected],
                "released": [],
                "reason": ("missing owner decision(s): " + ", ".join(missing) +
                           ". A1.1-C preparation-only + A2-E authorize no training. "
                           "Handoff refused."),
                "training_eligible_tokens": 0,
            }
        # Even under a hypothetical future A1.1-A, this milestone never trains:
        blocking = [f"{c['candidate_id']}: state={c.get('state')}" for c in selected
                    if c.get("state") != "ELIGIBLE_CANDIDATE"]
        if blocking:
            return {
                "authorized": False,
                "owner_decision": decision,
                "blocking_candidates": blocking,
                "released": [],
                "reason": "not all candidates are ELIGIBLE_CANDIDATE",
                "training_eligible_tokens": 0,
            }
        return {
            "authorized": True,
            "owner_decision": decision,
            "candidates": [c["candidate_id"] for c in selected],
            "released": [],
            "reason": ("authorization detected, but this milestone executes no "
                       "training (A2-E: no compute authorization). Candidates "
                       "registered for a future owner-authorized run."),
            "training_eligible_tokens": 0,
        }

    # ------------------------------------------------------------------ #
    def record_feedback(self, candidate_id: str, *, outcome: str,
                        evaluator: str, note: str = "") -> dict[str, Any]:
        if outcome not in ("accepted", "rejected", "replay_scheduled", "quarantined"):
            raise ValueError(f"unknown feedback outcome: {outcome!r}")
        record = {"schema": "taksh.dlcae.learning_feedback/1",
                  "candidate_id": candidate_id, "outcome": outcome,
                  "evaluator": evaluator, "note": note,
                  "recorded": utcnow_iso()}
        self.feedback.append(record)
        if outcome == "quarantined" and candidate_id in self.candidates:
            self.candidates[candidate_id] = {**self.candidates[candidate_id],
                                             "state": "REJECTED",
                                             "quarantined": True}
        return record

    def replay_manifest(self, *, sample_fraction: float = 0.2,
                        language_ids: list[str] | None = None) -> dict[str, Any]:
        """Build a replay manifest of *verified* candidates (never executed here)."""
        verified = [c for c in self.candidates.values()
                    if c.get("verification_state") == VerificationState.VERIFIED.value
                    and c.get("state") in ("ELIGIBLE_CANDIDATE", "QUEUED_FOR_OWNER_REVIEW")
                    and (not language_ids or c["language_id"] in language_ids)]
        return {
            "schema": "taksh.dlcae.replay_manifest/1",
            "sample_fraction": sample_fraction,
            "candidates": [c["candidate_id"] for c in verified],
            "candidate_count": len(verified),
            "executes_training": False,
            "note": "manifest only; no weights are read, created or updated here",
        }

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for c in self.candidates.values():
            counts[c.get("state", "NEW")] = counts.get(c.get("state", "NEW"), 0) + 1
        return {"candidates": len(self.candidates),
                "state_counts": counts,
                "feedback_records": len(self.feedback),
                "handoff": self.handoff_to_training([])}
