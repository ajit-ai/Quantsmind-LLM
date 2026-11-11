"""Language Adaptation Engine — REGISTERED → … → qualified, with demotion.

Qualification requires evidence and a passing regression gate; a regression
must demote a language's status. Registration never implies comprehension;
comprehension never implies qualification. Capability states require recorded
benchmark evidence with an explicit bar and a passing regression.

Karkain is a mandatory first-class target: DISCOVERED, KCC-blocked, zero
corpus, zero benchmark tasks. No qualification is claimed anywhere.
"""

from __future__ import annotations

from typing import Any

from taksh.dlcae.registry import DynamicLanguageRegistry, RegistryError
from taksh.dlcae.types import QualificationStatus, utcnow_iso
from taksh.dlcae.validation import ValidationEngine


class LanguageAdaptationEngine:
    def __init__(self, registry: DynamicLanguageRegistry,
                 validation: ValidationEngine) -> None:
        self.registry = registry
        self.validation = validation
        self.ledger: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def onboard(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Register a new language. Initial status is never capability."""
        registered = self.registry.register(entry)
        self._record("onboard", registered["id"], registered["qualification_status"])
        return registered

    def discover(self, language_id: str, *, provenance: dict[str, Any]) -> dict[str, Any]:
        """Mark an existing entry as DISCOVERED after provenance is recorded."""
        problems = []
        for field in ("source", "license_spdx"):
            if not provenance.get(field):
                problems.append(f"missing provenance: {field}")
        if problems:
            raise RegistryError(f"discovery requires provenance: {problems}")
        if str(provenance.get("license_spdx", "")).upper() == "UNKNOWN":
            raise RegistryError("discovery requires a recorded license decision; UNKNOWN is not accepted")
        entry = self.registry.promote(language_id, "DISCOVERED")
        self._record("discover", language_id, entry["qualification_status"],
                     extra={"provenance": provenance})
        return entry

    def start_comprehension(self, language_id: str) -> dict[str, Any]:
        entry = self.registry.promote(language_id, "COMPREHENSION_IN_PROGRESS")
        self._record("comprehension_start", language_id, entry["qualification_status"])
        return entry

    def qualify(self, language_id: str, target: str, *,
                evidence: dict[str, Any]) -> dict[str, Any]:
        """Promote to a capability state only with evidence; otherwise refused."""
        entry = self.registry.promote(language_id, target, evidence=evidence)
        self._record("qualify", language_id, entry["qualification_status"],
                     extra={"evidence": evidence})
        return entry

    def record_regression(self, language_id: str, *, reason: str,
                          evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """A failed regression gate demotes the language — mandatory, never silent."""
        entry = self.registry.demote(language_id, reason=reason, evidence=evidence)
        self._record("regression_demotion", language_id, entry["qualification_status"],
                     extra={"reason": reason, "evidence": evidence or {}})
        return entry

    def block(self, language_id: str, *, reason: str) -> dict[str, Any]:
        entry = self.registry.block(language_id, reason=reason)
        self._record("block", language_id, entry["qualification_status"], extra={"reason": reason})
        return entry

    def unblock(self, language_id: str, *, owner_reference: str) -> dict[str, Any]:
        entry = self.registry.unblock(language_id, owner_reference=owner_reference)
        self._record("unblock", language_id, entry["qualification_status"],
                     extra={"owner_reference": owner_reference})
        return entry

    # ------------------------------------------------------------------ #
    def missing_toolchains(self, language_id: str) -> list[str]:
        lane = self.validation.host_capabilities.get(language_id.lower())
        if language_id.lower() == "karkain":
            return ["kcc (0xC0000135 on this host)",
                    "KARK-BENCH: 0 instantiated tasks; qualification blocked"]
        if language_id.lower() == "python":
            missing = []
            if not lane.get("type_checker"):
                missing.append("type_checker (e.g. mypy) not installed")
            if lane.get("test_runner"):
                pass
            return missing
        return [f"no toolchain qualified for {language_id!r} on this host"]

    def status_report(self, language_id: str) -> dict[str, Any]:
        entry = self.registry.lookup(language_id)
        if entry is None:
            return {"language_id": language_id, "registered": False,
                    "note": "not registered — discovery required before comprehension"}
        status = entry.get("qualification_status", "REGISTERED")
        return {
            "language_id": language_id,
            "registered": True,
            "status": status,
            "origin": entry.get("origin"),
            "is_capability_state": status in {"FUNCTIONALLY_QUALIFIED", "ENGINEERING_QUALIFIED"},
            "missing_toolchains": self.missing_toolchains(language_id),
            "note": "registration/discovery/comprehension never imply capability",
        }

    def _record(self, event: str, language_id: str, status: str,
                extra: dict[str, Any] | None = None) -> None:
        self.ledger.append({
            "schema": "taksh.dlcae.adaptation_ledger/1",
            "event": event,
            "language_id": language_id,
            "status": status,
            "at": utcnow_iso(),
            **(extra or {}),
        })
