"""Dynamic language registry integration — no fixed catalogue ceiling.

Wraps the committed TAKSH registries (read-only) and adds a *runtime*
extension layer:

- the 53-language base (`configs/feasibility/l051_languages.yaml`) is
  immutable and never written;
- committed extension entries
  (`configs/foundation/l060r1_dynamic_language_registry.yaml`,
  Karkain = 54th first-class target) are read and merged;
- runtime registration adds *new* languages without touching either file or
  the schema — including languages nobody has heard of yet;
- capability status can never be assigned at registration time;
  promotion requires evaluation evidence; regression forces demotion;
- BLOCKED is terminal until an explicit owner reference unblocks it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from taksh.dlcae.types import CAPABILITY_STATES, QualificationStatus, utcnow_iso

REPO = Path(__file__).resolve().parent.parent.parent
BASE_REGISTRY = REPO / "configs" / "feasibility" / "l051_languages.yaml"
EXTENSION_REGISTRY = REPO / "configs" / "foundation" / "l060r1_dynamic_language_registry.yaml"

REQUIRED_ENTRY_FIELDS = [
    "id", "name", "aliases", "version_or_dialect", "extensions",
    "tokenizer_metadata", "grammar_or_spec_reference", "semantic_metadata",
    "compiler_or_runtime_integration", "execution_config",
    "stdlib_knowledge_reference", "documentation_and_provenance",
    "license_and_source_authorization", "qualification_status",
    "benchmark_associations", "adaptation_history",
]
VALID_STATUSES = frozenset(s.value for s in QualificationStatus)

# Demotion ladder applied on regression evidence. Capability states fall to
# COMPREHENSION_IN_PROGRESS; progress states fall to DISCOVERED.
_DEMOTE_TARGET = {
    QualificationStatus.ENGINEERING_QUALIFIED: QualificationStatus.FUNCTIONALLY_QUALIFIED,
    QualificationStatus.FUNCTIONALLY_QUALIFIED: QualificationStatus.COMPREHENSION_IN_PROGRESS,
    QualificationStatus.COMPREHENSION_IN_PROGRESS: QualificationStatus.DISCOVERED,
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def extension_entry_template(language_id: str, name: str, extensions: list[str],
                             *, source: str, license_spdx: str,
                             status: str = "REGISTERED") -> dict[str, Any]:
    """A complete, schema-valid extension entry for a newly registered language.

    ``license_spdx='UNKNOWN'`` is accepted on purpose: registration is an
    inventory act, not a rights grant.
    """
    return {
        "id": language_id,
        "name": name,
        "aliases": [language_id],
        "version_or_dialect": "unspecified — discovery pending",
        "extensions": list(extensions),
        "tokenizer_metadata": "reuses taksh_code_bpe_v1.json; unverified fit",
        "grammar_or_spec_reference": "none on file — discovery pending",
        "semantic_metadata": "none on file — discovery pending",
        "compiler_or_runtime_integration": "none on file — discovery pending",
        "execution_config": "none on file — discovery pending",
        "stdlib_knowledge_reference": "none on file — discovery pending",
        "documentation_and_provenance": source,
        "license_and_source_authorization": license_spdx,
        "qualification_status": status,
        "benchmark_associations": [],
        "adaptation_history": [],
    }


class RegistryError(Exception):
    pass


class DynamicLanguageRegistry:
    """Read-only committed view plus a runtime-registered extension layer."""

    def __init__(self) -> None:
        self._committed = self._load_committed()
        self._runtime: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def _load_committed(self) -> dict[str, dict[str, Any]]:
        registry: dict[str, dict[str, Any]] = {}
        base = load_yaml(BASE_REGISTRY)
        for lang in base.get("languages", []):
            registry[lang["id"]] = {
                "id": lang["id"], "name": lang.get("name", lang["id"]),
                "extensions": lang.get("extensions", []),
                "qualification_status": lang.get("qualification_status", "REGISTERED"),
                "class": "language", "origin": "base-53",
            }
        for tech in base.get("supporting_technologies", []):
            key = tech["name"].lower().replace(" ", "_").replace("-", "_")
            registry[key] = {"id": key, "name": tech["name"], "extensions": [],
                             "qualification_status": "SUPPORTING_TECHNOLOGY",
                             "class": "technology", "origin": "base-53"}
        ext = load_yaml(EXTENSION_REGISTRY)
        for entry in ext.get("entries", []):
            merged = dict(entry)
            merged["qualification_status"] = entry.get("qualification_status", "DISCOVERED")
            merged["origin"] = "committed-extension"
            merged["class"] = "language"
            registry[entry["id"]] = merged
        return registry

    # ------------------------------------------------------------------ #
    def all_ids(self) -> list[str]:
        return sorted(set(self._committed) | set(self._runtime))

    def lookup(self, language_id: str) -> dict[str, Any] | None:
        key = language_id.lower()
        if key in self._runtime:
            return copy.deepcopy(self._runtime[key])
        entry = self._committed.get(key)
        return copy.deepcopy(entry) if entry else None

    def discover(self, query: str) -> list[dict[str, Any]]:
        needle = query.lower().strip()
        if not needle:
            return []
        matches: list[dict[str, Any]] = []
        if needle.startswith("."):
            for entry in self._all_entries().values():
                if needle in [e.lower() for e in entry.get("extensions", [])]:
                    matches.append(entry)
            return matches
        for entry in self._all_entries().values():
            if needle in entry["id"] or needle in str(entry.get("name", "")).lower():
                matches.append(entry)
            elif any(needle in str(alias).lower() for alias in entry.get("aliases", [])):
                matches.append(entry)
        return matches

    def _all_entries(self) -> dict[str, dict[str, Any]]:
        merged = {k: copy.deepcopy(v) for k, v in self._committed.items()}
        merged.update(copy.deepcopy(self._runtime))
        return merged

    # ------------------------------------------------------------------ #
    def register(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Register a new language at runtime. Never grants a capability state."""
        missing = [f for f in REQUIRED_ENTRY_FIELDS
                   if entry.get(f) in (None, "", [])
                   and f not in ("adaptation_history", "benchmark_associations")]
        if "adaptation_history" not in entry:
            missing.append("adaptation_history")
        if "benchmark_associations" not in entry:
            missing.append("benchmark_associations")
        if missing:
            raise RegistryError(f"missing required fields for registration: {missing}")
        lid = str(entry["id"]).lower().strip()
        if not lid or any(c in lid for c in "/\\ "):
            raise RegistryError(f"invalid language id: {entry['id']!r}")
        if lid in self._committed or lid in self._runtime:
            raise RegistryError(f"language id already registered: {lid!r}")
        status = str(entry.get("qualification_status", "REGISTERED"))
        if status != QualificationStatus.REGISTERED.value:
            if status == QualificationStatus.DISCOVERED.value:
                if str(entry.get("documentation_and_provenance", "")).endswith(
                        "discovery pending"):
                    raise RegistryError(
                        "DISCOVERED requires real provenance, not the template placeholder")
            else:
                raise RegistryError(
                    f"registration never grants status {status!r}; "
                    f"only REGISTERED (or DISCOVERED with real provenance) is permitted")
        registered = dict(entry, id=lid)
        registered["origin"] = "runtime-registered"
        registered["class"] = "language"
        registered["adaptation_history"] = list(entry.get("adaptation_history", []))
        registered["registered_at"] = utcnow_iso()
        self._runtime[lid] = registered
        return copy.deepcopy(registered)

    # ------------------------------------------------------------------ #
    def status_of(self, language_id: str) -> str | None:
        entry = self.lookup(language_id)
        return entry.get("qualification_status") if entry else None

    def is_capability_state(self, language_id: str) -> bool:
        status = self.status_of(language_id)
        return status in {s.value for s in CAPABILITY_STATES}

    def promote(self, language_id: str, target: str, *,
                evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Promote a language. Capability states require recorded evaluation."""
        entry = self._require_entry(language_id)
        current = QualificationStatus(entry["qualification_status"])
        try:
            goal = QualificationStatus(target)
        except ValueError:
            raise RegistryError(f"unknown qualification status: {target!r}") from None

        if goal is QualificationStatus.BLOCKED:
            raise RegistryError("use block() to restrict a language — never promote into BLOCKED")
        if current is QualificationStatus.BLOCKED:
            raise RegistryError("BLOCKED is terminal until an owner unblocks it")
        allowed_forward = {
            QualificationStatus.REGISTERED: {QualificationStatus.DISCOVERED},
            QualificationStatus.DISCOVERED: {QualificationStatus.COMPREHENSION_IN_PROGRESS},
            QualificationStatus.COMPREHENSION_IN_PROGRESS: {
                QualificationStatus.FUNCTIONALLY_QUALIFIED},
            QualificationStatus.FUNCTIONALLY_QUALIFIED: {
                QualificationStatus.ENGINEERING_QUALIFIED},
            QualificationStatus.ENGINEERING_QUALIFIED: set(),
        }[current]
        if goal not in allowed_forward:
            raise RegistryError(f"illegal promotion {current.value} -> {goal.value}")

        if goal in CAPABILITY_STATES:
            if not evidence or evidence.get("evaluation_passed") is not True:
                raise RegistryError(
                    f"{goal.value} requires evaluation evidence with evaluation_passed=True")
            bar = evidence.get("benchmark_bar")
            if not isinstance(bar, str) or not bar.strip():
                raise RegistryError("capability promotion requires a recorded benchmark bar")
            if evidence.get("regression_passed") is not True:
                raise RegistryError("capability promotion requires a passing regression gate")
            evidence_note = {"evaluation_passed": True, "benchmark_bar": bar,
                             "regression_passed": True}
        else:
            evidence_note = {"provenance_recorded": True} if goal is QualificationStatus.DISCOVERED else {}

        entry = copy.deepcopy(entry)
        entry["qualification_status"] = goal.value
        entry["adaptation_history"] = list(entry.get("adaptation_history", [])) + [{
            "at": utcnow_iso(),
            "from": current.value,
            "to": goal.value,
            "evidence": evidence_note,
        }]
        self._write_back(language_id, entry)
        return entry

    def demote(self, language_id: str, *, reason: str,
               evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Mandatory demotion on regression evidence. Never silently accepted."""
        if not reason.strip():
            raise RegistryError("demotion requires a reason")
        entry = self._require_entry(language_id)
        current = QualificationStatus(entry["qualification_status"])
        target = _DEMOTE_TARGET.get(current)
        if target is None:
            target = QualificationStatus.DISCOVERED if current is QualificationStatus.DISCOVERED \
                else QualificationStatus.REGISTERED
        entry = copy.deepcopy(entry)
        entry["qualification_status"] = target.value
        entry["adaptation_history"] = list(entry.get("adaptation_history", [])) + [{
            "at": utcnow_iso(),
            "from": current.value,
            "to": target.value,
            "demotion_reason": reason,
            "evidence": evidence or {},
        }]
        self._write_back(language_id, entry)
        return entry

    def block(self, language_id: str, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise RegistryError("blocking requires a reason")
        entry = self._require_entry(language_id)
        entry = copy.deepcopy(entry)
        entry["qualification_status"] = QualificationStatus.BLOCKED.value
        entry["adaptation_history"] = list(entry.get("adaptation_history", [])) + [{
            "at": utcnow_iso(), "from": entry.get("qualification_status"),
            "to": "BLOCKED", "block_reason": reason}]
        self._write_back(language_id, entry)
        return entry

    def unblock(self, language_id: str, *, owner_reference: str) -> dict[str, Any]:
        if not (owner_reference or "").strip():
            raise RegistryError(
                "unblocking requires an explicit owner reference (owner-decision id)")
        entry = self._require_entry(language_id)
        entry = copy.deepcopy(entry)
        previous = entry["qualification_status"]
        entry["qualification_status"] = QualificationStatus.DISCOVERED.value
        entry["adaptation_history"] = list(entry.get("adaptation_history", [])) + [{
            "at": utcnow_iso(), "from": previous, "to": "DISCOVERED",
            "owner_reference": owner_reference}]
        self._write_back(language_id, entry)
        return entry

    # ------------------------------------------------------------------ #
    def _require_entry(self, language_id: str) -> dict[str, Any]:
        entry = self.lookup(language_id)
        if entry is None:
            raise RegistryError(f"language not registered: {language_id!r}")
        if entry.get("class") == "technology":
            raise RegistryError("supporting technologies are not languages; refusing")
        return entry

    def _write_back(self, language_id: str, entry: dict[str, Any]) -> None:
        key = language_id.lower()
        if key in self._runtime:
            self._runtime[key] = entry
        else:
            # Committed entries are copied into the runtime layer on first
            # mutation; committed files are never written.
            self._runtime[key] = entry

    def runtime_state(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.registry_runtime/1",
            "generated": utcnow_iso(),
            "committed_total": len(self._committed),
            "runtime_registered": sorted(self._runtime),
            "entries": {k: v for k, v in sorted(self._runtime.items())},
            "ceiling": None,
        }
