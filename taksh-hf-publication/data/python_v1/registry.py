"""Default-deny source registry for the Python data foundation (v1.2.0).

A source enters the corpus only through a versioned contract that declares
everything needed to trace it: an identity, a pinned content revision, licence
evidence with a hash, and an explicit owner approval. A source that does not
declare all of it is **denied**.

Three rules shape this module:

* Absence of evidence is a denial, never a permission. An unknown source, an
  unapproved source, an unresolved licence decision and a missing revision are
  all denials, and each carries a stable reason code.
* A repository's top-level licence file is not a decision. The contract must
  state the decision, the reviewer and the rationale, and the declared evidence
  file must exist inside the source and hash to the declared value.
* The allowlist never widens implicitly. ``policy.default_decision`` is
  ``deny`` and the contract loader refuses any other value.

This module implements policy enforcement only. It makes no legal
determination, and it does not resolve the roadmap's D1-D5 licensing
questions: those remain owner decisions and are reported as unresolved.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from data.licence_paths import path_within

CONTRACT_SCHEMA = "quantsmind.python_data_contract/1"
PIPELINE_VERSION = "python-v1.2.0"
APPROVED = "approved"
ALLOWED_KINDS = ("local_directory",)

REQUIRED_SOURCE_KEYS = ("id", "kind", "location", "revision", "license", "approval")
OPTIONAL_SOURCE_KEYS = (
    "project",
    "attribution",
    "allowed_extensions",
    "excluded_paths",
    "prune_dirs",
    "notes",
)
LICENSE_KEYS = ("spdx", "decision", "evidence", "evidence_sha256")
APPROVAL_KEYS = ("status", "reviewer", "rationale")

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ContractError(ValueError):
    """The contract is malformed. Raised before any source is touched."""


def canonical_json(value) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path) -> dict:
    """Load and validate a data contract. Raises :class:`ContractError`."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ContractError(f"{path}: contract must be a mapping")
    if data.get("schema") != CONTRACT_SCHEMA:
        raise ContractError(
            f"{path}: schema must be {CONTRACT_SCHEMA!r}, got {data.get('schema')!r}"
        )
    problems = validate_contract(data)
    if problems:
        raise ContractError(f"{path}: invalid contract: " + "; ".join(problems))
    return data


def validate_contract(cfg) -> list[str]:
    """Return every problem with a contract. An empty list means usable."""
    problems: list[str] = []
    for key in ("policy", "limits", "normalization", "quality", "dedup", "splits"):
        if key not in cfg:
            problems.append(f"missing top-level key '{key}'")
    policy = cfg.get("policy") or {}
    if policy.get("default_decision", "deny") != "deny":
        problems.append("policy.default_decision must remain 'deny'")
    if not policy.get("allowed_spdx"):
        problems.append("policy.allowed_spdx must not be empty")
    sources = cfg.get("sources")
    if sources is None:
        problems.append("missing 'sources'")
    elif not isinstance(sources, list):
        problems.append("'sources' must be a list")
    else:
        seen: set[str] = set()
        for index, source in enumerate(sources):
            problems.extend(validate_source(source, index, seen))
    return problems


def validate_source(source, index: int, seen_ids: set) -> list[str]:
    """Validate one source entry's shape. Approval itself is decided later."""
    if not isinstance(source, dict):
        return [f"sources[{index}] must be a mapping"]
    problems: list[str] = []
    sid = source.get("id")
    if not isinstance(sid, str) or not SOURCE_ID_RE.match(sid):
        problems.append(
            f"sources[{index}].id must match [a-z0-9][a-z0-9._-]*, got {sid!r}"
        )
    elif sid in seen_ids:
        problems.append(f"duplicate source id {sid!r}")
    else:
        seen_ids.add(sid)
    unknown = set(source) - set(REQUIRED_SOURCE_KEYS) - set(OPTIONAL_SOURCE_KEYS)
    if unknown:
        problems.append(f"sources[{index}] has unknown keys: {sorted(unknown)}")
    for key in REQUIRED_SOURCE_KEYS:
        if key not in source:
            problems.append(f"sources[{index}] missing required key '{key}'")
    kind = source.get("kind")
    if kind is not None and kind not in ALLOWED_KINDS:
        problems.append(
            f"sources[{index}].kind {kind!r} is not permitted; "
            f"allowed: {list(ALLOWED_KINDS)}"
        )
    licence = source.get("license")
    if isinstance(licence, dict):
        unknown_licence = set(licence) - set(LICENSE_KEYS)
        if unknown_licence:
            problems.append(
                f"sources[{index}].license has unknown keys: {sorted(unknown_licence)}"
            )
        for key in LICENSE_KEYS:
            if key not in licence:
                problems.append(f"sources[{index}].license missing '{key}'")
    elif licence is not None:
        problems.append(f"sources[{index}].license must be a mapping")
    approval = source.get("approval")
    if isinstance(approval, dict):
        unknown_approval = set(approval) - set(APPROVAL_KEYS)
        if unknown_approval:
            problems.append(
                f"sources[{index}].approval has unknown keys: {sorted(unknown_approval)}"
            )
        for key in ("status", "reviewer", "rationale"):
            if key not in approval:
                problems.append(f"sources[{index}].approval missing '{key}'")
    elif approval is not None:
        problems.append(f"sources[{index}].approval must be a mapping")
    return problems


def resolve_source_root(contract_path, source) -> Path:
    """Absolute path of a local source, resolved from the contract location."""
    location = source.get("location")
    if not isinstance(location, str) or not location:
        raise ContractError(f"source {source.get('id')!r} has no location")
    if "://" in location:
        raise ContractError(
            f"source {source.get('id')!r} location looks like a URL; only local "
            "directories are permitted and downloads are a separate action"
        )
    base = Path(contract_path).resolve().parent
    return (base / location).resolve()


def license_evidence_ok(source, root) -> tuple[bool, str]:
    """Verify the declared licence evidence file inside the source root.

    The evidence must live inside the source, must exist, and must hash to the
    declared value. A top-level licence file alone is never sufficient: the
    contract's explicit decision and this hash are both required.
    """
    licence = source.get("license") or {}
    evidence = licence.get("evidence")
    if not evidence:
        return False, "license_evidence_missing"
    root_path = Path(root).resolve()
    candidate = (root_path / evidence).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError:
        return False, "license_evidence_outside_source"
    if not candidate.is_file():
        return False, "license_evidence_missing"
    declared = licence.get("evidence_sha256")
    if not declared:
        return False, "license_evidence_hash_missing"
    if file_sha256(candidate) != declared:
        return False, "license_evidence_hash_mismatch"
    return True, "license_evidence_verified"


def source_decision(cfg, source, root=None) -> dict:
    """Decide whether one source may contribute. The default is denial."""
    policy = cfg.get("policy") or {}
    allowed_spdx = {str(s).lower() for s in policy.get("allowed_spdx", [])}
    licence = source.get("license") or {}
    approval = source.get("approval") or {}
    reasons: list[str] = []
    if source.get("kind") not in ALLOWED_KINDS:
        reasons.append("source_kind_not_permitted")
    if approval.get("status") != APPROVED:
        reasons.append("approval_not_granted")
    if licence.get("decision") != APPROVED:
        reasons.append("license_decision_unresolved")
    if str(licence.get("spdx") or "").lower() not in allowed_spdx:
        reasons.append("license_not_allowlisted")
    revision = source.get("revision")
    if not revision:
        reasons.append("revision_missing")
    elif not SHA256_RE.match(str(revision)):
        reasons.append("revision_not_pinned")
    if root is not None:
        ok, reason = license_evidence_ok(source, root)
        if not ok:
            reasons.append(reason)
    return {
        "id": source.get("id"),
        "kind": source.get("kind"),
        "project": source.get("project") or source.get("id"),
        "revision": revision,
        "license_spdx": licence.get("spdx"),
        "license_decision": licence.get("decision"),
        "approval_status": approval.get("status"),
        "approval_reviewer": approval.get("reviewer"),
        "allowed": not reasons,
        "reasons": reasons,
    }


def snapshot_digest(entries) -> str:
    """Content identity of a source snapshot.

    ``entries`` is an iterable of ``(relative_posix_path, file_sha256)``. The
    digest covers the exact file set and its contents, so a changed file
    changes the revision. No timestamp or absolute path participates.
    """
    lines = "".join(f"{rel}\t{digest}\n" for rel, digest in sorted(entries))
    return "sha256:" + sha256_text(lines)


def contract_fingerprint(cfg) -> str:
    """Content hash of the whole contract, used in the dataset manifest."""
    return "sha256:" + sha256_text(canonical_json(cfg))


def registry_fingerprint(cfg) -> str:
    """Hash of the source registry section only (identities and decisions)."""
    material = {
        "schema": CONTRACT_SCHEMA,
        "sources": [
            {
                "id": s.get("id"),
                "kind": s.get("kind"),
                "project": s.get("project") or s.get("id"),
                "revision": s.get("revision"),
                "license": s.get("license"),
                "approval": s.get("approval"),
                "excluded_paths": s.get("excluded_paths", []),
            }
            for s in cfg.get("sources", [])
        ],
    }
    return "sha256:" + sha256_text(canonical_json(material))
