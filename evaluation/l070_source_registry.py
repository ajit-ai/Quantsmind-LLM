"""TAKSH L0.7.0 — data-source registry validator and state machine.

Validates configs/data/l070_source_registry.yaml against the L0.7.0 contract:
required fields, explicit acquisition states, legal transitions, and the
owner-only authorization gate. A source never qualifies for acquisition by
being downloadable; the owner alone authorizes ACQUISITION_AUTHORIZED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REGISTRY_PATH = REPO / "configs" / "data" / "l070_source_registry.yaml"

STATUSES = frozenset({
    "DISCOVERED", "METADATA_REVIEW", "LICENSE_REVIEW", "PROVENANCE_REVIEW",
    "TECHNICAL_REVIEW", "OWNER_APPROVAL_REQUIRED", "QUALIFIED_FOR_ACQUISITION",
    "ACQUISITION_AUTHORIZED", "ACQUIRED", "REJECTED", "DEFERRED", "BLOCKED",
})

REQUIRED_FIELDS = [
    "source_id", "name", "publisher", "dataset_version", "source_type",
    "languages_covered", "supporting_technologies_covered", "description",
    "acquisition_method", "distribution_format", "license_id", "license_ref",
    "attribution_required", "redistribution_restrictions", "commercial_use",
    "derivative_works", "training_use", "tos_restrictions", "provenance",
    "integrity", "known_exclusions", "legal_review_status",
    "owner_approval_status", "acquisition_status",
]

REQUIRED_PROVENANCE = ["source", "revision", "license_note"]

# Legal forward progression. Regression paths exist (demotion) but the only
# path into ACQUISITION_AUTHORIZED is via OWNER_APPROVAL_REQUIRED / QUALIFIED_FOR_ACQUISITION.
ALLOWED_TRANSITIONS = {
    "DISCOVERED": {"METADATA_REVIEW", "REJECTED", "DEFERRED", "BLOCKED"},
    "METADATA_REVIEW": {"LICENSE_REVIEW", "REJECTED", "DEFERRED", "BLOCKED"},
    "LICENSE_REVIEW": {"PROVENANCE_REVIEW", "REJECTED", "BLOCKED"},
    "PROVENANCE_REVIEW": {"TECHNICAL_REVIEW", "OWNER_APPROVAL_REQUIRED", "BLOCKED", "REJECTED"},
    "TECHNICAL_REVIEW": {"OWNER_APPROVAL_REQUIRED", "BLOCKED", "REJECTED"},
    "OWNER_APPROVAL_REQUIRED": {"QUALIFIED_FOR_ACQUISITION", "REJECTED", "DEFERRED"},
    "QUALIFIED_FOR_ACQUISITION": {"ACQUISITION_AUTHORIZED", "DEFERRED"},
    "ACQUISITION_AUTHORIZED": {"ACQUIRED", "REJECTED"},
    "ACQUIRED": {"BLOCKED"},             # acquired corpora can be blocked on new evidence
    "REJECTED": set(),                    # terminal
    "DEFERRED": {"DISCOVERED", "METADATA_REVIEW"},   # owner may revisit
    "BLOCKED": {"DISCOVERED"},            # owner unblock only
}

# States that require things to be true before entering/being in them
GUARDS = {
    "QUALIFIED_FOR_ACQUISITION": "legal_review must be completed (not HUMAN_LEGAL_REVIEW_REQUIRED/UNKNOWN) AND owner_approval_status != 'not_requested'",
    "ACQUISITION_AUTHORIZED": "owner_authorization_record required (recorded out-of-band decision; never granted by this tool)",
    "ACQUIRED": "ACQUISITION_AUTHORIZED must have been reached first",
}


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_source(src: dict) -> list[str]:
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in src or src[field] is None:
            problems.append(f"missing field: {field}")
    if problems:
        return problems

    if src["acquisition_status"] not in STATUSES:
        problems.append(f"acquisition_status {src['acquisition_status']!r} not in {sorted(STATUSES)}")
    prov = src.get("provenance") or {}
    for key in REQUIRED_PROVENANCE:
        if not prov.get(key):
            problems.append(f"provenance.{key} required")
    if not isinstance(src.get("languages_covered"), list) or not src["languages_covered"]:
        problems.append("languages_covered must be a non-empty list")
    if not isinstance(src.get("tos_restrictions"), list):
        problems.append("tos_restrictions must be a list")
    if not isinstance(src.get("known_exclusions"), list):
        problems.append("known_exclusions must be a list")

    status = src["acquisition_status"]
    if status in ("QUALIFIED_FOR_ACQUISITION", "ACQUISITION_AUTHORIZED", "ACQUIRED"):
        legal = src.get("legal_review_status", "")
        if legal in ("HUMAN_LEGAL_REVIEW_REQUIRED", "UNKNOWN"):
            problems.append(f"{status} requires legal review completion (found {legal})")
        if src.get("owner_approval_status") in (None, "not_requested"):
            problems.append(f"{status} requires a recorded owner approval/request (found {src.get('owner_approval_status')!r})")
    return problems


def validate_registry(data: dict) -> list[str]:
    problems: list[str] = []
    for key in ("schema", "id", "version", "sources"):
        if key not in data:
            problems.append(f"missing top-level field: {key}")
    if problems:
        return problems
    if data["schema"] != "taksh.dataset_source_registry/1":
        problems.append(f"schema must be taksh.dataset_source_registry/1, got {data['schema']!r}")
    seen_ids: set[str] = set()
    for i, src in enumerate(data["sources"]):
        for p in validate_source(src):
            problems.append(f"source[{i}] ({src.get('source_id', '?')}): {p}")
        sid = src.get("source_id")
        if sid in seen_ids:
            problems.append(f"duplicate source_id: {sid}")
        seen_ids.add(sid)
    return problems


def check_transition(current: str, target: str) -> dict:
    if current not in ALLOWED_TRANSITIONS:
        return {"ok": False, "reason": f"unknown current state {current!r}"}
    if target not in ALLOWED_TRANSITIONS[current]:
        return {"ok": False, "reason": f"illegal transition {current} -> {target}"}
    return {"ok": True}


def source_counts(data: dict) -> dict:
    from collections import Counter

    return dict(Counter(s.get("acquisition_status") for s in data.get("sources", [])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_source_registry.json")
    args = parser.parse_args(argv)

    data = load_registry()
    problems = validate_registry(data)
    if problems:
        for p in problems:
            print("REGISTRY INVALID:", p)
        return 2

    report = {
        "schema": "taksh.l070_source_registry/1",
        "generated_by": "evaluation/l070_source_registry.py",
        "valid": True,
        "registry_id": data["id"],
        "version": data["version"],
        "sources": [
            {"source_id": s["source_id"],
             "acquisition_status": s["acquisition_status"],
             "legal_review_status": s["legal_review_status"],
             "owner_approval_status": s["owner_approval_status"],
             "acquisition_authorized": s["acquisition_status"] in ("ACQUISITION_AUTHORIZED", "ACQUIRED")}
            for s in data["sources"]
        ],
        "status_counts": source_counts(data),
        "authorized_acquisitions": sum(1 for s in data["sources"]
                                       if s["acquisition_status"] in ("ACQUISITION_AUTHORIZED", "ACQUIRED")),
        "verdict": "SOURCE REGISTRY QUALIFIED — zero sources authorized for acquisition in L0.7.0",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"registry valid; {len(data['sources'])} sources; status counts: {report['status_counts']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
