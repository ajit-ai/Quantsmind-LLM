"""TAKSH L0.7.2 — multilingual + Karkain coverage (freeze-aware).

Reports the real per-language state of the frozen corpus: python (derived from
the L0.7.2 snapshot) plus zero rows for everything else — never invented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l051_language_registry import load_registry  # noqa: E402

ARTIFACT = REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json"
RECORDS = REPO / "artifacts" / "l072_l04_freeze" / "records_meta.jsonl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l072_coverage.json")
    args = parser.parse_args(argv)

    registry = load_registry()
    snapshot = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    eligible = [r for r in real if not r.get("exclusion_reason")]
    eligible_py_tokens = sum(r["token_count"] for r in eligible if r["language"] == "python")

    entries = []
    for lang in registry["languages"]:
        lid = lang["id"]
        is_python = lid == "python"
        entries.append({
            "registry_id": lid,
            "name": lang["name"],
            "family": lang["category"],
            "l1_target": bool(lang.get("l1_target")),
            "registry_status": "REGISTERED",
            "corpus_samples": len(eligible) if is_python else 0,
            "corpus_tokens": eligible_py_tokens if is_python else 0,
            "quality_qualified_samples": len(eligible) if is_python else 0,
            "technically_eligible_samples": len(eligible) if is_python else 0,
            "training_eligible_samples": 0,
            "source_coverage": "internal-l04 (authorized, internal research only)" if is_python else "none qualified",
            "toolchain_available_here": lang.get("execution_available_here"),
            "blockers": ([] if is_python else ["no acquisition authorization; no corpus"]),
            "labels": "REAL_AUTHORIZED_CORPUS" if is_python else "UNVERIFIED_SOURCE",
        })
    entries.append({
        "registry_id": "karkain",
        "name": "Karkain",
        "family": "first-class extension (54th language)",
        "l1_target": False,
        "registry_status": "DISCOVERED",
        "corpus_samples": 0,
        "corpus_tokens": 0,
        "quality_qualified_samples": 0,
        "technically_eligible_samples": 0,
        "training_eligible_samples": 0,
        "source_coverage": "none — zero corpus; .kark pinned (SPEC 1.1.0 / compiler 1.1.0); kcc 0xC0000135",
        "toolchain_available_here": False,
        "blockers": ["kcc.exe 0xC0000135; exit-3 pipeline; owner authorization absent"],
        "labels": "BLOCKED_SOURCE",
    })

    qualifying_families = {}
    for e in entries:
        qualifying_families.setdefault(e["family"], []).append(e["registry_id"])

    report = {
        "schema": "taksh.l072_coverage/1",
        "generated_by": "evaluation/l072_coverage.py",
        "total_languages": len(entries),
        "languages_with_real_data": ["python"],
        "languages_with_zero_corpus": [e["registry_id"] for e in entries if e["corpus_tokens"] == 0],
        "entries": entries,
        "groups": qualifying_families,
        "synthetic_fixture_tokens_python": 16,
        "python_eligible_tokens": eligible_py_tokens,
        "python_snapshot": snapshot["snapshot_id"],
        "verdict": "PYTHON-ONLY REAL CORPUS; 52 + Karkain have ZERO qualified tokens; no multilingual claims",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"coverage: python {eligible_py_tokens:,} tokens / {len(eligible)} samples; 53 other languages + karkain at zero")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
