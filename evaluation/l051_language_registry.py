"""TAKSH L0.5.1 — multilingual language registry (WP1).

Loads and validates configs/feasibility/l051_languages.yaml, emits
reports/l051_language_registry.json with per-language entries and aggregate
statistics. Validation is strict: duplicate ids, missing fields, wrong
categories, not exactly 16 L1.0 targets, or supporting technologies appearing
under `languages` all abort.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REGISTRY_PATH = REPO / "configs" / "feasibility" / "l051_languages.yaml"

REQUIRED_FIELDS = [
    "id", "name", "category", "l1_target", "extensions", "runtime",
    "ecosystem", "execution_requirements", "execution_available_here",
    "tokenizer_status", "dataset_status", "benchmark_status", "limitations",
]
ALLOWED_CATEGORIES = frozenset({"l1_target", "enterprise", "scientific", "systems", "emerging"})
SUPPORTING_CATEGORIES = frozenset({
    "markup", "style", "data_interchange", "config", "query", "iac", "build",
    "shell", "schema", "hdl",
})
EXPECTED_L1_TARGETS = frozenset({
    "python", "javascript", "typescript", "c", "cpp", "java", "csharp", "go",
    "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
})
EXPECTED_SUPPORTING = frozenset({
    "HTML", "CSS", "SCSS", "JSON", "YAML", "XML", "TOML", "GraphQL",
    "Terraform HCL", "Nix", "CMake", "Makefiles", "Gradle DSL", "PowerShell",
    "Zsh", "OpenAPI", "Protocol Buffers", "Verilog", "SystemVerilog", "VHDL",
})


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_registry(data: dict) -> dict:
    problems: list[str] = []

    langs = data.get("languages", [])
    if len(langs) != 53:
        problems.append(f"expected exactly 53 languages, found {len(langs)}")

    ids = [l.get("id") for l in langs]
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        problems.append(f"duplicate language ids: {sorted(set(dupes))}")

    for lang in langs:
        missing = [f for f in REQUIRED_FIELDS if not lang.get(f) and lang.get(f) is not False]
        if missing:
            problems.append(f"language {lang.get('id', '??')} missing fields: {missing}")
        if lang.get("category") not in ALLOWED_CATEGORIES:
            problems.append(f"language {lang.get('id')}: category {lang.get('category')!r} not allowed")
        if not isinstance(lang.get("extensions"), list) or not lang["extensions"]:
            problems.append(f"language {lang.get('id')}: extensions missing or empty")

    l1 = [l for l in langs if l.get("l1_target") is True]
    l1_ids = frozenset(l["id"] for l in l1)
    if len(l1) != 16:
        problems.append(f"expected exactly 16 L1.0 targets, found {len(l1)}: {sorted(l1_ids)}")
    if l1_ids != EXPECTED_L1_TARGETS:
        problems.append(
            f"L1.0 target set mismatch: missing={sorted(EXPECTED_L1_TARGETS - l1_ids)} "
            f"extra={sorted(l1_ids - EXPECTED_L1_TARGETS)}"
        )

    for l in l1:
        if l["category"] not in ("l1_target",):
            problems.append(f"L1.0 target {l['id']} has non-l1_target category {l['category']!r}")

    supporting = data.get("supporting_technologies", [])
    supporting_names = [s.get("name") for s in supporting]
    if len(set(supporting_names)) != len(supporting_names):
        problems.append("duplicate supporting_technologies names")
    missing_st = EXPECTED_SUPPORTING - set(supporting_names)
    if missing_st:
        problems.append(f"missing supporting technologies: {sorted(missing_st)}")
    lang_names_lower = {l.get("id") or l.get("name") for l in langs}
    for s in supporting:
        if (s.get("name") or "").lower().replace(" ", "_") in lang_names_lower:
            problems.append(f"supporting technology {s.get('name')!r} also appears as a language")
        if s.get("category") not in SUPPORTING_CATEGORIES:
            problems.append(f"supporting {s.get('name')!r}: category {s.get('category')!r} not allowed")

    if problems:
        return {"valid": False, "problems": problems}
    return {"valid": True, "problems": []}


def summarize(data: dict) -> dict:
    langs = data["languages"]
    avail = Counter(str(l.get("execution_available_here")) for l in langs)
    ds = Counter(l.get("dataset_status") for l in langs)
    bench = Counter(l.get("benchmark_status") for l in langs)
    tok = Counter(l.get("tokenizer_status") for l in langs)
    cat = Counter(l.get("category") for l in langs)
    return {
        "categories": dict(cat),
        "execution_available_here": dict(avail),
        "dataset_status": dict(ds),
        "benchmark_status": dict(bench),
        "tokenizer_status": dict(tok),
        "l1_target_count": sum(1 for l in langs if l.get("l1_target")),
        "l1_target_ids": sorted(l["id"] for l in langs if l.get("l1_target")),
        "supporting_technologies": [s["name"] for s in data.get("supporting_technologies", [])],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_language_registry.json")
    args = parser.parse_args(argv)

    data = load_registry()
    result = validate_registry(data)
    if not result["valid"]:
        for problem in result["problems"]:
            print("REGISTRY INVALID:", problem)
        return 2

    summary = summarize(data)
    report = {
        "schema": "taksh.l051_language_registry/1",
        "generated_by": "evaluation/l051_language_registry.py",
        "valid": True,
        "language_count": 53,
        "l1_target_count": summary["l1_target_count"],
        "l1_target_ids": summary["l1_target_ids"],
        "categories": summary["categories"],
        "execution_available_here": summary["execution_available_here"],
        "dataset_status": summary["dataset_status"],
        "benchmark_status": summary["benchmark_status"],
        "tokenizer_status": summary["tokenizer_status"],
        "supporting_technologies": summary["supporting_technologies"],
        "languages": data["languages"],
        "supporting_technologies_detail": data["supporting_technologies"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"registry valid: 53 languages, {summary['l1_target_count']} L1.0 targets, "
          f"{len(summary['supporting_technologies'])} supporting technologies")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
