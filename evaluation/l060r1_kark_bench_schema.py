"""TAKSH L0.6.0-R1 — TAKSH-KARK-BENCH schema validator (WP5).

Defines the dedicated Karkain benchmark contract covering: syntax, semantics,
compilation, KIR, debugging, algorithms, standard-library usage, translation,
repository engineering, adaptation-to-new-features. Pins every result to an
explicit Karkain compiler and specification version. Separates raw-model
performance from compiler-assisted system performance.

The schema is a contract; no Karkain tasks are instantiated in this milestone
(execution would require a working KCC on the host, which is unavailable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCHEMA = {
    "schema": "taksh.kark_bench_schema/1",
    "name": "TAKSH-KARK-BENCH",
    "version": "1.0.0",
    "compiler_and_spec_pinning": {
        "required": True,
        "pins": {
            "karkain_compiler_version": "1.1.0 (windows/amd64, Stable Build; commit e612ede branch feature/v1.2.0-native-independence)",
            "karkain_spec_version": "1.1.0",
            "kcc_engine": "go reference compiler (pkg/lexer|parser|sema|codegen|ir/ssa)",
            "note": "every score must reference the exact compiler build it was produced with; scores without a pin are invalid",
        },
    },
    "performance_modes": ["model_only", "compiler_assisted"],
    "mode_semantics": {
        "model_only": "raw model output scored without any compiler or execution assistance; this is the model capability signal",
        "compiler_assisted": "model output compiled by KCC and scored against tests/execution/diagnostics; this is the system capability signal; never reported as model capability",
    },
    "categories": [
        {"id": "syntax", "description": "lexing/parsing conformance: tokens, declarations, control flow, functions, structs, enums, interfaces, generics, error syntax"},
        {"id": "semantics", "description": "name resolution, type checking, ownership/borrow rules, move semantics, generic constraint resolution"},
        {"id": "compilation", "description": "end-to-end compile of the program; diagnostics emitted, no false positives, error-code correctness (K000-K999)"},
        {"id": "kir", "description": "KIR v1 emission, structure verification (SSA dominance, phi placement), semantic preservation across lowering"},
        {"id": "debugging", "description": "error localization: given broken code, produce the corrected version or correctly identify the defect class"},
        {"id": "algorithms", "description": "algorithms and data structures: loops, conditionals, arrays/maps/structs, sorting/searching, recursion"},
        {"id": "stdlib", "description": "standard-library usage: string, collections, math, io, encoding, testing, generics"},
        {"id": "translation", "description": "cross-language translation involving Karkain: other language -> Karkain and Karkain -> other language"},
        {"id": "repository_engineering", "description": "multi-file tasks: imports, modules, build files, test discovery, small project evolution"},
        {"id": "adaptation", "description": "adaptation to new Karkain features: given a new or changed construct, correctly comprehend and use it"},
    ],
    "task_contract": {
        "required_fields": [
            "id", "category", "difficulty", "title", "statement",
            "entry_point", "kind", "tests", "execution", "timeout_s",
            "scoring", "held_out",
        ],
        "difficulty": ["easy", "medium", "hard"],
        "kind": ["function", "stdio", "sql_query", "compile_pass", "compile_fail", "kir_inspect", "debug_repair"],
        "execution": ["runtime", "language", "toolchain_version_pinned"],
        "timeout_s": "integer > 0",
        "scoring": "unit assertions, equal weight per assertion",
        "held_out": True,
        "contamination_rules": [
            "no Karkain benchmark task may appear in any training corpus without explicit owner sign-off",
            "Karkain conformance corpus (conformance/*.kark) may be used ONLY as references for schema validation, never as training targets or answer keys",
        ],
    },
    "qualification_bar": {
        "model_only": "NOT DEFINED YET — bar must be set before evaluation, never after results are seen",
        "compiler_assisted": "NOT DEFINED YET — bar must be set before evaluation, never after results are seen",
        "rule": "no threshold may be lowered after observing results; any qualification requires FUNCTIONALLY_QUALIFIED or ENGINEERING_QUALIFIED status in the dynamic registry",
    },
    "versioning": {
        "bench_schema_version": "1.0.0",
        "frozen": "alteration requires explicit owner sign-off and a new version; older results remain valid against the schema version they were collected under",
    },
    "instantiated_tasks": [],
    "instantiation_status": "0 tasks instantiated; schema defined and validated; samples require a healthy KCC on the host (unavailable on this machine: see local_toolchain_limitations in the dynamic registry)",
}

VALID_KIND = frozenset(SCHEMA["task_contract"]["kind"])
VALID_DIFFICULTY = frozenset(SCHEMA["task_contract"]["difficulty"])
CATEGORY_IDS = {c["id"] for c in SCHEMA["categories"]}


def validate_schema() -> list[str]:
    problems: list[str] = []
    cc = SCHEMA["compiler_and_spec_pinning"]
    if not cc.get("required") or not cc.get("pins", {}).get("karkain_compiler_version"):
        problems.append("compiler/spec pinning missing")
    if SCHEMA["performance_modes"] != ["model_only", "compiler_assisted"]:
        problems.append("performance_modes must list model_only and compiler_assisted, model_only first")
    for c in SCHEMA["categories"]:
        for field in ("id", "description"):
            if not c.get(field):
                problems.append(f"category missing {field}: {c}")
    for field in ("required_fields", "difficulty", "kind", "scoring", "contamination_rules"):
        if not SCHEMA["task_contract"].get(field):
            problems.append(f"task_contract missing {field}")
    if SCHEMA["instantiated_tasks"] != []:
        problems.append("instantiated_tasks must be empty until a healthy KCC is available")
    return problems


def validate_task(task: dict) -> list[str]:
    problems: list[str] = []
    for field in SCHEMA["task_contract"]["required_fields"]:
        if field not in task or task[field] in (None, "", []):
            problems.append(f"task {task.get('id', '??')} missing field: {field}")
    if task.get("category") not in CATEGORY_IDS:
        problems.append(f"task {task.get('id', '??')}: bad category {task.get('category')!r}")
    if task.get("difficulty") not in VALID_DIFFICULTY:
        problems.append(f"task {task.get('id', '??')}: bad difficulty {task.get('difficulty')!r}")
    if task.get("kind") not in VALID_KIND:
        problems.append(f"task {task.get('id', '??')}: bad kind {task.get('kind')!r}")
    if not isinstance(task.get("timeout_s"), int) or task["timeout_s"] <= 0:
        problems.append(f"task {task.get('id', '??')}: timeout_s must be a positive int")
    if task.get("held_out") is not True:
        problems.append(f"task {task.get('id', '??')}: held_out must be true")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l060r1_kark_bench_schema.json")
    args = parser.parse_args(argv)

    problems = validate_schema()
    if problems:
        for p in problems:
            print("KARK-BENCH SCHEMA INVALID:", p)
        return 2

    report = {
        "schema": "taksh.l060r1_kark_bench_schema/1",
        "generated_by": "evaluation/l060r1_kark_bench_schema.py",
        "schema_valid": True,
        "definition": SCHEMA,
        "declared_categories": [c["id"] for c in SCHEMA["categories"]],
        "instantiated_tasks": 0,
        "instantiation_status": SCHEMA["instantiation_status"],
        "conclusions": [
            "schema is complete and self-consistent; zero tasks instantiated because KCC is unavailable on this host",
            "no performance claim of any kind is made for Karkain capability in TAKSH",
            "the schema pins every future result to an explicit compiler + spec version",
            "model_only and compiler_assisted scores are always reported separately",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Karkain benchmark schema valid; 0 tasks instantiated (KCC unavailable on this host)")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
