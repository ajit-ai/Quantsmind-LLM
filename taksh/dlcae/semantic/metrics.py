"""Semantic-validation measurements for the A/B/C experiment framework.

Fixtures carry synthetic, known ground truth — never frozen benchmark tasks.
Measurements returned are engineering telemetry; no comparative winner is
declared here and no capability claim is produced.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

from taksh.dlcae.semantic.dependencies import analyze_python_dependencies
from taksh.dlcae.semantic.symbols import SymbolAnalysisEngine
from taksh.dlcae.semantic.typesys import TypeAnalysisEngine, types_compatible

# --------------------------------------------------------------------------- #
# Synthetic fixtures with known ground truth
# --------------------------------------------------------------------------- #
SYMBOL_FIXTURE = """
import os

CONSTANT = 42

def helper(x, y: int) -> int:
    return x + y

class Widget:
    def render(self, ctx: dict) -> str:
        return str(ctx)

class Widget2(Widget):
    pass

_UNUSED = 1
"""

SYMBOL_GROUND_TRUTH = {
    "function_decls": {"helper", "render"},
    "class_decls": {"Widget", "Widget2"},
    "module_imports": {"os"},
    "exported_functions_or_classes": {"helper", "Widget", "Widget2"},
    "private_excluded": {"_UNUSED"},
    "known_builtins_only": True,
}

UNRESOLVED_FIXTURE = "def f():\n    return nonexistent_symbol_to_detect\n"
EXPECTED_UNRESOLVED = {"nonexistent_symbol_to_detect"}

SHADOW_FIXTURE = "x = 1\ndef f():\n    x = 2\n    return x\n"
EXPECTED_SHADOW_COUNT = 1

DUPLICATE_FIXTURE = "def f():\n    return 1\ndef f():\n    return 2\n"
EXPECTED_DUPLICATE = "f"

TYPE_FIXTURE = (
    "def annotate(a: int, b: str) -> str:\n"
    "    c: int = a\n"
    "    return b\n"
)
EXPECTED_TYPE_KINDS = {"signature", "parameter", "return", "variable"}

CYCLE_SOURCES = {
    "cyc_a": "import cyc_b\nVALUE_A = 1\n",
    "cyc_b": "import cyc_c\nVALUE_B = 1\n",
    "cyc_c": "import cyc_a\nVALUE_C = 1\n",
}

MISSING_SOURCES = {
    "needs_missing": "import definitely_not_installed_pkg_xyz123\n",
}


def evaluate_symbol_detection() -> dict[str, Any]:
    """Precision/recall over the synthetic fixture's known ground truth."""
    t0 = time.perf_counter()
    engine = SymbolAnalysisEngine()
    analysis = engine.analyze(SYMBOL_FIXTURE, language_id="python",
                              file="fixture_symbols.py", module="fixture_symbols")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    symbols = analysis.symbols
    fn = {s["name"] for s in symbols if s["kind"] == "function"}
    classes = {s["name"] for s in symbols if s["kind"] == "class"}
    imports = {s["name"] for s in symbols if s["kind"] == "import"}
    exported = set(analysis.exports)
    truth_fn, truth_cls = SYMBOL_GROUND_TRUTH["function_decls"], SYMBOL_GROUND_TRUTH["class_decls"]
    tp = len((fn & truth_fn) | (classes & truth_cls))
    detected = fn | classes
    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(truth_fn | truth_cls)
    unresolved = evaluate_unresolved()
    shadows = evaluate_shadowing()
    duplicates = evaluate_duplicates()
    return {
        "schema": "taksh.dlcae.semantic_symbol_metrics/1",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": tp,
        "detected": sorted(detected),
        "imports_detected": sorted(imports & SYMBOL_GROUND_TRUTH["module_imports"]),
        "exports": sorted(exported),
        "unresolved": unresolved,
        "shadowing": shadows,
        "duplicates": duplicates,
        "duration_ms": elapsed_ms,
        "truth_source": "synthetic fixture (never frozen benchmarks)",
    }


def evaluate_unresolved() -> dict[str, Any]:
    analysis = SymbolAnalysisEngine().analyze(UNRESOLVED_FIXTURE, language_id="python",
                                              file="fixture_unresolved.py",
                                              module="fixture_unresolved")
    found = {u["name"] for u in analysis.unresolved}
    expected = EXPECTED_UNRESOLVED
    return {"expected": sorted(expected), "found": sorted(found),
            "all_expected_detected": expected <= found,
            "precision": round(len(found & expected) / len(found), 4) if found else 0.0}


def evaluate_shadowing() -> dict[str, Any]:
    analysis = SymbolAnalysisEngine().analyze(SHADOW_FIXTURE, language_id="python",
                                              file="fixture_shadow.py",
                                              module="fixture_shadow")
    return {"expected_shadow_count": EXPECTED_SHADOW_COUNT,
            "shadowing": analysis.shadowing,
            "shadow_detected": len(analysis.shadowing) >= EXPECTED_SHADOW_COUNT}


def evaluate_duplicates() -> dict[str, Any]:
    analysis = SymbolAnalysisEngine().analyze(DUPLICATE_FIXTURE, language_id="python",
                                              file="fixture_dup.py",
                                              module="fixture_dup")
    detected = {d["qualified_name"].rsplit(".", 1)[-1] for d in analysis.duplicates}
    return {"expected": EXPECTED_DUPLICATE, "detected": sorted(detected),
            "duplicate_detected": EXPECTED_DUPLICATE in detected}


def evaluate_type_relations() -> dict[str, Any]:
    analysis = TypeAnalysisEngine().analyze(TYPE_FIXTURE, language_id="python",
                                            file="fixture_types.py",
                                            module="fixture_types")
    kinds = {r["kind"] for r in analysis.relations}
    parameters = [r for r in analysis.relations if r["kind"] == "parameter"]
    unknown_or_inferred = [r for r in analysis.relations
                           if r.get("explicit_unknown") and r["kind"] == "variable"]
    return {
        "schema": "taksh.dlcae.semantic_type_metrics/1",
        "relations_found": len(analysis.relations),
        "kinds_found": sorted(kinds),
        "expected_kinds_present": EXPECTED_TYPE_KINDS <= kinds,
        "parameter_annotations": [{"name": p["name"], "type": p.get("type")}
                                  for p in parameters if p["name"].startswith("annotate")],
        "unannotated_count": len([r for r in analysis.relations
                                  if r.get("explicit_unknown")]),
        "discipline": analysis.discipline,
    }


def evaluate_dependency_graph() -> dict[str, Any]:
    cycle_graph = analyze_python_dependencies(CYCLE_SOURCES)
    missing_graph = analyze_python_dependencies(MISSING_SOURCES)
    return {
        "schema": "taksh.dlcae.semantic_dependency_metrics/1",
        "cycle_sources": sorted(CYCLE_SOURCES),
        "cycles_detected": cycle_graph.cycles,
        "cycle_detected": bool(cycle_graph.cycles),
        "missing_detector": [m["module"] for m in missing_graph.missing],
        "missing_dependency_detected": any(
            m["module"].startswith("definitely_not_installed") for m in missing_graph.missing),
        "installed_stdlib_recognized": any(e["to"] == "json" for e in
                                           analyze_python_dependencies({"uses_json": "import json\n"}).edges),
    }


def evaluate_reproducibility() -> dict[str, Any]:
    first = SymbolAnalysisEngine().analyze(SYMBOL_FIXTURE, language_id="python",
                                           file="repro.py", module="repro")
    second = SymbolAnalysisEngine().analyze(SYMBOL_FIXTURE, language_id="python",
                                            file="repro.py", module="repro")
    from taksh.dlcae.types import fingerprint

    f1 = fingerprint(first.to_dict())
    f2 = fingerprint(second.to_dict())
    return {"schema": "taksh.dlcae.reproducibility/1",
            "fingerprint_a": f1, "fingerprint_b": f2,
            "byte_identical": f1 == f2}


def evaluate_memory() -> dict[str, Any]:
    tracemalloc.start()
    SymbolAnalysisEngine().analyze(SYMBOL_FIXTURE, language_id="python",
                                   file="mem.py", module="mem")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"schema": "taksh.dlcae.memory_trace/1",
            "peak_bytes": peak, "tool": "tracemalloc"}


def evaluate_compatibility() -> dict[str, Any]:
    """types_compatible observations — matrix rows recorded, none as verdicts."""
    cases = [("int", "float"), ("bool", "int"), ("str", "int"),
             (None, "int"), ("int", None), ("Any", "str"), ("int", "int")]
    observations = {f"{a}->{b}": types_compatible(a, b)["observation"] for a, b in cases}
    return {"schema": "taksh.dlcae.type_compatibility/1",
            "observations": observations,
            "unknown_discipline_preserved": observations["None->int"] == "UNVERIFIED"}
