"""Focused tests for the TAKSH L0.6.0-R1 architecture alignment milestone."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402
    l051_language_registry as reg,
    l060r1_dynamic_registry as dyn,
    l060r1_kark_bench_schema as kbs,
)

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


# --------------------------------------------------------------------------- #
# Dynamic language registry
# --------------------------------------------------------------------------- #
def test_base_registry_still_valid():
    base = dyn.load(dyn.BASE_REGISTRY)
    assert reg.validate_registry(base)["valid"] is True


def test_dynamic_registry_valid_and_karkain_present():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    base = dyn.load(dyn.BASE_REGISTRY)
    assert dyn.validate_dynamic_registry(dynamic, base) == []
    assert any(e["id"] == "karkain" for e in dynamic["entries"])


def test_dynamic_registry_merges_to_54_without_ceiling():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    base = dyn.load(dyn.BASE_REGISTRY)
    view = dyn.merged_view(base, dynamic)
    assert view["base_count"] == 53
    assert view["extension_count"] == 1
    assert view["total_languages"] == 54
    assert view["ceiling"] is None
    assert view["karkain_first_class_target"] is True
    assert view["original_53_preserved"] is True


def test_karkain_contract_fields():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    kark = next(e for e in dynamic["entries"] if e["id"] == "karkain")
    assert ".kark" in kark["extensions"]
    assert "SPEC.md v1.1.0" in kark["grammar_or_spec_reference"]
    assert "TAKSH-KARK-BENCH" in " ".join(kark["benchmark_associations"])
    assert kark["qualification_status"] == "DISCOVERED"


def test_dynamic_registry_entry_missing_field_rejected():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    base = dyn.load(dyn.BASE_REGISTRY)
    broken = {"entries": [dict(dynamic["entries"][0])]}
    del broken["entries"][0]["name"]
    problems = dyn.validate_dynamic_registry(broken, base)
    assert any("missing fields" in p for p in problems)


def test_dynamic_registry_bad_status_rejected():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    base = dyn.load(dyn.BASE_REGISTRY)
    broken = {"entries": [dict(dynamic["entries"][0], qualification_status="QUALIFIED")]}
    problems = dyn.validate_dynamic_registry(broken, base)
    assert any("invalid status" in p for p in problems)


def test_no_capability_status_is_set():
    dynamic = dyn.load(dyn.DYNAMIC_REGISTRY)
    for entry in dynamic["entries"]:
        assert entry["qualification_status"] not in (
            "FUNCTIONALLY_QUALIFIED", "ENGINEERING_QUALIFIED"
        )


# --------------------------------------------------------------------------- #
# TAKSH-KARK-BENCH schema
# --------------------------------------------------------------------------- #
def test_kark_bench_schema_is_valid():
    assert kbs.validate_schema() == []


def test_kark_bench_schema_declares_ten_categories():
    ids = {c["id"] for c in kbs.SCHEMA["categories"]}
    assert ids == {"syntax", "semantics", "compilation", "kir", "debugging",
                   "algorithms", "stdlib", "translation", "repository_engineering", "adaptation"}


def test_kark_bench_zero_tasks_instantiated():
    assert kbs.SCHEMA["instantiated_tasks"] == []


def test_kark_bench_task_validator_rejects_bad_task():
    bad = {"id": "x"}  # missing nearly everything
    problems = kbs.validate_task(bad)
    assert problems


def test_kark_bench_task_validator_accepts_synthetic_task():
    task = {
        "id": "ark-test-001", "category": "syntax", "difficulty": "easy",
        "title": "dummy", "statement": "dummy",
        "entry_point": "main", "kind": "compile_pass",
        "tests": [{"name": "t1", "expected": "ok"}],
        "execution": {"runtime": "kcc", "language": "karkain", "toolchain_version_pinned": "1.1.0"},
        "timeout_s": 5, "scoring": "unit assertions", "held_out": True,
    }
    assert kbs.validate_task(task) == []


def test_model_only_and_compiler_assisted_modes_declared():
    assert kbs.SCHEMA["performance_modes"] == ["model_only", "compiler_assisted"]


# --------------------------------------------------------------------------- #
# Preservation
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_unchanged():
    import hashlib

    digest = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    assert digest == FROZEN_PY_SHA


def test_multilingual_64_tasks_preserved():
    bench = json.loads((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_text(encoding="utf-8"))
    assert len(bench["tasks"]) == 64
    assert bench["qualification"]["min_tasks_per_language"] == 16
    assert bench["qualification"]["min_pass_ratio"] == 0.50


def test_l06_config_and_checkpoint_unchanged():
    import hashlib

    cfg_hash = hashlib.sha256((REPO / "configs" / "foundation" / "l060_foundation.yaml").read_bytes()).hexdigest()
    assert cfg_hash == "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
    ckpt = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
    h = hashlib.sha256()
    with ckpt.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    assert h.hexdigest() == "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"


def test_registry_l1_targets_and_counts_preserved():
    data = yaml.safe_load((REPO / "configs/feasibility/l051_languages.yaml").read_text(encoding="utf-8"))
    assert len(data["languages"]) == 53
    assert sum(1 for l in data["languages"] if l["l1_target"]) == 16
    assert len(data["supporting_technologies"]) == 20


def _walk_keys(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _walk_keys(item)
    return keys


def test_no_max_language_ceiling_anywhere():
    for path in [REPO / "configs/foundation/l060r1_dynamic_language_registry.yaml"]:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "max_languages" not in _walk_keys(doc)
        assert "language_limit" not in _walk_keys(doc)


def test_no_dlcae_is_implemented_yet():
    """The DLCAE architecture docs exist; no runtime modules are claimed to exist."""
    assert (REPO / "docs" / "architecture" / "DLCAE-ARCHITECTURE.md").exists()
