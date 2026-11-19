"""Focused tests for TAKSH L0.7.6 — master roadmap and release governance."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.roadmap import roadmap_validator  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_CFG_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
L051_SHA = "075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d"

REGISTRY = REPO / "configs" / "roadmap" / "taksh-v3-roadmap.yaml"
ROADMAP_MD = REPO / "docs" / "TAKSH-MASTER-ROADMAP-v3.0.0.md"
GOVERNANCE_MD = REPO / "docs" / "TAKSH-RELEASE-GOVERNANCE.md"
CHANGELOG_MD = REPO / "docs" / "TAKSH-ROADMAP-CHANGELOG.md"


def load():
    return roadmap_validator.load_registry(REGISTRY)


# --------------------------------------------------------------------------- #
# Registry structure
# --------------------------------------------------------------------------- #
def test_registry_valid_and_complete():
    data = load()
    assert roadmap_validator.validate_registry(data) == []
    assert len(data["milestones"]) == 54
    assert len(data["releases"]) == 4


def test_milestone_ids_unique():
    data = load()
    ids = [m["id"] for m in data["milestones"]]
    assert len(set(ids)) == len(ids)


def test_status_vocabulary_enforced():
    data = load()
    statuses = {m["status"] for m in data["milestones"]}
    assert statuses <= roadmap_validator.VALID_STATUSES


def test_every_release_covers_only_known_stages():
    data = load()
    milestone_stages = {m["id"]: m["stage"] for m in data["milestones"]}
    for r in data["releases"]:
        assert r["stage"] in {"L0", "L1", "L2", "L3"}
        for mid in r["milestones"]:
            assert milestone_stages[mid] == r["stage"]


def test_all_milestones_mapped_to_a_release():
    data = load()
    mapped = {mid for r in data["releases"] for mid in r["milestones"]}
    assert mapped == {m["id"] for m in data["milestones"]}


def test_dependencies_exist_and_acyclic():
    data = load()
    id_set = {m["id"] for m in data["milestones"]}
    for m in data["milestones"]:
        assert set(m["dependencies"]) <= id_set
    assert roadmap_validator.validate_registry(data) == []


def test_qualified_milestones_carry_evidence():
    data = load()
    for m in data["milestones"]:
        if m["status"] in ("QUALIFIED", "CONDITIONAL_PASS"):
            assert m["evidence"]
            for ev in m["evidence"]:
                assert (REPO / ev).exists(), ev


def test_current_baseline_statuses_honest():
    data = load()
    by_id = {m["id"]: m for m in data["milestones"]}
    assert by_id["L0.7.5"]["status"] == "BLOCKED"
    assert by_id["L0.7.2"]["status"] == "BLOCKED"
    assert by_id["L0.8.0"]["status"] == "BLOCKED"
    assert by_id["L0.7.6"]["status"] == "IN_PROGRESS"
    assert by_id["L0.6.2"]["status"] == "CONDITIONAL_PASS"
    assert by_id["L1.0.0"]["status"] == "NOT_STARTED"
    assert by_id["L2.10"]["status"] == "NOT_STARTED"
    assert by_id["L3.7"]["status"] == "NOT_STARTED"


def test_roadmap_validator_catches_unknown_status():
    data = copy.deepcopy(load())
    data["milestones"][0]["status"] = "DONE"
    assert any("invalid status" in p for p in roadmap_validator.validate_registry(data))


def test_roadmap_validator_catches_missing_dependency():
    data = copy.deepcopy(load())
    data["milestones"][0]["dependencies"] = ["L9.99"]
    assert any("does not exist" in p for p in roadmap_validator.validate_registry(data))


def test_roadmap_validator_catches_duplicate_ids():
    data = copy.deepcopy(load())
    data["milestones"].append(copy.deepcopy(data["milestones"][0]))
    assert any("duplicate milestone ids" in p for p in roadmap_validator.validate_registry(data))


def test_roadmap_validator_catches_cycle():
    data = copy.deepcopy(load())
    by_id = {m["id"]: m for m in data["milestones"]}
    by_id["L0.6.0"]["dependencies"] = ["L0.7.2"]
    by_id["L0.7.2"]["dependencies"] = ["L0.6.0"]
    assert any("cycle" in p for p in roadmap_validator.validate_registry(data))


def test_roadmap_validator_catches_release_milestone_mismatch():
    data = copy.deepcopy(load())
    v3 = next(r for r in data["releases"] if r["id"] == "v3.0.0")
    by_id = {m["id"]: m for m in data["milestones"]}
    v3["milestones"].append("L2.0")
    assert any("expects stage" in p for p in roadmap_validator.validate_registry(data))


# --------------------------------------------------------------------------- #
# Roadmap markdown and governance documents
# --------------------------------------------------------------------------- #
def test_roadmap_markdown_exists_and_lists_releases():
    text = ROADMAP_MD.read_text(encoding="utf-8")
    for rel in ("v1.0.0", "v2.0.0", "v3.0.0", "L0.8.0", "L3.7", "L1.0.0", "L2.10"):
        assert rel in text


def test_roadmap_markdown_states_no_capability_claim():
    text = ROADMAP_MD.read_text(encoding="utf-8").lower()
    assert "0/30" in text
    assert "not a proven" in text or "claims" in text


def test_governance_defines_status_vocabulary_and_boundaries():
    text = GOVERNANCE_MD.read_text(encoding="utf-8")
    for token in ("NOT_STARTED", "QUALIFIED", "GA", "DEFERRED",
                  "authorization boundaries", "Release-decision rule", "Frozen benchmarks"):
        assert token.lower() in text.lower()


def test_changelog_records_this_revision():
    text = CHANGELOG_MD.read_text(encoding="utf-8")
    assert "2026-11-11" in text
    assert "L0.7.6" in text
    assert "roadmap authority" in text.lower() or "forward planning" in text.lower()


def test_roadmap_deliverables_exist():
    for path in (ROADMAP_MD, GOVERNANCE_MD, CHANGELOG_MD, REGISTRY):
        assert path.exists(), path


def test_roadmap_validator_cli_exits_zero(tmp_path):
    out = tmp_path / "validation.json"
    rc = roadmap_validator.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["milestone_count"] == 54


def test_validator_fails_without_roadmap_md(monkeypatch, tmp_path):
    monkeypatch.setattr(roadmap_validator, "ROADMAP_MD", tmp_path / "nonexistent.md")
    assert any("roadmap markdown missing" in p for p in roadmap_validator.validate_registry(load()))


# --------------------------------------------------------------------------- #
# Preservation anchors (unchanged by this milestone)
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_checkpoint_preserved():
    digest = hashlib.sha256()
    with (REPO / "runs" / "v180_sft" / "checkpoint_last.pt").open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA


def test_foundation_config_preserved():
    assert hashlib.sha256((REPO / "configs" / "foundation" / "l060_foundation.yaml").read_bytes()).hexdigest() == FOUNDATION_CFG_SHA


def test_registry_53_languages_preserved():
    assert hashlib.sha256((REPO / "configs" / "feasibility" / "l051_languages.yaml").read_bytes()).hexdigest() == L051_SHA


def test_owner_decision_record_contains_only_authorized_decisions():
    """The L0.7.6 test asserted a blank record; L0.8.0-PRE recorded A1.1-C and A2-E only."""
    text = (REPO / "docs" / "L075_L080_OWNER_DECISION_RECORD.md").read_text(encoding="utf-8")
    checked = [line for line in text.splitlines() if "- [x]" in line]
    assert {line.strip() for line in checked} == {
        "- [x] **A1.1-C \u2014 Authorize preparation and preflight only; no training.**",
        "- [x] **A2-E \u2014 No compute authorization yet.**",
    }
    for line in text.splitlines():
        assert not any(s in line for s in ("- [x] **A1.1-A", "- [x] **A1.1-B", "- [x] **A2-A", "- [x] **A2-B", "- [x] **A2-C", "- [x] **A2-D"))
    assert "TAKSH-OWNER-DECISION-2026-11-11" in text


def test_karkain_remains_undiscovered_for_training():
    import yaml as _y

    strategy = _y.safe_load((REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"
    assert strategy["identity"]["spec_version_pinned"] == "1.1.0"
