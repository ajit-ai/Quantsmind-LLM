import hashlib

import pytest

from taksh.architect import (
    ArchitectArtifacts,
    ArchitectContract,
    ArchitectTaskKind,
    Adr,
    Diagram,
    Risk,
)
from taksh.architect.render import render_architect_prompt
from taksh.architect.validate import validate_artifacts
from taksh.architect.review import review_design


def request_hash(seed: str = "request") -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def make_artifacts() -> ArchitectArtifacts:
    return ArchitectArtifacts(
        summary="Local queue worker with replay and rollback support.",
        source_request_sha256=request_hash(),
        assumptions=["delivery at least once"],
        decisions=[
            Adr(
                id="ADR-001",
                title="Append-only queue",
                status="proposed",
                decision="Use an append-only queue",
                rationale="Replay and audit are mandatory",
                consequences=["storage growth"],
            )
        ],
        diagrams=[
            Diagram(kind="c4", type="C4Container", source="C4Container\nSystem_Boundary(a, \"API\")")
        ],
        api_specs=[{"openapi": "3.0.3", "paths": {}, "info": {"title": "worker"}}],
        data_models=[{"name": "Job", "depends_on": ["Queue"]}],
        migration={"rollback": "reprovision previous revision", "dependencies": []},
        risks=[Risk(category="availability", severity="HIGH", likelihood="MEDIUM", mitigation="rollback and alerts")],
    )


def test_contract_fingerprint_is_stable_and_content_derived():
    contract = ArchitectContract(requirements=[{"id": "R1", "text": "retain audit logs"}])
    other = ArchitectContract(requirements=[{"id": "R1", "text": "retain audit logs"}])
    changed = ArchitectContract(requirements=[{"id": "R1", "text": "different"}])
    assert contract.fingerprint == other.fingerprint
    assert contract.fingerprint != changed.fingerprint


def test_contract_rejects_invalid_review_level_and_unknown_task_kind():
    with pytest.raises(ValueError):
        ArchitectContract(review_level="maybe")
    with pytest.raises(ValueError):
        ArchitectContract(task_kind="not-a-kind")


def test_contract_roundtrip_preserves_task_kind_and_requested_artifacts():
    contract = ArchitectContract(task_kind=ArchitectTaskKind.MIGRATION)
    rebuilt = ArchitectContract.from_dict(contract.to_dict())
    assert rebuilt == contract
    assert rebuilt.requested_artifacts


def test_render_prompt_is_deterministic_and_contains_rules():
    contract = ArchitectContract(
        requirements=[{"id": "R1", "text": "must retry failed jobs"}],
        constraints=["no external network at runtime"],
        languages=["python"],
    )
    first = render_architect_prompt(contract)
    second = render_architect_prompt(contract)
    assert first == second
    assert "Contract: qm.architect_design/1" in first
    assert "must retry failed jobs" in first
    assert "no external network" in first
    assert "model-generated output is not verified output" in first


def test_artifacts_roundtrip_stable():
    artifacts = make_artifacts()
    rebuilt = ArchitectArtifacts.from_dict(artifacts.to_dict())
    assert rebuilt.to_dict() == artifacts.to_dict()
    assert rebuilt.fingerprint == artifacts.fingerprint


def test_valid_artifacts_produce_no_blocking_issues():
    issues = validate_artifacts(make_artifacts(), requirements=[{"id": "R1", "text": "replay failed jobs"}])
    assert all(issue.severity != "HIGH" for issue in issues)


def test_validate_rejects_empty_summary_and_short_hash():
    artifacts = make_artifacts()
    artifacts.summary = " "
    artifacts.source_request_sha256 = "nothex"
    categories = {issue.category for issue in validate_artifacts(artifacts)}
    assert {"missing_summary", "request_hash"} <= categories


def test_validate_rejects_duplicate_adrs_and_bad_status():
    artifacts = make_artifacts()
    artifacts.decisions.append(
        Adr(id="ADR-001", title="t", status="maybe", decision="d", rationale="r")
    )
    categories = {issue.category for issue in validate_artifacts(artifacts)}
    assert "duplicate_adr" in categories
    assert "adr_status" in categories


def test_validate_detects_bad_diagram_and_missing_api_marker():
    artifacts = make_artifacts()
    artifacts.diagrams[0].source = "this is not mermaid"
    artifacts.api_specs = [{"not_a_spec": True}]
    categories = {issue.category for issue in validate_artifacts(artifacts)}
    assert "diagram_syntax" in categories
    assert "api_spec" in categories


def test_validate_detects_dependency_cycle():
    artifacts = make_artifacts()
    artifacts.data_models = [
        {"name": "A", "depends_on": ["B"]},
        {"name": "B", "depends_on": ["A"]},
    ]
    categories = {issue.category for issue in validate_artifacts(artifacts)}
    assert "dependency_cycle" in categories


def test_ambiguous_requirement_without_open_question_is_flagged():
    artifacts = make_artifacts()
    issues = validate_artifacts(
        artifacts,
        requirements=["maybe use a queue, should retain audit logs"],
    )
    assert any(issue.category == "ambiguity_unresolved" for issue in issues)


def test_review_flags_missing_failure_modes_and_security_gap():
    artifacts = make_artifacts()
    artifacts.risks = []
    artifacts.migration = {}
    artifacts.api_specs = [{"openapi": "3.0.3", "paths": {}}]
    findings = review_design(artifacts)
    categories = {finding.category for finding in findings}
    assert "operability" in categories
    assert "security" in categories


def test_review_flags_missing_adr_and_diagram():
    artifacts = make_artifacts()
    artifacts.decisions = []
    artifacts.diagrams = []
    categories = {finding.category for finding in review_design(artifacts)}
    assert "design_rationale" in categories
    assert "structure" in categories
