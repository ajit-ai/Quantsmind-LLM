"""Focused tests for the L1.2.0 training-authorization packet.

The tests are read-only with respect to training state. They never train,
acquire data, install packages, create a checkpoint or publish anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def test_packet_is_a_request_not_an_authorization():
    from evaluation.l120_training_authorization import AUTHORIZATION_POLICY_KEYS, load_packet

    packet = load_packet()
    assert packet["schema"] == "taksh.l120_training_authorization/1"
    assert packet["program_label"] == "1.2.0"
    assert packet["status"] in ALLOWED
    assert packet["status"] == "BLOCKED"
    for key in AUTHORIZATION_POLICY_KEYS:
        assert packet["policy"][key] is False
    assert packet["policy"]["preparation_only"] is True
    assert all(
        decision["status"] == "NOT GRANTED"
        for decision in packet["owner_decisions"]
    )


def test_model_and_tokenizer_contract_gates_pass():
    from evaluation.l120_training_authorization import inspect

    report = inspect()
    model = report["gates"]["model"]
    tokenizer = report["gates"]["tokenizer"]

    assert model["status"] == "PASS"
    assert model["id"] == "taksh-foundation-111m"
    assert model["parameters"] == model["expected_parameters"] == 110_658_816
    assert model["sha256_lf"] == model["recorded_sha256_lf"]
    assert tokenizer["status"] == "PASS"
    assert tokenizer["vocab_size"] == tokenizer["expected_vocab_size"] == 5_822
    assert tokenizer["sha256_raw"] == tokenizer["recorded_sha256"]


def test_corpus_gates_record_licence_and_scale_blockers():
    from evaluation.l120_training_authorization import inspect

    report = inspect()
    licences = report["gates"]["corpus_licences"]
    scale = report["gates"]["corpus_scale"]

    assert licences["status"] == "BLOCKED"
    assert licences["checked"] == 9
    assert licences["verified"] == 7
    assert licences["verified_with_quarantine"] == 2
    assert licences["unresolved_ids"] == ["psf-black", "pypa-setuptools"]
    assert licences["required_unresolved"] == 0

    assert scale["status"] == "BLOCKED"
    assert scale["observed_tokens"] == 2_239_771
    assert scale["required_tokens"] == 100_000_000
    assert scale["token_shortfall"] == 97_760_229
    assert scale["observed_repositories"] == 9
    assert scale["required_repositories"] == 20
    assert scale["repository_shortfall"] == 11


def test_corpus_payload_is_blocked_on_absent_governed_materialization():
    from evaluation.l120_training_authorization import inspect

    payload = inspect()["gates"]["corpus_payload"]
    assert payload["status"] == "BLOCKED"
    assert payload["manifest_present"] is True
    assert payload["token_metadata_present"] is True
    assert payload["payload_present"] is False
    assert payload["tokens_present"] is False
    assert payload["frozen_snapshot_present"] is False
    assert payload["data_card_present"] is True


def test_compute_and_checkpoint_remain_blocked():
    from evaluation.l120_training_authorization import inspect

    report = inspect()
    compute = report["gates"]["compute"]
    checkpoint = report["gates"]["checkpoint"]

    assert compute["status"] == "BLOCKED"
    assert compute["authorization"] == "NOT GRANTED"
    assert compute["max_budget_usd"] == 0
    assert compute["gpu_hours_authorized"] == 0

    assert checkpoint["status"] == "BLOCKED"
    assert checkpoint["exists"] is False
    assert checkpoint["creation_authorized"] is False
    forbidden = checkpoint["forbidden_artifacts"]
    assert [item["id"] for item in forbidden] == ["v180_sft"]
    assert "GEOMETRY_INCOMPATIBLE" in forbidden[0]["reason"]


def test_report_never_claims_training_capability_or_publication():
    from evaluation.l120_training_authorization import inspect

    report = inspect()
    assert report["schema"] == "taksh.l120_training_authorization_report/1"
    assert report["status"] in ALLOWED
    assert report["status"] == "BLOCKED"
    assert report["verdict"] == "TRAINING_NOT_AUTHORIZED"
    assert report["ready_to_train"] is False
    assert report["training_executed"] is False
    assert report["checkpoint_created"] is False
    assert report["authorization"]["authorization_flags_clear"] is True
    assert report["policy"] == {
        "no_training": True,
        "no_checkpoint": True,
        "no_dataset_acquisition": True,
        "no_network_access": True,
        "no_publication": True,
        "no_capability_claim": True,
    }
    assert report["blockers"]
    assert len(report["owner_decisions"]) == 5


def test_report_is_deterministic():
    from evaluation.l120_training_authorization import inspect

    assert inspect() == inspect()


def test_fail_closed_when_authorization_flag_is_true_without_decision(tmp_path):
    from evaluation.l120_training_authorization import (
        TrainingAuthorizationError,
        inspect,
        load_packet,
    )

    packet = load_packet()
    packet["policy"]["training_authorized"] = True
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingAuthorizationError, match="D-TRAIN-1"):
        inspect(path)


def test_invalid_owner_decision_status_is_rejected(tmp_path):
    from evaluation.l120_training_authorization import (
        TrainingAuthorizationError,
        inspect,
        load_packet,
    )

    packet = load_packet()
    packet["owner_decisions"][0]["status"] = "APPROVED"
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingAuthorizationError, match="invalid status"):
        inspect(path)


def test_cli_writes_report_and_exits_nonzero_while_blocked(tmp_path, capsys):
    from evaluation.l120_training_authorization import main

    out = tmp_path / "authorization.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "ready to train: False" in captured.out
    assert "TRAINING_NOT_AUTHORIZED" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.l120_training_authorization_report/1"
    assert written["ready_to_train"] is False
