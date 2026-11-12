"""Focused tests for the distribution/publication readiness assessment.

These tests are read-only with respect to the repository and never publish or
contact a network service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def test_report_schema_and_allowed_statuses():
    from evaluation.publication_readiness import evaluate

    report = evaluate()
    assert report["schema"] == "taksh.publication_readiness/1"
    assert report["product"]["name"] == "Taksh"
    assert report["channels"], "at least one channel must be assessed"
    for channel in report["channels"]:
        assert channel["status"] in ALLOWED, channel["id"]
        assert channel["owner_gate"], channel["id"]


def test_no_publication_or_network_action_is_claimed():
    from evaluation.publication_readiness import evaluate

    policy = evaluate()["policy"]
    assert policy["publication_authorized"] is False
    assert policy["publication_executed"] is False
    assert policy["network_access"] == "none"
    assert policy["training_authorized"] is False
    assert policy["compute_authorized"] is False
    assert policy["model_weights_published"] is False
    assert policy["datasets_published"] is False


def test_evaluation_is_deterministic():
    from evaluation.publication_readiness import evaluate

    first = evaluate()
    second = evaluate()
    assert first == second


def test_hugging_face_model_is_blocked_without_canonical_checkpoint():
    from evaluation.publication_readiness import evaluate

    report = evaluate()
    channel = next(c for c in report["channels"] if c["id"] == "huggingface_model")
    canonical = any(
        item["exists"] and item["role"].startswith("canonical_candidate")
        for item in channel["evidence"]["checkpoints"]
    )
    if canonical:
        assert channel["status"] in ALLOWED
    else:
        assert channel["status"] == "BLOCKED"
        assert report["capability"]["model_availability"] == "MODEL_UNAVAILABLE"
        assert channel["evidence"]["transformers_compatible"] is False
        assert any("canonical taksh-foundation-111m checkpoint" in item
                   for item in channel["missing"])


def test_pypi_is_partial_and_contains_no_weights():
    from evaluation.publication_readiness import evaluate

    channel = next(c for c in evaluate()["channels"] if c["id"] == "pypi")
    assert channel["status"] == "PARTIAL"
    assert channel["evidence"]["contains_weights"] is False
    assert channel["evidence"]["contains_data"] is False
    assert any("credential" in item.lower() for item in channel["missing"])


def test_local_openai_surface_exists_but_is_chat_only():
    from evaluation.publication_readiness import evaluate

    channel = next(
        c for c in evaluate()["channels"] if c["id"] == "openai_compatible_local"
    )
    assert channel["status"] == "PARTIAL"
    assert channel["evidence"]["surface_complete"] is True
    assert any("/v1/models" in route for route in channel["evidence"]["routes"])
    assert channel["evidence"]["tool_calling"].startswith("rejected")
    assert channel["evidence"]["streaming"].startswith("rejected")
    assert channel["evidence"]["checkpoint_argument"].startswith("--ckpt")


def test_agentic_clients_are_blocked_by_missing_tool_calling():
    from evaluation.publication_readiness import evaluate

    channels = {c["id"]: c for c in evaluate()["channels"]}
    for channel_id in ("cline", "opencode"):
        channel = channels[channel_id]
        assert channel["status"] == "BLOCKED"
        assert channel["evidence"]["tool_calling"].startswith("not implemented")
        assert any("tool-calling" in item for item in channel["missing"])


def test_openrouter_is_not_started():
    from evaluation.publication_readiness import evaluate

    channel = next(c for c in evaluate()["channels"] if c["id"] == "openrouter")
    assert channel["status"] == "NOT STARTED"
    assert channel["evidence"]["submission_status"] == "none"


def test_capability_is_not_evaluated_and_history_is_preserved():
    from evaluation.publication_readiness import evaluate

    capability = evaluate()["capability"]
    assert capability["actual_coding_capability"] == "NOT EVALUATED"
    assert capability["frozen_python_benchmark"]["historical_result"].startswith("0/30")
    history = capability["historical_m7"]
    if history["exists"]:
        assert "0/40" in history["locked_syntax"]
        assert "0/8" in history["functional"]
        assert history["evidence_level"] == 0


def test_verdict_blocks_model_and_dataset_publication():
    from evaluation.publication_readiness import evaluate

    verdict = evaluate()["verdict"]
    assert verdict["code"] == "NO_MODEL_OR_DATASET_CHANNEL_READY"
    assert verdict["model"] == "BLOCKED"
    assert verdict["dataset"] == "BLOCKED"
    assert verdict["hosted_api"] == "NOT STARTED"


def test_cli_writes_report_to_requested_path(tmp_path, capsys):
    from evaluation.publication_readiness import main

    out = tmp_path / "publication.json"
    rc = main(["--out", str(out)])
    assert rc == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "taksh.publication_readiness/1"
    captured = capsys.readouterr().out
    assert "MODEL_UNAVAILABLE" in captured
    assert "NO_MODEL_OR_DATASET_CHANNEL_READY" in captured


def test_taksh_cli_publication_and_capability_commands(capsys):
    from taksh.cli import main

    assert main(["publication"]) is None
    out = capsys.readouterr().out
    assert "Hugging Face model repository" in out
    assert "NO_MODEL_OR_DATASET_CHANNEL_READY" in out

    assert main(["capability"]) is None
    out = capsys.readouterr().out
    assert "MODEL_UNAVAILABLE" in out
    assert "NOT EVALUATED" in out
