"""Focused tests for the Hugging Face export preparation tool.

The tool is dry-run by default and this suite never writes model weights,
creates a checkpoint, trains, uploads, or contacts a network service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

EXPECTED_PARAMETERS = 110_658_816


def _tokenizer_info():
    from evaluation.hf_export import load_tokenizer_info

    return load_tokenizer_info()


def _hf_config(parameter_count: int = EXPECTED_PARAMETERS):
    from evaluation.hf_export import (
        load_foundation_document,
        load_model_config,
        to_hf_config,
    )

    return to_hf_config(
        load_model_config(),
        foundation_document=load_foundation_document(),
        tokenizer_info=_tokenizer_info(),
        parameter_count=parameter_count,
        checkpoint_sha256="a" * 64,
    )


def test_hf_config_maps_the_canonical_geometry():
    hf_config = _hf_config()
    assert hf_config["hidden_size"] == 768
    assert hf_config["intermediate_size"] == 3072
    assert hf_config["num_hidden_layers"] == 12
    assert hf_config["num_attention_heads"] == 12
    assert hf_config["num_key_value_heads"] == 6
    assert hf_config["max_position_embeddings"] == 2048
    assert hf_config["vocab_size"] == 5822
    assert hf_config["tie_word_embeddings"] is True
    assert hf_config["pad_token_id"] == 0
    assert hf_config["bos_token_id"] == 1
    assert hf_config["eos_token_id"] == 2
    assert hf_config["unk_token_id"] == 3
    assert hf_config["architectures"] == ["QMModel"]
    assert hf_config["taksh"]["transformers_auto_load"] is False
    assert hf_config["taksh"]["load_with"] == "evaluation.hf_export.load_export"
    assert hf_config["taksh"]["parameter_count"] == EXPECTED_PARAMETERS


def test_parameter_count_is_exact():
    from evaluation.hf_export import count_parameters, load_model_config

    assert count_parameters(load_model_config()) == EXPECTED_PARAMETERS


def test_native_config_round_trips_through_hf_config():
    from evaluation.hf_export import native_model_config_from_hf

    native = native_model_config_from_hf(_hf_config())
    assert native["d_model"] == 768
    assert native["ffn_dim"] == 3072
    assert native["n_layers"] == 12
    assert native["n_heads"] == 12
    assert native["n_kv_heads"] == 6
    assert native["max_seq_len"] == 2048
    assert native["vocab_size"] == 5822
    assert native["tie_embeddings"] is True


def test_native_config_rejects_incomplete_hf_config():
    from evaluation.hf_export import ExportError, native_model_config_from_hf

    with pytest.raises(ExportError, match="hidden_size"):
        native_model_config_from_hf({"vocab_size": 5822})


def test_missing_checkpoint_is_blocked_and_writes_nothing(tmp_path):
    from evaluation.hf_export import plan_export

    output_dir = tmp_path / "export"
    plan = plan_export(tmp_path / "missing.pt", output_dir=output_dir)

    assert plan.status == "BLOCKED"
    assert "checkpoint" in plan.missing
    assert plan.hf_config["taksh"]["transformers_auto_load"] is False
    assert not output_dir.exists(), "a dry-run plan must not create the output directory"
    assert all(entry["written"] is False for entry in plan.files)
    assert all(entry["sha256"] is None for entry in plan.files)


def test_export_file_kinds_are_classified():
    from evaluation.hf_export import plan_export

    plan = plan_export("does-not-exist.pt", output_dir="unused")
    kinds = {entry["path"]: entry["kind"] for entry in plan.files}

    assert kinds["model.safetensors"] == "weights"
    assert kinds["config.json"] == "config"
    assert kinds["tokenizer.json"] == "tokenizer"
    assert kinds["tokenizer_config.json"] == "tokenizer_config"
    assert kinds["special_tokens_map.json"] == "tokenizer_config"
    assert kinds["README.md"] == "model_card"
    assert kinds["SHA256SUMS"] == "checksums"
    assert kinds["taksh_export_manifest.json"] == "manifest"


def test_blocked_plan_cannot_be_executed(tmp_path):
    from evaluation.hf_export import ExportError, execute_export, plan_export

    output_dir = tmp_path / "export"
    plan = plan_export(tmp_path / "missing.pt", output_dir=output_dir)

    with pytest.raises(ExportError, match="blocked"):
        execute_export(plan, state_dict={})
    assert not output_dir.exists()


def test_model_card_reports_no_capability_and_preserves_history():
    from evaluation.hf_export import build_model_card, load_model_config

    hf_config = _hf_config()
    card = build_model_card(
        model_cfg=load_model_config(),
        hf_config=hf_config,
        checkpoint_path="runs/does-not-exist.pt",
        checkpoint_sha256="a" * 64,
        tokenizer_info=_tokenizer_info(),
        compatibility={"compatible": True, "checkpoint_sha256": "a" * 64},
    )

    assert "**Actual coding capability: NOT EVALUATED.**" in card
    assert "M7 locked syntax | 0/40 syntax-valid" in card
    assert "M7 functional probes | 0/8 passed" in card
    assert "Frozen Python benchmark | 0/30" in card
    assert "No evaluation report was supplied" in card
    assert "evaluation.hf_export import load_export" in card
    assert "AutoModelForCausalLM" in card
    assert "Nothing was uploaded." in card
    assert "110,658,816" in card
    # The card must not invent a score or claim the export is independently safe.
    assert "No claim of code-generation quality" in card
    assert "No production, security, legal or safety qualification." in card


def test_cli_dry_run_blocks_without_a_checkpoint(tmp_path, capsys):
    from evaluation.hf_export import main

    output_dir = tmp_path / "export"
    code = main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--out",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "status: BLOCKED" in captured.out
    assert "BLOCKED: checkpoint" in captured.out
    assert not output_dir.exists()


def test_cli_json_plan_is_machine_readable(tmp_path, capsys):
    from evaluation.hf_export import main

    code = main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--out",
            str(tmp_path / "export"),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    plan = json.loads(captured.out)
    assert plan["schema"] == "taksh.hf_export_plan/1"
    assert plan["status"] == "BLOCKED"
    assert plan["publication"] == {
        "uploaded": False,
        "network_access": "none",
        "authorized": False,
    }


def test_load_export_refuses_a_missing_directory(tmp_path):
    from evaluation.hf_export import ExportError, load_export

    with pytest.raises(ExportError, match="config not found"):
        load_export(tmp_path / "absent")


def test_verify_export_reports_missing_manifest(tmp_path):
    from evaluation.hf_export import verify_export

    result = verify_export(tmp_path)
    assert result["status"] == "BLOCKED"
    assert "taksh_export_manifest.json is missing" in result["problems"]


def test_verify_export_requires_all_export_files(tmp_path):
    from evaluation.hf_export import verify_export

    (tmp_path / "taksh_export_manifest.json").write_text(
        json.dumps({"schema": "taksh.hf_export/1", "files": []}),
        encoding="utf-8",
    )
    result = verify_export(tmp_path)

    assert result["status"] == "BLOCKED"
    assert "missing required file: model.safetensors" in result["problems"]
    assert "missing required file: README.md" in result["problems"]
