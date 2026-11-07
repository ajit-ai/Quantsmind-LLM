"""Taksh v1.5.0 qualification tests: evaluation surfaces, geometry, objective,
instruction contract, serving reliability, and preservation.

Everything is offline, CPU-bounded and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.python_eval.metrics import aggregate
from evaluation.python_eval.sandbox import detect_local_isolation
from evaluation.python_eval.surfaces import (
    SurfaceError,
    assemble_program,
    check_contract,
    surface_of,
)
from evaluation.python_eval.schema import load_cases
from training.instruction import (
    InstructionContractError,
    build_supervised_sequence,
    load_examples,
    validate_example,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V14_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v1.json"
V15_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v15.json"
FIXTURES = REPO_ROOT / "training" / "fixtures" / "instruction_synthetic_v1.jsonl"
TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


# --------------------------------------------------------------------------- #
# A - evaluation surfaces
# --------------------------------------------------------------------------- #
def test_v14_suite_is_unchanged():
    cases = load_cases(V14_CASES)
    assert len(cases) == 20
    assert "surface" not in cases[0]


def test_v15_suite_declares_surfaces():
    cases = load_cases(V15_CASES)
    surfaces = [case["surface"] for case in cases]
    assert set(surfaces) == {"base_completion", "instruction_to_code", "chat_instruction"}
    assert len(cases) == 16


def test_surface_of_derives_from_prompt_format():
    assert surface_of({"prompt_format": "code_completion"}) == "base_completion"
    assert surface_of({"prompt_format": "chat_instruction"}) == "instruction_to_code"
    assert surface_of({"surface": "chat_instruction"}) == "chat_instruction"
    with pytest.raises(SurfaceError):
        surface_of({"surface": "telepathy"})


def test_completion_assembles_prefix_and_continuation():
    case = {"surface": "base_completion", "prompt": "def add(a, b):\n"}
    program = assemble_program(case, "    return a + b\n")
    assert program == "def add(a, b):\n    return a + b\n"
    compile(program, "<test>", "exec")


def test_instruction_surface_is_not_assembled_with_a_prefix():
    case = {"surface": "instruction_to_code", "prompt": "Write a function."}
    assert assemble_program(case, "def f():\n    return 1\n") == "def f():\n    return 1\n"


def test_instruction_surface_requires_an_instruction_qualified_checkpoint():
    case = {"surface": "instruction_to_code", "prompt": "x"}
    allowed, reason = check_contract(case, {"completion": True, "instruction": False})
    assert allowed is False
    assert "instruction" in reason
    allowed, _ = check_contract(case, {"completion": True, "instruction": True})
    assert allowed is True


def test_chat_surface_requires_the_matching_contract():
    case = {"surface": "chat_instruction", "prompt": "x", "prompt_contract": "qm.chat_instruction/2"}
    allowed, reason = check_contract(
        case, {"chat": True, "prompt_contract": "qm.chat_instruction/1"}
    )
    assert allowed is False and "contract" in reason
    allowed, _ = check_contract(
        case, {"chat": True, "prompt_contract": "qm.chat_instruction/2"}
    )
    assert allowed is True


def test_base_completion_needs_no_qualification():
    assert check_contract({"surface": "base_completion", "prompt": "x"}, None)[0] is True


def test_metrics_keep_denominators_and_nulls():
    records = [
        {"outcome": "pass", "syntax_valid": True, "surface": "base_completion"},
        {"outcome": "fail", "syntax_valid": False, "surface": "base_completion",
         "failure_category": "syntax_error"},
        {"outcome": "refused", "surface": "chat_instruction",
         "failure_category": "contract_mismatch"},
    ]
    metrics = aggregate(records)
    assert metrics["cases_total"] == 3
    assert metrics["cases_evaluated"] == 2
    assert metrics["contract_mismatch_count"] == 1
    assert metrics["functional_pass_rate"] is None
    assert metrics["by_surface"]["base_completion"]["total"] == 2


def test_metrics_report_empty_and_truncated_generations():
    records = [
        {"outcome": "fail", "syntax_valid": False, "empty_output": True,
         "truncated": False},
        {"outcome": "fail", "syntax_valid": False, "empty_output": False,
         "truncated": True},
    ]
    metrics = aggregate(records)
    assert metrics["empty_output_count"] == 1
    assert metrics["truncated_count"] == 1
    assert metrics["truncation_rate"] == 0.5


# --------------------------------------------------------------------------- #
# B - checkpoint / tokenizer / geometry compatibility
# --------------------------------------------------------------------------- #
def test_geometry_inspector_reports_a_matching_triple():
    from training.geometry import inspect

    report = inspect(
        checkpoint_path=REPO_ROOT / "runs" / "m7_pretrain" / "checkpoint_last.pt",
        tokenizer_path=TOKENIZER,
        model_config_path=REPO_ROOT / "configs" / "model" / "code_m3_9m.yaml",
    )
    assert report.problems == []
    assert report.compatible is True
    assert report.model_vocab == 5822
    assert report.embedding_rows == 5822
    assert report.tokenizer_vocab == 5822
    assert report.weight_tensors > 0
    assert report.checkpoint_step == 300


def test_geometry_inspector_fails_closed_on_a_missing_tokenizer(tmp_path):
    from training.geometry import inspect

    report = inspect(tokenizer_path=tmp_path / "nope.json")
    assert report.compatible is False
    assert any("tokenizer" in problem for problem in report.problems)


def test_geometry_inspector_fails_closed_on_a_vocab_mismatch():
    from training.geometry import inspect

    report = inspect(
        checkpoint_path=REPO_ROOT / "runs" / "m7_pretrain" / "checkpoint_last.pt",
        tokenizer_path=TOKENIZER,
        model_config_path=REPO_ROOT / "configs" / "model" / "code_30m.yaml",
    )
    assert report.compatible is False
    assert any("vocab" in problem for problem in report.problems)


def test_geometry_inspector_reports_the_orphan_s3_checkpoint():
    from training.geometry import inspect

    report = inspect(
        checkpoint_path=REPO_ROOT / "runs" / "code_pretrain" / "checkpoint_last.pt",
        tokenizer_path=TOKENIZER,
    )
    assert report.compatible is False
    assert report.tokenizer_vocab == 5822
    assert report.problems


def test_geometry_require_compatible_raises():
    from training.geometry import CompatibilityError, require_compatible

    with pytest.raises(CompatibilityError):
        require_compatible(type("R", (), {"compatible": False, "problems": ["x"]})())


# --------------------------------------------------------------------------- #
# C - training objective and masks
# --------------------------------------------------------------------------- #
def test_causal_shift_is_exactly_once():
    from training.objective import shift_for_causal

    ids = torch.tensor([1, 2, 3, 4])
    inputs, labels = shift_for_causal(ids, pad_id=0)
    assert torch.equal(inputs, torch.tensor([1, 2, 3]))
    assert torch.equal(labels, torch.tensor([2, 3, 4]))


def test_padding_targets_are_masked():
    from training.objective import IGNORE_INDEX, shift_for_causal

    ids = torch.tensor([5, 6, 0, 0])
    _, labels = shift_for_causal(ids, pad_id=0)
    assert labels.tolist()[-2:] == [IGNORE_INDEX, IGNORE_INDEX]


def test_loss_ignores_masked_positions():
    import math

    from training.objective import IGNORE_INDEX, token_mean_loss

    logits = torch.zeros(1, 3, 10)
    labels = torch.tensor([[1, IGNORE_INDEX, IGNORE_INDEX]])
    summed, count = token_mean_loss(logits, labels)
    assert count == 1
    # Uniform logits over 10 classes: the single supervised target costs ln(10).
    assert abs(float(summed) - math.log(10)) < 1e-5


# --------------------------------------------------------------------------- #
# D - instruction contract
# --------------------------------------------------------------------------- #
def test_instruction_examples_load_and_validate():
    examples = load_examples(FIXTURES)
    assert len(examples) == 4
    for example in examples:
        assert validate_example(example) == []


def test_production_mode_rejects_unapproved_fixtures():
    for example in load_examples(FIXTURES):
        problems = validate_example(example, production=True)
        assert problems, "a synthetic fixture must not pass production mode"


def test_instruction_schema_requires_provenance():
    problems = validate_example(
        {"example_id": "x", "language": "python", "instruction": "a",
         "response": "b", "provenance": {}, "authorization": "pending",
         "contamination": "authored", "prompt_contract": "qm.chat_instruction/2",
         "token_budget": 64, "split": "train"}
    )
    assert any("provenance" in problem for problem in problems)


def test_supervised_sequence_masks_the_prompt(tokenizer):
    example = load_examples(FIXTURES)[0]
    sequence = build_supervised_sequence(tokenizer, example, max_length=128)
    assert sequence["supervised_tokens"] > 0
    assert sequence["response_tokens"] > 0
    assert sequence["prompt_tokens"] > 0
    supervised = [label for label in sequence["labels"] if label != -100]
    assert len(supervised) == sequence["supervised_tokens"]


def test_supervised_sequence_truncates_within_budget(tokenizer):
    example = dict(load_examples(FIXTURES)[0])
    example["token_budget"] = 12
    sequence = build_supervised_sequence(tokenizer, example, max_length=128)
    assert len(sequence["ids"]) == 12
    assert sequence["truncated"] is True


def test_unsupported_prompt_contract_is_refused(tokenizer):
    example = dict(load_examples(FIXTURES)[0])
    example["prompt_contract"] = "qm.plain_instruction/1"
    with pytest.raises(InstructionContractError):
        build_supervised_sequence(tokenizer, example)


# --------------------------------------------------------------------------- #
# E - serving reliability
# --------------------------------------------------------------------------- #
@pytest.fixture()
def serving_client(monkeypatch):
    from starlette.testclient import TestClient

    from serving import openai_server

    class FakeTokenizer:
        SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

        def token_to_id(self, token):
            return self.SPECIALS.get(token)

        def get_vocab_size(self):
            return 5822

        def encode(self, text):
            self._text = text
            return self

        @property
        def ids(self):
            return [ord(c) % 5000 for c in self._text]

        def decode(self, ids):
            return "".join(chr(i % 5000) for i in ids)

    class FakeModel:
        def __init__(self):
            self.cfg = {"max_seq_len": 4096, "vocab_size": 5822, "name": "qm-code-tiny"}
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter([self._param])

        def num_params(self, non_embedding=True):
            return 495_808

        def eval(self):
            return self

    openai_server.state.model = FakeModel()
    openai_server.state.model_cfg = dict(openai_server.state.model.cfg)
    openai_server.state.tokenizer = FakeTokenizer()
    openai_server.state.model_id = "qm-code-tiny"
    openai_server.state.checkpoint_step = 200
    openai_server.state.api_key = None
    openai_server.state.ckpt_path = None
    openai_server.state.contract_id = "qm.plain_instruction/1"
    # monkeypatch restores the previous values, so this fixture cannot leak a
    # prompt surface into another test module. The surface is declared for the
    # tokenizer this fixture installs, matching how load_state scopes it.
    monkeypatch.setattr(openai_server.state, "prompt_surface", "base")
    monkeypatch.setattr(openai_server.state, "surface_tokenizer",
                        openai_server.state.tokenizer)
    monkeypatch.setattr(openai_server.state, "chat_qualified", False)

    captured = {}

    def fake_generate_ids(model, tokenizer, prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "text": "def answer():\n    return 1\n",
            "token_ids": [1, 2, 3],
            "finish_reason": "stop",
            "prompt_tokens": len(prompt),
            "completion_tokens": 3,
            "eos_id": 2,
        }

    monkeypatch.setattr(openai_server, "generate_ids", fake_generate_ids)
    return TestClient(openai_server.app), captured


def test_unknown_model_id_is_rejected(serving_client):
    client, _ = serving_client
    response = client.post(
        "/v1/chat/completions",
        json={"model": "not-loaded", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"


def test_loaded_model_id_is_accepted(serving_client):
    client, _ = serving_client
    response = client.post(
        "/v1/chat/completions",
        json={"model": "qm-code-tiny", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "qm-code-tiny"


def test_base_checkpoint_is_served_raw_text(serving_client):
    client, captured = serving_client
    client.post(
        "/v1/chat/completions",
        json={"model": "qm-code-tiny", "messages": [{"role": "user", "content": "add"}]},
    )
    assert captured["prompt"] == "add"


def test_model_discovery_still_works(serving_client):
    client, _ = serving_client
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qm-code-tiny"


def test_unsupported_capabilities_are_still_refused(serving_client):
    client, _ = serving_client
    for field, value in (
        ("stream", True),
        ("tools", [{"type": "function", "function": {"name": "f"}}]),
        ("response_format", {"type": "json_object"}),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qm-code-tiny",
                  "messages": [{"role": "user", "content": "hi"}],
                  field: value},
        )
        assert response.status_code == 400, field


# --------------------------------------------------------------------------- #
# F - functional evaluation safety
# --------------------------------------------------------------------------- #
def test_isolation_detection_is_read_only_and_reports_absence():
    report = detect_local_isolation()
    assert "mechanisms" in report
    assert isinstance(report["functional_execution_available"], bool)
    assert report["policy"]
    if not report["functional_execution_available"]:
        for name, entry in report["mechanisms"].items():
            assert entry["state"] in ("absent", "present_unusable")


def test_functional_execution_stays_blocked_without_a_mechanism():
    from evaluation.python_eval.sandbox import default_policy, execute_python

    result = execute_python("print('x')", default_policy())
    assert result.blocked is True
    assert result.executed is False


# --------------------------------------------------------------------------- #
# G - corpus authorization readiness
# --------------------------------------------------------------------------- #
def test_readiness_report_is_read_only_and_default_deny():
    from data.python_v1.readiness import assess

    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["approved"] == 0
    assert report["production_training_authorized"] is False
    assert report["default_decision"] == "deny"
    assert report["decision_register"]["D3"]


def test_readiness_report_does_not_mutate_the_contract():
    from data.python_v1.readiness import assess
    from data.python_v1.registry import load_contract

    path = REPO_ROOT / "data" / "contracts" / "python_v1.yaml"
    before = path.read_bytes()
    assess(path)
    assert path.read_bytes() == before
    assert load_contract(path)["sources"] == []


# --------------------------------------------------------------------------- #
# H - preservation
# --------------------------------------------------------------------------- #
def test_shipped_contract_still_approves_no_source():
    from data.python_v1.registry import load_contract

    assert load_contract(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")["sources"] == []


def test_v14_evaluation_suite_is_untouched():
    payload = json.loads(V14_CASES.read_text(encoding="utf-8"))
    assert payload["suite"]["id"] == "python_capability"
    assert len(payload["cases"]) == 20
    assert all("surface" not in case for case in payload["cases"])


def test_tokenizer_artifact_and_model_geometry_are_unchanged():
    from training.geometry import inspect

    report = inspect(
        checkpoint_path=REPO_ROOT / "runs" / "m7_pretrain" / "checkpoint_last.pt",
        tokenizer_path=TOKENIZER,
        model_config_path=REPO_ROOT / "configs" / "model" / "code_m3_9m.yaml",
    )
    assert report.embedding_rows == 5822
    assert report.output_projection_rows == 5822
    assert report.context_window == 2048
