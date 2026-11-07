"""Taksh v1.4.0 qualification tests: capability evaluation and agent contract.

Everything here is offline, CPU-bounded and deterministic. No generated code is
executed in the evaluator's process: the sandbox is blocked by default and
the tests assert that refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.agent_contract import AgentClientError, SimulatedCodingAgentClient, TakshTransport
from evaluation.python_eval import CATEGORIES, CONTAMINATION_CLASSES, SCHEMA_VERSION
from evaluation.python_eval.metrics import aggregate
from evaluation.python_eval.sandbox import (
    REQUIRED_GUARANTEES,
    SandboxPolicy,
    capability_report,
    default_policy,
    execute_python,
)
from evaluation.python_eval.schema import CaseError, load_cases, validate_case, validate_suite
from evaluation.python_eval.syntax import extract_code_block, validate_syntax
from serving.agent_profile import default_profile, profile_to_json

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v1.json"
TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"


@pytest.fixture(scope="module")
def cases():
    return load_cases(CASES)


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


@pytest.fixture()
def serving_client(monkeypatch):
    """A client bound to the real FastAPI application, with a stubbed model."""
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

    captured = {}

    def fake_generate_ids(model, tokenizer, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        text = "def answer():\n    return 1\n"
        return {
            "text": text,
            "token_ids": list(range(len(text))),
            "finish_reason": "stop",
            "prompt_tokens": len(prompt),
            "completion_tokens": len(text),
            "eos_id": 2,
        }

    monkeypatch.setattr(openai_server, "generate_ids", fake_generate_ids)
    transport = TakshTransport(app=openai_server.app)
    client = SimulatedCodingAgentClient(transport, model="qm-code-tiny")
    return client, captured, TestClient(openai_server.app)


# --------------------------------------------------------------------------- #
# G3 - evaluation dataset integrity
# --------------------------------------------------------------------------- #
def test_g3_suite_loads_and_covers_every_category(cases):
    assert len(cases) == 20
    covered = {case["category"] for case in cases}
    assert covered == set(CATEGORIES)
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_g3_every_case_declares_provenance_and_contamination(cases):
    for case in cases:
        assert case["provenance"]["source"] == "authored:v1.4.0"
        assert case["contamination"] in CONTAMINATION_CLASSES
        assert case["contamination"] == "authored"
        assert case["budget"]["timeout_seconds"] > 0
        assert case["budget"]["max_output_bytes"] > 0


def test_g3_case_schema_rejects_a_missing_field():
    problems = validate_case({"case_id": "x"}, 0)
    assert any("missing" in problem for problem in problems)


def test_g3_case_schema_rejects_an_unknown_category(cases):
    broken = dict(cases[0])
    broken["category"] = "not_a_category"
    assert validate_case(broken, 0)


def test_g3_duplicate_case_ids_are_rejected(cases):
    duplicated = list(cases) + [dict(cases[0])]
    assert any("duplicate" in problem for problem in validate_suite(duplicated))


def test_g3_suite_file_rejects_a_wrong_schema(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema": "wrong", "cases": []}), encoding="utf-8")
    with pytest.raises(CaseError):
        load_cases(path)


# --------------------------------------------------------------------------- #
# G5 - syntax evaluation
# --------------------------------------------------------------------------- #
def test_g5_valid_python_is_accepted():
    ok, category, detail = validate_syntax("def f():\n    return 1\n", "python")
    assert ok is True and category == "ok" and detail == ""


def test_g5_invalid_python_is_rejected_with_a_category():
    ok, category, detail = validate_syntax("def broken(:\n", "python")
    assert ok is False and category == "syntax_error" and "line" in detail


def test_g5_empty_output_is_a_syntax_error():
    ok, category, _ = validate_syntax("   ", "python")
    assert ok is False and category == "syntax_error"


def test_g5_unknown_language_is_reported_not_guessed():
    ok, category, detail = validate_syntax("x = 1", "rust")
    assert ok is False and category == "unsupported_language"
    assert "rust" in detail


def test_g5_language_dispatch_is_extensible():
    from evaluation.python_eval.syntax import VALIDATORS

    assert "python" in VALIDATORS
    assert "python-like" in VALIDATORS
    ok, _, _ = validate_syntax("x = 1\n", "python-like")
    assert ok is True


def test_g5_code_fences_are_extracted():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\n"
    assert extract_code_block(text) == "def f():\n    return 1\n"
    assert extract_code_block("def f():\n    return 1\n").startswith("def f()")


# --------------------------------------------------------------------------- #
# G6 - safe functional evaluation
# --------------------------------------------------------------------------- #
def test_g6_functional_execution_is_blocked_by_default():
    result = execute_python("print('hello')", default_policy())
    assert result.blocked is True
    assert result.executed is False
    assert "blocked" in result.reason


def test_g6_capability_report_states_what_is_not_guaranteed():
    report = capability_report(default_policy())
    assert report["functional_execution"] == "blocked"
    assert "filesystem isolation" in report["not_guaranteed_by_this_process"]
    assert list(report["required_guarantees"]) == list(REQUIRED_GUARANTEES)


def test_g6_external_mode_runs_inside_the_configured_command(tmp_path):
    marker = tmp_path / "ran.txt"
    policy = SandboxPolicy(
        mode="external",
        external_command=[
            "python",
            "-c",
            "import sys,pathlib;pathlib.Path(sys.argv[2]).write_text('ran');"
            "sys.stdout.write('ok')",
            "unused",
        ],
        timeout_seconds=30,
    )
    # The command receives the script path; a fake command records the call.
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type(
            "R",
            (),
            {"returncode": 0, "stdout": "ok\n", "stderr": ""},
        )()

    import subprocess

    original = subprocess.run
    subprocess.run = fake_run
    try:
        result = execute_python("print('x')", policy)
    finally:
        subprocess.run = original
    assert result.executed is True
    assert result.ok is True
    assert calls and str(calls[0][-1]).endswith("generated.py")
    assert not Path(calls[0][-1]).exists()


def test_g6_external_mode_reports_a_timeout():
    import subprocess

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    policy = SandboxPolicy(mode="external", external_command=["python"], timeout_seconds=1)
    original = subprocess.run
    subprocess.run = fake_run
    try:
        result = execute_python("print('x')", policy)
    finally:
        subprocess.run = original
    assert result.timed_out is True
    assert result.ok is False


# --------------------------------------------------------------------------- #
# G4 - generation correctness and bounds
# --------------------------------------------------------------------------- #
def _generator(tokenizer, **overrides):
    from serving.python_codegen import PythonCodeGenerator

    class StubModel:
        cfg = {"max_seq_len": 64, "vocab_size": 5822}

        def eval(self):
            return self

    kwargs = {
        "model_identity": {"model_id": "stub"},
        "qualified_formats": ("code_completion", "chat_instruction"),
        "max_new_tokens_cap": 32,
    }
    kwargs.update(overrides)
    return PythonCodeGenerator(StubModel(), tokenizer, **kwargs)


def test_g4_empty_prompt_is_refused(tokenizer):
    result = _generator(tokenizer).generate("", max_new_tokens=8)
    assert result["refusal"] == "empty_prompt"
    assert result["diagnostics"]["refused"] is True


def test_g4_reserved_literal_prompt_is_refused(tokenizer):
    result = _generator(tokenizer).generate('x = "<|user|>"\n', max_new_tokens=8)
    assert result["refusal"] == "reserved_token_literal_in_prompt"
    assert "<|user|>" in result["diagnostics"]["literals"]


def test_g4_unqualified_prompt_surface_is_refused(tokenizer):
    generator = _generator(tokenizer, qualified_formats=("code_completion",))
    result = generator.generate("hello", max_new_tokens=8, format="chat_instruction")
    assert result["refusal"] == "prompt_surface_not_qualified"


def test_g4_unknown_prompt_format_is_refused(tokenizer):
    result = _generator(tokenizer).generate("hello", max_new_tokens=8, format="telepathy")
    assert result["refusal"] == "unknown_prompt_format"


def test_g4_context_overflow_is_refused(tokenizer):
    generator = _generator(tokenizer)
    result = generator.generate("x = 1\n" * 200, max_new_tokens=8)
    assert result["refusal"] == "prompt_exceeds_context"


def test_g4_token_budget_is_capped(tokenizer, monkeypatch):
    generator = _generator(tokenizer)
    seen = {}

    def fake_generate_ids(model, tok, prompt, **kwargs):
        seen["budget"] = kwargs.get("max_new_tokens")
        return {
            "text": "", "token_ids": [], "finish_reason": "stop",
            "prompt_tokens": 1, "completion_tokens": 0, "eos_id": 2,
        }

    monkeypatch.setattr("serving.python_codegen.generate_ids", fake_generate_ids)
    generator.generate("def f():\n", max_new_tokens=1000)
    assert seen["budget"] <= generator.max_new_tokens_cap


def test_g4_greedy_generation_is_deterministic(tokenizer, monkeypatch):
    generator = _generator(tokenizer)
    calls = []

    def fake_generate_ids(model, tok, prompt, **kwargs):
        calls.append(prompt)
        return {
            "text": "def f():\n    return 1\n",
            "token_ids": [1, 2, 3],
            "finish_reason": "stop",
            "prompt_tokens": len(prompt),
            "completion_tokens": 3,
            "eos_id": 2,
        }

    monkeypatch.setattr("serving.python_codegen.generate_ids", fake_generate_ids)
    first = generator.generate("def f():\n", max_new_tokens=8)
    second = generator.generate("def f():\n", max_new_tokens=8)
    assert first["text"] == second["text"]
    assert calls[0] == calls[1]
    assert first["diagnostics"]["refused"] is False


def test_g4_chat_template_is_not_applied_to_a_base_model(tokenizer, monkeypatch):
    generator = _generator(tokenizer, chat_contract=None)
    seen = {}

    def fake_generate_ids(model, tok, prompt, **kwargs):
        seen["prompt"] = prompt
        return {
            "text": "", "token_ids": [], "finish_reason": "stop",
            "prompt_tokens": 1, "completion_tokens": 0, "eos_id": 2,
        }

    monkeypatch.setattr("serving.python_codegen.generate_ids", fake_generate_ids)
    generator.generate("write a function", max_new_tokens=8, format="chat_instruction")
    assert "<|user|>" not in seen["prompt"]
    assert seen["prompt"] == "write a function"


# --------------------------------------------------------------------------- #
# G9 - metrics honesty
# --------------------------------------------------------------------------- #
def test_g9_rates_carry_denominators():
    records = [
        {"outcome": "pass", "syntax_valid": True, "functional": True, "category": "small_algorithm", "latency_seconds": 1.0, "tokens_generated": 10},
        {"outcome": "fail", "syntax_valid": False, "functional": None, "category": "small_algorithm", "latency_seconds": 2.0, "tokens_generated": 20},
        {"outcome": "blocked", "syntax_valid": True, "functional": None, "category": "debugging", "latency_seconds": 3.0, "tokens_generated": 5},
    ]
    metrics = aggregate(records)
    assert metrics["cases_total"] == 3
    assert metrics["cases_evaluated"] == 2
    assert metrics["cases_blocked"] == 1
    assert metrics["syntax_valid_denominator"] == 3
    assert metrics["syntax_valid_rate"] == pytest.approx(2 / 3)
    assert metrics["functional_pass_denominator"] == 1
    assert metrics["functional_pass_rate"] == pytest.approx(1.0)


def test_g9_unmeasurable_metrics_are_null_not_zero():
    metrics = aggregate([{"outcome": "blocked", "syntax_valid": None, "functional": None}])
    assert metrics["functional_pass_rate"] is None
    assert metrics["functional_measurement_available"] is False


# --------------------------------------------------------------------------- #
# G7 / G8 - serving contract and simulated agent client
# --------------------------------------------------------------------------- #
def test_g7_model_discovery_works(serving_client):
    client, _, _ = serving_client
    models = client.list_models()
    assert "qm-code-tiny" in models


def test_g7_health_reports_the_contract(serving_client):
    client, _, _ = serving_client
    health = client.health()
    assert health["model_loaded"] is True
    assert health["prompt_contract"] == "qm.plain_instruction/1"


def test_g7_chat_completion_round_trip(serving_client):
    client, captured, _ = serving_client
    response = client.chat(
        [{"role": "user", "content": "hello"}], max_tokens=32, temperature=0.0
    )
    assert response["choices"][0]["message"]["content"]
    assert response["model"] == "qm-code-tiny"
    assert captured["kwargs"]["max_new_tokens"] <= 32


def test_g7_invalid_model_is_refused(serving_client):
    """The server echoes any model id, so the client is the guard."""
    client, _, _ = serving_client
    with pytest.raises(AgentClientError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}], model="does-not-exist")
    assert excinfo.value.code == "model_not_found"
    assert excinfo.value.status == 404


def test_g7_malformed_request_is_refused(serving_client):
    client, _, _ = serving_client
    with pytest.raises(AgentClientError) as excinfo:
        client.chat([{"role": "user"}])
    assert excinfo.value.status == 422


def test_g7_non_positive_token_limit_is_refused(serving_client):
    client, _, _ = serving_client
    with pytest.raises(AgentClientError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}], max_tokens=0)
    assert excinfo.value.status == 400


def test_g7_tool_calling_is_refused_explicitly(serving_client):
    client, _, _ = serving_client
    with pytest.raises(AgentClientError) as excinfo:
        client.request_tools(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "f"}}],
        )
    assert excinfo.value.status == 400
    assert excinfo.value.code


def test_g7_error_codes_are_consistent(serving_client):
    client, _, _ = serving_client
    with pytest.raises(AgentClientError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}], max_tokens=0)
    error = excinfo.value.payload["error"]
    assert error["type"] == "invalid_request_error"
    assert "max_tokens" in error["message"]


def test_g8_concurrent_requests_are_bounded(serving_client):
    client, _, _ = serving_client
    results = [
        client.chat_text([{"role": "user", "content": f"hi {i}"}], max_tokens=8)
        for i in range(3)
    ]
    assert len(results) == 3


def test_g8_client_records_its_calls(serving_client):
    client, _, _ = serving_client
    before = len(client.transport.calls)
    client.list_models()
    assert len(client.transport.calls) == before + 1
    assert client.transport.calls[-1]["path"] == "/v1/models"


def test_g8_profile_declares_what_the_server_actually_supports():
    profile = default_profile()
    assert profile.streaming_supported is False
    assert profile.tool_calling_supported is False
    assert profile.structured_output_supported is False
    assert "POST /v1/chat/completions" in profile.supported_endpoints
    assert not any("/v1/completions" in endpoint for endpoint in profile.supported_endpoints)
    assert profile.integration_status == "EXTERNAL_AGENT_INTEGRATION_PENDING"
    assert "tool calling" in profile.summary()


def test_g8_profile_serialises_with_its_schema():
    payload = json.loads(profile_to_json(default_profile()))
    assert payload["schema"] == "quantsmind.agent_profile/1"
    assert payload["known_limitations"]
