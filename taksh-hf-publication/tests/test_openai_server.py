"""Tests for the OpenAI-compatible local API.

Inference is mocked: no checkpoint weights are loaded and no CPU generation runs.
These assert request/response schema, role handling, model discovery, structured
errors, context limits and termination semantics only.

Routes are still exercised through the real FastAPI app via ``TestClient``, so
routing, middleware, pydantic validation and status codes are genuinely covered.
"""

import pytest

pytest.importorskip("torch")

import torch
from fastapi.testclient import TestClient

from serving import openai_server
from serving.openai_server import app


class FakeTokenizer:
    """One token per character, so prompt token counts are exactly predictable."""

    SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self):
        self._text = ""

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
        # Wide enough that the fixed system prompt plus a test message fits; the
        # context-limit tests narrow this per-test.
        self.cfg = {"max_seq_len": 4096, "vocab_size": 5822, "name": "qm-code-tiny"}
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def num_params(self, non_embedding=True):
        return 495_808

    def eval(self):
        return self


@pytest.fixture
def client(monkeypatch):
    openai_server.state.model = FakeModel()
    openai_server.state.model_cfg = dict(openai_server.state.model.cfg)
    openai_server.state.tokenizer = FakeTokenizer()
    openai_server.state.model_id = "qm-code-tiny"
    openai_server.state.checkpoint_step = 200
    openai_server.state.api_key = None
    openai_server.state.ckpt_path = None

    captured = {}

    def fake_generate_ids(model, tokenizer, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        finish = captured.get("force_finish", "stop")
        n = min(kwargs.get("max_new_tokens", 1), 5)
        text = "".join(chr(97 + (i % 26)) for i in range(n))
        return {
            "text": text,
            "token_ids": list(range(len(text))),
            "finish_reason": finish,
            "prompt_tokens": len(prompt),
            "completion_tokens": len(text),
            "eos_id": 2,
        }

    monkeypatch.setattr(openai_server, "generate_ids", fake_generate_ids)
    return TestClient(app), captured


# --------------------------------------------------------------------------- #
# GET /v1/models
# --------------------------------------------------------------------------- #
def test_models_lists_real_identifier(client):
    c, _ = client
    r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["id"] == "qm-code-tiny"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "quantsmind"
    assert isinstance(entry["created"], int)


def test_models_reports_context_step_and_params(client):
    c, _ = client
    meta = c.get("/v1/models").json()["data"][0]["quantsmind"]
    assert meta["context_window"] == 4096
    assert meta["checkpoint_step"] == 200
    assert meta["n_params_non_embedding"] == 495_808


def test_models_empty_when_nothing_loaded(client):
    c, _ = client
    openai_server.state.model = None
    body = c.get("/v1/models").json()
    assert body["data"] == []
    assert "unavailable_reason" in body


# --------------------------------------------------------------------------- #
# POST /v1/chat/completions - response schema
# --------------------------------------------------------------------------- #
def test_chat_completion_schema(client):
    c, _ = client
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "qm-code-tiny"

    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert choice["logprobs"] is None

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] >= 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_requested_model_is_echoed(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"model": "some-alias", "messages": [{"role": "user", "content": "hi"}]},
    )
    # v1.5.0: an unknown model id is refused instead of echoed.
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# Role handling
# --------------------------------------------------------------------------- #
def test_system_and_user_roles_reach_the_prompt(client):
    c, captured = client
    c.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "add two numbers"},
            ]
        },
    )
    prompt = captured["prompt"]
    assert "You are terse." in prompt
    assert "add two numbers" in prompt


def test_legacy_contract_refuses_multi_turn(client):
    """v1 cannot express roles, so it must refuse instead of emitting delimiters."""
    c, captured = client
    r = c.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ]
        },
    )
    assert r.status_code == 400
    assert "legacy contract" in r.json()["error"]["message"]
    assert "prompt" not in captured


def test_v2_assistant_turn_preserved_in_history_order(client, monkeypatch):
    from training.tokenizer_contract import build_contract_tokenizer

    c, captured = client
    monkeypatch.setattr(
        openai_server.state,
        "tokenizer",
        build_contract_tokenizer("tokenizer/qm_code_bpe_v0.json"),
        raising=False,
    )
    r = c.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ]
        },
    )
    assert r.status_code == 200
    prompt = captured["prompt"]
    assert prompt.index("first") < prompt.index("second") < prompt.index("third")
    assert "<|assistant|>second<|end_of_turn|>" in prompt
    assert prompt.endswith("<|assistant|>")


def test_unsupported_role_rejected(client):
    c, _ = client
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_role"
    assert "tool" in err["message"]


def test_system_only_messages_rejected(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "only a system prompt"}]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages"


def test_empty_messages_rejected(client):
    c, _ = client
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 422


# --------------------------------------------------------------------------- #
# Structured errors and rejection of unsupported input
# --------------------------------------------------------------------------- #
def test_streaming_rejected_clearly(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "streaming_unsupported"
    assert err["param"] == "stream"


@pytest.mark.parametrize(
    "param,value",
    [
        ("tools", [{"type": "function"}]),
        ("functions", [{"name": "f"}]),
        ("tool_choice", "auto"),
        ("n", 2),
        ("logprobs", True),
        ("logit_bias", {"1": 1}),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_unsupported_params_rejected_not_ignored(client, param, value):
    """An ignored param would let a caller believe a tool loop or n>1 is running."""
    c, _ = client
    body = {"messages": [{"role": "user", "content": "hi"}], param: value}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 400, f"{param} must be rejected, not silently dropped"
    err = r.json()["error"]
    assert err["code"] == "unsupported_parameter"
    assert err["param"] == param


def test_error_envelope_shape(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert isinstance(err["message"], str)


def test_errors_do_not_leak_paths_or_tracebacks(client):
    c, _ = client
    openai_server.state.ckpt_path = openai_server.Path("runs/secret/internal_ckpt.pt")
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "bad", "content": "x"}]})
    text = r.text
    assert "Traceback" not in text
    assert "internal_ckpt" not in text
    assert ".pt" not in text


def test_no_endpoint_loads_a_checkpoint_from_a_request(client):
    """Regression guard: there must be no request-driven weight loading."""
    c, _ = client
    for route in ("/load", "/v1/load", "/reload"):
        for method in (c.get, c.post):
            resp = method(route, params={"ckpt": "runs/code_pretrain_tiny/checkpoint_last.pt"})
            assert resp.status_code == 404, f"{method.__name__} {route} must not exist"


# --------------------------------------------------------------------------- #
# Context limits
# --------------------------------------------------------------------------- #
def test_context_length_exceeded(client):
    c, _ = client
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 9000}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert "4096" in err["message"]


def test_max_tokens_clamped_to_remaining_context(client):
    """With a narrow window and a long prompt, the budget must shrink to fit."""
    c, captured = client
    from serving.openai_server import ChatMessage

    message = [{"role": "user", "content": "x" * 1000}]
    # Size the window from the real prompt so the assertion does not depend on the
    # exact length of the built-in system prompt.
    prompt_tokens = len(openai_server.build_prompt([ChatMessage(**m) for m in message]))
    openai_server.state.model_cfg["max_seq_len"] = prompt_tokens + 50

    c.post("/v1/chat/completions", json={"messages": message, "max_tokens": 1000})
    kwargs = captured["kwargs"]
    assert kwargs["max_new_tokens"] <= 50
    assert kwargs["max_new_tokens"] >= 1


def test_max_tokens_ceiling_applied(client):
    c, captured = client
    c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10**9},
    )
    assert captured["kwargs"]["max_new_tokens"] <= openai_server.MAX_TOKENS_CEILING


def test_non_positive_max_tokens_rejected(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "max_tokens"


# --------------------------------------------------------------------------- #
# Termination semantics
# --------------------------------------------------------------------------- #
def test_finish_reason_stop_is_reported(client):
    c, _ = client
    body = c.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    ).json()
    assert body["choices"][0]["finish_reason"] == "stop"


def test_finish_reason_length_is_reported(client):
    c, captured = client
    captured["force_finish"] = "length"
    body = c.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    ).json()
    assert body["choices"][0]["finish_reason"] == "length"


# --------------------------------------------------------------------------- #
# Sampling options reach the generator
# --------------------------------------------------------------------------- #
def test_sampling_options_are_forwarded(client):
    c, captured = client
    c.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["\n\n"],
            "seed": 7,
        },
    )
    kwargs = captured["kwargs"]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["stop"] == ["\n\n"]
    assert kwargs["seed"] == 7


def test_single_stop_string_is_normalised_to_list(client):
    c, captured = client
    c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stop": "END"},
    )
    assert captured["kwargs"]["stop"] == ["END"]


def test_defaults_applied_when_omitted(client):
    c, captured = client
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    kwargs = captured["kwargs"]
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 1.0


# --------------------------------------------------------------------------- #
# health + auth
# --------------------------------------------------------------------------- #
def test_health_reports_loaded_state(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == "qm-code-tiny"


def test_chat_requires_loaded_model(client):
    c, _ = client
    openai_server.state.model = None
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_bearer_auth_enforced_when_key_set(client):
    c, _ = client
    openai_server.state.api_key = "s3cret"
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"

    ok = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 200


def test_auth_is_off_by_default(client):
    c, _ = client
    assert openai_server.state.api_key is None
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Startup safety defaults
# --------------------------------------------------------------------------- #
def test_default_host_is_loopback():
    """Local-only must be the default; remote binding requires an explicit opt-in.

    The default now lives in the shared `server_security.add_standard_args` helper so
    all three servers agree, so that is what is asserted here.
    """
    import argparse

    from serving.server_security import add_standard_args, resolve_bind_host

    parser = argparse.ArgumentParser()
    add_standard_args(parser, default_port=8002)
    args = parser.parse_args(["--ckpt", "x.pt"])

    assert args.host == "127.0.0.1"
    assert args.allow_remote is False
    assert args.port == 8002
    assert resolve_bind_host(args.host, args.allow_remote) == "127.0.0.1"


# --------------------------------------------------------------------------- #
# Integration: real checkpoint through the existing loader.
#
# Loads weights and asserts schema only. It never runs generation, so it stays fast
# and cannot produce a fabricated capability claim. Skipped when weights are absent.
# --------------------------------------------------------------------------- #
REPO_ROOT = openai_server.Path(__file__).resolve().parent.parent
VERIFIED_CKPT = REPO_ROOT / "runs" / "code_pretrain_tiny" / "checkpoint_last.pt"
VERIFIED_TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"


@pytest.mark.skipif(
    not (VERIFIED_CKPT.exists() and VERIFIED_TOKENIZER.exists()),
    reason="verified checkpoint/tokenizer not present locally",
)
def test_verified_checkpoint_loads_and_is_discoverable():
    from tokenizers import Tokenizer

    from serving.local_inference import load_model

    model, cfg = load_model(str(VERIFIED_CKPT), "cpu")
    tok = Tokenizer.from_file(str(VERIFIED_TOKENIZER))

    # state-dict compatibility and tokenizer fit are the two real compatibility gates.
    assert cfg["vocab_size"] == tok.get_vocab_size(), (
        "tokenizer vocab must not exceed the model embedding width"
    )
    ids = tok.encode("def add(a, b):\n    return a + b\n").ids
    assert max(ids) < cfg["vocab_size"]
    assert model.num_params() == 495_808

    openai_server.state.model = model
    openai_server.state.model_cfg = cfg
    openai_server.state.tokenizer = tok
    openai_server.state.model_id = cfg.get("name")
    openai_server.state.checkpoint_step = 200
    openai_server.state.api_key = None
    openai_server.state.ckpt_path = VERIFIED_CKPT

    body = TestClient(app).get("/v1/models").json()
    entry = body["data"][0]
    assert entry["id"] == "qm-code-tiny"
    assert entry["quantsmind"]["context_window"] == 512
    assert entry["quantsmind"]["vocab_size"] == 5822
    assert entry["quantsmind"]["n_params_non_embedding"] == 495_808
    assert entry["created"] > 0


def test_stale_base_surface_does_not_govern_a_different_tokenizer(monkeypatch):
    """A surface decided for a pretraining checkpoint must not leak onward."""
    from training.tokenizer_contract import build_contract_tokenizer

    tokenizer = build_contract_tokenizer("tokenizer/qm_code_bpe_v0.json")
    monkeypatch.setattr(openai_server.state, "tokenizer", tokenizer, raising=False)
    monkeypatch.setattr(openai_server.state, "prompt_surface", "base")
    monkeypatch.setattr(
        openai_server.state, "surface_tokenizer", object(), raising=False
    )
    prompt = openai_server.build_prompt(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
    )
    assert "<|user|>" in prompt and "<|assistant|>" in prompt
    assert prompt.index("first") < prompt.index("second") < prompt.index("third")


def test_base_surface_accepts_dict_messages(monkeypatch):
    tokenizer = object()
    monkeypatch.setattr(openai_server.state, "tokenizer", tokenizer, raising=False)
    monkeypatch.setattr(openai_server.state, "prompt_surface", "base")
    monkeypatch.setattr(
        openai_server.state, "surface_tokenizer", tokenizer, raising=False
    )
    assert openai_server.build_prompt(
        [{"role": "user", "content": "add"}]
    ) == "add"
    with pytest.raises(ValueError):
        openai_server.build_prompt(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ]
        )