"""Regression tests for A1: no HTTP request may trigger checkpoint loading.

The original defect, in both ``serving/completion_server.py`` and
``serving/review_server.py``, was a ``POST /load`` route that took a ``ckpt`` query
parameter and called ``torch.load(weights_only=False)`` on it, with no authentication
and ``--host`` defaulting to ``0.0.0.0``. Any client that could reach the port could
execute arbitrary code by pointing ``ckpt`` at a crafted pickle.

These tests assert the vulnerability cannot be reintroduced silently. They never build
a payload checkpoint and never execute model inference.
"""

import inspect

import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient

from serving import completion_server, review_server
from serving import server_security as sec

LEGACY_MODULES = [completion_server, review_server]


# --------------------------------------------------------------------------- #
# 1. No request-triggered checkpoint loading
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_no_load_route_exists(mod):
    paths = [r.path for r in mod.app.routes if hasattr(r, "path")]
    assert "/load" not in paths, f"{mod.__name__} still exposes a /load route"


@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_load_route_absent_for_every_http_verb(mod):
    client = TestClient(mod.app)
    for method in (client.get, client.post, client.put, client.delete, client.patch):
        for path in ("/load", "/v1/load", "/reload", "/load_model"):
            resp = method(path, params={"ckpt": "evil.pt"})
            assert resp.status_code == 404, f"{method.__name__} {path} must not exist"


@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_no_handler_takes_a_checkpoint_or_tokenizer_parameter(mod):
    """Belt and braces: no request handler may accept a filesystem path."""
    offenders = []
    for route in mod.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        for name in inspect.signature(endpoint).parameters:
            lowered = name.lower()
            if "ckpt" in lowered or "checkpoint" in lowered or "tokenizer" in lowered:
                offenders.append(f"{route.path}.{endpoint.__name__}({name})")
    assert not offenders, f"handlers still accept filesystem paths: {offenders}"


@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_checkpoint_cannot_be_supplied_in_a_request_body(mod):
    """A `ckpt` field in the body must be inert: no route may act on it.

    503 (model not loaded) is the expected outcome and is itself the proof -- if the
    field had been honoured, the server would have attempted to deserialise it.
    """
    client = TestClient(mod.app)
    mod._model = None
    body = {"ckpt": "evil.pt", "tokenizer": "evil.json", "prefix": "x", "code": "x = 1"}
    for path in ("/complete", "/review", "/load"):
        resp = client.post(path, json=body)
        assert resp.status_code in (404, 422, 503), f"{path} accepted unexpected input"
        assert "evil" not in resp.text
    assert mod._model is None, "a request must never populate server state"
    assert mod._tokenizer is None, "a request must never populate server state"


def test_openai_server_also_free_of_load_routes():
    from serving import openai_server

    paths = [r.path for r in openai_server.app.routes if hasattr(r, "path")]
    assert "/load" not in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths


@pytest.mark.parametrize(
    "mod", LEGACY_MODULES + [__import__("serving.openai_server", fromlist=["x"])],
    ids=lambda m: m.__name__,
)
def test_handlers_do_not_call_torch_load(mod):
    """No route handler may deserialise a checkpoint, whatever the argument source."""
    for route in mod.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        src = inspect.getsource(endpoint)
        assert "torch.load" not in src, f"{route.path} calls torch.load"
        assert "load_model" not in src, f"{route.path} calls load_model"
        assert "load_trusted_state" not in src, f"{route.path} calls load_trusted_state"


# --------------------------------------------------------------------------- #
# 2. Loopback by default, remote requires explicit opt-in
# --------------------------------------------------------------------------- #
def test_is_loopback_recognises_localhost_forms():
    for host in ("127.0.0.1", "localhost", "::1"):
        assert sec.is_loopback(host) is True
    for host in ("0.0.0.0", "192.168.1.10", "10.0.0.5", "example.com", ""):
        assert sec.is_loopback(host) is False


def test_resolve_bind_host_defaults_to_loopback():
    assert sec.resolve_bind_host("127.0.0.1", allow_remote=False) == "127.0.0.1"


def test_resolve_bind_host_refuses_remote_without_optin():
    for host in ("0.0.0.0", "192.168.1.10"):
        with pytest.raises(SystemExit):
            sec.resolve_bind_host(host, allow_remote=False)


def test_resolve_bind_host_allows_remote_with_optin():
    assert sec.resolve_bind_host("0.0.0.0", allow_remote=True) == "0.0.0.0"
    assert sec.resolve_bind_host("192.168.1.10", allow_remote=True) == "192.168.1.10"


def test_resolve_bind_host_empty_host_is_not_silently_loopback():
    with pytest.raises(SystemExit):
        sec.resolve_bind_host("", allow_remote=False)


def test_legacy_server_argparse_defaults_to_loopback():
    """The old default was 0.0.0.0. The shared flag must default to 127.0.0.1."""
    import argparse

    parser = argparse.ArgumentParser()
    sec.add_standard_args(parser, default_port=8000)
    args = parser.parse_args(["--ckpt", "x.pt"])
    assert args.host == "127.0.0.1"
    assert args.allow_remote is False


@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_no_server_binds_zero_by_default(mod):
    src = inspect.getsource(mod)
    assert 'default="0.0.0.0"' not in src, f"{mod.__name__} still defaults to 0.0.0.0"


# --------------------------------------------------------------------------- #
# 3. Remote-access safeguards
# --------------------------------------------------------------------------- #
def test_is_authorized_allows_everything_without_a_key():
    assert sec.is_authorized(None, None) is True
    assert sec.is_authorized("Bearer anything", None) is True


def test_is_authorized_requires_exact_bearer_token():
    assert sec.is_authorized("Bearer s3cret", "s3cret") is True
    for bad in (None, "", "Bearer wrong", "s3cret", "Basic s3cret", "bearer s3cret"):
        assert sec.is_authorized(bad, "s3cret") is False


def test_bearer_middleware_absent_without_key():
    assert sec.bearer_auth_middleware(None) is None
    assert sec.bearer_auth_middleware("") is None
    assert sec.bearer_auth_middleware("k") is not None


def test_install_auth_reports_whether_it_took_effect():
    from fastapi import FastAPI

    assert sec.install_auth(FastAPI(), None) is False
    assert sec.install_auth(FastAPI(), "k") is True


def test_unauthorized_response_shape():
    resp = sec.unauthorized_response()
    assert resp.status_code == 401
    assert "invalid_api_key" in resp.body.decode()


# --------------------------------------------------------------------------- #
# 4. Hardened checkpoint deserialisation
# --------------------------------------------------------------------------- #
def test_torch_load_checkpoint_prefers_weights_only(monkeypatch):
    """weights_only=True is attempted first; the unsafe path is a loud last resort."""
    from serving import local_inference

    calls = []

    def fake_load(path, map_location=None, weights_only=False):
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("simulated unsafe-pickle rejection")
        return {"ok": True}

    monkeypatch.setattr(local_inference.torch, "load", fake_load)
    result = local_inference.torch_load_checkpoint("x.pt")

    assert calls[0] is True, "safe deserialisation must be attempted first"
    assert calls[-1] is False, "documented fallback must still load the checkpoint"
    assert result == {"ok": True}


def test_torch_load_checkpoint_does_not_fall_back_when_safe_load_works(monkeypatch):
    from serving import local_inference

    calls = []

    def fake_load(path, map_location=None, weights_only=False):
        calls.append(weights_only)
        return {"model": {}}

    monkeypatch.setattr(local_inference.torch, "load", fake_load)
    local_inference.torch_load_checkpoint("x.pt")
    assert calls == [True], "must not weaken to weights_only=False unnecessarily"


def test_real_checkpoint_still_loads_through_hardened_loader():
    """The fallback must not break real loading. Metadata only, no inference."""
    from pathlib import Path

    from serving.local_inference import load_model, torch_load_checkpoint

    root = Path(__file__).resolve().parent.parent
    ckpt = root / "runs" / "code_pretrain_tiny" / "checkpoint_last.pt"
    if not ckpt.exists():
        pytest.skip("verified checkpoint not present locally")

    blob = torch_load_checkpoint(str(ckpt))
    assert "model" in blob and "step" in blob

    model, cfg = load_model(str(ckpt), "cpu")
    assert cfg["vocab_size"] == 5822
    assert model.num_params() == 495_808


# --------------------------------------------------------------------------- #
# 5. Existing behaviour preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_health_contract_preserved(mod):
    client = TestClient(mod.app)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert body["model_loaded"] is False  # nothing loaded in a test process


@pytest.mark.parametrize("mod", LEGACY_MODULES, ids=lambda m: m.__name__)
def test_inference_routes_still_present(mod):
    paths = [r.path for r in mod.app.routes if hasattr(r, "path")]
    expected = "/complete" if mod is completion_server else "/review"
    assert expected in paths
    assert "/health" in paths


def test_complete_returns_503_when_model_not_loaded():
    client = TestClient(completion_server.app)
    completion_server._model = None
    resp = client.post("/complete", json={"prefix": "def add(a, b):"})
    assert resp.status_code == 503
    # The error must point at the new startup mechanism, not a removed /load route.
    assert "Call /load" not in resp.text
    assert "--ckpt" in resp.text


def test_review_returns_503_when_model_not_loaded():
    client = TestClient(review_server.app)
    review_server._model = None
    resp = client.post("/review", json={"code": "x = 1"})
    assert resp.status_code == 503


def test_static_analysis_still_works():
    """The regex linter is unchanged; /review must still surface real findings."""
    issues = review_server.static_analysis("eval(user_input)", "python")
    assert any("eval" in i.message for i in issues)
    assert all(i.suggestion for i in issues)


def test_review_response_schema_preserved():
    assert set(review_server.ReviewResponse.model_fields) == {"issues", "summary", "model"}
    assert set(review_server.ReviewIssue.model_fields) == {
        "severity",
        "line",
        "message",
        "suggestion",
    }


def test_completion_response_schema_preserved():
    assert set(completion_server.CompletionResponse.model_fields) == {
        "completions",
        "scores",
        "model",
    }
    fields = completion_server.CompletionRequest.model_fields
    for name in ("prefix", "suffix", "language", "max_tokens", "temperature", "top_k"):
        assert name in fields, f"/complete contract lost {name}"
    assert "ckpt" not in fields and "tokenizer" not in fields