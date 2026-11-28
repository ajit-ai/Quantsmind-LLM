"""Independence guards for the local Taksh coding backend.

These tests do not prove the absence of every possible network call. They pin
the structural property that matters: the coding API path is local-only, has no
provider fallback, no hosted base URL, no request-driven weight load, and the
declared external providers cannot generate anything.

The rule is simple: Taksh is the only model. A VSCode client is a client, not a
second model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_API_MODULES = [
    "serving/openai_server.py",
    "serving/local_inference.py",
    "serving/fim.py",
    "serving/tool_protocol.py",
    "serving/server_security.py",
]
NETWORK_CLIENTS = {
    "aiohttp",
    "httpx",
    "openai",
    "requests",
    "socket",
    "urllib",
    "urllib3",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("relative", LOCAL_API_MODULES)
def test_local_api_path_imports_no_network_client(relative):
    roots = _imported_roots(REPO_ROOT / relative)
    assert not (roots & NETWORK_CLIENTS), f"{relative} imports {roots & NETWORK_CLIENTS}"


def test_local_api_does_not_import_the_provider_registry():
    """No external-provider fallback can be reached from the coding API."""
    roots = _imported_roots(REPO_ROOT / "serving" / "openai_server.py")
    assert "providers" not in roots


def test_coding_api_exposes_only_local_routes():
    from serving.openai_server import app

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert {
        "/v1/models",
        "/health",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/fim/completions",
    } <= paths
    forbidden = {"/load", "/v1/load", "/reload", "/v1/proxy", "/proxy"}
    assert not (paths & forbidden)


def test_declared_external_providers_cannot_generate():
    pytest.importorskip("torch")
    from serving.providers import (
        GenerationConfig,
        GenerationRequest,
        ProviderUnavailableError,
        get_provider,
    )

    request = GenerationRequest(prompt="def f():", config=GenerationConfig(max_new_tokens=1))
    for name in ("qwen2.5-coder-7b-instruct", "qwen3-coder-30b-a3b-instruct"):
        provider = get_provider(name)
        capabilities = provider.capabilities()
        assert capabilities.available is False
        assert capabilities.offline is True
        with pytest.raises(ProviderUnavailableError):
            provider.generate(request)


def test_no_remote_model_base_url_is_configured_in_the_coding_server():
    """The OpenAI wire format is a local protocol, not an OpenAI dependency."""
    from serving import openai_server

    assert not hasattr(openai_server.state, "base_url")
    assert not hasattr(openai_server.state, "api_base")
    assert openai_server.state.model is None or hasattr(
        openai_server.state.model, "parameters"
    )
