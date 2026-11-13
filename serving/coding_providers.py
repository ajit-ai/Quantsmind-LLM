"""Provider implementations for the TAKSH coding agent.

Mocks are explicitly labelled test providers. They are not evidence of model
capability.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderModelIdentity,
    ProviderNotFoundError,
    ProviderUnavailableError,
    list_providers,
    register_provider,
)


class Transport(Protocol):
    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]: ...


def _urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailableError(f"provider is unreachable: {exc.reason}") from exc
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError("provider returned invalid JSON") from exc


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    context_window: int = 8192
    offline: bool = False
    streaming: bool = False

    @classmethod
    def from_env(cls) -> dict[str, "ProviderConfig"]:
        configs: dict[str, ProviderConfig] = {}
        if os.environ.get("TAKSH_OPENAI_API_KEY"):
            configs["openai-compatible"] = ProviderConfig(
                provider="openai-compatible",
                model=os.environ.get("TAKSH_OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("TAKSH_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ["TAKSH_OPENAI_API_KEY"],
                context_window=int(os.environ.get("TAKSH_OPENAI_CONTEXT_WINDOW", "8192")),
            )
        configs["ollama"] = ProviderConfig(
            provider="ollama",
            model=os.environ.get("TAKSH_OLLAMA_MODEL", ""),
            base_url=os.environ.get("TAKSH_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            context_window=int(os.environ.get("TAKSH_OLLAMA_CONTEXT_WINDOW", "8192")),
            offline=True,
        )
        return configs


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, config: ProviderConfig | None = None, transport: Transport | None = None) -> None:
        self.config = config or ProviderConfig.from_env().get("openai-compatible") or ProviderConfig(self.name, "", api_key=None)
        self.transport = transport or _urllib_transport
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id=self.config.model or "unconfigured",
            revision="configured",
            source=self.config.base_url or "unconfigured",
            license_or_rights="provider license applies; not enumerated here",
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        available = bool(self.config.api_key and self.config.model)
        return ProviderCapabilities(
            provider=self.name,
            available=available,
            offline=False,
            modes=("chat", "text", "streaming") if available else (),
            max_context_tokens=self.config.context_window,
            notes="Configured OpenAI-compatible endpoint; no credential is embedded or logged.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        if not self.config.api_key:
            raise ProviderUnavailableError("openai-compatible provider credential is not configured")
        started = time.perf_counter()
        url = (self.config.base_url or "").rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_new_tokens,
            "stream": False,
        }).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        data = self.transport(url, body, headers, request.config.timeout_seconds or 120.0)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason=str(data.get("choices", [{}])[0].get("finish_reason", "unknown")),
            prompt_tokens=int(data.get("usage", {}).get("prompt_tokens", 0)),
            completion_tokens=int(data.get("usage", {}).get("completion_tokens", 0)),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "provider-output"},
            metadata={"endpoint": self.config.base_url},
        )


class OllamaProvider:
    name = "ollama"

    def __init__(self, config: ProviderConfig | None = None, transport: Transport | None = None) -> None:
        self.config = config or ProviderConfig.from_env().get("ollama") or ProviderConfig(self.name, "")
        self.transport = transport or _urllib_transport
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id=self.config.model or "unconfigured",
            revision="local",
            source=self.config.base_url or "http://127.0.0.1:11434",
            license_or_rights="local model license applies; user is responsible for model rights",
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            available=bool(self.config.model),
            offline=True,
            modes=("chat",) if self.config.model else (),
            max_context_tokens=self.config.context_window,
            notes="Ollama local endpoint; configured model must be accessible locally.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        if not self.config.model:
            raise ProviderUnavailableError("ollama model is not configured; set TAKSH_OLLAMA_MODEL")
        started = time.perf_counter()
        url = (self.config.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
        body = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": {"temperature": request.config.temperature, "num_predict": request.config.max_new_tokens},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        data = self.transport(url, body, headers, request.config.timeout_seconds or 120.0)
        text = data.get("message", {}).get("content", "")
        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason=str(data.get("done_reason", "stop")),
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "provider-output"},
            metadata={"endpoint": self.config.base_url, "ollama": True},
        )


class DeterministicCodingProvider:
    """Deterministic test provider. Not evidence of model capability."""

    name = "deterministic-mock"

    def __init__(self) -> None:
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id="deterministic-mock-1",
            revision="builtin",
            source="test-double",
            license_or_rights="test double; no model rights claimed",
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            available=True,
            offline=True,
            modes=("chat", "code"),
            max_context_tokens=8192,
            notes="Deterministic mock provider for conformance tests; not a language model.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        prompt = request.prompt.lower()
        if "bug" in prompt or "fix" in prompt:
            text = "def broken(a, b):\n    return a + b\n"
        elif "test" in prompt:
            text = "def test_example():\n    assert True\n"
        else:
            text = "change proposal"
        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            latency_seconds=0.0,
            repair={"enabled": bool(request.config.binding_repair), "status": "mock"},
            metadata={"deterministic": True},
        )


def _openai_factory(**kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(transport=kwargs.get("transport"))


def _ollama_factory(**kwargs: Any) -> OllamaProvider:
    return OllamaProvider(transport=kwargs.get("transport"))


def register_coding_providers() -> None:
    for name, factory in (
        ("openai-compatible", _openai_factory),
        ("ollama", _ollama_factory),
        ("deterministic-mock", lambda **kwargs: DeterministicCodingProvider()),
    ):
        try:
            register_provider(name, factory)
        except ValueError:
            pass


register_coding_providers()
