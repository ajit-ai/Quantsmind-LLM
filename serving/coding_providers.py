"""Provider implementations for the TAKSH coding agent.

Mocks are explicitly labelled test providers. They are not evidence of model
capability.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderError,
    ProviderModelIdentity,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    list_providers,
    redact_secrets,
    register_provider,
)


class Transport(Protocol):
    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]: ...


class ProbeTransport(Protocol):
    def __call__(self, url: str, timeout: float) -> dict[str, Any]: ...


def _secret_values(headers: dict[str, str] | None = None) -> list[str]:
    values = [
        os.environ.get("TAKSH_OPENAI_API_KEY", ""),
        os.environ.get("TAKSH_OLLAMA_API_KEY", ""),
    ]
    for header, value in (headers or {}).items():
        if header.lower() == "authorization":
            values.append(value.removeprefix("Bearer ").strip())
    return [value for value in values if value]


def _http_detail(exc: urllib.error.HTTPError, headers: dict[str, str] | None = None) -> str:
    try:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - diagnostic only
        detail = ""
    return redact_secrets(detail, _secret_values(headers))


def _urllib_request_json(
    request: urllib.request.Request,
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = _http_detail(exc, headers)
        suffix = f": {detail}" if detail else ""
        if exc.code in (401, 403):
            raise ProviderAuthenticationError(
                f"provider rejected the configured credential (HTTP {exc.code}){suffix}"
            ) from exc
        if exc.code == 429:
            raise ProviderRateLimitError(f"provider rate limit reached (HTTP 429){suffix}") from exc
        if 500 <= exc.code < 600:
            raise ProviderServerError(f"provider server error (HTTP {exc.code}){suffix}") from exc
        raise ProviderResponseError(f"provider returned HTTP {exc.code}{suffix}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderTimeoutError(f"provider request timed out after {timeout}s") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise ProviderTimeoutError(f"provider request timed out after {timeout}s") from exc
        raise ProviderUnavailableError(
            f"provider is unreachable: {redact_secrets(reason, _secret_values(headers))}"
        ) from exc

    if not payload:
        raise ProviderResponseError("provider returned an empty response body")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("provider returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ProviderResponseError("provider returned a non-object JSON response")
    return data


def _urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    return _urllib_request_json(request, timeout, headers=headers)


def _urllib_probe(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    return _urllib_request_json(request, timeout)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    context_window: int = 8192
    offline: bool = False
    streaming: bool = False

    def validate(self) -> "ProviderConfig":
        if not self.provider:
            raise ValueError("provider must be set")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.base_url:
            parsed = urllib.parse.urlparse(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"base_url must be an http(s) URL, got {self.base_url!r}")
        return self

    @classmethod
    def from_env(cls) -> dict[str, "ProviderConfig"]:
        configs: dict[str, ProviderConfig] = {}
        if os.environ.get("TAKSH_OPENAI_API_KEY"):
            configs["openai-compatible"] = ProviderConfig(
                provider="openai-compatible",
                model=os.environ.get("TAKSH_OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("TAKSH_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ["TAKSH_OPENAI_API_KEY"],
                context_window=_env_int("TAKSH_OPENAI_CONTEXT_WINDOW", 8192),
            )
        configs["ollama"] = ProviderConfig(
            provider="ollama",
            model=os.environ.get("TAKSH_OLLAMA_MODEL", ""),
            base_url=os.environ.get("TAKSH_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            context_window=_env_int("TAKSH_OLLAMA_CONTEXT_WINDOW", 8192),
            offline=True,
        )
        return configs


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = (
            config
            or ProviderConfig.from_env().get("openai-compatible")
            or ProviderConfig(self.name, "", api_key=None)
        )
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
        configured = bool(self.config.api_key and self.config.model and self.config.base_url)
        if configured:
            status = "configured"
        elif not self.config.api_key:
            status = "unauthenticated"
        else:
            status = "misconfigured"
        return ProviderCapabilities(
            provider=self.name,
            available=configured,
            offline=False,
            configured=configured,
            status=status,
            modes=("chat", "text", "streaming") if configured else (),
            max_context_tokens=self.config.context_window,
            notes="Configured OpenAI-compatible endpoint; no credential is embedded or logged.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        self.config.validate()
        if not self.config.api_key:
            raise ProviderAuthenticationError(
                "openai-compatible provider credential is not configured; set TAKSH_OPENAI_API_KEY"
            )
        if not self.config.model:
            raise ProviderUnavailableError(
                "openai-compatible model is not configured; set TAKSH_OPENAI_MODEL"
            )
        started = time.perf_counter()
        url = (self.config.base_url or "").rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_new_tokens,
            "stream": False,
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        data = self.transport(url, body, headers, request.config.timeout_seconds or 120.0)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response contains no choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("provider returned empty or non-text content")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason=str(first.get("finish_reason", "unknown")),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "provider-output"},
            metadata={"endpoint": self.config.base_url},
        )


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        transport: Transport | None = None,
        probe_transport: ProbeTransport | None = None,
    ) -> None:
        self.config = config or ProviderConfig.from_env().get("ollama") or ProviderConfig(self.name, "")
        self.transport = transport or _urllib_transport
        self.probe_transport = probe_transport or _urllib_probe
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
        configured = bool(self.config.model)
        return ProviderCapabilities(
            provider=self.name,
            available=configured,
            offline=True,
            configured=configured,
            status="configured" if configured else "misconfigured",
            modes=("chat",) if configured else (),
            max_context_tokens=self.config.context_window,
            notes="Ollama local endpoint; reachability and model presence require an explicit probe.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        self.config.validate()
        if not self.config.model:
            raise ProviderUnavailableError(
                "ollama model is not configured; set TAKSH_OLLAMA_MODEL"
            )
        started = time.perf_counter()
        url = (self.config.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
        body = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": {
                "temperature": request.config.temperature,
                "num_predict": request.config.max_new_tokens,
            },
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        data = self.transport(url, body, headers, request.config.timeout_seconds or 120.0)

        if data.get("error"):
            detail = redact_secrets(str(data["error"]))
            if "not found" in detail.lower() or "no such model" in detail.lower():
                raise ProviderUnavailableError(
                    f"ollama model {self.config.model!r} is not available locally: {detail}"
                )
            raise ProviderResponseError(f"ollama returned an error: {detail}")

        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("ollama returned empty or non-text content")

        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason=str(data.get("done_reason", "stop")),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "provider-output"},
            metadata={"endpoint": self.config.base_url, "ollama": True},
        )

    def probe(self, timeout: float = 5.0) -> dict[str, Any]:
        """Explicitly check the Ollama service and configured model.

        This method is never called by ``capabilities()`` or ordinary
        diagnostics, so importing TAKSH never contacts a local service.
        """
        self.config.validate()
        url = (self.config.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/tags"
        data = self.probe_transport(url, timeout)
        raw_models = data.get("models") if isinstance(data.get("models"), list) else []
        names: list[str] = []
        for item in raw_models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if name:
                    names.append(str(name))
        model = self.config.model
        return {
            "provider": self.name,
            "endpoint": self.config.base_url,
            "reachable": True,
            "model": model,
            "model_present": bool(model and model in names),
            "models": sorted(names),
        }


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
            configured=True,
            status="test_only",
            test_only=True,
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
    return OpenAICompatibleProvider(
        config=kwargs.get("config"),
        transport=kwargs.get("transport"),
    )


def _ollama_factory(**kwargs: Any) -> OllamaProvider:
    return OllamaProvider(
        config=kwargs.get("config"),
        transport=kwargs.get("transport"),
        probe_transport=kwargs.get("probe_transport"),
    )


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
