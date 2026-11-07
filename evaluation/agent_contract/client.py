"""A simulated coding-agent client for the Taksh serving contract.

The client is written the way an external agent would write it: it sends
JSON, it reads JSON, it treats an error body as an error, and it has a
timeout. The transport is injectable, so the same client code runs against
the real FastAPI application in-process (no sockets, no network).

What the client deliberately does *not* do is pretend to be more capable
than the server: there is no tool-calling helper, no streaming reader, and no
retry loop that would hide a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class AgentClientError(RuntimeError):
    """The server refused a request, or answered with something unusable."""

    def __init__(self, message: str, *, status: int | None = None,
                 code: str | None = None, payload=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.payload = payload


@dataclass
class TakshTransport:
    """In-process transport over the FastAPI application."""

    app: object
    api_key: str | None = None
    calls: list = field(default_factory=list)

    def post(self, path: str, payload: dict) -> tuple:
        from fastapi.testclient import TestClient

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with TestClient(self.app) as client:
            response = client.post(path, json=payload, headers=headers)
        self.calls.append({"method": "POST", "path": path, "payload": payload})
        return response.status_code, response.json()

    def get(self, path: str) -> tuple:
        from fastapi.testclient import TestClient

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with TestClient(self.app) as client:
            response = client.get(path, headers=headers)
        self.calls.append({"method": "GET", "path": path})
        return response.status_code, response.json()


class SimulatedCodingAgentClient:
    """The subset of an OpenAI-style client a coding agent actually needs."""

    def __init__(self, transport, model: str = "qm-code-m3-9m",
                 timeout_seconds: int = 120):
        self.transport = transport
        self.model = model
        self.timeout_seconds = int(timeout_seconds)
        self._models: list | None = None

    # ------------------------------------------------------------------ errors
    @staticmethod
    def _raise_for_error(status: int, payload) -> None:
        if status < 400:
            return
        error = (payload or {}).get("error") or {}
        raise AgentClientError(
            error.get("message") or f"request failed with status {status}",
            status=status,
            code=error.get("code"),
            payload=payload,
        )

    def served_models(self, refresh: bool = False) -> list:
        """Model identifiers the server reports, cached after the first call.

        The Taksh server echoes whatever ``model`` a request carries instead of
        rejecting an unknown identifier, so the *client* is the place that has
        to notice a model it cannot use. That check lives here.
        """
        if refresh or self._models is None:
            status, payload = self.transport.get("/v1/models")
            self._raise_for_error(status, payload)
            self._models = [item.get("id") for item in payload.get("data", [])]
        return list(self._models)

    # ------------------------------------------------------------------ calls
    def list_models(self) -> list:
        return self.served_models(refresh=True)

    def health(self) -> dict:
        status, payload = self.transport.get("/health")
        self._raise_for_error(status, payload)
        return payload

    def chat(
        self,
        messages: list,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        stop=None,
    ) -> dict:
        """Send a chat completion and return the parsed response."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        requested = payload["model"]
        if requested not in self.served_models():
            raise AgentClientError(
                f"model {requested!r} is not served by this endpoint",
                status=404,
                code="model_not_found",
            )
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if stop is not None:
            payload["stop"] = stop
        status, response = self.transport.post("/v1/chat/completions", payload)
        self._raise_for_error(status, response)
        if not isinstance(response, dict) or "choices" not in response:
            raise AgentClientError(
                "response is not a chat-completion object", status=status,
                payload=response,
            )
        return response

    def chat_text(self, messages: list, **kwargs) -> str:
        """The convenience an agent actually uses: the assistant text."""
        response = self.chat(messages, **kwargs)
        choices = response.get("choices") or []
        if not choices:
            raise AgentClientError("response contained no choices", payload=response)
        return (choices[0].get("message") or {}).get("content", "")

    def request_tools(self, messages: list, tools: list, **kwargs) -> dict:
        """Attempt a tool-calling request.

        The Taksh server rejects these explicitly, and this helper surfaces
        that rejection instead of hiding it. It exists so the contract test
        can assert the *refusal* is clean.
        """
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "tools": tools,
        }
        status, response = self.transport.post("/v1/chat/completions", payload)
        self._raise_for_error(status, response)
        return response
