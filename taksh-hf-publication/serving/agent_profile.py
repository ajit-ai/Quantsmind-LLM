"""Experimental Taksh provider profile for coding-agent integration (v1.4.0).

The profile is a *declaration* of what a local Taksh deployment actually
offers, written so an external coding agent (QuantsMind-Agent, OpenCode,
Cline) can be configured against it without being told something the server
does not do.

Two rules shape it:

* **Nothing unsupported is advertised.** The OpenAI-compatible server in
  ``serving/openai_server.py`` exposes ``GET /v1/models``, ``GET /health``
  and ``POST /v1/chat/completions``. It has no plain-completion endpoint, it
  returns non-streaming responses, and it *rejects* tool, function, n,
  logprobs and response_format requests with an explicit error. The profile
  says exactly that.
* **Tool calling is reported as unsupported**, not as "maybe". A profile that
  implied tool support would cause an agent to send tool requests and then
  fail at runtime.

The integration status is ``EXTERNAL_AGENT_INTEGRATION_PENDING``: the
contract tests in ``evaluation/agent_contract`` exercise the wire format
against the real application, but QuantsMind-Agent lives in another
repository, so real interoperability has not been demonstrated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

PROFILE_SCHEMA_VERSION = "quantsmind.agent_profile/1"
INTEGRATION_STATUS_PENDING = "EXTERNAL_AGENT_INTEGRATION_PENDING"


@dataclass
class AgentProviderProfile:
    """What a local Taksh deployment offers to a coding agent."""

    provider_id: str = "taksh-local"
    model_id: str = "qm-code-m3-9m"
    api_style: str = "openai_chat_completions"
    base_url: str = "http://127.0.0.1:8000"
    supported_endpoints: list = field(default_factory=lambda: [
        "GET /v1/models",
        "GET /health",
        "POST /v1/chat/completions",
    ])
    request_types: list = field(default_factory=lambda: ["chat_completion"])
    context_window: int = 512
    max_generation_tokens: int = 256
    max_generation_tokens_ceiling: int = 4096
    streaming_supported: bool = False
    streaming_status: str = "not_implemented: responses are always a single JSON object"
    tool_calling_supported: bool = False
    tool_calling_status: str = (
        "unsupported: the server rejects tools/functions/tool_choice with an "
        "explicit error rather than ignoring them"
    )
    structured_output_supported: bool = False
    structured_output_status: str = (
        "unsupported: response_format is rejected with an explicit error"
    )
    authentication: str = "optional bearer token via TAKSH_API_KEY; the middleware is always active"
    recommended_timeouts: dict = field(default_factory=lambda: {
        "connect_seconds": 5,
        "read_seconds": 120,
        "overall_seconds": 180,
    })
    known_limitations: list = field(default_factory=lambda: [
        "The model is a 9.36M-parameter base model; it is not instruction-tuned.",
        "No plain /v1/completions endpoint: use /v1/chat/completions.",
        "Responses are never streamed.",
        "No tool calling and no structured output.",
        "The base model is served as raw text; the chat template is not applied.",
        "No KV cache: generation re-runs a full forward pass per token, so long "
        "generations are slow on CPU.",
    ])
    integration_status: str = INTEGRATION_STATUS_PENDING
    notes: str = (
        "Experimental profile. The wire contract is exercised offline against "
        "the real application; interoperability with QuantsMind-Agent itself "
        "has not been demonstrated."
    )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema"] = PROFILE_SCHEMA_VERSION
        return payload

    def summary(self) -> str:
        return (
            f"{self.provider_id}/{self.model_id} | endpoints: "
            f"{', '.join(self.supported_endpoints)} | streaming: "
            f"{self.streaming_supported} | tool calling: "
            f"{self.tool_calling_supported} | integration: "
            f"{self.integration_status}"
        )


def default_profile(model_id: str = "qm-code-m3-9m", **overrides) -> AgentProviderProfile:
    """The profile for a local Taksh deployment."""
    profile = AgentProviderProfile(model_id=model_id)
    for key, value in overrides.items():
        setattr(profile, key, value)
    return profile


def profile_to_json(profile: AgentProviderProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True)
