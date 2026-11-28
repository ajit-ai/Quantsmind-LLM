"""Declared Taksh provider profile for coding-agent integration.

The profile is a *declaration* of what a local Taksh deployment actually
offers, written so an external coding agent (QuantsMind-Agent, OpenCode,
Cline, a VSCode extension) can be configured against it without being told
something the server does not do.

Three rules shape it:

* **Nothing unsupported is advertised.** The server exposes model discovery,
  health, chat completions, legacy text completions and a native FIM endpoint.
  It streams over SSE. It parses a documented text tool-call block. It does
  *not* produce structured output, and it does not have native tool tokens or
  tool-use training.
* **Protocol support is not capability.** ``tool_call_parsing_supported`` is
  true because the server accepts ``tools`` and understands the documented
  reply shape. ``tool_calling_supported`` remains false because the checkpoint
  has never been trained to emit a call; an agent that depends on tool calls
  will stall. Conflating those two would be the dishonest version of this file.
* **Streaming is real streaming.** Deltas come from the shared token loop, not
  from replaying a finished string. Binding repair is not applied to streamed
  text because repair is a whole-completion transform; the profile says so.

The integration status is ``EXTERNAL_AGENT_INTEGRATION_PENDING``: the wire
contract is exercised offline against the real application, but interoperability
with an external agent product has not been demonstrated end to end.
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
    model_id: str = "qm-code-v160-15m"
    api_style: str = "openai_chat_completions"
    base_url: str = "http://127.0.0.1:8002"
    supported_endpoints: list = field(default_factory=lambda: [
        "GET /v1/models",
        "GET /health",
        "POST /v1/chat/completions",
        "POST /v1/completions",
        "POST /v1/fim/completions",
    ])
    request_types: list = field(default_factory=lambda: [
        "chat_completion",
        "text_completion",
        "fim_completion",
    ])
    context_window: int = 2048
    max_generation_tokens: int = 256
    max_generation_tokens_ceiling: int = 4096
    streaming_supported: bool = True
    streaming_status: str = (
        "implemented: Server-Sent Events from the shared token loop. Binding "
        "repair is not applied to streamed text because it is a whole-completion "
        "transform."
    )
    tool_calling_supported: bool = False
    tool_call_parsing_supported: bool = True
    tool_calling_status: str = (
        "protocol only: tools are rendered into the prompt and a documented "
        "<tool_call> block is parsed back. The checkpoint has no tool-use "
        "training and is not expected to emit one. tool_choice accepts only "
        "'auto' and 'none'; streaming with tools is refused."
    )
    structured_output_supported: bool = False
    structured_output_status: str = (
        "unsupported: response_format is rejected with an explicit error"
    )
    authentication: str = "optional bearer token via QM_API_KEY; the middleware is always active"
    recommended_timeouts: dict = field(default_factory=lambda: {
        "connect_seconds": 5,
        "read_seconds": 120,
        "overall_seconds": 180,
    })
    known_limitations: list = field(default_factory=lambda: [
        "The current checkpoint scores 0/20 on the frozen Native v0.1 baseline; "
        "the API does not create coding capability.",
        "2,048-token context window; injected repository context must be small.",
        "No native tool-use training: the tool block is a protocol shim only.",
        "No FIM training: the FIM endpoints use literal text sentinels, not "
        "tokenizer control tokens, and no completion quality is claimed.",
        "The shipped v1 tokenizers cannot express roles, so multi-turn chat is "
        "refused unless the server is started with --flatten-history.",
        "Binding repair is applied only to buffered responses, never to streams.",
    ])
    integration_status: str = INTEGRATION_STATUS_PENDING
    notes: str = (
        "Experimental profile. The wire contract is exercised offline against "
        "the real application; interoperability with an external agent product "
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
            f"{self.tool_calling_supported} (parsing: "
            f"{self.tool_call_parsing_supported}) | integration: "
            f"{self.integration_status}"
        )


def default_profile(model_id: str = "qm-code-v160-15m", **overrides) -> AgentProviderProfile:
    """The profile for a local Taksh deployment."""
    profile = AgentProviderProfile(model_id=model_id)
    for key, value in overrides.items():
        setattr(profile, key, value)
    return profile


def profile_to_json(profile: AgentProviderProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, sort_keys=True)
