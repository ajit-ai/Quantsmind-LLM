"""OpenAI-compatible local API for Taksh.

Exposes the minimum surface an OpenAI SDK client needs:

    GET  /v1/models
    POST /v1/chat/completions   (stream=false only; streaming is a follow-up slice)

Design notes
------------
* Weights are loaded **once at process start** from a CLI path. There is no HTTP
  endpoint accepting a checkpoint path. The existing ``/load`` endpoints on
  ``completion_server`` and ``review_server`` call ``torch.load(weights_only=False)``
  on a request-supplied path with no auth and bind 0.0.0.0 by default, which is
  arbitrary code execution. This module deliberately does not reproduce that.

* Binds 127.0.0.1 unless ``--allow-remote`` is passed explicitly.

* Optional bearer auth via ``QM_API_KEY``. Unset means keyless, which is safe only
  because the listener is loopback by default.

* The blocking torch decode runs in a worker thread, so the event loop never stalls.

* Unsupported parameters are rejected with a structured 400 rather than silently
  ignored, so a caller never mistakes an ignored request for a honoured one.

* Model-level limitations (no KV cache, untrained chat markers, small vocab, silent
  left-truncation) are in docs/OPEN_ISSUES.md and the README. This server is
  API-shaped plumbing over an experimental checkpoint; it does not make the
  checkpoint capable.
"""

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taksh import __version__
from training.chat_template import SYSTEM_PROMPT, render_chat
from serving.local_inference import generate_ids, torch_load_checkpoint
from serving.server_security import (
    add_standard_args,
    api_key_from_env,
    is_authorized,
    load_trusted_state,
    serve,
)

app = FastAPI(title="Taksh - OpenAI-compatible API", version=__version__)

# Real OpenAI parameters this slice does NOT implement. Silently ignoring them would
# let a caller believe a tool loop, n>1 sampling or logprob ranking is happening.
UNSUPPORTED_PARAMS = {
    "tools": "tool calling is not implemented in this slice",
    "functions": "legacy function calling is not implemented in this slice",
    "tool_choice": "tool calling is not implemented in this slice",
    "n": "n>1 sampling is not implemented in this slice",
    "logprobs": "logprob output is not implemented in this slice",
    "top_logprobs": "logprob output is not implemented in this slice",
    "logit_bias": "logit_bias is not implemented in this slice",
    "response_format": "structured output is not implemented in this slice",
    "stream_options": "streaming is not implemented in this slice",
}

DEFAULT_MAX_TOKENS = 256
MAX_TOKENS_CEILING = 4096


class State:
    model = None
    model_cfg = None
    tokenizer = None
    model_id = "unloaded"
    checkpoint_step = None
    ckpt_path = None
    api_key = None
    contract_id = None
    #: "base" for a pretraining checkpoint, "chat" for an instruction-qualified one.
    prompt_surface = "chat"
    #: The tokenizer object the surface decision was derived for. If a different
    #: tokenizer is installed, the surface is re-derived instead of serving a
    #: stale process-global decision.
    surface_tokenizer = None
    chat_qualified = False


state = State()


def _err(status, message, etype, param=None, code=None):
    """OpenAI-shaped error envelope. Never leaks paths, secrets or tracebacks."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "param": param, "code": code}},
    )


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Optional bearer auth. No-op when QM_API_KEY is unset (loopback default).

    The decision itself lives in ``server_security.is_authorized`` so all three
    servers agree on it.
    """
    if not is_authorized(request.headers.get("authorization"), state.api_key):
        return _err(
            401,
            "Invalid or missing API key.",
            "invalid_request_error",
            code="invalid_api_key",
        )
    return await call_next(request)


def require_loaded():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool | None = False
    user: str | None = None  # standard opaque client id; accepted and ignored

    # Declared explicitly so the handler can *reject* them. Without these fields
    # pydantic would drop them as unknown extras and the request would appear to
    # succeed while the caller's intent (tool loop, n>1, logprobs) was ignored.
    tools: Any = None
    functions: Any = None
    tool_choice: Any = None
    n: Any = None
    logprobs: Any = None
    top_logprobs: Any = None
    logit_bias: Any = None
    response_format: Any = None
    stream_options: Any = None


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model discovery reporting the real loaded identifier."""
    if state.model is None:
        return {
            "object": "list",
            "data": [],
            "unavailable_reason": "No model is loaded. Start the server with --ckpt.",
        }

    created = 0
    if state.ckpt_path is not None and state.ckpt_path.exists():
        created = int(state.ckpt_path.stat().st_mtime)

    return {
        "object": "list",
        "data": [
            {
                "id": state.model_id,
                "object": "model",
                "created": created,
                "owned_by": "quantsmind",
                "root": state.model_id,
                "parent": None,
                "permission": [],
                "quantsmind": {
                    "checkpoint_step": state.checkpoint_step,
                    "context_window": state.model_cfg.get("max_seq_len"),
                    "vocab_size": state.model_cfg.get("vocab_size"),
                    "n_params_non_embedding": state.model.num_params(),
                    "prompt_contract": state.contract_id,
                },
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": state.model is not None,
        "model_id": state.model_id,
        "prompt_contract": state.contract_id,
        "context_window": (state.model_cfg or {}).get("max_seq_len"),
    }


def _field(message, name, default=""):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _base_prompt(messages):
    """Serve raw text to a pretraining checkpoint, refusing turn shapes."""
    user_turns = [
        _field(m, "content") for m in messages if _field(m, "role") == "user"
    ]
    assistant_turns = [
        _field(m, "content") for m in messages if _field(m, "role") == "assistant"
    ]
    if len(user_turns) != 1 or assistant_turns:
        # A base completion model cannot represent turns, so a multi-turn
        # history is refused rather than silently flattened.
        raise ValueError(
            "a base checkpoint serves a single user turn; multi-turn input "
            "needs a checkpoint qualified on the chat contract"
        )
    system_parts = [
        _field(m, "content") for m in messages if _field(m, "role") == "system"
    ]
    parts = list(system_parts) + [user_turns[0]]
    return "\n\n".join(part for part in parts if part)


def build_prompt(messages):
    """Render through the contract-aware renderer for the loaded checkpoint.

    v1.1.0 removed the hand-built legacy "<|system|>..." string: the markers
    were never tokens, so multi-turn input silently became byte fragments. The
    renderer now refuses shapes the loaded contract cannot express.

    v1.5.0 adds the prompt *surface*: a base (pretraining) checkpoint is
    served raw text, because wrapping an instruction in a chat template the
    model never saw during training is a prompt-surface mismatch, not a
    capability. A checkpoint that declares chat qualification keeps the
    canonical renderer.

    The surface is scoped to the tokenizer it was derived for. A caller that
    installs a different tokenizer (a reload, a contract-specific test) gets a
    surface derived from that tokenizer rather than the previous checkpoint's
    decision, so no process-global state can leak across models.
    """
    from training.tokenizer_contract import (
        CONTRACT_V1,
        CONTRACT_V2,
        TokenizerContractError,
        detect_contract,
    )

    tokenizer = state.tokenizer
    if tokenizer is not None and tokenizer is not getattr(
        state, "surface_tokenizer", None
    ):
        try:
            contract = detect_contract(tokenizer)
        except (AttributeError, TokenizerContractError):
            contract = None
        if contract in (CONTRACT_V1, CONTRACT_V2):
            return render_chat(tokenizer, messages, default_system=SYSTEM_PROMPT)
    if getattr(state, "prompt_surface", "chat") == "base":
        return _base_prompt(messages)
    return render_chat(state.tokenizer, messages, default_system=SYSTEM_PROMPT)


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    require_loaded()

    if payload.stream:
        return _err(
            400,
            "Streaming is not implemented in this slice. Set stream=false.",
            "invalid_request_error",
            param="stream",
            code="streaming_unsupported",
        )

    present = [k for k in UNSUPPORTED_PARAMS if getattr(payload, k, None) is not None]
    if present:
        name = present[0]
        return _err(
            400,
            f"Parameter '{name}' is not supported: {UNSUPPORTED_PARAMS[name]}.",
            "invalid_request_error",
            param=name,
            code="unsupported_parameter",
        )

    for message in payload.messages:
        if message.role not in ("system", "user", "assistant"):
            return _err(
                400,
                f"Unsupported role '{message.role}'. Use 'system', 'user' or 'assistant'.",
                "invalid_request_error",
                param="messages",
                code="unsupported_role",
            )

    # v1.5.0: an unknown model identifier is refused instead of being echoed.
    # The previous behaviour accepted any string and answered with it, so a
    # client could believe it was talking to a model that was never loaded.
    if payload.model is not None and payload.model != state.model_id:
        return _err(
            400,
            f"Model '{payload.model}' is not loaded. This endpoint serves "
            f"'{state.model_id}'.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
        )

    try:
        prompt = build_prompt(payload.messages)
    except ValueError as exc:
        return _err(400, str(exc), "invalid_request_error", param="messages")

    max_seq_len = state.model_cfg["max_seq_len"]

    max_tokens = payload.max_tokens if payload.max_tokens is not None else DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        return _err(
            400, "max_tokens must be positive.", "invalid_request_error", param="max_tokens"
        )
    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)

    prompt_token_count = len(state.tokenizer.encode(prompt).ids)
    if prompt_token_count >= max_seq_len:
        return _err(
            400,
            f"Prompt is {prompt_token_count} tokens; the context window is {max_seq_len}. "
            "Shorten the conversation.",
            "invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )

    # Clamp the generation budget to the room actually left in the window.
    effective_max_tokens = min(max_tokens, max_seq_len - prompt_token_count)

    stop = payload.stop
    if isinstance(stop, str):
        stop = [stop]

    device = str(next(state.model.parameters()).device)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generate_ids(
            state.model,
            state.tokenizer,
            prompt,
            max_new_tokens=effective_max_tokens,
            temperature=payload.temperature if payload.temperature is not None else 0.2,
            top_p=payload.top_p if payload.top_p is not None else 1.0,
            stop=stop,
            presence_penalty=payload.presence_penalty or 0.0,
            frequency_penalty=payload.frequency_penalty or 0.0,
            device=device,
            seed=payload.seed,
        ),
    )

    completion_tokens = result["completion_tokens"]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or state.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "total_tokens": result["prompt_tokens"] + completion_tokens,
        },
    }


def load_state(ckpt: str, tokenizer: str) -> None:
    """Load weights at startup. Only reachable from the CLI, never from a request."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg, tok = load_trusted_state(ckpt, tokenizer, device)

    step = None
    try:
        step = torch.load(ckpt, map_location="cpu", weights_only=False).get("step")
    except Exception:  # pragma: no cover - metadata only, never fatal
        step = None

    # v1.1.0: the served contract is a property of the checkpoint, not of the
    # server's mood. A /2 checkpoint must match the tokenizer exactly; a legacy
    # /1 file is explicitly identified as qm.plain_instruction/1 and warned about.
    from training.checkpoint_schema import SCHEMA_V2, verify_against
    from training.tokenizer_contract import CONTRACT_V1, detect_contract

    metadata = torch_load_checkpoint(ckpt)
    provenance = metadata.get("provenance") or {}
    # v1.5.0: the prompt surface follows the checkpoint's recorded training
    # mode. A pretraining checkpoint is a completion model; serving it through
    # a chat template it never saw is a prompt-surface mismatch.
    training_mode = provenance.get("training_mode", "pretrain")
    state.prompt_surface = "chat" if training_mode == "instruction" else "base"
    state.surface_tokenizer = tok
    state.chat_qualified = bool(provenance.get("chat_qualified", False))
    if metadata.get("schema") == SCHEMA_V2:
        state.contract_id = detect_contract(tok)
        verify_against(
            metadata["provenance"], tokenizer=tok, contract_id=state.contract_id
        )
    else:
        state.contract_id = CONTRACT_V1
        print(
            "WARNING: checkpoint has no provenance contract (schema /1); serving "
            f"under the documented legacy contract {CONTRACT_V1}. Do not treat "
            "this as a v2 artifact."
        )

    state.model = model
    state.model_cfg = model_cfg
    state.tokenizer = tok
    state.model_id = model_cfg.get("name", "quantsmind-model")
    state.checkpoint_step = step
    state.ckpt_path = Path(ckpt)
    print(
        f"loaded model_id={state.model_id} step={step} params={model.num_params():,} "
        f"vocab(model)={model_cfg.get('vocab_size')} vocab(tokenizer)={tok.get_vocab_size()} "
        f"context={model_cfg.get('max_seq_len')} device={device}"
    )


def main():
    p = argparse.ArgumentParser(description="Taksh OpenAI-compatible local API")
    add_standard_args(p, default_port=8002)
    args = p.parse_args()

    load_state(args.ckpt, args.tokenizer)
    state.api_key = api_key_from_env()

    serve(app, p, args, "OpenAI-compatible API", auth_enabled=bool(state.api_key))


if __name__ == "__main__":
    main()