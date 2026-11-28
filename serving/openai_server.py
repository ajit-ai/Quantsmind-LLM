"""OpenAI-compatible local API for Taksh.

Exposes the surface a local coding client needs:

    GET  /v1/models
    POST /v1/chat/completions   (buffered or SSE-streamed)
    POST /v1/completions        (legacy text completion; ``suffix`` selects FIM)
    POST /v1/fim/completions    (Taksh-native fill-in-the-middle)
    GET  /health

Design notes
------------
* Weights are loaded **once at process start** from a CLI path. There is no HTTP
  endpoint accepting a checkpoint path. The existing ``/load`` endpoints on
  ``completion_server`` and ``review_server`` called ``torch.load(weights_only=False)``
  on a request-supplied path with no auth and bind 0.0.0.0 by default, which is
  arbitrary code execution. This module deliberately does not reproduce that.

* Binds 127.0.0.1 unless ``--allow-remote`` is passed explicitly.

* Optional bearer auth via ``QM_API_KEY``. Unset means keyless, which is safe only
  because the listener is loopback by default.

* The blocking torch decode runs in a worker thread, so the event loop never
  stalls. Streaming bridges each blocking ``next()`` through the executor for the
  same reason.

* Unsupported parameters are rejected with a structured 400 rather than silently
  ignored, so a caller never mistakes an ignored request for a honoured one.

* **Tool calling is a text protocol, not a trained capability.** The caller's
  schemas are rendered into the prompt and a documented ``<tool_call>`` block is
  parsed back into the OpenAI shape. Invalid or absent blocks are returned as
  ordinary text with a diagnostic; nothing is invented. ``tool_choice`` accepts
  only ``auto`` and ``none`` because a forced call cannot be guaranteed.

* **FIM is a text protocol too.** The tokenizer has no FIM control tokens, so
  ``<|fim_prefix|>`` etc. are literal text. The checkpoint is not FIM-trained.

* Model-level limitations (small vocab, 2,048-token context, no native tool or
  FIM training, a checkpoint that scores 0/20 on the frozen baseline) are in
  ``docs/OPEN_ISSUES.md`` and ``docs/TAKSH-LOCAL-API.md``. This server is
  API-shaped plumbing over an experimental checkpoint; it does not make the
  checkpoint capable.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taksh import __version__
from training.chat_template import SYSTEM_PROMPT, render_chat
from serving.fim import FIM_PROTOCOL_ID, protocol_note as fim_protocol_note, render_fim
from serving.local_inference import generate_ids, stream_text, torch_load_checkpoint
from serving.server_security import (
    add_standard_args,
    api_key_from_env,
    is_authorized,
    load_trusted_state,
    serve,
)
from serving.tool_protocol import (
    TOOL_PROTOCOL_ID,
    ToolProtocolError,
    parse_tool_call,
    render_tools,
    validate_tool_choice,
    validate_tools,
)

app = FastAPI(title="Taksh - OpenAI-compatible API", version=__version__)

# Real OpenAI parameters this slice does NOT implement. Silently ignoring them would
# let a caller believe a tool loop, n>1 sampling or logprob ranking is happening.
# ``tools`` and ``tool_choice`` are no longer here: they are supported as the text
# protocol documented above. ``parallel_tool_calls`` remains unsupported because
# only the first valid block is parsed.
UNSUPPORTED_PARAMS = {
    "functions": "legacy function calling is not implemented; use 'tools'",
    "function_call": "legacy function calling is not implemented; use 'tool_choice'",
    "n": "n>1 sampling is not implemented in this slice",
    "logprobs": "logprob output is not implemented in this slice",
    "top_logprobs": "logprob output is not implemented in this slice",
    "logit_bias": "logit_bias is not implemented in this slice",
    "response_format": "structured output is not implemented in this slice",
    "stream_options": (
        "stream_options is not implemented; streamed usage is reported in the "
        "final chunk"
    ),
    "parallel_tool_calls": (
        "parallel tool calls are not implemented; at most one call is parsed"
    ),
    "echo": "echo is not implemented in this slice",
}

DEFAULT_MAX_TOKENS = 256
MAX_TOKENS_CEILING = 4096


class RequestError(ValueError):
    """A request the server refuses with a structured 400."""

    def __init__(self, message, *, param=None, code="unsupported_parameter", status=400):
        super().__init__(message)
        self.param = param
        self.code = code
        self.status = status


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
    #: Opt-in compatibility mode. The shipped tokenizers cannot express roles,
    #: so a v1 contract refuses multi-turn input. With this flag the server
    #: serialises the history as plain labelled text instead, which is a
    #: compatibility shim, not the training template. Default off.
    flatten_history = False


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
    name: str | None = None
    tool_call_id: str | None = None


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
    tools: list | None = None
    tool_choice: Any = None

    # Declared explicitly so the handler can *reject* them. Without these fields
    # pydantic would drop them as unknown extras and the request would appear to
    # succeed while the caller's intent was ignored.
    functions: Any = None
    function_call: Any = None
    parallel_tool_calls: Any = None
    n: Any = None
    logprobs: Any = None
    top_logprobs: Any = None
    logit_bias: Any = None
    response_format: Any = None
    stream_options: Any = None


class CompletionRequest(BaseModel):
    """Legacy text completion. ``suffix`` turns it into a FIM request."""

    model: str | None = None
    prompt: str | list[str] | None = None
    suffix: str | None = None
    max_tokens: int | None = 16
    temperature: float | None = 0.2
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool | None = False
    user: str | None = None

    n: Any = None
    logprobs: Any = None
    echo: Any = None


class FIMRequest(BaseModel):
    prefix: str
    suffix: str = ""
    max_tokens: int | None = 64
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool | None = False


CAPABILITIES = {
    "chat_completions": True,
    "text_completions": True,
    "streaming": True,
    "fim": True,
    "tool_call_parsing": True,
    "native_tool_training": False,
    "binding_repair_streaming": False,
    "multi_turn_native": False,
}


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
                    "capabilities": dict(CAPABILITIES),
                    "fim_protocol": FIM_PROTOCOL_ID,
                    "tool_protocol": TOOL_PROTOCOL_ID,
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
        "capabilities": dict(CAPABILITIES),
        "fim_protocol": FIM_PROTOCOL_ID,
        "tool_protocol": TOOL_PROTOCOL_ID,
        "flatten_history": bool(state.flatten_history),
    }


def _field(message, name, default=""):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _normalise_messages(messages):
    normalised = []
    for message in messages:
        normalised.append({
            "role": _field(message, "role"),
            "content": _field(message, "content") or "",
        })
    return normalised


def _flatten_history(messages):
    """Serialise a multi-turn history as plain labelled text.

    Only used when ``--flatten-history`` is enabled. The shipped v1 tokenizers
    cannot express roles, so the alternative is refusing every real chat client
    on its second turn. Plain ASCII labels are honest text; invented special
    tokens would be silently shredded.
    """
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if len(turns) <= 1:
        return messages
    system = [m["content"] for m in messages if m["role"] == "system"]
    transcript = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in turns
    )
    preamble = "\n\n".join(part for part in system + ["Conversation transcript:", transcript] if part)
    return [{"role": "user", "content": preamble}]


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


def build_prompt(messages, tools=None, tool_choice="auto"):
    """Render through the contract-aware renderer for the loaded checkpoint.

    v1.1.0 removed the hand-built legacy "<|system|>..." string: the markers
    were never tokens, so multi-turn input silently became byte fragments. The
    renderer now refuses shapes the loaded contract cannot express.

    v1.5.0 adds the prompt *surface*: a base (pretraining) checkpoint is
    served raw text, because wrapping an instruction in a chat template the
    model never saw during training is a prompt-surface mismatch, not a
    capability. A checkpoint that declares chat qualification keeps the
    canonical renderer.

    v1.6.0 (this slice) adds an opt-in ``--flatten-history`` compatibility
    mode and a tool-schema block. Neither changes the model; both change what
    text the model is shown. The tool block is literal text, and the parser
    treats its absence as "no call", never as a failure to be papered over.
    """
    from training.tokenizer_contract import (
        CONTRACT_V1,
        CONTRACT_V2,
        TokenizerContractError,
        detect_contract,
    )

    messages = _normalise_messages(messages)
    if getattr(state, "flatten_history", False):
        messages = _flatten_history(messages)
    if tools and tool_choice != "none":
        messages = list(messages) + [
            {"role": "system", "content": render_tools(tools)}
        ]

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


def _check_unsupported(payload):
    present = [k for k in UNSUPPORTED_PARAMS if getattr(payload, k, None) is not None]
    if present:
        name = present[0]
        raise RequestError(
            f"Parameter '{name}' is not supported: {UNSUPPORTED_PARAMS[name]}.",
            param=name,
        )


def _check_model(model_id):
    # v1.5.0: an unknown model identifier is refused instead of being echoed.
    # The previous behaviour accepted any string and answered with it, so a
    # client could believe it was talking to a model that was never loaded.
    if model_id is not None and model_id != state.model_id:
        raise RequestError(
            f"Model '{model_id}' is not loaded. This endpoint serves "
            f"'{state.model_id}'.",
            param="model",
            code="model_not_found",
        )


def _normalise_stop(stop):
    if isinstance(stop, str):
        return [stop]
    return stop


def _prepare_generation(
    *,
    prompt,
    model_id,
    max_tokens,
    temperature,
    top_p,
    stop,
    seed,
    presence_penalty,
    frequency_penalty,
    default_max_tokens,
    default_temperature,
    default_top_p,
):
    """Shared validation for every generation endpoint.

    Returns a dict so the buffered, streaming and FIM handlers cannot drift in
    their context, stop or model checks. The prompt is passed explicitly: a
    Pydantic request model is not a scratch pad, and smuggling it through an
    undeclared attribute is exactly the kind of implicit state this repo avoids.
    """
    require_loaded()
    _check_model(model_id)

    max_seq_len = state.model_cfg["max_seq_len"]

    max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    if max_tokens <= 0:
        raise RequestError(
            "max_tokens must be positive.", param="max_tokens", code="invalid_max_tokens"
        )
    max_tokens = min(max_tokens, MAX_TOKENS_CEILING)

    prompt_token_count = len(state.tokenizer.encode(prompt).ids)
    if prompt_token_count >= max_seq_len:
        raise RequestError(
            f"Prompt is {prompt_token_count} tokens; the context window is {max_seq_len}. "
            "Shorten the conversation.",
            param="messages",
            code="context_length_exceeded",
        )

    effective_max_tokens = min(max_tokens, max_seq_len - prompt_token_count)
    device = str(next(state.model.parameters()).device)

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_token_count,
        "max_tokens": effective_max_tokens,
        "temperature": temperature if temperature is not None else default_temperature,
        "top_p": top_p if top_p is not None else default_top_p,
        "stop": _normalise_stop(stop),
        "seed": seed,
        "presence_penalty": presence_penalty or 0.0,
        "frequency_penalty": frequency_penalty or 0.0,
        "device": device,
    }


def _prepare_chat(payload):
    _check_unsupported(payload)

    for message in payload.messages:
        if message.role not in ("system", "user", "assistant"):
            raise RequestError(
                f"Unsupported role '{message.role}'. Use 'system', 'user' or 'assistant'.",
                param="messages",
                code="unsupported_role",
            )

    _check_model(payload.model)

    tools = None
    tool_choice = "auto"
    if payload.tools is not None:
        try:
            tools = validate_tools(payload.tools)
            tool_choice = validate_tool_choice(payload.tool_choice, tools)
        except ToolProtocolError as exc:
            raise RequestError(str(exc), param=exc.param, code="invalid_tools") from exc

    if payload.stream and tools and tool_choice != "none":
        raise RequestError(
            "stream=true with tools is not supported in this slice: a streamed "
            "text protocol cannot be parsed into a tool_calls message without "
            "buffering the whole completion. Use stream=false for tool requests.",
            param="stream",
            code="unsupported_combination",
        )

    try:
        prompt = build_prompt(payload.messages, tools=tools, tool_choice=tool_choice)
    except ValueError as exc:
        raise RequestError(str(exc), param="messages", code="invalid_messages") from exc

    prepared = _prepare_generation(
        prompt=prompt,
        model_id=payload.model,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stop=payload.stop,
        seed=payload.seed,
        presence_penalty=payload.presence_penalty,
        frequency_penalty=payload.frequency_penalty,
        default_max_tokens=DEFAULT_MAX_TOKENS,
        default_temperature=0.2,
        default_top_p=1.0,
    )
    prepared["tools"] = tools
    prepared["tool_choice"] = tool_choice
    prepared["flattened_history"] = bool(
        getattr(state, "flatten_history", False)
        and len([m for m in payload.messages if m.role in ("user", "assistant")]) > 1
    )
    return prepared


def _sse(payload) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


async def _stream_sse(prepared, *, object_name, include_role):
    """Bridge the blocking token generator into an SSE response.

    Each ``next()`` runs in the executor, so a slow CPU forward pass never blocks
    the event loop. The generator is closed in ``finally`` so a client disconnect
    does not leave a decoded sequence running.
    """
    loop = asyncio.get_running_loop()
    created = int(time.time())
    completion_id = f"{'chatcmpl' if include_role else 'cmpl'}-{uuid.uuid4().hex}"
    model_id = state.model_id
    generator = stream_text(
        state.model,
        state.tokenizer,
        prepared["prompt"],
        max_new_tokens=prepared["max_tokens"],
        temperature=prepared["temperature"],
        top_p=prepared["top_p"],
        stop=prepared["stop"],
        presence_penalty=prepared["presence_penalty"],
        frequency_penalty=prepared["frequency_penalty"],
        device=prepared["device"],
        seed=prepared["seed"],
    )
    sentinel = object()

    def next_event():
        try:
            return next(generator)
        except StopIteration:
            return sentinel

    def chunk(delta, finish_reason=None, usage=None):
        choice = {
            "index": 0,
            "finish_reason": finish_reason,
        }
        if include_role:
            choice["delta"] = delta
            choice["logprobs"] = None
        else:
            choice["text"] = delta.get("content", "")
            choice["logprobs"] = None
        body = {
            "id": completion_id,
            "object": object_name,
            "created": created,
            "model": model_id,
            "choices": [choice],
        }
        if usage is not None:
            body["usage"] = usage
        return _sse(body)

    try:
        if include_role:
            yield chunk({"role": "assistant", "content": ""})
        while True:
            event = await loop.run_in_executor(None, next_event)
            if event is sentinel:
                break
            if event["type"] == "delta":
                yield chunk({"content": event["text"]} if include_role else {"content": event["text"]})
            else:
                usage = {
                    "prompt_tokens": prepared["prompt_tokens"],
                    "completion_tokens": event["completion_tokens"],
                    "total_tokens": prepared["prompt_tokens"] + event["completion_tokens"],
                }
                yield chunk({}, finish_reason=event["finish_reason"], usage=usage)
        yield "data: [DONE]\n\n"
    finally:
        generator.close()


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    try:
        prepared = _prepare_chat(payload)
    except RequestError as exc:
        return _err(exc.status, str(exc), "invalid_request_error", param=exc.param, code=exc.code)

    if payload.stream:
        return StreamingResponse(
            _stream_sse(
                prepared,
                object_name="chat.completion.chunk",
                include_role=True,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generate_ids(
            state.model,
            state.tokenizer,
            prepared["prompt"],
            max_new_tokens=prepared["max_tokens"],
            temperature=prepared["temperature"],
            top_p=prepared["top_p"],
            stop=prepared["stop"],
            presence_penalty=prepared["presence_penalty"],
            frequency_penalty=prepared["frequency_penalty"],
            device=prepared["device"],
            seed=prepared["seed"],
        ),
    )

    content = result["text"]
    tool_call = None
    tool_error = None
    if prepared["tools"] and prepared["tool_choice"] != "none":
        content, tool_call, tool_error = parse_tool_call(content, prepared["tools"])

    message = {"role": "assistant", "content": content}
    if tool_call is not None:
        message["tool_calls"] = [tool_call]

    completion_tokens = result["completion_tokens"]
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or state.model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_call is not None else result["finish_reason"],
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "total_tokens": result["prompt_tokens"] + completion_tokens,
        },
        "quantsmind": {
            "prompt_contract": state.contract_id,
            "history_serialization": (
                "plain_text_transcript/1" if prepared["flattened_history"] else "native"
            ),
            "tool_protocol": TOOL_PROTOCOL_ID if prepared["tools"] else None,
            "binding_repair": result.get("repair"),
        },
    }
    if tool_error is not None:
        body["quantsmind"]["tool_call_error"] = tool_error
    return body


def _completion_prompt(payload):
    if isinstance(payload.prompt, list):
        raise RequestError(
            "prompt arrays are not implemented; send one prompt string.",
            param="prompt",
            code="unsupported_parameter",
        )
    if not isinstance(payload.prompt, str) or not payload.prompt:
        raise RequestError(
            "'prompt' must be a non-empty string.",
            param="prompt",
            code="invalid_prompt",
        )
    return render_fim(payload.prompt, payload.suffix or "")


@app.post("/v1/completions")
async def completions(payload: CompletionRequest):
    try:
        _check_unsupported(payload)
        prompt = _completion_prompt(payload)
    except RequestError as exc:
        return _err(exc.status, str(exc), "invalid_request_error", param=exc.param, code=exc.code)

    try:
        prepared = _prepare_generation(
            prompt=prompt,
            model_id=payload.model,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop,
            seed=payload.seed,
            presence_penalty=None,
            frequency_penalty=None,
            default_max_tokens=16,
            default_temperature=0.2,
            default_top_p=1.0,
        )
    except RequestError as exc:
        return _err(exc.status, str(exc), "invalid_request_error", param=exc.param, code=exc.code)

    if payload.stream:
        return StreamingResponse(
            _stream_sse(prepared, object_name="text_completion", include_role=False),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generate_ids(
            state.model,
            state.tokenizer,
            prepared["prompt"],
            max_new_tokens=prepared["max_tokens"],
            temperature=prepared["temperature"],
            top_p=prepared["top_p"],
            stop=prepared["stop"],
            presence_penalty=prepared["presence_penalty"],
            frequency_penalty=prepared["frequency_penalty"],
            device=prepared["device"],
            seed=prepared["seed"],
        ),
    )

    body = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": payload.model or state.model_id,
        "choices": [
            {
                "text": result["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "quantsmind": {
            "fim_protocol": FIM_PROTOCOL_ID if payload.suffix else None,
            "fim_note": fim_protocol_note() if payload.suffix else None,
            "binding_repair": result.get("repair"),
        },
    }
    return body


@app.post("/v1/fim/completions")
async def fim_completions(payload: FIMRequest):
    prompt = render_fim(payload.prefix, payload.suffix)
    try:
        prepared = _prepare_generation(
            prompt=prompt,
            model_id=None,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            stop=payload.stop,
            seed=payload.seed,
            presence_penalty=None,
            frequency_penalty=None,
            default_max_tokens=64,
            default_temperature=0.0,
            default_top_p=1.0,
        )
    except RequestError as exc:
        return _err(exc.status, str(exc), "invalid_request_error", param=exc.param, code=exc.code)

    if payload.stream:
        return StreamingResponse(
            _stream_sse(prepared, object_name="fim.completion.chunk", include_role=False),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generate_ids(
            state.model,
            state.tokenizer,
            prepared["prompt"],
            max_new_tokens=prepared["max_tokens"],
            temperature=prepared["temperature"],
            top_p=prepared["top_p"],
            stop=prepared["stop"],
            presence_penalty=prepared["presence_penalty"],
            frequency_penalty=prepared["frequency_penalty"],
            device=prepared["device"],
            seed=prepared["seed"],
        ),
    )

    return {
        "id": f"fim-{uuid.uuid4().hex}",
        "object": "fim.completion",
        "created": int(time.time()),
        "model": state.model_id,
        "choices": [
            {
                "text": result["text"],
                "index": 0,
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "quantsmind": {
            "protocol": FIM_PROTOCOL_ID,
            "note": fim_protocol_note(),
            "binding_repair": result.get("repair"),
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

    # Same trusted startup boundary as load_trusted_state: operator CLI path.
    metadata = torch_load_checkpoint(ckpt, allow_unsafe_trusted=True)
    provenance = metadata.get("provenance") or {}
    # v1.5.0: the prompt surface follows the checkpoint's recorded training
    # mode. A pretraining checkpoint is a completion model; serving it through
    # a chat template it never saw is a prompt-surface mismatch.
    training_mode = provenance.get("training_mode", "pretrain")
    state.prompt_surface = "chat" if training_mode == "instruction" else "base"
    state.surface_tokenizer = tok
    state.chat_qualified = bool(provenance.get("chat_qualified", False))
    if metadata.get("schema") == SCHEMA_V2:
        from training.tokenizer_contract import tokenizer_file_fingerprint

        state.contract_id = detect_contract(tok)
        # Provenance records the tokenizer *file* digest whenever training knew
        # the path; pass both identities so the served artifact is checked
        # against the same bytes it was trained with.
        verify_against(
            metadata["provenance"],
            tokenizer=tok,
            tokenizer_sha256=tokenizer_file_fingerprint(tokenizer),
            contract_id=state.contract_id,
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
    p.add_argument(
        "--flatten-history",
        action="store_true",
        help=(
            "compatibility mode: serialise multi-turn chat as plain labelled text "
            "for tokenizers that cannot express roles. Off by default because it "
            "is not the training template."
        ),
    )
    args = p.parse_args()

    load_state(args.ckpt, args.tokenizer)
    state.api_key = api_key_from_env()
    state.flatten_history = bool(args.flatten_history)

    serve(app, p, args, "OpenAI-compatible API", auth_enabled=bool(state.api_key))


if __name__ == "__main__":
    main()
