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
from serving.code_chat import CODE_SYSTEM_PROMPT, format_prompt
from serving.local_inference import generate_ids
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
        "context_window": (state.model_cfg or {}).get("max_seq_len"),
    }


def build_prompt(messages):
    """Convert OpenAI chat messages into this model's prompt format.

    Reuses ``serving.code_chat.format_prompt`` (already covered by
    tests/test_code_serving.py) for the single user-turn case and extends it for
    multi-turn histories, which ``format_prompt`` does not model.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role in ("user", "assistant")]

    if not turns:
        raise ValueError("messages must contain at least one user or assistant turn")

    # Only delegate to format_prompt when there is no caller-supplied system prompt.
    # format_prompt hardcodes CODE_SYSTEM_PROMPT, so delegating unconditionally
    # would silently discard the caller's system message.
    if len(turns) == 1 and turns[0].role == "user" and not system_parts:
        return format_prompt(turns[0].content, "python")

    system = "\n".join(system_parts) if system_parts else CODE_SYSTEM_PROMPT
    parts = [f"<|system|>{system}\n"]
    for turn in turns:
        tag = "<|user|>" if turn.role == "user" else "<|assistant|>"
        parts.append(f"{tag}\n{turn.content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


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