"""TAKSH L0.6.2 — local OpenAI-compatible inference API (engine adapter).

Adapts :mod:`serving.inference_engine` to HTTP using the project's existing
FastAPI conventions. A *subset* of the OpenAI chat-completions API is
supported; anything unsupported is rejected with a structured 400 — never
silently ignored.

Binding properties:
  * the engine is injected at app-creation time; creating the app never
    loads a checkpoint;
  * no endpoint accepts a filesystem path or model selection — the loaded
    model is the operator's startup choice only;
  * external providers are NOT wired in: ``provider`` is always ``"local"``;
  * ``/v1/cancel`` cancels an in-flight request cooperatively;
  * streaming uses Server-Sent Events when ``stream=true``;
  * errors are the OpenAI-shaped envelope and never leak tracebacks or paths;
  * capability claims are never made by this surface.

Supported subset of POST /v1/chat/completions:
  model (optional, echoed), messages[], max_tokens, temperature, top_p,
  top_k (non-standard extension, quantsmind), presence_penalty,
  frequency_penalty, seed, stop, stream.
Anything else in UNSUPPORTED_PARAMS is rejected 400. This is not full OpenAI
API compatibility and no conformance claim is made beyond this subset.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.inference_engine import (  # noqa: E402
    DEFAULT_CONTEXT_LIMIT,
    ConfigurationError,
    ContextOverflowError,
    InferenceRequest,
    LoadState,
    ModelLoader,
    SamplingParams,
    memory_snapshot,
)
from serving.openai_server import UNSUPPORTED_PARAMS  # noqa: E402 — one registry, no duplication


def _err(status: int, message: str, etype: str, param: Optional[str] = None,
         code: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype,
                           "param": param, "code": code}},
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Any] = None        # validated below
    stream: Optional[bool] = False
    user: Optional[str] = None

    # Declared so they can be *rejected* explicitly (see UNSUPPORTED_PARAMS).
    tools: Any = None
    functions: Any = None
    tool_choice: Any = None
    n: Any = None
    logprobs: Any = None
    top_logprobs: Any = None
    logit_bias: Any = None
    response_format: Any = None
    stream_options: Any = None


class NativeGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 0
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[list[str]] = None
    language_id: Optional[str] = None
    language_hint: Optional[str] = None
    task_id: Optional[str] = None
    mode: Optional[str] = "general"
    truncation_policy: Optional[str] = "reject"
    timeout_seconds: Optional[float] = None


def _last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str) and message.content.strip():
                return message.content
    raise ConfigurationError("messages must include at least one non-empty user turn")


def _validate_stop(stop: Any, engine_max: int = 8) -> list[str]:
    if stop is None:
        return []
    items = stop if isinstance(stop, list) else [stop]
    if not all(isinstance(s, str) for s in items):
        raise ConfigurationError("stop must be a string or a list of strings")
    items = [s for s in items if s]
    if len(items) > engine_max:
        raise ConfigurationError(f"at most {engine_max} stop sequences")
    return items


def _sampling_from(request: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=0.0 if request.temperature is None else float(request.temperature),
        top_k=0 if request.top_k is None else int(request.top_k),
        top_p=1.0 if request.top_p is None else float(request.top_p),
        presence_penalty=0.0 if request.presence_penalty is None else float(request.presence_penalty),
        frequency_penalty=0.0 if request.frequency_penalty is None else float(request.frequency_penalty),
        seed=request.seed,
        max_new_tokens=64 if request.max_tokens is None else int(request.max_tokens),
    )


def create_app(engine: Optional[ModelLoader] = None) -> FastAPI:
    app = FastAPI(title="Taksh — local inference API", version="1.0.0")
    state: dict[str, Any] = {"engine": engine, "created": time.time()}

    # ---- structured error envelopes ------------------------------------------ #
    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        return _err(422, "request validation failed",
                    "invalid_request_error",
                    param=".".join(str(p) for p in first.get("loc", [])[1:]) or None,
                    code="validation_error")

    @app.exception_handler(ConfigurationError)
    async def config_handler(_: Request, exc: ConfigurationError):
        return _err(400, str(exc), "invalid_request_error", code="configuration_error")

    @app.exception_handler(ContextOverflowError)
    async def overflow_handler(_: Request, exc: ContextOverflowError):
        return _err(400, str(exc), "invalid_request_error", code="context_overflow")

    # ---- helpers --------------------------------------------------------------- #
    def require_ready() -> ModelLoader:
        eng = state["engine"]
        if eng is None or eng.state is not LoadState.READY:
            status = eng.state.value if eng is not None else "NOT_LOADED"
            raise _ResourceHttp(status)
        return eng

    class _ResourceHttp(Exception):
        def __init__(self, status: str):
            self.status = status

    @app.exception_handler(_ResourceHttp)
    async def blocked_handler(_: Request, exc: _ResourceHttp):
        return _err(503,
                    f"model is not ready (state={exc.status}); operator must load it explicitly",
                    "resource_error", code="resource_blocked")

    # ---- introspection ---------------------------------------------------------- #
    @app.get("/health")
    async def health():
        eng = state["engine"]
        return {
            "status": "ok" if (eng is not None and eng.state is LoadState.READY) else "degraded",
            "version": "1.0.0",
            "provider": "local",
            "model_loaded": eng is not None and eng.state is LoadState.READY,
            "load_state": eng.state.value if eng is not None else "NOT_LOADED",
            "external_providers": "disabled",
        }

    @app.get("/v1/resources")
    async def resources():
        snap = memory_snapshot()
        eng = state["engine"]
        return {
            "provider": "local",
            "device": eng.device if eng else "cpu",
            "memory": snap,
            "labels": {"memory": "measured", "estimates_not_guarantees": True},
            "inflight_requests": eng.cancel_controller.inflight() if eng else [],
        }

    @app.get("/v1/models")
    async def models():
        eng = state["engine"]
        if eng is None or eng.state is not LoadState.READY:
            state_reason = eng.state.value if eng is not None else "NOT_LOADED"
            return {"object": "list", "data": [],
                    "unavailable_reason": f"model not loaded (state={state_reason}); operator must load explicitly"}
        assert eng.model_cfg is not None
        return {
            "object": "list",
            "provider": "local",
            "data": [{
                "id": "taksh-foundation-111m",
                "object": "model",
                "owned_by": "quantsmind",
                "created": 0,
                "permission": [],
                "quantsmind": {
                    "checkpoint_fingerprint": eng.load_report.get("checkpoint_sha256"),
                    "model_config_fingerprint": eng.load_report.get("model_config_fingerprint"),
                    "tokenizer_sha256": eng.load_report.get("tokenizer_sha256"),
                    "context_window": eng.model_cfg.get("max_seq_len", DEFAULT_CONTEXT_LIMIT),
                    "vocab_size": eng.model_cfg.get("vocab_size"),
                    "parameters": eng.load_report.get("parameters"),
                    "capability_claim": None,
                    "note": "experimental checkpoint; capability not assessed by this API",
                },
            }],
        }

    # ---- generation ------------------------------------------------------------- #
    def build_request(body: ChatCompletionRequest, request_id: str) -> InferenceRequest:
        for name, reason in UNSUPPORTED_PARAMS.items():
            if getattr(body, name, None) is not None:
                raise ConfigurationError(f"parameter {name!r} is not supported: {reason}")
        user_prompt = _last_user_message(body.messages)
        system_parts = [m.content for m in body.messages if m.role == "system"]
        prompt = "\n\n".join([p for p in [*system_parts, user_prompt] if p.strip()])
        sampling = _sampling_from(body).validate()
        return InferenceRequest(
            prompt=prompt,
            sampling=sampling,
            stop=_validate_stop(body.stop),
            request_id=request_id,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        eng = require_ready()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        req = build_request(body, request_id)

        if body.stream:
            return StreamingResponse(
                sse_stream(eng, req),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id, "Cache-Control": "no-cache"},
            )

        started = time.perf_counter()
        chunks = list(eng.stream(req))           # single pass; no re-decode
        final = next((c for c in reversed(chunks) if c.final), None)
        if final is None or final.response is None:
            raise ConfigurationError("engine produced no terminal record")
        resp = final.response
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model or "taksh-foundation-111m",
            "provider": "local",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": resp.text},
                "finish_reason": resp.finish_reason,
            }],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.prompt_tokens + resp.completion_tokens,
            },
            "latency_seconds": round(time.perf_counter() - started, 6),
            "quantsmind": {
                "finish_reason": resp.finish_reason,
                "deterministic": resp.sampling["deterministic"],
                "truncated": resp.truncated,
                "device": resp.device,
                "capability_claim": None,
            },
        }

    async def sse_stream(eng: ModelLoader, req: InferenceRequest) -> Iterator[str]:
        # Run the blocking decode loop in a worker thread is unnecessary for
        # correctness; the loop yields cooperatively between tokens.
        created = int(time.time())
        yield _sse({"id": req.request_id, "object": "chat.completion.chunk",
                    "created": created, "model": "taksh-foundation-111m", "provider": "local",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        completed = 0
        for chunk in eng.stream(req):
            if chunk.final:
                assert chunk.response is not None
                completed = chunk.response.completion_tokens
                yield _sse({"id": req.request_id, "object": "chat.completion.chunk",
                            "created": created, "model": "taksh-foundation-111m",
                            "provider": "local",
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": chunk.response.finish_reason}],
                            "usage": {"prompt_tokens": chunk.response.prompt_tokens,
                                      "completion_tokens": completed,
                                      "total_tokens": chunk.response.prompt_tokens + completed}})
                break
            if chunk.text:
                yield _sse({"id": req.request_id, "object": "chat.completion.chunk",
                            "created": created, "model": "taksh-foundation-111m",
                            "provider": "local",
                            "choices": [{"index": 0, "delta": {"content": chunk.text},
                                         "finish_reason": None}]})
        yield "data: [DONE]\n\n"

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @app.post("/v1/generate")
    async def native_generate(body: NativeGenerateRequest):
        eng = require_ready()
        request_id = f"req-{uuid.uuid4().hex[:16]}"
        language_hint = body.language_hint or body.language_id
        req = InferenceRequest(
            prompt=body.prompt,
            sampling=SamplingParams(
                temperature=float(body.temperature) if body.temperature is not None else 0.0,
                top_k=int(body.top_k) if body.top_k is not None else 0,
                top_p=float(body.top_p) if body.top_p is not None else 1.0,
                presence_penalty=float(body.presence_penalty) if body.presence_penalty is not None else 0.0,
                frequency_penalty=float(body.frequency_penalty) if body.frequency_penalty is not None else 0.0,
                seed=body.seed,
                max_new_tokens=int(body.max_new_tokens) if body.max_new_tokens is not None else 64,
            ).validate(),
            stop=_validate_stop(body.stop),
            language_id=body.language_id,
            language_hint=language_hint,
            task_id=body.task_id,
            mode=body.mode or "general",
            truncation_policy=body.truncation_policy or "reject",
            timeout_seconds=body.timeout_seconds,
            request_id=request_id,
        )
        chunks = list(eng.stream(req))
        final = next((c for c in reversed(chunks) if c.final), None)
        if final is None or final.response is None:
            raise ConfigurationError("engine produced no terminal record")
        resp = final.response
        return {
            "id": request_id,
            "provider": "local",
            "text": resp.text,
            "finish_reason": resp.finish_reason,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "truncated": resp.truncated,
            "context_limit": resp.context_limit,
            "utilization": round(resp.utilization, 4),
            "labels": {"token_counts": "measured", "utilization": "measured"},
            "trace": resp.trace.to_eval_record(),
        }

    @app.post("/v1/cancel")
    async def cancel(payload: dict):
        eng = state["engine"]
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            return _err(400, "request_id is required", "invalid_request_error",
                        code="configuration_error")
        if eng is None:
            return {"cancelled": False, "reason": "engine not initialised"}
        return {"cancelled": eng.cancel_controller.cancel(request_id),
                "request_id": request_id, "provider": "local"}

    return app
