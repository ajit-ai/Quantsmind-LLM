"""Native TAKSH provider adapter over serving.inference_engine.

This adapter is selectable but remains unqualified until a compatible trained
111M checkpoint exists and is evaluated. Missing or incompatible checkpoints are
reported as unavailable; there is no fallback checkpoint.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from serving.inference_engine import (
    ConfigurationError,
    GenerationFailedError,
    InferenceRequest,
    InferenceResponse,
    LoadState,
    ModelLoader,
    ResourceBlockedError,
    SamplingParams,
)
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ProviderCapabilities,
    ProviderError,
    ProviderModelIdentity,
    ProviderUnavailableError,
    register_provider,
)


def _native_paths():
    checkpoint = os.environ.get("TAKSH_NATIVE_CHECKPOINT") or None
    tokenizer = os.environ.get("TAKSH_NATIVE_TOKENIZER") or None
    return checkpoint, tokenizer


class NativeTakshProvider:
    name = "native-taksh"

    def __init__(self, checkpoint_path: str | Path | None = None, tokenizer_path: str | Path | None = None, *, device: str = "cpu") -> None:
        checkpoint, tokenizer = _native_paths()
        self.checkpoint_path = Path(checkpoint_path or checkpoint) if (checkpoint_path or checkpoint) else None
        self.tokenizer_path = Path(tokenizer_path or tokenizer) if (tokenizer_path or tokenizer) else None
        self.device = device
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id="taksh-foundation-111m",
            revision="unverified",
            source=str(self.checkpoint_path) if self.checkpoint_path is not None else "unconfigured",
            license_or_rights="source Apache-2.0; native weights not cleared for redistribution",
            artifact_sha256=None,
            tokenizer=str(self.tokenizer_path) if self.tokenizer_path is not None else None,
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        configured = bool(self.checkpoint_path and self.tokenizer_path)
        available = configured and Path(self.checkpoint_path).is_file() and Path(self.tokenizer_path).is_file()
        return ProviderCapabilities(
            provider=self.name,
            available=available,
            offline=True,
            modes=("local",) if available else (),
            max_context_tokens=2048,
            notes="Experimental/unqualified native backend; architecture compatibility and evaluation were not established.",
        )

    def _build_loader(self) -> ModelLoader:
        if not self.checkpoint_path or not self.tokenizer_path:
            raise ProviderUnavailableError("native checkpoint/tokenizer are not configured")
        if not Path(self.checkpoint_path).is_file():
            raise ProviderUnavailableError(f"native checkpoint not found: {self.checkpoint_path}")
        if not Path(self.tokenizer_path).is_file():
            raise ProviderUnavailableError(f"native tokenizer not found: {self.tokenizer_path}")
        return ModelLoader(self.checkpoint_path, self.tokenizer_path, self.device)

    def _require_paths(self):
        identity = self.identity()
        if not self.checkpoint_path or not self.tokenizer_path:
            raise ProviderUnavailableError(
                "native checkpoint/tokenizer are not configured; set TAKSH_NATIVE_CHECKPOINT and TAKSH_NATIVE_TOKENIZER"
            )
        if not Path(self.checkpoint_path).is_file():
            raise ProviderUnavailableError(f"native checkpoint not found: {self.checkpoint_path}")
        if not Path(self.tokenizer_path).is_file():
            raise ProviderUnavailableError(f"native tokenizer not found: {self.tokenizer_path}")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        started = time.perf_counter()
        self._require_paths()
        loader = self._build_loader()
        report = loader.preflight()
        if report.get("decision") != "PROCEED":
            raise ProviderUnavailableError(
                "native checkpoint blocked by preflight: " + "; ".join(report.get("problems", ["not ready"]))
            )
        try:
            loaded = loader.load(allow_full_model=True)
        except (ResourceBlockedError, ConfigurationError, GenerationFailedError, Exception) as exc:
            raise ProviderUnavailableError(f"native engine load failed: {type(exc).__name__}: {exc}") from exc
        if isinstance(loaded, dict) and loaded.get("state") != LoadState.READY.value:
            raise ProviderUnavailableError(str(loaded.get("reason") or "native engine did not become READY"))

        params = SamplingParams(
            temperature=float(request.config.temperature),
            top_k=int(request.config.top_k) if request.config.top_k >= 0 else 0,
            top_p=float(request.config.top_p),
            seed=request.config.seed,
            max_new_tokens=int(request.config.max_new_tokens),
        ).validate()
        engine_request = InferenceRequest(
            prompt=request.prompt,
            sampling=params,
            stop=list(request.config.stop or []),
            timeout_seconds=request.config.timeout_seconds,
            mode="model_only",
        )
        try:
            response = loader.generate(engine_request)
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailableError(f"native generation failed: {type(exc).__name__}: {exc}") from exc
        return GenerationResult(
            provider=self.name,
            model_id="taksh-foundation-111m",
            revision=Path(self.checkpoint_path).name,
            text=response.text,
            raw_text=response.text,
            finish_reason=response.finish_reason,
            prompt_tokens=int(response.prompt_tokens),
            completion_tokens=int(response.completion_tokens),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "native-engine"},
            metadata={"checkpoint": str(self.checkpoint_path), "artifact_compatible": True},
        )


try:
    register_provider("native-taksh", NativeTakshProvider)
except ValueError:
    pass
