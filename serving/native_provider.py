"""Native TAKSH provider adapter over serving.foundation_inference.

This adapter selects the L0.6 foundation implementation. It fails closed when
checkpoint/tokenizer are missing or incompatible, and never falls back to an
unrelated checkpoint or calls a hosted provider.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from serving.foundation_inference import (
    ConfigurationError,
    CheckpointIncompatibleError,
    GenerationFailedError,
    generate_bounded,
    load_foundation,
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

    def __init__(self, checkpoint_path=None, tokenizer_path=None, *, device: str = "cpu") -> None:
        env_checkpoint, env_tokenizer = _native_paths()
        checkpoint = checkpoint_path or env_checkpoint
        tokenizer = tokenizer_path or env_tokenizer
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        self.tokenizer_path = Path(tokenizer) if tokenizer else None
        self.device = device
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id="taksh-foundation-111m",
            revision="unqualified",
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
            notes="Native L0.6 foundation backend; not capability-qualified without a compatible trained checkpoint and evaluation.",
        )

    def _require_paths(self):
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
        try:
            model, tokenizer, meta = load_foundation(str(self.checkpoint_path), str(self.tokenizer_path), self.device)
            result = generate_bounded(
                model,
                tokenizer,
                request.prompt,
                max_new_tokens=min(request.config.max_new_tokens, 256),
                temperature=request.config.temperature,
                top_k=request.config.top_k,
                device=self.device,
                seed=request.config.seed,
            )
        except (ConfigurationError, CheckpointIncompatibleError, GenerationFailedError) as exc:
            raise ProviderUnavailableError(f"native inference failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - fail closed, never fallback
            raise ProviderUnavailableError(f"native inference failed: {type(exc).__name__}: {exc}") from exc

        return GenerationResult(
            provider=self.name,
            model_id="taksh-foundation-111m",
            revision=Path(self.checkpoint_path).name,
            text=str(result.get("text", "")),
            raw_text=str(result.get("raw_text", result.get("text", ""))),
            finish_reason=str(result.get("finish_reason", "unknown")),
            prompt_tokens=int(result.get("prompt_tokens", 0) or 0),
            completion_tokens=int(result.get("completion_tokens", 0) or 0),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "native-foundation"},
            metadata={"checkpoint": str(self.checkpoint_path), "meta": meta},
        )


try:
    register_provider("native-taksh", NativeTakshProvider)
except ValueError:
    pass
