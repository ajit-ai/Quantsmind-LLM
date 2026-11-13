"""Native TAKSH provider adapter over serving.foundation_inference.

This adapter selects the L0.6 foundation implementation. It fails closed when
checkpoint/tokenizer are missing or incompatible, and never falls back to an
unrelated checkpoint or calls a hosted provider.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

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
    redact_secrets,
    register_provider,
)

DEFAULT_MAX_CONTEXT = 2048
HARD_MAX_NEW_TOKENS = 256


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
        self._probe_status: str | None = None
        self._probe_detail = ""
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

    def _set_status(self, status: str, detail: str = "") -> None:
        self._probe_status = status
        self._probe_detail = redact_secrets(detail)

    def artifact_status(self) -> dict[str, Any]:
        """Cheap status that never loads model weights."""
        if not self.checkpoint_path or not self.tokenizer_path:
            return {
                "status": "checkpoint_missing",
                "detail": (
                    "native checkpoint/tokenizer are not configured; set "
                    "TAKSH_NATIVE_CHECKPOINT and TAKSH_NATIVE_TOKENIZER"
                ),
            }
        if not self.checkpoint_path.is_file():
            return {
                "status": "checkpoint_missing",
                "detail": f"native checkpoint not found: {self.checkpoint_path}",
            }
        if not self.tokenizer_path.is_file():
            return {
                "status": "tokenizer_missing",
                "detail": f"native tokenizer not found: {self.tokenizer_path}",
            }
        if self._probe_status:
            return {"status": self._probe_status, "detail": self._probe_detail}
        return {
            "status": "configured_unverified",
            "detail": "artifacts are present; compatibility is verified only by an explicit probe or generation",
        }

    def capabilities(self) -> ProviderCapabilities:
        state = self.artifact_status()["status"]
        artifacts_present = state not in ("checkpoint_missing", "tokenizer_missing")
        available = artifacts_present and state != "checkpoint_incompatible"
        return ProviderCapabilities(
            provider=self.name,
            available=available,
            offline=True,
            configured=bool(self.checkpoint_path and self.tokenizer_path),
            status=state,
            modes=("local",) if available else (),
            max_context_tokens=DEFAULT_MAX_CONTEXT,
            notes=(
                "Native L0.6 foundation backend; available means artifacts are present, "
                "not that capability is qualified. No fallback checkpoint is used."
            ),
        )

    def _require_paths(self):
        if not self.checkpoint_path or not self.tokenizer_path:
            self._set_status("checkpoint_missing")
            raise ProviderUnavailableError(
                "native checkpoint/tokenizer are not configured; set "
                "TAKSH_NATIVE_CHECKPOINT and TAKSH_NATIVE_TOKENIZER"
            )
        if not Path(self.checkpoint_path).is_file():
            self._set_status("checkpoint_missing", f"native checkpoint not found: {self.checkpoint_path}")
            raise ProviderUnavailableError(f"native checkpoint not found: {self.checkpoint_path}")
        if not Path(self.tokenizer_path).is_file():
            self._set_status("tokenizer_missing", f"native tokenizer not found: {self.tokenizer_path}")
            raise ProviderUnavailableError(f"native tokenizer not found: {self.tokenizer_path}")

    def probe(self) -> dict[str, Any]:
        """Explicitly load and validate the native artifacts.

        This is intentionally not called by ``capabilities()`` or ordinary
        diagnostics because it instantiates the model.
        """
        self._require_paths()
        try:
            model, tokenizer, meta = load_foundation(
                str(self.checkpoint_path), str(self.tokenizer_path), self.device
            )
        except CheckpointIncompatibleError as exc:
            self._set_status("checkpoint_incompatible", str(exc))
            return {"status": "checkpoint_incompatible", "detail": redact_secrets(str(exc))}
        except Exception as exc:  # noqa: BLE001 - fail closed
            self._set_status("runtime_failure", f"{type(exc).__name__}: {exc}")
            return {
                "status": "runtime_failure",
                "detail": redact_secrets(f"{type(exc).__name__}: {exc}"),
            }
        self._set_status("compatible")
        return {"status": "compatible", "meta": meta, "parameters": model.num_params()}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        started = time.perf_counter()
        self._require_paths()
        try:
            model, tokenizer, meta = load_foundation(
                str(self.checkpoint_path), str(self.tokenizer_path), self.device
            )
        except CheckpointIncompatibleError as exc:
            self._set_status("checkpoint_incompatible", str(exc))
            raise ProviderUnavailableError(f"native checkpoint is incompatible: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - fail closed, never fallback
            self._set_status("runtime_failure", f"{type(exc).__name__}: {exc}")
            raise ProviderUnavailableError(
                f"native checkpoint failed to load: {type(exc).__name__}: {exc}"
            ) from exc

        max_seq = int(meta.get("max_seq_len") or DEFAULT_MAX_CONTEXT)
        try:
            prompt_tokens = len(tokenizer.encode(request.prompt).ids)
        except Exception as exc:  # noqa: BLE001 - tokenizer failures are structured
            self._set_status("runtime_failure", f"tokenizer encode failed: {type(exc).__name__}: {exc}")
            raise ProviderUnavailableError(f"native tokenizer encode failed: {exc}") from exc
        if prompt_tokens >= max_seq:
            raise ProviderUnavailableError(
                f"prompt length {prompt_tokens} tokens exceeds native context limit {max_seq}"
            )
        max_new_tokens = min(
            request.config.max_new_tokens,
            HARD_MAX_NEW_TOKENS,
            max(1, max_seq - prompt_tokens),
        )

        try:
            result = generate_bounded(
                model,
                tokenizer,
                request.prompt,
                max_new_tokens=max_new_tokens,
                temperature=request.config.temperature,
                top_k=request.config.top_k,
                device=self.device,
                seed=request.config.seed,
            )
        except ConfigurationError as exc:
            raise ProviderUnavailableError(f"native generation configuration rejected: {exc}") from exc
        except GenerationFailedError as exc:
            self._set_status("runtime_failure", str(exc))
            raise ProviderUnavailableError(f"native generation failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - fail closed, never fallback
            self._set_status("runtime_failure", f"{type(exc).__name__}: {exc}")
            raise ProviderUnavailableError(
                f"native inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._set_status("compatible")
        return GenerationResult(
            provider=self.name,
            model_id="taksh-foundation-111m",
            revision=Path(self.checkpoint_path).name,
            text=str(result.get("text", "")),
            raw_text=str(result.get("raw_text", result.get("text", ""))),
            finish_reason=str(result.get("finish_reason", "unknown")),
            prompt_tokens=int(result.get("prompt_tokens", prompt_tokens) or prompt_tokens),
            completion_tokens=int(result.get("completion_tokens", 0) or 0),
            latency_seconds=time.perf_counter() - started,
            repair={"enabled": bool(request.config.binding_repair), "status": "native-foundation"},
            metadata={"checkpoint": str(self.checkpoint_path), "meta": meta},
        )


try:
    register_provider("native-taksh", NativeTakshProvider)
except ValueError:
    pass
