"""Model-independent coding-provider contract for Taksh.

This module defines the replaceable provider layer. It is deliberately narrow:
the concrete built-in local implementations are the existing Taksh S2 provider
and the Seed-Coder adapter. Other foundation-model adapters are represented as
placeholder providers that report availability and identity but never download
weights or call a hosted service.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class ProviderError(Exception):
    """Base provider failure."""


class ProviderNotFoundError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic generation options shared by providers."""

    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = 20261023
    timeout_seconds: float | None = 120.0
    binding_repair: bool = True
    language: str = "python"

    def validated(self) -> "GenerationConfig":
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when set")
        return self


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validated(self) -> "GenerationRequest":
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        self.config.validated()
        return self


@dataclass(frozen=True)
class ProviderModelIdentity:
    provider: str
    model_id: str
    revision: str
    source: str
    license_or_rights: str
    artifact_sha256: str | None = None
    tokenizer: str | None = None

    def validated(self) -> "ProviderModelIdentity":
        if not self.provider:
            raise ValueError("provider must be set")
        if not self.model_id:
            raise ValueError("model_id must be set")
        if not self.revision:
            raise ValueError("revision must be set")
        if not self.source:
            raise ValueError("source must be set")
        if not self.license_or_rights:
            raise ValueError("license_or_rights must be set")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    available: bool
    offline: bool
    modes: tuple[str, ...] = ()
    max_context_tokens: int | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modes"] = list(self.modes)
        return data


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model_id: str
    revision: str
    text: str
    raw_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    repair: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelProvider(Protocol):
    name: str

    def identity(self) -> ProviderModelIdentity:
        ...

    def capabilities(self) -> ProviderCapabilities:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


class ExternalModelProvider:
    """Architecture placeholder for a foundation model.

    It intentionally cannot generate. It records identity, license and
    capability facts, but it never downloads weights, calls a network
    endpoint, or converts checkpoints.
    """

    name = "external-foundation-placeholder"

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        revision: str,
        source: str,
        license_or_rights: str,
        artifact_sha256: str | None = None,
        tokenizer: str | None = None,
        modes: tuple[str, ...] = ("offline-placeholder",),
        max_context_tokens: int | None = None,
        notes: str = "",
    ) -> None:
        self._identity = ProviderModelIdentity(
            provider=provider,
            model_id=model_id,
            revision=revision,
            source=source,
            license_or_rights=license_or_rights,
            artifact_sha256=artifact_sha256,
            tokenizer=tokenizer,
        ).validated()
        self._capabilities = ProviderCapabilities(
            provider=provider,
            available=False,
            offline=True,
            modes=modes,
            max_context_tokens=max_context_tokens,
            notes=notes or "Placeholder only; no weight download or hosted execution is authorized.",
        )

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        raise ProviderUnavailableError(
            f"provider '{self._identity.provider}' is not available: "
            "foundation adoption is not authorized and no execution path is configured"
        )


class TakshS2Provider:
    """Existing from-scratch S2 model, wrapped without changing behavior."""

    name = "taksh-s2"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str,
        source: str = "local S2 checkpoint",
        license_or_rights: str = "source Apache-2.0; model weights not cleared for redistribution",
        artifact_sha256: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model must be provided")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.model = model
        self.tokenizer = tokenizer
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id=model_id,
            revision=revision,
            source=source,
            license_or_rights=license_or_rights,
            artifact_sha256=artifact_sha256,
            tokenizer=tokenizer_path,
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            available=True,
            offline=True,
            modes=("local", "raw", "binding_repair"),
            max_context_tokens=int(getattr(self.model, "cfg", {}).get("max_seq_len", 512)),
            notes="Original from-scratch S2 path; behavior is preserved by wrapping generate_ids.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        from serving import local_inference

        started = time.perf_counter()
        try:
            result = local_inference.generate_ids(
                self.model,
                self.tokenizer,
                request.prompt,
                max_new_tokens=request.config.max_new_tokens,
                temperature=request.config.temperature,
                top_k=request.config.top_k,
                top_p=request.config.top_p,
                stop=request.config.stop,
                device="cpu",
                seed=request.config.seed,
                binding_repair=request.config.binding_repair,
            )
        except Exception as exc:  # provider boundary: raw output is not replaced
            raise ProviderError(f"taksh-s2 generation failed: {type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started
        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=str(result.get("text", "")),
            raw_text=str(result.get("raw_text", result.get("text", ""))),
            finish_reason=str(result.get("finish_reason", "unknown")),
            prompt_tokens=int(result.get("prompt_tokens", 0)),
            completion_tokens=int(result.get("completion_tokens", 0)),
            latency_seconds=elapsed,
            repair=dict(result.get("repair", {})),
            metadata={
                "prompt_metadata": dict(request.metadata),
                "language": request.config.language,
            },
        )


SEED_CODER_PROVIDER = "seed-coder-8b-instruct"
SEED_CODER_MODEL_ID = "ByteDance-Seed/Seed-Coder-8B-Instruct"
SEED_CODER_REVISION = "1c5af72afa21c22c9272a0b60a3a404195e88fcd"
SEED_CODER_SOURCE = "https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct"
SEED_CODER_LICENSE = "MIT; pending final legal review"
_SEED_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_artifact_hashes(model_path: Path, *, include_weights: bool = False) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for name in _SEED_METADATA_FILES:
        candidate = model_path / name
        hashes[name] = _file_sha256(candidate) if candidate.is_file() else None
    if include_weights:
        weights = [model_path / "model.safetensors", *sorted(model_path.glob("model-*.safetensors"))]
        weights = [p for p in weights if p.is_file()]
        hashes["weights_sha256"] = "\n".join(_file_sha256(p) for p in weights) if weights else None
    return hashes


def _load_seed_resources(model_path: Path, *, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    load_kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype, **load_kwargs)
    if device == "cpu":
        model.to("cpu")
    elif device == "cuda":
        model.to("cuda")
    elif device == "auto":
        try:
            import torch

            model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            pass
    model.eval()
    return model, tokenizer


class SeedCoderProvider:
    """Concrete local Seed-Coder adapter using the D2 provider contract.

    It never downloads weights. If the approved local checkpoint is absent, it
    reports unavailable rather than silently switching to S2.
    """

    name = SEED_CODER_PROVIDER

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        revision: str = SEED_CODER_REVISION,
        model_id: str = SEED_CODER_MODEL_ID,
        source: str = SEED_CODER_SOURCE,
        license_or_rights: str = SEED_CODER_LICENSE,
        device: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        expected_artifact_sha256: str | None = None,
        record_weight_hashes: bool = False,
    ) -> None:
        if revision != SEED_CODER_REVISION:
            raise ValueError(f"Seed-Coder revision must be {SEED_CODER_REVISION}")
        if model_id != SEED_CODER_MODEL_ID:
            raise ValueError(f"Seed-Coder model ID must be {SEED_CODER_MODEL_ID}")
        if trust_remote_code:
            raise ValueError("trust_remote_code is not permitted in D3")
        if not local_files_only:
            raise ValueError("local-only loading is required")
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self.dtype = dtype
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id=model_id,
            revision=revision,
            source=source,
            license_or_rights=license_or_rights,
            artifact_sha256=expected_artifact_sha256,
            tokenizer=str(self.model_path / "tokenizer.json") if self.model_path is not None else None,
        ).validated()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._artifact_hashes = (
            _seed_artifact_hashes(self.model_path, include_weights=record_weight_hashes)
            if self.model_path is not None and self.model_path.exists()
            else {}
        )

    def artifact_hashes(self) -> dict[str, str | None]:
        return dict(self._artifact_hashes)

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def _available(self) -> bool:
        return bool(
            self.model_path is not None
            and self.model_path.is_dir()
            and importlib.util.find_spec("transformers") is not None
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            available=self._available(),
            offline=True,
            modes=("local", "chat_template", "raw", "binding_repair"),
            max_context_tokens=32768,
            notes=(
                "Local Seed-Coder adapter. Exact approved revision only; "
                "missing checkpoint reports unavailable and never falls back to S2."
            ),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.model_path is None:
            raise ProviderUnavailableError("local Seed-Coder model directory is not supplied")
        if not self.model_path.is_dir():
            raise ProviderUnavailableError(f"local Seed-Coder model directory not found: {self.model_path}")
        if importlib.util.find_spec("transformers") is None:
            raise ProviderUnavailableError("transformers is required for the Seed-Coder adapter")
        try:
            self._model, self._tokenizer = _load_seed_resources(
                self.model_path, device=self.device, dtype=self.dtype
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Seed-Coder local load failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._loaded = True

    def _render_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.validated()
        started = time.perf_counter()
        self._ensure_loaded()
        prompt = self._render_prompt(request.prompt)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device if self.device != "auto" else "cpu")
            input_ids = inputs["input_ids"]
            input_length = int(input_ids.shape[-1])
            if request.config.seed is not None:
                try:
                    import torch

                    torch.manual_seed(request.config.seed)
                except Exception:
                    pass
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": request.config.max_new_tokens,
                "do_sample": request.config.temperature > 0,
            }
            if request.config.temperature > 0:
                generate_kwargs.update(
                    {
                        "temperature": request.config.temperature,
                        "top_k": request.config.top_k,
                        "top_p": request.config.top_p,
                    }
                )
            pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            if request.config.stop:
                generate_kwargs["stop_strings"] = list(request.config.stop)
            outputs = self._model.generate(**inputs, **generate_kwargs)
            completion_ids = outputs[0][input_length:]
            raw = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        except Exception as exc:
            raise ProviderError(f"seed-coder generation failed: {type(exc).__name__}: {exc}") from exc

        text = raw
        repair: dict[str, Any] = {
            "enabled": bool(request.config.binding_repair),
            "applied": False,
            "status": "repaired" if request.config.binding_repair else "disabled",
            "reason": "repair enabled" if request.config.binding_repair else "repair disabled",
            "mapping": {},
            "changed": False,
            "raw_preserved": True,
        }
        if request.config.binding_repair:
            try:
                from training.binding_repair import repair_from_prompt_info

                info = repair_from_prompt_info(request.prompt, raw)
                text = info.completion
                repair.update(
                    {
                        "applied": bool(info.changed),
                        "status": info.status,
                        "reason": info.reason,
                        "mapping": dict(info.mapping),
                        "changed": bool(info.changed),
                    }
                )
            except Exception as exc:
                repair.update(
                    {
                        "applied": False,
                        "status": "repair_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "changed": False,
                    }
                )
                text = raw

        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=raw,
            finish_reason="unknown",
            prompt_tokens=input_length,
            completion_tokens=len(completion_ids),
            latency_seconds=time.perf_counter() - started,
            repair=repair,
            metadata={
                "prompt_metadata": dict(request.metadata),
                "language": request.config.language,
                "model_path": str(self.model_path),
                "artifact_hashes": self.artifact_hashes(),
            },
        )


_PROVIDER_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_provider(name: str, factory: Callable[..., Any]) -> None:
    if not name or name in _PROVIDER_FACTORIES:
        raise ValueError(f"provider is empty or already registered: {name!r}")
    _PROVIDER_FACTORIES[name] = factory


def list_providers() -> list[str]:
    return sorted(_PROVIDER_FACTORIES)


def get_provider(name: str, **kwargs: Any) -> Any:
    factory = _PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise ProviderNotFoundError(f"unknown provider: {name!r}")
    return factory(**kwargs)


def qualification_metadata(
    provider: ModelProvider,
    *,
    suite_version: str,
    suite_sha256: str,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GenerationConfig()
    identity = provider.identity().as_dict()
    identity["validated"] = True
    return {
        "provider_identity": identity,
        "capabilities": provider.capabilities().as_dict(),
        "suite": {"version": suite_version, "sha256": suite_sha256},
        "generation_config": asdict(cfg),
    }


def generate_with_timeout(provider: ModelProvider, request: GenerationRequest) -> GenerationResult:
    request.validated()
    timeout = request.config.timeout_seconds
    if timeout is None:
        return provider.generate(request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(provider.generate, request)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise ProviderTimeoutError(
                f"provider '{provider.name}' exceeded timeout of {timeout} seconds"
            ) from exc


register_provider("taksh-s2", TakshS2Provider)
register_provider(
    SEED_CODER_PROVIDER,
    lambda model_path=None, **kwargs: SeedCoderProvider(model_path=model_path, **kwargs),
)
register_provider(
    "qwen2.5-coder-7b-instruct",
    lambda **kwargs: ExternalModelProvider(
        provider="qwen2.5-coder-7b-instruct",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
        source="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        license_or_rights="Apache-2.0; legal review still required before redistribution",
        modes=("offline-placeholder",),
        max_context_tokens=131072,
        notes="Close Apache-2.0 fallback; placeholder only.",
    ),
)
register_provider(
    "qwen3-coder-30b-a3b-instruct",
    lambda **kwargs: ExternalModelProvider(
        provider="qwen3-coder-30b-a3b-instruct",
        model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        source="https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        license_or_rights="Apache-2.0; hosted/reference use only until hardware and legal review",
        modes=("offline-placeholder", "hosted-reference"),
        max_context_tokens=262144,
        notes="Hosted/reference candidate; placeholder only.",
    ),
)
