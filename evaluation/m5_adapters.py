"""Model adapters for M5 evaluation.

One interface, three implementations, so the runner never special-cases a
model family.

``QMCkptAdapter``
    QuantsMind M3/M4 checkpoints. Reuses ``training.generate`` and
    ``training.checkpoint`` -- the M3/M4 loading and generation paths are not
    reimplemented here.

``ExternalHFAdapter``
    a Hugging Face causal LM, loaded with ``transformers``. Optional: the
    import happens inside ``preflight``, so the existing M3/M4 test suite does
    not gain a new hard dependency.

No adapter calls a hosted inference API. Every path is local.

Every adapter reports ``can_evaluate`` before anything is loaded, with a reason.
That ordering matters on a 4 GiB host: the refusal has to happen *before* the
weights are touched, not after an OOM.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ADAPTER_VERSION = "quantsmind.m5_adapter/1"


class AdapterUnavailable(RuntimeError):
    """The model cannot be evaluated here, and the reason is explicit."""


@dataclass
class AdapterInfo:
    """Standardized identity and resource record. Every adapter returns one."""

    adapter: str
    model_id: str
    family: str
    tokenizer_id: str | None = None
    checkpoint_path: str | None = None
    artifact_sha256: str | None = None
    revision: str | None = None
    parameters: int | None = None
    context_length: int | None = None
    vocab_size: int | None = None
    parameters_source: str = "unknown"
    license_id: str | None = None
    license_verified: bool = False
    load_seconds: float | None = None
    artifact_bytes: int | None = None
    peak_rss_bytes: int | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_snapshot() -> dict:
    from training.preflight import disk_space, peak_rss_bytes, system_memory, torch_facts

    memory = system_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_facts(),
        "memory": memory,
        "disk": disk_space("."),
        "peak_rss_bytes": peak_rss_bytes(),
    }


# --------------------------------------------------------------------------- #
# Base contract
# --------------------------------------------------------------------------- #
class BaseAdapter:
    """The contract every adapter implements."""

    name = "base"
    family = "unknown"

    def preflight(self) -> tuple[bool, str]:
        """``(can_evaluate, reason)``. Must not load the model."""
        raise NotImplementedError

    def info(self) -> AdapterInfo:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 96, **kwargs) -> dict:
        """Return ``{'text', 'token_ids', 'finish_reason', 'seconds', ...}``."""
        raise NotImplementedError

    def release(self) -> None:
        """Free resources. Safe to call more than once."""
        raise NotImplementedError

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# --------------------------------------------------------------------------- #
# QuantsMind checkpoints (M3 / M4)
# --------------------------------------------------------------------------- #
class QMCkptAdapter(BaseAdapter):
    """Loads a QuantsMind checkpoint through the M3/M4 generation path."""

    name = "qm_ckpt"
    family = "quantsmind_llama"

    def __init__(self, checkpoint_path, tokenizer_path=None, device="cpu",
                 model_config_path=None):
        self.checkpoint_path = str(checkpoint_path)
        # Exposed so the runner can label results without knowing the adapter type.
        self.model_id = str(checkpoint_path)
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.model_config_path = model_config_path
        self._model = None
        self._info = None

    def preflight(self):
        from training.preflight import system_memory

        path = Path(self.checkpoint_path)
        if not path.exists():
            return False, f"checkpoint not found: {path}"
        size_mb = path.stat().st_size / 2**20
        memory = system_memory()
        available_mb = (memory["available_bytes"] or 0) / 2**20
        if available_mb and available_mb < size_mb:
            return False, (
                f"insufficient RAM: checkpoint is {size_mb:.0f} MiB and only "
                f"{available_mb:.0f} MiB is available"
            )
        if self.tokenizer_path and not Path(self.tokenizer_path).exists():
            return False, f"tokenizer not found: {self.tokenizer_path}"
        return True, "available"

    def load(self) -> None:
        from training.generate import (
            GenerationError, load_checkpoint_for_generation, load_tokenizer,
        )

        ok, reason = self.preflight()
        if not ok:
            raise AdapterUnavailable(reason)
        started = time.perf_counter()
        try:
            model, info = load_checkpoint_for_generation(
                self.checkpoint_path, device=self.device)
        except GenerationError as exc:
            raise AdapterUnavailable(str(exc)) from exc

        tokenizer_path = self.tokenizer_path or (
            info.get("tokenizer_identity") or {}).get("path")
        try:
            tokenizer, vocab = load_tokenizer(tokenizer_path, info["vocab_size"])
        except GenerationError as exc:
            raise AdapterUnavailable(str(exc)) from exc

        self._model = model
        self._tokenizer = tokenizer
        self._load_seconds = time.perf_counter() - started
        self._info = AdapterInfo(
            adapter=self.name,
            model_id=self.checkpoint_path,
            family=self.family,
            tokenizer_id=str(tokenizer_path),
            checkpoint_path=self.checkpoint_path,
            artifact_sha256=sha256_file(self.checkpoint_path),
            parameters=info.get("parameters"),
            context_length=info.get("max_seq_len"),
            vocab_size=vocab,
            parameters_source="measured from the loaded state_dict",
            load_seconds=round(self._load_seconds, 3),
            artifact_bytes=Path(self.checkpoint_path).stat().st_size,
            extra={
                "checkpoint_step": info.get("checkpoint_step"),
                "tokens_processed": info.get("tokens_processed"),
                "val_loss": info.get("val_loss_at_checkpoint"),
                "interrupted_run": info.get("interrupted"),
                "device": self.device,
            },
        )

    def info(self) -> AdapterInfo:
        if self._info is None:
            raise AdapterUnavailable("model not loaded; call load() first")
        return self._info

    def generate(self, prompt: str, max_new_tokens: int = 96,
                 greedy: bool = True, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, seed: int | None = None,
                 stop: list | None = None) -> dict:
        if self._model is None:
            raise AdapterUnavailable("model not loaded; call load() first")
        from training.generate import generate_text

        started = time.perf_counter()
        result = generate_text(
            self._model, self._tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, greedy=greedy, stop=stop, seed=seed,
            device=self.device,
        )
        elapsed = time.perf_counter() - started
        tokens = max(result["completion_tokens"], 1)
        return {
            "text": result["completion"],
            "token_ids": result["token_ids"],
            "finish_reason": result["finish_reason"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "seconds": round(elapsed, 4),
            "tokens_per_second": round(tokens / elapsed, 2) if elapsed else None,
        }

    def release(self) -> None:
        self._model = None
        self._tokenizer = None


# --------------------------------------------------------------------------- #
# External open-weight candidates
# --------------------------------------------------------------------------- #
class ExternalHFAdapter(BaseAdapter):
    """A Hugging Face causal LM, loaded locally with ``transformers``.

    Every refusal happens in :meth:`preflight`, before weights are touched:
    missing dependency, absent weights, insufficient RAM, missing licence
    record. Downloading is never attempted implicitly -- ``allow_download`` must
    be set explicitly by an operator who has decided that is appropriate.
    """

    name = "external_hf"
    family = "external_open_weight"

    def __init__(self, model_id, revision=None, device="cpu",
                 license_record=None, allow_download=False,
                 dtype="auto", local_files_only=None, trust_remote_code=False):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.license_record = license_record or {}
        self.allow_download = allow_download
        self.dtype = dtype
        # Default to local-only unless the operator explicitly permits fetching.
        self.local_files_only = (
            not allow_download if local_files_only is None else local_files_only
        )
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None
        self._info = None

    # ---- preflight -------------------------------------------------------
    def _transformers_available(self):
        try:
            import transformers  # noqa: PLC0415
            return transformers.__version__
        except Exception:  # noqa: BLE001
            return None

    def _snapshot_path(self) -> Path | None:
        """Locate local weights without triggering a download."""
        try:
            import os

            from huggingface_hub import try_to_load_from_cache

            found = try_to_load_from_cache(
                self.model_id, "model.safetensors", revision=self.revision
            )
            if isinstance(found, str) and Path(found).exists():
                return Path(found)
        except Exception:  # noqa: BLE001 - absence is the answer, not an error
            pass
        for env in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
            base = os.environ.get(env) if (os := __import__("os")) else None
            if not base:
                continue
            candidate = Path(base) / "hub" / self.model_id.replace("/", "--")
            if candidate.exists():
                for name in ("model.safetensors", "pytorch_model.bin"):
                    if (candidate / name).exists():
                        return candidate / name
        return None

    def preflight(self):
        from training.preflight import system_memory

        version = self._transformers_available()
        if not version:
            return False, (
                "the 'transformers' package is not installed; external "
                "evaluation cannot run"
            )

        weights = self._snapshot_path()
        if weights is None:
            if self.allow_download:
                return True, (
                    f"download permitted by operator; would fetch {self.model_id} "
                    f"(revision {self.revision or 'default'})"
                )
            return False, (
                f"no local weights for {self.model_id!r} and download is not "
                "permitted by this run (allow_download=False); nothing was "
                "fetched"
            )

        size_mb = weights.stat().st_size / 2**20
        memory = system_memory()
        available_mb = (memory["available_bytes"] or 0) / 2**20
        # fp16 weights plus a small activation margin. Deliberately conservative.
        required_mb = size_mb * 1.3
        if available_mb and required_mb > available_mb:
            return False, (
                f"insufficient RAM: {weights.name} is {size_mb:.0f} MiB, needs "
                f"about {required_mb:.0f} MiB with activations, and only "
                f"{available_mb:.0f} MiB is available"
            )

        if not self.license_record.get("license_id"):
            return False, (
                "no licence record supplied; an external model may not be "
                "evaluated or redistributed without a verified licence"
            )
        return True, (
            f"transformers {version}; {weights.name} {size_mb:.0f} MiB; "
            f"{available_mb:.0f} MiB available"
        )

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        ok, reason = self.preflight()
        if not ok:
            raise AdapterUnavailable(reason)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        dtype = {
            "float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype, "auto")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device).eval()
        self._load_seconds = time.perf_counter() - started

        parameters = sum(p.numel() for p in self._model.parameters())
        self._info = AdapterInfo(
            adapter=self.name, model_id=self.model_id, family=self.family,
            tokenizer_id=self.model_id, revision=self.revision,
            parameters=parameters,
            context_length=getattr(self._model.config, "max_position_embeddings", None),
            vocab_size=getattr(self._model.config, "vocab_size", None),
            parameters_source="measured from the loaded model",
            license_id=self.license_record.get("license_id"),
            license_verified=bool(self.license_record.get("verified")),
            load_seconds=round(self._load_seconds, 3),
            extra={
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
                "device": self.device,
            },
        )

    def info(self) -> AdapterInfo:
        if self._info is None:
            raise AdapterUnavailable("model not loaded; call load() first")
        return self._info

    def generate(self, prompt: str, max_new_tokens: int = 96,
                 greedy: bool = True, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, seed: int | None = None,
                 stop: list | None = None) -> dict:
        if self._model is None:
            raise AdapterUnavailable("model not loaded; call load() first")
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_k=top_k if (top_k and not greedy) else None,
                top_p=top_p if not greedy else None,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        finished = int(generated[-1].item()) == int(
            self._tokenizer.eos_token_id
        ) if generated.numel() else False
        tokens = max(len(generated), 1)
        return {
            "text": text,
            "token_ids": generated.tolist(),
            "finish_reason": "stop" if finished else "length",
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.numel()),
            "seconds": round(elapsed, 4),
            "tokens_per_second": round(tokens / elapsed, 2) if elapsed else None,
        }

    def release(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()