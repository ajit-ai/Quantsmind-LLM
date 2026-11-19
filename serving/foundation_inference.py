"""TAKSH L0.6.0 — structured inference surface for the foundation model.

Reuses the verified serving stack (serving/local_inference.py) and adds:
* typed structured inference errors (no bare tracebacks)
* a thin, documented API for loading a foundation checkpoint + tokenizer
* explicit checkpoint/tokenizer identity (file SHA-256) and template validation
* deterministic, bounded generation with explicit prompt/context budget limits

Never instantiates the 111M foundation artifact outside of authorized runs;
smoke tests use tiny synthetic configurations.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Optional

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.local_inference import generate_ids, load_model  # noqa: E402


class InferenceError(RuntimeError):
    """Base class for structured inference failures."""


class ConfigurationError(InferenceError):
    """The inference request itself is invalid (limits out of range, bad args)."""


class CheckpointIncompatibleError(InferenceError):
    """Checkpoint or tokenizer failed to load compatibly."""


class GenerationFailedError(InferenceError):
    """Generation failed mid-stream (OOM, numerical issue, interrupt)."""


def _sha256_file(path: str | Path, *, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_foundation(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """Load checkpoint + matching tokenizer as a (model, tokenizer, meta) tuple.

    Any failure is surfaced as CheckpointIncompatibleError with a clean,
    operator-readable message. No silent fallback to any other model.
    """
    try:
        checkpoint_sha = _sha256_file(checkpoint_path)
    except OSError as exc:
        raise CheckpointIncompatibleError(f"checkpoint sha256 failed: {checkpoint_path}: {exc}") from exc

    try:
        model, cfg = load_model(checkpoint_path, device)
    except FileNotFoundError as exc:
        raise CheckpointIncompatibleError(f"checkpoint not found: {checkpoint_path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CheckpointIncompatibleError(
            f"checkpoint {checkpoint_path} failed to load: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        tokenizer_sha = _sha256_file(tokenizer_path)
    except OSError as exc:
        raise CheckpointIncompatibleError(f"tokenizer sha256 failed: {tokenizer_path}: {exc}") from exc

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointIncompatibleError(
            f"tokenizer {tokenizer_path} failed to load: {type(exc).__name__}: {exc}"
        ) from exc

    if tokenizer.get_vocab_size() != int(cfg["vocab_size"]):
        raise CheckpointIncompatibleError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} != checkpoint vocab {cfg['vocab_size']}"
        )

    try:
        from training.chat_template import validate_chat_template

        markers = validate_chat_template(tokenizer)
    except Exception as exc:  # noqa: BLE001 - template contract failure is fatal to serving
        raise CheckpointIncompatibleError(
            f"tokenizer {tokenizer_path} does not satisfy the chat-template contract: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        total_parameters = sum(int(p.numel()) for p in model.parameters())
    except Exception:  # noqa: BLE001 - diagnostic only
        total_parameters = None
    meta = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": tokenizer.get_vocab_size(),
        "parameters": model.num_params(),
        "total_parameters": total_parameters,
        "device": device,
        "max_seq_len": int(cfg.get("max_seq_len", 0)),
        "model_name": str(cfg.get("name") or cfg.get("model_name") or "taksh-native"),
        "model_config": dict(cfg),
        "template_markers": markers,
    }
    return model, tokenizer, meta


def generate_bounded(model, tokenizer, prompt: str, *, max_new_tokens: int = 64,
                     temperature: float = 0.0, top_k: int = 50,
                     device: str = "cpu", seed: Optional[int] = None) -> dict:
    """Deterministic, bounded generation with explicit, enforced limits.

    Enforces:
      * max_new_tokens <= 256 (hard cap; the serving default is lower)
      * prompt truncation to model max_seq_len (via generate_ids)
      * deterministic argmax (temperature == 0) or seeded sampling
    Raises GenerationFailedError on any mid-stream failure.
    """
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 256:
        raise ConfigurationError(f"max_new_tokens must be in [1, 256], got {max_new_tokens}")
    if temperature < 0.0 or temperature > 2.0:
        raise ConfigurationError(f"temperature must be in [0.0, 2.0], got {temperature}")
    try:
        prompt_tokens = len(tokenizer.encode(prompt).ids)
    except Exception as exc:  # noqa: BLE001 - tokenizer failures are structured, not bare
        raise ConfigurationError(
            f"prompt tokenization failed: {type(exc).__name__}: {exc}"
        ) from exc
    max_seq_len = int(getattr(model, "cfg", {}).get("max_seq_len", 0))
    if max_seq_len <= 0:
        raise ConfigurationError("model exposes no valid max_seq_len; refusing to generate")
    if prompt_tokens >= max_seq_len:
        raise ConfigurationError(
            f"prompt of {prompt_tokens} tokens exceeds model context limit {max_seq_len}"
        )
    if prompt_tokens + max_new_tokens > max_seq_len:
        raise ConfigurationError(
            f"prompt({prompt_tokens}) + max_new_tokens({max_new_tokens}) "
            f"exceeds context limit {max_seq_len}; lower the budget or shorten the prompt"
        )
    if seed is not None:
        torch.manual_seed(seed)
    try:
        return generate_ids(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_k=top_k,
            device=device,
            seed=seed,
            binding_repair=True,
        )
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - CPU-only today
        raise GenerationFailedError(f"out of memory during generation: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - structured, never bare
        raise GenerationFailedError(f"generation failed: {type(exc).__name__}: {exc}") from exc


def max_context_note() -> str:
    return (
        "generate_ids truncates to model.cfg['max_seq_len'] for the KV-cache path "
        "and the no-cache path slices x[:, -max_seq_len:] for scoring; prompts "
        "never exceed the configured context"
    )
