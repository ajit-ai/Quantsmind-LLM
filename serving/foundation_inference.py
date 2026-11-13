"""TAKSH L0.6.0 — structured inference surface for the foundation model.

Reuses the verified serving stack (serving/local_inference.py) and adds:
* typed structured inference errors (no bare tracebacks)
* a thin, documented API for loading a foundation checkpoint + tokenizer
* deterministic, bounded generation with explicit limits

Never instantiates the 111M foundation artifact outside of authorized runs;
smoke tests use tiny synthetic configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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


def load_foundation(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """Load checkpoint + matching tokenizer as a (model, tokenizer, meta) tuple.

    Any failure is surfaced as CheckpointIncompatibleError with a clean,
    operator-readable message. No silent fallback to any other model.
    """
    try:
        model, cfg = load_model(checkpoint_path, device)
    except FileNotFoundError as exc:
        raise CheckpointIncompatibleError(f"checkpoint not found: {checkpoint_path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CheckpointIncompatibleError(
            f"checkpoint {checkpoint_path} failed to load: {type(exc).__name__}: {exc}"
        ) from exc
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
    meta = {
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "parameters": model.num_params(),
        "device": device,
        "max_seq_len": int(cfg.get("max_seq_len", 0)),
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
