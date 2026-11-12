"""TAKSH — native generation request contract (L1.1.1).

Fail-closed input validation shared by the local inference engine and the
providers. Every invalid parameter raises ``TypeError`` or ``ValueError`` with
a message naming the offender — never a silent auto-fix.

Semantics (matching ``serving.providers.GenerationConfig.validated``):

- ``max_new_tokens``: int >= 1
- ``temperature``: float in [0.0, 2.0] (0.0 == greedy, dispatched to argmax)
- ``top_k``: int >= 0 (0 disables top-k)
- ``top_p``: float in (0.0, 1.0]
- ``stop``: None | str | iterable of str (each non-empty)
- ``presence_penalty`` / ``frequency_penalty``: float in [-2.0, 2.0]
- ``seed``: None | int (not bool)
- ``prompt``: non-empty str (empty or whitespace-only is rejected)
"""

from __future__ import annotations

from typing import Any


class GenerationContractError(ValueError):
    """A generation-time parameter violates the serving contract."""


def validate_generation_request(*,
                                prompt: Any,
                                max_new_tokens: Any,
                                temperature: Any,
                                top_k: Any,
                                top_p: Any,
                                stop: Any,
                                presence_penalty: Any,
                                frequency_penalty: Any,
                                seed: Any) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationContractError(
            f"prompt must be a non-empty string, got {type(prompt).__name__} "
            f"with len={len(prompt) if isinstance(prompt, str) else 'n/a'}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise GenerationContractError(
            f"max_new_tokens must be an int, got {type(max_new_tokens).__name__}")
    if max_new_tokens < 1:
        raise GenerationContractError(
            f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise GenerationContractError(
            f"temperature must be a number, got {type(temperature).__name__}")
    if not 0.0 <= float(temperature) <= 2.0:
        raise GenerationContractError(
            f"temperature must be in [0.0, 2.0], got {temperature}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise GenerationContractError(
            f"top_k must be an int (>=0; 0 disables), got {type(top_k).__name__}")
    if top_k < 0:
        raise GenerationContractError(f"top_k must be >= 0, got {top_k}")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise GenerationContractError(
            f"top_p must be a number, got {type(top_p).__name__}")
    if not 0.0 < float(top_p) <= 1.0:
        raise GenerationContractError(
            f"top_p must be in (0.0, 1.0], got {top_p}")
    for name, value in (("presence_penalty", presence_penalty),
                        ("frequency_penalty", frequency_penalty)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GenerationContractError(
                f"{name} must be a number, got {type(value).__name__}")
        if not -2.0 <= float(value) <= 2.0:
            raise GenerationContractError(
                f"{name} must be in [-2.0, 2.0], got {value}")
    for name, value in (("temperature", temperature),
                        ("top_k", top_k),
                        ("top_p", top_p),
                        ("presence_penalty", presence_penalty),
                        ("frequency_penalty", frequency_penalty)):
        fv = float(value)
        if fv != fv or fv in (float("inf"), float("-inf")):  # NaN/Inf
            raise GenerationContractError(f"{name} must be finite, got {value}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise GenerationContractError(
            f"seed must be an int or null, got {type(seed).__name__}")
    if stop is not None:
        stop_list = [stop] if isinstance(stop, str) else list(stop)
        for item in stop_list:
            if not isinstance(item, str) or not item:
                raise GenerationContractError(
                    f"stop sequences must be non-empty strings, got {item!r}")
        if len(stop_list) > 8:
            raise GenerationContractError(
                f"at most 8 stop sequences are supported, got {len(stop_list)}")
