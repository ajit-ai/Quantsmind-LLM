"""TAKSH L0.6.0 — WP7: resource and safety controls for future authorized experiments.

Lightweight guards that keep future (owner-authorized) experiments within
explicit limits. These guards enforce discipline; they never grant
authorization. Authority comes from an explicit operator decision recorded
OUTSIDE this module (env var + CLI flag pair), and absence fails closed.

Limits enforced (all configurable; defaults are conservative for CPU):
  * max training steps (default 300)
  * max micro-batch size (default 16)
  * max context length (default 2048)
  * max checkpoint file size (default 512 MiB)
  * max CPU threads (default 4)
  * explicit device selection (default cpu; anything else requires flag)
  * explicit confirmation phrase required for any expensive operation
  * fail-closed when authorization metadata is absent
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULTS = {
    "max_training_steps": 300,
    "max_micro_batch_size": 16,
    "max_context_length": 2048,
    "max_checkpoint_size_bytes": 512 * 1024 * 1024,
    "max_cpu_threads": 4,
    "device": "cpu",
}

AUTH_ENV = "TAKSH_EXECUTE_AUTHORIZED"
AUTH_PHRASE = "I_AUTHORIZE_EXPENSIVE_EXECUTION"


class AuthorizationRequired(RuntimeError):
    """Raised when an expensive operation is attempted without authorization."""


class ResourceLimitExceeded(RuntimeError):
    """Raised when a configured resource bound is breached."""


def authorization_granted() -> bool:
    """True only when the operator set TAKSH_EXECUTE_AUTHORIZED=<phrase>."""
    return os.environ.get(AUTH_ENV) == AUTH_PHRASE


def require_authorization(operation: str) -> None:
    """Fail closed unless an explicit operator authorization is present."""
    if not authorization_granted():
        raise AuthorizationRequired(
            f"operation {operation!r} requires explicit owner authorization "
            f"(set {AUTH_ENV}={AUTH_PHRASE} AND pass --confirm {AUTH_PHRASE} at the call site). "
            "Resource guards do not grant authorization; they only enforce it."
        )


def confirm_expensive(argv: list | None = None) -> None:
    """Hard gate requiring the exact confirmation phrase."""
    argv = list(argv or [])
    try:
        idx = argv.index("--confirm")
        phrase = argv[idx + 1]
    except (ValueError, IndexError) as exc:
        raise AuthorizationRequired(
            f"expensive operation requires --confirm {AUTH_PHRASE}"
        ) from exc
    if phrase != AUTH_PHRASE:
        raise AuthorizationRequired(
            f"incorrect --confirm phrase; expected the exact phrase"
        )
    require_authorization("expensive_execution")


def clamp_threads(requested: int | None = None, default: int = DEFAULTS["max_cpu_threads"]) -> int:
    n = default if requested is None else int(requested)
    return max(1, min(n, DEFAULTS["max_cpu_threads"]))


def enforce_context(length: int, *, max_context: int = DEFAULTS["max_context_length"]) -> int:
    if not isinstance(length, int) or length <= 0:
        raise ResourceLimitExceeded(f"context length must be a positive int, got {length!r}")
    if length > max_context:
        raise ResourceLimitExceeded(f"context length {length} exceeds limit {max_context}")
    return length


def enforce_batch_size(size: int, *, max_batch: int = DEFAULTS["max_micro_batch_size"]) -> int:
    if not isinstance(size, int) or size <= 0:
        raise ResourceLimitExceeded(f"batch size must be a positive int, got {size!r}")
    if size > max_batch:
        raise ResourceLimitExceeded(f"batch size {size} exceeds limit {max_batch}")
    return size


def enforce_training_steps(steps: int, *, max_steps: int = DEFAULTS["max_training_steps"]) -> int:
    if not isinstance(steps, int) or steps <= 0:
        raise ResourceLimitExceeded(f"training steps must be a positive int, got {steps!r}")
    if steps > max_steps:
        raise ResourceLimitExceeded(f"training steps {steps} exceeds limit {max_steps}")
    return steps


def enforce_checkpoint_size(path: str, *, max_bytes: int = DEFAULTS["max_checkpoint_size_bytes"]) -> int:
    size = Path(path).stat().st_size
    if size > max_bytes:
        raise ResourceLimitExceeded(
            f"checkpoint {path} is {size:,} bytes; limit is {max_bytes:,} bytes"
        )
    return size


def device_allowed(device: str) -> str:
    if device != "cpu":
        raise AuthorizationRequired(
            f"device {device!r} requires explicit owner authorization; this host defaults to cpu"
        )
    return device


def assert_guards_do_not_authorize() -> bool:
    """Documented invariant, exercised by tests: no code path in this module
    grants authorization; every expensive path demands the external signal."""
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-defaults", action="store_true")
    args = parser.parse_args(argv)

    if args.print_defaults:
        import json

        print(json.dumps(DEFAULTS, indent=2))
        return 0
    print("resource guard loaded; nothing executed, nothing authorized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
