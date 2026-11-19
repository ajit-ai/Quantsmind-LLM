"""Shared serving security primitives.

Used by ``completion_server``, ``review_server`` and ``openai_server`` so the
security posture cannot drift between them.

The rule these helpers exist to enforce: **checkpoint paths come from trusted
startup configuration only.** No HTTP request may ever name a file that gets
deserialised.
"""

import os
import sys
from pathlib import Path

from fastapi.responses import JSONResponse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def is_loopback(host: str) -> bool:
    return (host or "").lower() in LOOPBACK_HOSTS


def resolve_bind_host(host: str, allow_remote: bool) -> str:
    """Return the host to bind, refusing non-loopback binding without opt-in.

    Raises SystemExit rather than warning, because a misconfigured server that
    silently listens on every interface is exactly the failure this guards.
    """
    if allow_remote:
        return "0.0.0.0" if host in ("", "0.0.0.0") else host
    if not is_loopback(host):
        raise SystemExit(
            f"Refusing to bind {host!r}. These servers are local-only by default "
            "because they expose an unauthenticated inference surface on your "
            "network. Pass --allow-remote to bind 0.0.0.0 deliberately."
        )
    return host


def api_key_from_env(var: str = "QM_API_KEY"):
    key = os.environ.get(var)
    return key or None


def is_authorized(supplied_header, api_key) -> bool:
    """Single source of truth for the bearer-token decision.

    No configured key means no check, which is only acceptable on a loopback
    listener. The compare is constant-time so the key is not leaked byte by byte.
    """
    if not api_key:
        return True
    import hmac

    return hmac.compare_digest(supplied_header or "", f"Bearer {api_key}")


def unauthorized_response():
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Invalid or missing API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
    )


def bearer_auth_middleware(api_key):
    """Build an ASGI middleware enforcing ``Authorization: Bearer <key>``.

    Returns None when no key is configured, which is only acceptable because the
    listener is loopback by default.
    """

    if not api_key:
        return None

    from starlette.middleware.base import BaseHTTPMiddleware

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not is_authorized(request.headers.get("authorization"), api_key):
                return unauthorized_response()
            return await call_next(request)

    return BearerAuthMiddleware


def install_auth(app, api_key):
    """Attach bearer auth to a FastAPI app if a key is configured."""
    middleware = bearer_auth_middleware(api_key)
    if middleware is not None:
        app.add_middleware(middleware)
        return True
    return False


def resolve_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_serving_contract(ckpt_path: str, tokenizer) -> str:
    """Prove the checkpoint and tokenizer belong to the same prompt contract.

    Schema /2 must match exactly: a ``ContractMismatch`` propagates and the server
    refuses to start. Schema /1 predates contracts and is served under the
    documented legacy contract, with a warning that it is not a v2 artifact.
    """
    from training.checkpoint_schema import SCHEMA_V2, verify_against
    from training.tokenizer_contract import CONTRACT_V1, detect_contract

    from serving.local_inference import torch_load_checkpoint

    # Trusted startup boundary: operator-supplied path, may need unsafe loading
    # for legacy checkpoints that predate safe-serialisation support.
    metadata = torch_load_checkpoint(str(ckpt_path), allow_unsafe_trusted=True)
    detected = detect_contract(tokenizer)
    if metadata.get("schema") == SCHEMA_V2:
        verify_against(metadata["provenance"], tokenizer=tokenizer, contract_id=detected)
        return detected
    print(
        "WARNING: checkpoint has no provenance contract (schema /1); serving under "
        f"the documented legacy contract {CONTRACT_V1}"
    )
    return CONTRACT_V1


def load_trusted_state(ckpt: str, tokenizer: str, device: str):
    """Load a checkpoint from operator-supplied startup configuration.

    Returns ``(model, model_cfg, tokenizer)``. The path is trusted because it came
    from the process command line, never from an HTTP request.
    """
    from tokenizers import Tokenizer

    from serving.local_inference import load_model

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    tok_path = Path(tokenizer)
    if not tok_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tokenizer}")

    # Explicit trusted-artifact boundary: this startup path comes from the
    # operator's command line, never from a request, so legacy checkpoints
    # that require weights_only=False may load here — and only here.
    model, model_cfg = load_model(str(ckpt_path), device, allow_unsafe_trusted=True)
    tok = Tokenizer.from_file(str(tok_path))

    model_vocab = int(model_cfg["vocab_size"])
    tok_vocab = tok.get_vocab_size()
    if tok_vocab > model_vocab:
        raise SystemExit(
            f"Tokenizer vocab {tok_vocab} exceeds model vocab_size {model_vocab}; "
            "token ids would index outside the embedding table."
        )

    verify_serving_contract(str(ckpt_path), tok)
    return model, model_cfg, tok


def add_standard_args(parser, default_port: int):
    """Attach the shared security-relevant CLI flags to a server's parser."""
    parser.add_argument("--ckpt", required=True, help="checkpoint path, loaded at startup")
    parser.add_argument("--tokenizer", default="tokenizer/taksh_code_bpe_v1.json")
    parser.add_argument("--host", default="127.0.0.1", help="default is loopback only")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="bind a non-loopback interface; only safe behind auth and a firewall",
    )
    return parser


def serve(app, parser, args, server_name: str, auth_enabled: bool):
    """Resolve binding, print a startup banner, and run uvicorn."""
    import uvicorn

    host = resolve_bind_host(args.host, args.allow_remote)
    print(
        f"{server_name} on http://{host}:{args.port} "
        f"(auth={'on' if auth_enabled else 'off'}, remote={'yes' if args.allow_remote else 'no'})"
    )
    if not auth_enabled and not is_loopback(host):
        print(
            "WARNING: no QM_API_KEY is set while binding a non-loopback interface. "
            "Anyone who can reach this port can call the model."
        )
    uvicorn.run(app, host=host, port=args.port)