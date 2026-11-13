"""Provider diagnostics without exposing credentials."""

from __future__ import annotations

from typing import Any

from serving.providers import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    get_provider,
    list_providers,
    redact_secrets,
)

from serving import coding_providers  # noqa: F401 - registers coding providers
from serving import native_provider  # noqa: F401 - registers native-taksh


def _status_from_error(exc: Exception) -> str:
    if isinstance(exc, ProviderAuthenticationError):
        return "unauthenticated"
    if isinstance(exc, ProviderTimeoutError):
        return "runtime_failure"
    if isinstance(exc, ProviderRateLimitError):
        return "runtime_failure"
    if isinstance(exc, ProviderError):
        return "unavailable"
    return "runtime_failure"


def diagnostic(name: str, *, probe: bool = False) -> dict[str, Any]:
    registered = list_providers()
    if name not in registered:
        return {
            "provider": name,
            "available": False,
            "configured": False,
            "status": "unknown_provider",
            "detail": f"unknown provider: {name}",
            "credential_handling": "not exposed",
        }
    try:
        provider = get_provider(name)
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": name,
            "available": False,
            "configured": False,
            "status": "uninstantiable",
            "detail": redact_secrets(f"{type(exc).__name__}: {exc}"),
            "credential_handling": "not exposed",
        }

    try:
        identity = provider.identity().as_dict()
        capabilities = provider.capabilities().as_dict()
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": name,
            "available": False,
            "configured": False,
            "status": "diagnostic_failed",
            "detail": redact_secrets(f"{type(exc).__name__}: {exc}"),
            "credential_handling": "not exposed",
        }

    status = str(capabilities.get("status") or ("configured" if capabilities.get("available") else "unavailable"))
    detail = ""
    if hasattr(provider, "artifact_status") and callable(provider.artifact_status):
        try:
            artifact = provider.artifact_status()
            status = str(artifact.get("status", status))
            detail = str(artifact.get("detail", ""))
        except Exception as exc:  # noqa: BLE001
            status = "runtime_failure"
            detail = redact_secrets(f"{type(exc).__name__}: {exc}")

    if probe and hasattr(provider, "probe") and callable(provider.probe):
        try:
            probe_result = provider.probe()
            if isinstance(probe_result, dict):
                status = str(probe_result.get("status", status))
                detail = str(probe_result.get("detail", detail))
        except Exception as exc:  # noqa: BLE001
            status = _status_from_error(exc)
            detail = redact_secrets(f"{type(exc).__name__}: {exc}")

    return {
        "provider": identity.get("provider", name),
        "model_id": identity.get("model_id"),
        "revision": identity.get("revision"),
        "source": redact_secrets(identity.get("source", "")),
        "license_or_rights": identity.get("license_or_rights"),
        "available": bool(capabilities.get("available", False)),
        "configured": bool(capabilities.get("configured", False)),
        "offline": capabilities.get("offline"),
        "status": status,
        "test_only": bool(capabilities.get("test_only", False)),
        "modes": capabilities.get("modes", []),
        "max_context_tokens": capabilities.get("max_context_tokens"),
        "notes": capabilities.get("notes", ""),
        "detail": redact_secrets(detail),
        "credential_handling": "not exposed",
    }


def diagnostics(*, probe: bool = False) -> dict[str, Any]:
    registered = list_providers()
    return {name: diagnostic(name, probe=probe) for name in registered}
