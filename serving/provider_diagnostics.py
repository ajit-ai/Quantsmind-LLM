"""Provider diagnostics without exposing credentials."""

from __future__ import annotations

from typing import Any

from serving.providers import ProviderError, get_provider, list_providers


def diagnostic(name: str) -> dict[str, Any]:
    try:
        provider = get_provider(name)
    except ProviderError as exc:
        return {"provider": name, "available": False, "status": "unknown_provider", "detail": str(exc)}

    try:
        identity = provider.identity().as_dict()
        capabilities = provider.capabilities().as_dict()
    except Exception as exc:  # noqa: BLE001
        return {"provider": name, "available": False, "status": "diagnostic_failed", "detail": f"{type(exc).__name__}: {exc}"}

    return {
        "provider": identity.get("provider", name),
        "model_id": identity.get("model_id"),
        "revision": identity.get("revision"),
        "source": identity.get("source"),
        "license_or_rights": identity.get("license_or_rights"),
        "available": capabilities.get("available", False),
        "offline": capabilities.get("offline"),
        "modes": capabilities.get("modes", []),
        "max_context_tokens": capabilities.get("max_context_tokens"),
        "notes": capabilities.get("notes", ""),
        "credential_handling": "not exposed",
    }


def diagnostics() -> dict[str, Any]:
    registered = list_providers()
    return {name: diagnostic(name) for name in registered}
