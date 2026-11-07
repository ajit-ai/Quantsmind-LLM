"""Shared language identifiers for Taksh serving surfaces."""

from __future__ import annotations

SUPPORTED_LANGUAGES = frozenset({
    "auto",
    "python",
    "javascript",
    "typescript",
    "java",
    "csharp",
    "cpp",
    "c",
    "go",
    "rust",
    "ruby",
    "shell",
    "yaml",
    "toml",
    "dockerfile",
    "sql",
    "unknown",
})

_ALIASES = {
    "c#": "csharp",
    "cs": "csharp",
    "dotnet": "csharp",
    ".net": "csharp",
    "ts": "typescript",
    "js": "javascript",
    "c++": "cpp",
    "sh": "shell",
    "bash": "shell",
    "yml": "yaml",
}


def normalize_language(value: str | None) -> str:
    normalized = (value or "unknown").strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_LANGUAGES else "unknown"


def is_supported_language(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    return normalized in SUPPORTED_LANGUAGES


def prompt_language(value: str | None) -> str:
    normalized = normalize_language(value)
    return {
        "csharp": "C#",
        "cpp": "C++",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "yaml": "YAML",
        "toml": "TOML",
        "sql": "SQL",
        "auto": "unknown",
    }.get(normalized, normalized)
