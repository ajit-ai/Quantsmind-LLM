"""Language dispatch and syntax validation (Taksh v1.4.0).

Validators are registered per language identifier, so the engine is not
Python-specific. Python is the only language with a validator in this
version; a second language needs a registration, not a new engine.
"""

from __future__ import annotations

import ast
import re

#: language identifier -> validator callable ``(source) -> (ok, category, detail)``
VALIDATORS: dict = {}


def register(language: str):
    """Decorator registering a syntax validator for a language."""

    def wrapper(function):
        VALIDATORS[language] = function
        return function

    return wrapper


def known_languages() -> list:
    return sorted(VALIDATORS)


def validate_syntax(source: str, language: str = "python"):
    """Validate generated source for a language.

    Returns ``(ok, category, detail)``. ``category`` is ``"ok"``,
    ``"syntax_error"`` or ``"unsupported_language"``.
    """
    validator = VALIDATORS.get(language)
    if validator is None:
        return False, "unsupported_language", f"no validator for {language!r}"
    return validator(source)


@register("python")
def _validate_python(source: str):
    if not isinstance(source, str):
        return False, "syntax_error", "output is not text"
    if not source.strip():
        return False, "syntax_error", "empty output"
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax_error", f"line {exc.lineno}: {exc.msg}"
    except (ValueError, MemoryError, RecursionError) as exc:
        return False, "syntax_error", type(exc).__name__
    return True, "ok", ""


@register("python-like")
def _validate_python_like(source: str):
    """Validator for a Python-like dialect: same parser, relaxed rules."""
    return _validate_python(source)


def extract_code_block(text: str) -> str:
    """Extract the first fenced code block, or return the text unchanged.

    Models frequently wrap code in markdown fences; the evaluator needs the
    code itself, and the formatting-compliance metric looks at the raw text.
    """
    if not isinstance(text, str):
        return ""
    match = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text
