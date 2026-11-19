"""Python normalisation and quality filtering (v1.2.0).

Normalisation is deliberately minimal, because a Python corpus is a *code*
corpus and most "cleanups" are semantic changes:

``line_endings``
    CRLF and lone CR become LF. Lossless: Python's tokenizer treats all three
    as line boundaries, and a corpus that mixes them is not reproducible
    across platforms.
``bom``
    A leading UTF-8 byte-order mark is removed. Lossless for Python source:
    the BOM is not part of the program text.
``everything else``
    Untouched. Comments, docstrings, blank lines, trailing whitespace inside
    string literals, identifier names and indentation are preserved exactly.
    NFKC normalisation is *not* applied: it rewrites identifiers and string
    contents, which would change program semantics.

Quality filtering is a first-failure rule set with stable reason codes, so a
rejected file always reports the same reason for the same input. Secrets are
detected and the file is excluded; the matching content is never copied into a
log, a manifest or a report.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import zlib
from dataclasses import dataclass

from training.tokenizer_contract import CONTROL_TOKENS

#: The v1.0.0 special tokens. The v1.1.0 control range is imported from its
#: single source of truth so the two never drift apart.
BASE_SPECIAL_TOKENS = ("<bos>", "<eos>", "<unk>", "<pad>")

#: Reserved spellings that must not appear in a training sample. A literal
#: such as ``"<eos>"`` in ordinary source text is encoded as the *reserved id*,
#: and the reserved ids of the v1.1.0 control range are role markers. A file
#: that contains one of these spellings would therefore inject a control
#: marker into the token stream, and a message containing ``"<|user|>"`` would
#: forge a role boundary under the ``qm.chat_instruction/2`` contract.
RESERVED_LITERALS = tuple(BASE_SPECIAL_TOKENS) + tuple(CONTROL_TOKENS)

#: Every reason code this module can produce. Stable identifiers: they are
#: written into manifests and compared across runs.
#: Patterns that look like instructions embedded in source data. A training
#: corpus that contains them can teach the model to obey text found in files
#: rather than in the prompt, so such files are excluded rather than cleaned.
EMBEDDED_INSTRUCTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:the\s+)?previous\s+instructions"),
    re.compile(r"(?i)disregard\s+(?:the\s+)?(?:above|previous)"),
    re.compile(r"(?i)you\s+are\s+now\s+(?:a|an|the)\b"),
    re.compile(r"(?im)^\s*###\s*instruction\s*:"),
    re.compile(r"(?im)^\s*system\s*:\s*\S"),
)

EXCLUSION_REASONS = (
    "empty_file",
    "too_small",
    "too_large",
    "unsafe_metadata",
    "generated_code",
    "vendored_dependency",
    "minified_artifact",
    "excessive_repetition",
    "low_information",
    "embedded_secret",
    "embedded_instruction",
    "reserved_token_literal",
    "syntax_error",
    "too_complex",
    "trivial_sample",
    "too_few_lines",
)

#: Credential shapes. Extended beyond ``data.ingest_internal.SECRET_PATTERNS``
#: (which is coupled to the historical internal-ingest path) with the common
#: token formats found in public repositories. Only the *kind* is ever
#: reported, never the matched text.
SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|client[_-]?secret"
            r"|password|passwd|pwd)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)

_BIDI_CONTROLS = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the quality rules for one file."""

    ok: bool
    reason: str | None = None
    detail: str = ""


def normalize_source(text: str, config: dict | None = None) -> tuple[str, tuple]:
    """Normalise source text. Returns ``(text, applied_rule_names)``."""
    config = config or {}
    applied = []
    if config.get("strip_bom", True) and text.startswith("\ufeff"):
        text = text[1:]
        applied.append("bom_stripped")
    newline = config.get("newline", "\n")
    if newline == "\n":
        if "\r\n" in text:
            text = text.replace("\r\n", "\n")
            applied.append("crlf_to_lf")
        if "\r" in text:
            text = text.replace("\r", "\n")
            applied.append("cr_to_lf")
    return text, tuple(applied)


def detect_secrets(text: str) -> list:
    """Return the kinds of credential patterns present. Never the matches."""
    found = []
    for kind, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            found.append(kind)
    return found


def detect_reserved_literals(text: str, literals=RESERVED_LITERALS) -> list:
    """Reserved token spellings present in the text.

    The reserved spelling is reported, never the surrounding content: these
    strings are known constants, so no sample data is disclosed.
    """
    return [literal for literal in literals if literal in text]


def _token_stats(text: str) -> tuple[int, int]:
    """(total significant tokens, distinct token strings)."""
    total = 0
    distinct = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (
                tokenize.NAME,
                tokenize.NUMBER,
                tokenize.STRING,
                tokenize.OP,
            ):
                total += 1
                distinct.add(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return 0, 0
    return total, len(distinct)


def _line_stats(text: str) -> tuple[int, int, int]:
    lines = text.splitlines()
    if not lines:
        return 0, 0, 0
    longest = max(len(line) for line in lines)
    average = sum(len(line) for line in lines) / len(lines)
    return len(lines), longest, average


def assess_python(
    text: str,
    rel_path: str,
    quality: dict | None = None,
    *,
    generated_markers=(),
    excluded_path_patterns=(),
) -> QualityResult:
    """Apply the quality rules to one normalised Python file.

    The first failing rule wins, so the reported reason is deterministic for a
    given input and configuration.
    """
    quality = quality or {}
    min_bytes = int(quality.get("min_bytes", 200))
    max_bytes = int(quality.get("max_bytes", 200_000))
    min_lines = int(quality.get("min_lines", 5))
    min_unique_ratio = float(quality.get("min_unique_token_ratio", 0.10))
    minified_line = int(quality.get("minified_line_length", 400))
    minified_average = float(quality.get("minified_average_line_length", 200))
    repetition_ratio = float(quality.get("max_compression_ratio", 0.08))

    if not text.strip():
        return QualityResult(False, "empty_file", rel_path)
    size = len(text.encode("utf-8"))
    if size < min_bytes:
        return QualityResult(False, "too_small", f"{size} bytes")
    if size > max_bytes:
        return QualityResult(False, "too_large", f"{size} bytes")
    if _CONTROL_CHARS.search(text) or _BIDI_CONTROLS.search(text):
        return QualityResult(False, "unsafe_metadata", "control characters present")

    head = text[:400]
    for marker in generated_markers or quality.get("generated_markers", ()):
        if re.search(marker, head, re.IGNORECASE):
            return QualityResult(False, "generated_code", marker)
    for pattern in excluded_path_patterns or quality.get("excluded_path_patterns", ()):
        if re.search(pattern, rel_path, re.IGNORECASE):
            return QualityResult(False, "vendored_dependency", pattern)

    line_count, longest, average = _line_stats(text)
    if longest > minified_line and average > minified_average:
        return QualityResult(
            False, "minified_artifact", f"longest line {longest}, average {average:.0f}"
        )
    if size > 4096 and len(zlib.compress(text.encode("utf-8"), 9)) / size < repetition_ratio:
        return QualityResult(False, "excessive_repetition", "highly compressible content")

    secrets = detect_secrets(text)
    if secrets:
        return QualityResult(False, "embedded_secret", ",".join(sorted(secrets)))

    for pattern in EMBEDDED_INSTRUCTION_PATTERNS:
        if pattern.search(text):
            return QualityResult(False, "embedded_instruction", pattern.pattern[:40])

    reserved = detect_reserved_literals(
        text, quality.get("reserved_literals") or RESERVED_LITERALS
    )
    if reserved:
        # A reserved spelling in the text becomes a reserved id in the token
        # stream. For the control range that means a forged role boundary, so
        # the file is excluded rather than rewritten.
        return QualityResult(False, "reserved_token_literal", ",".join(reserved))

    total, distinct = _token_stats(text)
    if total and distinct / total < min_unique_ratio:
        return QualityResult(
            False, "low_information", f"{distinct}/{total} distinct tokens"
        )

    try:
        tree = ast.parse(text, filename=rel_path)
    except SyntaxError as exc:
        return QualityResult(False, "syntax_error", f"line {exc.lineno}: {exc.msg}")
    except (ValueError, MemoryError) as exc:
        return QualityResult(False, "syntax_error", type(exc).__name__)
    except RecursionError:
        return QualityResult(False, "too_complex", "recursion limit")

    body = [node for node in tree.body if not isinstance(node, ast.Expr)]
    if not body:
        return QualityResult(False, "trivial_sample", "no executable statements")
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return QualityResult(False, "trivial_sample", "single pass statement")
    if line_count < min_lines:
        return QualityResult(False, "too_few_lines", f"{line_count} lines")
    return QualityResult(True)
