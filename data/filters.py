"""Content filters shared by the corpus pipelines.

``scrub_pii`` is token-aware. The earlier version applied four regexes to raw
text, so a Python literal such as ``9876543210`` or a constant named
``ABCDE1234F`` was rewritten even though neither is personal data. That changes
program semantics in a corpus that is supposed to be valid source code, so code
input is now scrubbed only inside STRING and COMMENT tokens: the places where
personal data actually appears in a repository. NUMBER and NAME tokens are never
touched.
"""

import re
from typing import NamedTuple

AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")

PII_PATTERNS = (
    ("AADHAAR", AADHAAR, "[AADHAAR]"),
    ("PAN", PAN, "[PAN]"),
    ("EMAIL", EMAIL, "[EMAIL]"),
    ("PHONE", PHONE, "[PHONE]"),
)


class ScrubResult(NamedTuple):
    """Outcome of a scrub, kept alongside the text for auditing."""

    clean: str
    mode: str  # "code" | "prose" | "preserved"
    findings: tuple  # ((kind, count), ...) in PII_PATTERNS order
    preserved_reason: str | None = None


def _scrub_plain(text: str):
    """Whole-text regex scrub. Only for prose, never for source code."""
    findings = []
    out = text
    for kind, pattern, replacement in PII_PATTERNS:
        out, n = pattern.subn(replacement, out)
        if n:
            findings.append((kind, n))
    return out, findings


def _scrubbable_spans(text: str):
    """Line/column spans of STRING, COMMENT and f-string text tokens.

    Returns None when the text cannot be tokenised at all.
    """
    import io
    import tokenize

    kinds = {tokenize.STRING, tokenize.COMMENT}
    # Python 3.12+ splits f-strings into FSTRING_START/MIDDLE/END. The MIDDLE
    # pieces hold literal text, so PII there must still be scrubbed.
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        kinds.add(fstring_middle)

    spans = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in kinds:
                spans.append((tok.start, tok.end, tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None
    return spans


def _is_python_code(text: str) -> bool:
    """True when ``text`` is parseable Python.

    ``tokenize`` is far too lenient to make this call: ordinary prose such as
    "Call 9876543210 now." tokenises without error. The real parser is used so
    that prose falls through to the legacy whole-text scrub.
    """
    try:
        compile(text, "<probe>", "exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def scrub_pii_report(text: str, *, mode: str = "auto") -> ScrubResult:
    """Scrub personal data without altering code semantics.

    ``mode``:
        ``"code"``  -- scrub only inside string/comment tokens; if the text is
                       not tokenisable as Python it is **preserved unchanged**
                       and flagged, because a safe distinction cannot be made.
        ``"prose"`` -- legacy whole-text regex scrub, for non-code input.
        ``"auto"``  -- ``"code"`` when the text parses as Python, else
                       ``"prose"``.

    Limits, stated plainly: the patterns are India-specific (Aadhaar, PAN), a
    phone test is "10 digits starting 6-9", and nothing here detects names,
    addresses, IPs, JWTs or connection strings. Absent pattern coverage, personal
    data reaches the corpus.
    """
    if mode not in ("auto", "code", "prose"):
        raise ValueError(f"unknown scrub mode: {mode}")

    if mode in ("auto", "code"):
        if mode == "code" or _is_python_code(text):
            spans = _scrubbable_spans(text)
            if spans is None:
                return ScrubResult(
                    text, "preserved", (), "not_tokenizable_as_python"
                )
            # Rewrite only the string/comment tokens, right to left so earlier
            # offsets stay valid.
            lines = text.splitlines(keepends=True)
            starts = [0]
            for line in lines:
                starts.append(starts[-1] + len(line))

            def offset(pos):
                row, col = pos
                return starts[row - 1] + col

            counts = {k: 0 for k, _, _ in PII_PATTERNS}
            out = text
            for (srow, scol), (erow, ecol), literal in reversed(spans):
                start, end = offset((srow, scol)), offset((erow, ecol))
                replaced = literal
                for kind, pattern, replacement in PII_PATTERNS:
                    replaced, n = pattern.subn(replacement, replaced)
                    counts[kind] += n
                if replaced != literal:
                    out = out[:start] + replaced + out[end:]
            findings = tuple((k, counts[k]) for k, _, _ in PII_PATTERNS if counts[k])
            return ScrubResult(out, "code", findings, None)
        if mode == "code":
            return ScrubResult(text, "preserved", (), "not_tokenizable_as_python")

    clean, findings = _scrub_plain(text)
    return ScrubResult(clean, "prose", tuple(findings), None)


def scrub_pii(text: str, *, mode: str = "auto") -> str:
    """Backwards-compatible wrapper returning only the scrubbed text."""
    return scrub_pii_report(text, mode=mode).clean


def quality_check(
    text: str,
    min_chars: int = 200,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.6,
) -> tuple[bool, str]:
    if not (min_chars <= len(text) <= max_chars):
        return False, "length"
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < min_alpha_ratio:
        return False, "alpha_ratio"
    return True, ""


class ExactDedup:
    def __init__(self):
        self._seen: set[str] = set()

    def add(self, text: str) -> bool:
        key = hashlib_key(text)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def __len__(self) -> int:
        return len(self._seen)


def hashlib_key(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def word_shingles(text: str, k: int = 8) -> set:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _shingle_hash(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big")


class MinHashDedup:
    def __init__(self, num_perms: int = 64, bands: int = 16, threshold: float = 0.8):
        assert num_perms % bands == 0
        import numpy as np

        self.np = np
        self.num_perms = num_perms
        self.rows = num_perms // bands
        self.threshold = threshold
        self.signatures_index: dict = {}
        rng = np.random.default_rng(42)
        self._a = rng.integers(1, 2**61, size=num_perms, dtype=np.uint64)
        self._b = rng.integers(0, 2**61, size=num_perms, dtype=np.uint64)
        self._p = np.uint64((1 << 61) - 1)

    def signature(self, text: str):
        sh = self.np.fromiter(
            (_shingle_hash(s) for s in word_shingles(text)), dtype=self.np.uint64
        )
        if sh.size == 0:
            return self.np.array([], dtype=self.np.uint64)
        return ((self._a[:, None] * sh[None, :] + self._b[:, None]) % self._p).min(axis=1)

    def add(self, text: str) -> bool:
        """True if ``text`` is new, False if a near-duplicate is already known.

        Every band key is registered, not just the first unseen one. The previous
        version only indexed a signature under the *first* band whose key was new,
        so a later near-duplicate that collided on any other band could never be
        found -- recall collapsed. With all bands indexed, a candidate is compared
        against every signature that shares a band with it, which is the standard
        LSH banding construction.
        """
        sig = self.signature(text)
        if sig.size == 0:
            return True
        duplicate = False
        for band in range(self.num_perms // self.rows):
            key = (band, sig[band * self.rows : (band + 1) * self.rows].tobytes())
            existing = self.signatures_index.get(key)
            if existing is None:
                self.signatures_index[key] = sig
            elif jaccard(sig, existing) >= self.threshold:
                duplicate = True
                break
        return not duplicate

    def __len__(self) -> int:
        return len(self.signatures_index)


def jaccard(s1, s2) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return float((s1 == s2).mean())


class Decontaminator:
    def __init__(self, forbidden_texts: list, k: int = 8):
        self.k = k
        self.forbidden: set = set()
        for t in forbidden_texts:
            self.forbidden |= word_shingles(t, k)

    def is_contaminated(self, text: str) -> bool:
        return bool(word_shingles(text, self.k) & self.forbidden)
