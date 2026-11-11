"""Incremental Comprehension Engine — progressive L1-L7 comprehension.

Levels:

    L1  character representation      bytes -> codepoint summary
    L2  token comprehension           token stream + offsets
    L3  lexical / identifier          keywords, literals, operators
    L4  syntax / structure            declarations, control flow, scopes
    L5  semantic / type               annotations, imports, constraints
    L6  instruction / task            intent -> step decomposition
    L7  execution and verification    runs, tests, diagnostics

Design rules (mandatory):

- Character-level (L1) processing is a *research hypothesis*, not a mandated
  default; granularity is configurable (``granularity``) and measurable.
- Tokenization success never implies syntax, semantics or capability.
- Python ``ast`` is used as a real structural oracle for Python documents;
  everything beyond that stays UNVERIFIED unless a Validation Engine
  verdict is attached.
- Incomplete/uncertain/conflicting observations are preserved, never
  silently discarded.
- Verified knowledge is never discarded by incremental updates; only
  explicit invalidation (Knowledge Formation Engine) removes it.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Iterable

from taksh.dlcae.types import Observation, Provenance, VerificationState, fingerprint, utcnow_iso

PYTHON_KEYWORDS = frozenset({
    "def", "class", "return", "if", "elif", "else", "for", "while", "try",
    "except", "finally", "with", "import", "from", "and", "or", "not", "in",
    "is", "None", "True", "False", "lambda", "pass", "break", "continue",
    "raise", "yield", "async", "await", "match", "case", "global", "nonlocal",
    "assert", "del", "as",
})
C_KEYWORDS = frozenset({
    "int", "char", "float", "double", "void", "struct", "enum", "if", "else",
    "for", "while", "return", "sizeof", "const", "static", "typedef", "switch",
    "case", "default", "break", "continue", "do", "long", "short", "unsigned",
})
# Synthetic seed for tests ONLY — NOT a validated Karkain keyword list.
KARK_SYNTHETIC_KEYWORDS = frozenset({"fn", "func", "let", "const", "if", "else", "match"})
_LANGUAGE_KEYWORDS = {
    "python": PYTHON_KEYWORDS,
    "c": C_KEYWORDS,
    "karkain": KARK_SYNTHETIC_KEYWORDS,
}

_TOKEN_RE = re.compile(
    r"""
      (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|f"(?:[^"\\]|\\.)*"|f'(?:[^'\\]|\\.)*')
    | (?P<number>\b\d+(?:\.\d+)?\b)
    | (?P<comment>\#[^\n]*|//[^\n]*|/\*.*?\*/)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<op>[+\-*/%=!<>&|^~]{1,3}|[{}()\[\];:,.@])
    | (?P<ws>\s+)
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass
class ComprehensionBundle:
    """Leveled, JSON-serializable comprehension result for one document."""

    document_id: str
    language_id: str
    source_sha256: str
    granularity: str
    created: str = field(default_factory=utcnow_iso)
    levels: dict[int, list[Observation]] = field(default_factory=dict)
    level_status: dict[int, str] = field(default_factory=dict)
    verified: list[str] = field(default_factory=list)      # observation_ids
    verification_status: str = "UNVERIFIED"                # UNVERIFIED | PARTIAL | VERIFIED
    task_decomposition: list[dict[str, Any]] = field(default_factory=list)
    note: str = "comprehension output is not capability evidence"

    def observations(self, level: int | None = None) -> list[Observation]:
        if level is None:
            return [o for lv in sorted(self.levels) for o in self.levels[lv]]
        return list(self.levels.get(level, ()))

    def fingerprint(self) -> str:
        return fingerprint({"document_id": self.document_id,
                            "language_id": self.language_id,
                            "source_sha256": self.source_sha256,
                            "levels": {str(k): [o.to_dict() for o in v]
                                       for k, v in sorted(self.levels.items())}})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.comprehension_bundle/1",
            "document_id": self.document_id,
            "language_id": self.language_id,
            "source_sha256": self.source_sha256,
            "granularity": self.granularity,
            "created": self.created,
            "levels": {str(k): [o.to_dict() for o in v] for k, v in sorted(self.levels.items())},
            "level_status": {str(k): v for k, v in sorted(self.level_status.items())},
            "verified": list(self.verified),
            "verification_status": self.verification_status,
            "task_decomposition": list(self.task_decomposition),
            "note": self.note,
        }


def _tokenize(text: str) -> list[tuple[str, int, int, str]]:
    """Generic, lossless token scan: (kind, start, end, text)."""
    tokens: list[tuple[str, int, int, str]] = []
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        kind = match.lastgroup or "other"
        start, end = match.start(), match.end()
        if kind == "ws":
            pos = end
            continue
        tokens.append((kind, start, end, text[start:end]))
    return tokens


def _python_ast_summary(text: str) -> dict[str, Any]:
    """Parse with the real Python grammar; report or honestly fail."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {"parsed": False, "error": f"{exc.msg} (line {exc.lineno})",
                "error_line": exc.lineno}
    dump = {
        "parsed": True,
        "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        "classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        "imports": sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)),
        "annotated_assignments": sum(isinstance(n, ast.AnnAssign) for n in ast.walk(tree)),
        "returns": sum(isinstance(n, ast.Return) for n in ast.walk(tree)),
        "branches": sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.Match))
                        for n in ast.walk(tree)),
    }
    return dump


def _balanced(text: str) -> tuple[bool, dict[str, int]]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairs.items()}
    stack: list[str] = []
    in_str: str | None = None
    depth = {k: 0 for k in ("(", "[", "{")}
    for ch in text:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch in pairs:
            stack.append(ch)
        elif ch in closing:
            if not stack or stack.pop() != closing[ch]:
                return False, depth
        elif ch in depth:
            depth[ch] += 1
    return (not stack), depth


class IncrementalComprehensionEngine:
    """Progressive, level-gated comprehension of source documents."""

    LEVELS = {
        1: "character_representation",
        2: "token_comprehension",
        3: "lexical_identifier",
        4: "syntax_structure",
        5: "semantic_type",
        6: "instruction_task",
        7: "execution_verification",
    }

    def __init__(self, *, granularity: str = "token",
                 provenance: Provenance | None = None) -> None:
        if granularity not in ("char", "token", "line"):
            raise ValueError(f"unsupported granularity: {granularity!r}")
        self.granularity = granularity
        self._default_provenance = provenance

    # ------------------------------------------------------------------ #
    def comprehend(self, document_id: str, text: str, language_id: str, *,
                   max_level: int = 7) -> ComprehensionBundle:
        if not (1 <= max_level <= 7):
            raise ValueError("max_level must be in [1, 7]")
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prov = self._default_provenance or Provenance(
            record_id=f"src-{source_hash[:16]}", source=document_id,
            source_kind="repository", license_spdx="owner-internal")
        bundle = ComprehensionBundle(document_id=document_id,
                                     language_id=language_id,
                                     source_sha256=source_hash,
                                     granularity=self.granularity)
        for level in range(1, max_level + 1):
            obs = getattr(self, f"_level_{level}")(text, language_id, prov)
            bundle.levels[level] = obs
            bundle.level_status[level] = (
                "FAILED" if any(o.kind == "parse_failure" for o in obs)
                else ("UNCERTAIN" if any(o.uncertain for o in obs) else "COMPLETE")
            )
            if bundle.level_status[level] == "FAILED" and level >= 4:
                # deeper syntactic levels cannot proceed past a syntax failure
                for remaining in range(level + 1, max_level + 1):
                    bundle.level_status[remaining] = "BLOCKED"
                bundle.note = ("comprehension halted at syntax validation failure; "
                               "deeper levels BLOCKED — never guessed")
                break
        return bundle

    def resume_from(self, bundle: ComprehensionBundle, text: str,
                    language_id: str, *, levels: Iterable[int]) -> ComprehensionBundle:
        """Re-run specific levels without discarding verified observations."""
        existing = {o.observation_id: o for o in bundle.observations()}
        prov = self._default_provenance or Provenance(
            record_id=f"src-{bundle.source_sha256[:16]}", source=bundle.document_id,
            source_kind="repository", license_spdx="owner-internal")
        for level in sorted(levels):
            kept = [o for o in bundle.levels.get(level, ())
                    if o.state is VerificationState.VERIFIED]
            new_obs = getattr(self, f"_level_{level}")(text, language_id, prov)
            bundle.levels[level] = kept + new_obs
            combined = bundle.levels[level]
            bundle.level_status[level] = (
                "FAILED" if any(o.kind == "parse_failure" for o in combined)
                else ("UNCERTAIN" if any(o.uncertain for o in combined) else "COMPLETE"))
        bundle.verified = [oid for oid, o in existing.items()
                           if o.state is VerificationState.VERIFIED]
        # preserve prior verified ids that were re-verified in this pass
        for level_observations in bundle.levels.values():
            for o in level_observations:
                if o.state is VerificationState.VERIFIED and o.observation_id not in bundle.verified:
                    bundle.verified.append(o.observation_id)
        return bundle

    # ------------------------------------------------------------------ #
    def _level_1(self, text, language_id, prov):
        data = text.encode("utf-8")
        codepoints = len(text)
        printable = sum(1 for ch in text if ch.isprintable())
        ascii_bytes = sum(1 for b in data if b < 128)
        return [Observation(
            level=1, kind="character_representation",
            detail={"bytes": len(data), "codepoints": codepoints,
                    "utf8_ascii_fraction": round(ascii_bytes / max(len(data), 1), 4),
                    "printable_fraction": round(printable / max(codepoints, 1), 4),
                    "granularity": self.granularity,
                    "l1_hypothesis": "char-level processing is a measured research hypothesis"},
            provenance=prov, confidence=1.0)]

    def _level_2(self, text, language_id, prov):
        tokens = _tokenize(text)
        by_kind: dict[str, int] = {}
        for kind, _, _, _ in tokens:
            by_kind[kind] = by_kind.get(kind, 0) + 1
        sample = tokens[:8]
        return [Observation(
            level=2, kind="token_comprehension",
            detail={"token_count": len(tokens), "by_kind": by_kind,
                    "sample_offsets": [[s, e] for _, s, e, _ in sample]},
            provenance=prov, confidence=1.0)]

    def _level_3(self, text, language_id, prov):
        tokens = _tokenize(text)
        keywords = _LANGUAGE_KEYWORDS.get(language_id.lower())
        idents = [t for t in tokens if t[0] == "ident"]
        keyword_hits = sorted(t[3] for t in idents if keywords and t[3] in keywords)
        obs: list[Observation] = [Observation(
            level=3, kind="lexical_identifier",
            detail={"identifiers": len(idents),
                    "distinct_identifiers": len({t[3] for t in idents}),
                    "keywords_configured": bool(keywords),
                    "keyword_hits": keyword_hits,
                    "keyword_source": ("builtin-seed:%s" % language_id.lower())
                    if keywords else "no keyword seed available for language"},
            provenance=prov,
            confidence=1.0 if keywords else 0.5,
            uncertain=not keywords)]
        if not keywords:
            obs[0].state = VerificationState.INFERRED
        return obs

    def _level_4(self, text, language_id, prov):
        obs: list[Observation] = []
        ok, depth = _balanced(text)
        if language_id.lower() == "python":
            summary = _python_ast_summary(text)
            obs.append(Observation(
                level=4, kind="syntax_structure",
                detail={"brackets_balanced": ok, **summary},
                provenance=prov,
                confidence=1.0 if summary.get("parsed") else 0.0,
                uncertain=not summary.get("parsed")))
            if not summary.get("parsed"):
                obs.append(Observation(
                    level=4, kind="parse_failure",
                    detail={"error": summary.get("error"),
                            "error_line": summary.get("error_line")},
                    provenance=prov, confidence=0.0))
        else:
            obs.append(Observation(
                level=4, kind="structural_heuristic",
                detail={"brackets_balanced": ok, "depth": depth,
                        "note": "no grammar/parser qualified for this language on this host; "
                                "structure inferred heuristically — UNCERTAIN, never VERIFIED"},
                provenance=prov, confidence=0.4, uncertain=True,
                state=VerificationState.INFERRED))
        return obs

    def _level_5(self, text, language_id, prov):
        if language_id.lower() == "python":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return [Observation(
                    level=5, kind="semantic_type",
                    detail={"status": "UNVERIFIED", "reason": "syntax failure upstream"},
                    provenance=prov, confidence=0.0, uncertain=True)]
            annotations = sum(isinstance(n, ast.AnnAssign) for n in ast.walk(tree))
            imports = [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            return [Observation(
                level=5, kind="semantic_type",
                detail={"annotated_assignments": annotations,
                        "import_count": len(imports),
                        "note": "annotation presence measured; no type checker installed; "
                                "semantic correctness UNVERIFIED"},
                provenance=prov, confidence=0.6, uncertain=True)]
        return [Observation(
            level=5, kind="semantic_type",
            detail={"status": "UNVERIFIED",
                    "reason": f"no semantic oracle qualified for {language_id!r} on this host"},
            provenance=prov, confidence=0.0, uncertain=True,
            state=VerificationState.INFERRED)]

    def _level_6(self, text, language_id, prov):
        steps: list[dict[str, Any]] = []
        lowered = text.lower()
        if "def " in lowered or "fn " in lowered or "func " in lowered:
            steps.append({"step": "implement_function", "basis": "function declaration detected"})
        if re.search(r"\b(if|else|match|case)\b", lowered):
            steps.append({"step": "handle_branching", "basis": "conditional construct detected"})
        if re.search(r"\b(for|while)\b", lowered):
            steps.append({"step": "iterate_collection", "basis": "loop construct detected"})
        if not steps:
            steps.append({"step": "unknown_intent", "basis": "no construct signature matched; "
                          "INFERRED only with low confidence"})
        obs: list[Observation] = [Observation(
            level=6, kind="task_intent",
            detail={"decomposition": steps,
                    "single_responsibility_hypothesis": len(steps) == 1,
                    "note": "intent decomposition is INFERRED; never verified without tests"},
            provenance=prov, confidence=0.35, uncertain=True,
            state=VerificationState.INFERRED)]
        return obs

    def _level_7(self, text, language_id, prov):
        return [Observation(
            level=7, kind="execution_verification",
            detail={"status": "UNVERIFIED",
                    "reason": "no execution oracle engaged in this comprehension pass; "
                              "see Validation Engine for toolchain verdicts"},
            provenance=prov, confidence=0.0, uncertain=True)]
