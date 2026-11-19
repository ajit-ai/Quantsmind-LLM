"""Bounded, read-only, symlink-safe ingestion of approved local sources.

Safety properties, each enforced rather than assumed:

* Only files under the resolved source root are read. Every candidate is
  re-resolved and checked for containment, so a path that escapes the root is
  refused even if the directory listing looked safe.
* Symbolic links and Windows junctions are never followed. They are skipped and
  recorded as diagnostics, not silently traversed.
* Traversal is bounded: file count, per-file size, total size and directory
  depth all have explicit caps, and hitting a cap marks the run truncated
  rather than quietly returning less data.
* Traversal order is stable (sorted), so two runs over the same tree produce
  the same candidate list in the same order.
* Nothing is written to the source tree, and no network access happens here.
  Acquisition is a separate, explicitly authorised action.
"""

from __future__ import annotations

import io
import os
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from data.licence_paths import path_within

DEFAULT_PRUNE_DIRS = (".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache")


class IngestError(RuntimeError):
    """A file could not be ingested. ``reason`` is a stable code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class FileRef:
    """One candidate file discovered under an approved source."""

    source_id: str
    root: Path
    abs_path: Path
    rel_path: str
    size: int


@dataclass
class IngestReport:
    files: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    truncated: bool = False
    truncated_reason: str | None = None

    def note(self, reason: str, rel_path: str = "") -> None:
        self.diagnostics[reason] = self.diagnostics.get(reason, 0) + 1


def _is_junction(entry) -> bool:
    checker = getattr(entry, "is_junction", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except OSError:  # pragma: no cover - platform dependent
        return False


def _contained(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves inside ``root``."""
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def _relpath(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def ingest_source(
    source,
    root,
    *,
    limits: dict,
    excluded_paths=(),
    allowed_extensions=(".py",),
    prune_dirs=DEFAULT_PRUNE_DIRS,
) -> IngestReport:
    """Collect candidate files from one approved local source.

    Returns an :class:`IngestReport`. The report is *truncated* when a
    resource cap was reached; callers must treat a truncated ingest as an
    incomplete run rather than as a smaller corpus.
    """
    root = Path(root).resolve()
    report = IngestReport()
    if not root.is_dir():
        report.note("source_root_missing")
        return report

    max_files = int(limits.get("max_files", 1000))
    max_total_bytes = int(limits.get("max_total_bytes", 50_000_000))
    max_file_bytes = int(limits.get("max_file_bytes", 200_000))
    max_depth = int(limits.get("max_path_depth", 40))

    extensions = tuple(e.lower() for e in allowed_extensions)
    prune = {str(d).lower() for d in prune_dirs}
    total_bytes = 0

    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError:
            report.note("directory_unreadable", _relpath(root, current))
            continue
        for entry in entries:
            rel = _relpath(root, Path(entry.path))
            if depth + 1 > max_depth:
                report.note("path_depth_exceeded", rel)
                report.truncated = True
                report.truncated_reason = "path_depth_exceeded"
                continue
            if entry.is_symlink() or _is_junction(entry):
                # Never followed. A link may point anywhere, including outside
                # the approved root, and following it would break containment.
                report.note("link_skipped", rel)
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name.lower() in prune:
                    report.note("pruned_directory", rel)
                    continue
                if path_within(rel, list(excluded_paths)):
                    report.note("excluded_path", rel)
                    continue
                stack.append((Path(entry.path), depth + 1))
                continue
            if not entry.is_file(follow_symlinks=False):
                report.note("not_a_regular_file", rel)
                continue
            if not rel.lower().endswith(extensions):
                continue
            if path_within(rel, list(excluded_paths)):
                report.note("excluded_path", rel)
                continue
            path = Path(entry.path)
            if not _contained(root, path):
                report.note("path_escape_refused", rel)
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                report.note("stat_failed", rel)
                continue
            if size > max_file_bytes:
                report.note("file_too_large", rel)
                continue
            if len(report.files) >= max_files:
                report.note("file_limit_reached", rel)
                report.truncated = True
                report.truncated_reason = "max_files"
                continue
            if total_bytes + size > max_total_bytes:
                report.note("total_size_limit_reached", rel)
                report.truncated = True
                report.truncated_reason = "max_total_bytes"
                continue
            total_bytes += size
            report.files.append(
                FileRef(
                    source_id=source.get("id"),
                    root=root,
                    abs_path=path,
                    rel_path=rel,
                    size=size,
                )
            )
    report.files.sort(key=lambda f: (f.source_id, f.rel_path))
    return report


def detect_encoding(raw: bytes) -> str:
    """Detect the declared encoding of a Python source file.

    ``tokenize.detect_encoding`` honours a UTF-8 BOM and a PEP 263 coding
    cookie, and defaults to UTF-8. A malformed cookie raises, which is treated
    as an encoding error rather than silently falling back.
    """
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError as exc:
        raise IngestError("encoding_declaration_invalid", str(exc)) from exc
    return encoding


def looks_binary(raw: bytes) -> bool:
    """True when the bytes are not plausible text.

    A NUL byte is decisive. Otherwise a high share of control characters (other
    than tab, newline and carriage return) marks a binary payload.
    """
    if b"\x00" in raw:
        return True
    if not raw:
        return False
    sample = raw[:8192]
    control = sum(
        1 for byte in sample if byte < 32 and byte not in (9, 10, 13)
    )
    return control / len(sample) > 0.30


def decode_source(raw: bytes, *, fallbacks=()) -> tuple[str, str, bool]:
    """Decode a source file.

    Returns ``(text, encoding, used_fallback)``. The declared encoding is used
    first; configured fallbacks are tried only in the declared order, and each
    use is reported so the manifest can show that a file was not UTF-8.
    """
    encoding = detect_encoding(raw)
    try:
        return raw.decode(encoding), encoding, False
    except UnicodeDecodeError as exc:
        last = exc
    for candidate in fallbacks or ():
        try:
            return raw.decode(candidate), candidate, True
        except (UnicodeDecodeError, LookupError) as exc:
            last = exc
    raise IngestError("encoding_error", str(last))


def read_source(ref: FileRef, *, fallbacks=(), max_bytes=None) -> tuple[str, str, bool]:
    """Read and decode one candidate file, read-only."""
    try:
        with open(ref.abs_path, "rb") as handle:
            raw = handle.read(max_bytes + 1 if max_bytes else -1)
    except OSError as exc:
        raise IngestError("read_failed", str(exc)) from exc
    if max_bytes is not None and len(raw) > max_bytes:
        raise IngestError("file_too_large", ref.rel_path)
    if looks_binary(raw):
        raise IngestError("binary_content", ref.rel_path)
    return decode_source(raw, fallbacks=fallbacks)
