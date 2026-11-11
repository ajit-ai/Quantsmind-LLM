"""TAKSH L0.7.2 — Phase A: source-aware re-slicing.

Transforms the bundled repository-level records of the internal l04 corpus
into complete, traceable code samples <= the configured context limit (2,048).

Hierarchy:
  1. standalone <|doc|> lines (verified present: records end with them; they delimit files)
  2. single records are kept whole (m3-train slice is already one file per record)
  3. oversized files: ast top-level units packed greedily into <=2048 fragments
     (complete units only; order preserved; "module header" lands in fragment 0; NO duplication)
  4. monolithic units >2048 -> quarantine (never silently truncated)

Parent-child lineage is preserved end to end. No file boundary is invented:
separators are facts of the record content; structural boundaries come from
CPython's own parser.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import l071_corpus_record as cr  # noqa: E402

CONTEXT_LIMIT = 2048
SAFETY_TOKENS = 64          # conservative packing margin
BYTES_PER_TOKEN_EST = 3.5
TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"

_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(TOKENIZER))
    return _TOKENIZER


def tok_count(text: str) -> int:
    return len(tokenizer().encode(text).ids)


def standalone_separators(text: str) -> list[int]:
    return [i for i, line in enumerate(text.split("\n")) if line.strip() == "<|doc|>"]


def split_segments(text: str) -> list[tuple[int, int, str]]:
    """Split at standalone separator lines. Returns (start_line, end_line, text) 0-based spans."""
    lines = text.split("\n")
    seps = standalone_separators(text)
    segments = []
    prev = 0
    for sep in seps:
        seg = lines[prev:sep]
        if any(l.strip() for l in seg):
            segments.append((prev, sep, "\n".join(seg)))
        prev = sep + 1
    tail = lines[prev:]
    if any(l.strip() for l in tail):
        segments.append((prev, len(lines), "\n".join(tail)))
    return segments


def top_level_spans(text: str) -> list[tuple[int, int, str]]:
    """Line spans (0-based, end-exclusive) of complete top-level statements."""
    tree = ast.parse(text)
    spans = []
    for node in tree.body:
        start = node.lineno - 1
        end = node.end_lineno
        kind = type(node).__name__
        spans.append((start, end, kind))
    return spans


def _member_span(child) -> tuple[int, int, str]:
    starts = [d.lineno - 1 for d in getattr(child, "decorator_list", [])]
    start = min(starts + [child.lineno - 1])
    return (start, child.end_lineno, type(child).__name__)


def subdivide_class_unit(unit_text: str, limit: int = CONTEXT_LIMIT) -> tuple[list[dict], list[dict]]:
    """Split an oversized ClassDef at MEMBER boundaries.

    Fragment 0 = class statement + docstring + as many members as fit.
    Continuation fragments = class statement lines only (no repeated docstring)
    + complete members. Every fragment parses standalone and stays under the
    limit. The class statement lines are duplicated per fragment (tracked as
    context duplication); the docstring is never duplicated.
    """
    tree = ast.parse(unit_text)
    node = tree.body[0]
    assert isinstance(node, ast.ClassDef)
    lines = unit_text.split("\n")
    class_start = node.lineno - 1

    body = node.body
    first = body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        doc = first
        stmt_end = doc.lineno - 1            # class statement lines end before the docstring
        frag0_header_end = doc.end_lineno    # fragment 0 keeps the docstring
        members = [_member_span(c) for c in body[1:]]
    else:
        stmt_end = first.lineno - 1
        frag0_header_end = stmt_end
        members = [_member_span(c) for c in body]

    stmt_lines = lines[class_start:stmt_end]
    frag0_header = lines[class_start:frag0_header_end]

    fragments: list[dict] = []
    quarantines: list[dict] = []
    pending_members: list[tuple[int, int, str]] = []
    pending_start: int | None = None

    def token_len(text: str) -> int:
        return tok_count(text)

    def seal(members_sel, is_first):
        if not members_sel and not is_first:
            return None
        header = frag0_header if is_first else stmt_lines
        end = members_sel[-1][1] if members_sel else (frag0_header_end if is_first else stmt_end)
        if members_sel:
            start = members_sel[0][0]
            text = "\n".join(header + lines[start:end])
        else:
            text = "\n".join(header)
        return {"text": text,
                "fragment_index": len(fragments),
                "method": "class_member_subdivision"}

    for member in members:
        n = token_len("\n".join(lines[member[0]:member[1]]))
        if n > limit:
            if pending_members:
                fragments.append(seal(pending_members, not fragments))
                pending_members = []
            quarantines.append({"text": "\n".join(lines[member[0]:member[1]]),
                                "kind": member[2], "token_count": n,
                                "reason": "monolithic_member_exceeds_context"})
            continue
        candidate = pending_members + [member]
        cand_text = "\n".join((frag0_header if not fragments else stmt_lines) +
                              lines[candidate[0][0]:candidate[-1][1]])
        if token_len(cand_text) <= limit - SAFETY_TOKENS:
            pending_members = candidate
        else:
            fragments.append(seal(pending_members, not fragments))
            pending_members = [member]
    if pending_members or not fragments:
        if pending_members:
            fragments.append(seal(pending_members, not fragments))
        elif is_empty := (frag0_header and not members):
            fragments.append(seal([], True))

    final_fragments, final_quarantines = [], list(quarantines)
    for i, frag in enumerate(fragments):
        if frag is None:
            continue
        n = token_len(frag["text"])
        if n > limit:
            final_quarantines.append({"text": frag["text"], "kind": "class_fragment",
                                      "token_count": n, "reason": "fragment_exceeds_context_after_seal"})
            continue
        try:
            ast.parse(frag["text"])
        except SyntaxError:
            final_quarantines.append({"text": frag["text"], "kind": "class_fragment",
                                      "token_count": n, "reason": "fragment_not_parseable"})
            continue
        final_fragments.append({**frag, "fragment_index": len(final_fragments), "token_count": n})
    return final_fragments, final_quarantines


def pack_units(text: str, limit: int = CONTEXT_LIMIT) -> tuple[list[dict], list[dict]]:
    """Greedy-pack complete top-level units into fragments. Returns (fragments, quarantines)."""
    lines = text.split("\n")
    spans = top_level_spans(text)
    fragments: list[dict] = []
    quarantined: list[dict] = []

    if not spans:
        return [{"text": text, "fragment_index": 0, "method": "whole_file"}], []

    current_start: int | None = None
    current_units: list[str] = []
    current_lines = 0

    def seal(start: int | None, end_excl: int):
        if start is None:
            return
        frag_text = "\n".join(lines[start:end_excl])
        fragments.append({"text": frag_text, "fragment_index": len(fragments),
                          "method": "structural_subdivision"})

    for i, (start, end, kind) in enumerate(spans):
        unit_text = "\n".join(lines[start:end])
        if tok_count(unit_text) > limit:
            # seal whatever is pending, then subdivide or quarantine this unit
            if current_start is not None:
                seal(current_start, spans[i - 1][1] if i else 0)
                current_start, current_units, current_lines = None, [], 0
            if kind == "ClassDef":
                sub_frags, sub_quar = subdivide_class_unit(unit_text, limit)
                fragments.extend(sub_frags)
                quarantined.extend(sub_quar)
            else:
                quarantined.append({"text": unit_text, "kind": kind,
                                    "token_count": tok_count(unit_text),
                                    "reason": "monolithic_unit_exceeds_context"})
            continue
        if current_start is None:
            current_start = start
            current_units = [kind]
            current_lines = end - start
            continue
        start_idx = spans[i - 1][1]
        current_text = "\n".join(lines[current_start:end])
        if tok_count(current_text) <= limit - SAFETY_TOKENS:
            current_units.append(kind)
            current_lines = end - current_start
        else:
            seal(current_start, spans[i - 1][1])
            current_start = start
            current_units = [kind]
            current_lines = end - start
    seal(current_start, spans[-1][1] if spans else 0)

    # every sealed fragment must independently parse and fit
    final_fragments, final_quarantines = [], list(quarantined)
    for i, frag in enumerate(fragments):
        text_frag = frag["text"]
        n = tok_count(text_frag)
        if n > limit:
            final_quarantines.append({"text": text_frag, "kind": "fragment",
                                      "token_count": n, "reason": "fragment_exceeds_context_after_seal"})
            continue
        try:
            ast.parse(text_frag)
        except SyntaxError:
            final_quarantines.append({"text": text_frag, "kind": "fragment",
                                      "token_count": n, "reason": "fragment_not_parseable"})
            continue
        final_fragments.append({**frag, "fragment_index": len(final_fragments), "token_count": n})
    return final_fragments, final_quarantines


def reslice_record(rec: dict, text: str) -> tuple[list[dict], list[dict]]:
    """Derive child samples from one input record. Returns (children, quarantines)."""
    parent_fp = rec["original_fingerprint"]
    children: list[dict] = []
    quarantines: list[dict] = []

    segments = split_segments(text)
    for seg_index, (start, end, seg_text) in enumerate(segments):
        if standalone_separators(text):
            method = "marker_split"
        else:
            method = "single_record"
        seg_fp = hashlib.sha256(seg_text.encode("utf-8")).hexdigest()
        seg_tokens = tok_count(seg_text)

        if seg_tokens > CONTEXT_LIMIT:
            try:
                fragments, quar = pack_units(seg_text)
            except SyntaxError:
                quar = [{"text": seg_text, "kind": "file",
                         "token_count": seg_tokens,
                         "reason": "file_not_parseable_for_subdivision"}]
                fragments = []
            for frag in fragments:
                children.append(make_child(rec, seg_index, start, end, frag, seg_fp,
                                           f"structural_subdivision_from_{method}"))
            for q in quar:
                q.update({"parent_sample_id": rec["sample_id"],
                          "parent_fingerprint": parent_fp,
                          "file_index": seg_index,
                          "segment_lines": [start, end],
                          "extraction_method": method})
                quarantines.append(q)
        else:
            ok = True
            try:
                ast.parse(seg_text.replace("\ufeff", ""))
            except SyntaxError:
                ok = False
            if not ok:
                quarantines.append({"parent_sample_id": rec["sample_id"],
                                    "parent_fingerprint": parent_fp,
                                    "file_index": seg_index,
                                    "segment_lines": [start, end],
                                    "extraction_method": method,
                                    "token_count": seg_tokens,
                                    "text": seg_text,
                                    "reason": "file_not_parseable_quarantine"})
                continue
            children.append(make_child(rec, seg_index, start, end,
                                       {"text": seg_text, "fragment_index": 0,
                                        "method": method, "token_count": seg_tokens},
                                       seg_fp, method))
    return children, quarantines


def make_child(parent: dict, file_index: int, start: int, end: int,
               frag: dict, segment_fp: str, method: str) -> dict:
    frag_text = frag["text"]
    child_fp = hashlib.sha256(frag_text.encode("utf-8")).hexdigest()
    base_ids = f"{parent['sample_id']}:{file_index}:{frag['fragment_index']}"
    sample_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"taksh:l072:{base_ids}"))
    history = list(parent.get("transformation_history", []))
    history.append({"step": "reslice", "version": "l072-freeze/1",
                    "detail": f"method={method} file_index={file_index} fragment_index={frag['fragment_index']} lines=[{start},{end})"})
    child = cr.build_record(
        source_id=parent["source_id"],
        source_version=parent["source_version"],
        original_ref=f"{parent['original_ref']}#file{file_index}:fragment{frag['fragment_index']}",
        original_index=parent["original_index"],
        text=frag_text,
        language=parent["language"],
        language_family=parent["language_family"],
        license_id=parent["license_id"],
        license_decision_ref=parent["license_decision_ref"],
        provenance_ref=parent["provenance_ref"],
        tokenizer_id=parent["tokenizer_id"],
        tokenizer_version=parent["tokenizer_version"],
        tokenizer_fingerprint=parent["tokenizer_fingerprint"],
        token_count=frag.get("token_count") or tok_count(frag_text),
        snapshot_ref=parent.get("snapshot_ref", "ungated-local"),
        content_format=parent.get("content_format", "utf-8 text"),
        synthetic=False,
    )
    child["sample_id"] = sample_id
    child["parent_sample_id"] = parent["sample_id"]
    child["parent_original_fingerprint"] = parent["original_fingerprint"]
    child["file_index"] = file_index
    child["fragment_index"] = frag["fragment_index"]
    child["extraction_method"] = method
    child["segment_fingerprint"] = segment_fp
    child["segment_line_span"] = [start, end]
    child["transformation_history"] = history
    child["exclusion_reason"] = None
    child["split"] = "unassigned"
    child["_text"] = frag_text   # transient: stripped before records hit disk
    return child
