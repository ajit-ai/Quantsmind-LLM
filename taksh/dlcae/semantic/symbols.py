"""Symbol analysis — language adapters share a neutral record format.

Records declarations, definition/reference relationships, scope boundaries,
shadowing, unresolved identifiers, import/export relationships, module
boundaries, duplicate declarations, provenance and source locations.

Language-specific rules live in adapters (``PythonSymbolAdapter`` implemented
for real via ``ast``); the shared engine only sequences adapters. Unsupported
languages report explicit UNVERIFIED/BLOCKED states — never invented symbols.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import re
from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.semantic.contract import Diagnostic, source_sha256

ADAPTER_VERSION = "symbols/1.0.0"
_BUILTIN_NAMES = frozenset(dir(builtins)) | frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)


@dataclass
class SymbolAnalysis:
    language_id: str
    adapter: str
    adapter_version: str
    source_sha256: str
    file: str
    module: str
    status: str = "UNVERIFIED"          # PASS | FAIL | PARTIAL | BLOCKED | UNVERIFIED
    symbols: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    scopes: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    shadowing: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.symbol_analysis/1",
            "language_id": self.language_id,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "source_sha256": self.source_sha256,
            "file": self.file,
            "module": self.module,
            "status": self.status,
            "symbols": self.symbols,
            "references": self.references,
            "scopes": self.scopes,
            "imports": self.imports,
            "exports": self.exports,
            "duplicates": self.duplicates,
            "shadowing": self.shadowing,
            "unresolved": self.unresolved,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "unsupported": self.unsupported,
        }


def _loc(node: ast.AST) -> dict[str, int]:
    return {"line": getattr(node, "lineno", 1),
            "column": getattr(node, "col_offset", 0) + 1}


def _qualname(stack: list[tuple[str, str]], name: str) -> str:
    return ".".join([scope for scope, _ in stack] + [name])


class PythonSymbolAdapter:
    """Real symbol analysis for Python documents via the ``ast`` oracle."""

    language_ids = ("python",)
    name = "python-ast"
    version = ADAPTER_VERSION

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> SymbolAnalysis:
        result = SymbolAnalysis(language_id="python", adapter=self.name,
                                adapter_version=self.version,
                                source_sha256=source_sha256, file=file, module=module)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            result.status = "FAIL"
            result.diagnostics.append(Diagnostic(
                category="syntax_error",
                message=f"{exc.msg} (line {exc.lineno})",
                severity="ERROR", line=exc.lineno, column=exc.offset,
                source=self.name, tool_version=self.version,
                evidence_classification="TOOL_VERIFIED"))
            return result

        defined: list[dict[str, Any]] = []       # all definition records, in order
        shadow_entries: list[dict[str, Any]] = []

        def enclosing(stack) -> str:
            return ".".join(s for s, _ in stack) or "<module>"

        def add_symbol(name, kind, node, stack, *, exported=None):
            qname = f"{enclosing(stack)}.{name}" if stack else name
            record = {
                "name": name,
                "qualified_name": qname,
                "kind": kind,
                "scope": enclosing(stack),
                "line": getattr(node, "lineno", 1),
                "column": getattr(node, "col_offset", 0) + 1,
                "exported": (exported if exported is not None
                             else (not stack and not name.startswith("_"))),
            }
            result.symbols.append(record)
            # shadowing: the same name already defined in a shorter (outer) scope
            for prior in defined:
                if prior["name"] == name and self._is_outer(prior["scope"], record["scope"]):
                    shadow_entries.append({
                        "name": name,
                        "scope": record["scope"],
                        "shadows": prior["scope"],
                        "line": record["line"],
                    })
            defined.append(record)
            return record

        def note_reference(name, stack, node):
            result.references.append({
                "name": name,
                "scope": enclosing(stack),
                "line": getattr(node, "lineno", 1),
                "column": getattr(node, "col_offset", 0) + 1,
            })

        def visit(node, stack):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_symbol(node.name, "function", node, stack)
                fn_stack = stack + [(node.name, "function")]
                result.scopes.append({"path": enclosing(fn_stack), "kind": "function",
                                      "line": getattr(node, "lineno", 1)})
                params = list(node.args.posonlyargs) + list(node.args.args)
                params += [node.args.vararg] if node.args.vararg else []
                params += node.args.kwonlyargs
                params += [node.args.kwarg] if node.args.kwarg else []
                for param in params:
                    if param is None or param.arg == "_":
                        continue
                    add_symbol(param.arg, "parameter", param, fn_stack)
                for deco in node.decorator_list:
                    visit(deco, stack)   # decorators evaluated in the enclosing scope
                for child in ast.iter_child_nodes(node):
                    if child in node.decorator_list:
                        continue          # already visited in the enclosing scope
                    visit(child, fn_stack)
                return
            if isinstance(node, ast.ClassDef):
                add_symbol(node.name, "class", node, stack)
                cls_stack = stack + [(node.name, "class")]
                result.scopes.append({"path": enclosing(cls_stack), "kind": "class",
                                      "line": getattr(node, "lineno", 1)})
                for base in node.bases:
                    visit(base, cls_stack)
                for child in ast.iter_child_nodes(node):
                    visit(child, cls_stack)
                return
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append({"module": alias.name,
                                           "asname": alias.asname,
                                           "line": getattr(node, "lineno", 1),
                                           "level": 0, "kind": "import"})
                    add_symbol(alias.asname or alias.name.split(".")[0],
                               "import", node, stack)
                return
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_name = "." * (node.level or 0) + (node.module or "")
                    result.imports.append({"module": module_name,
                                           "asname": alias.asname,
                                           "line": getattr(node, "lineno", 1),
                                           "level": node.level or 0,
                                           "name": alias.name,
                                           "kind": "from_import"})
                    if alias.name != "*":
                        add_symbol(alias.asname or alias.name, "import", node, stack)
                return
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                add_symbol(node.target.id, "variable", node.target, stack)   # definition, not reference
                for child in ast.iter_child_nodes(node):
                    visit(child, stack)
                return
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name):
                            add_symbol(child.id, "variable", child, stack)
                for child in ast.iter_child_nodes(node):
                    visit(child, stack)
                return
            if isinstance(node, ast.arg):
                return  # parameters already recorded at function entry
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                note_reference(node.id, stack, node)
                return
            for child in ast.iter_child_nodes(node):
                visit(child, stack)

        visit(tree, [])

        # duplicates: re-definition of the same qualified function/class name
        # (local variable re-binding is idiomatic and is not a duplicate declaration)
        by_qname: dict[str, list[dict[str, Any]]] = {}
        for record in result.symbols:
            if record["kind"] not in ("function", "class"):
                continue
            by_qname.setdefault(record["qualified_name"], []).append(record)
        for qname, records in by_qname.items():
            if len(records) > 1:
                result.duplicates.append({
                    "qualified_name": qname,
                    "occurrences": [{"kind": r["kind"], "line": r["line"]} for r in records]})
                for dup in records[1:]:
                    result.diagnostics.append(Diagnostic(
                        category="warning",
                        message=f"duplicate declaration: {qname}",
                        severity="WARNING", line=dup["line"], column=dup["column"],
                        symbol=qname, source=self.name, tool_version=self.version,
                        evidence_classification="TOOL_VERIFIED"))

        # resolve references against nearest scope, then module, then builtins
        defined_by_name: dict[str, list[dict[str, Any]]] = {}
        for record in result.symbols:
            defined_by_name.setdefault(record["name"], []).append(record)
        unresolved: list[dict[str, Any]] = []
        for reference in result.references:
            match = self._resolve(reference, defined_by_name)
            if match is not None:
                reference["resolution"] = match["qualified_name"]
                reference["confidence"] = 0.9 if match["scope"] == reference["scope"] else 0.75
            elif reference["name"] in _BUILTIN_NAMES:
                reference["resolution"] = f"builtins.{reference['name']}"
                reference["confidence"] = 1.0
            else:
                reference["resolution"] = None
                reference["confidence"] = 0.0
                unresolved.append(reference)
        result.unresolved = [{"name": r["name"], "line": r["line"],
                              "column": r["column"]} for r in unresolved]
        result.shadowing = shadow_entries
        result.exports = sorted({r["name"] for r in result.symbols
                                 if r["scope"] == "<module>" and not r["name"].startswith("_")
                                 and r["kind"] in ("function", "class", "variable", "import")})
        result.status = "PASS" if not unresolved else "PARTIAL"
        if unresolved:
            result.unsupported.append(
                f"{len(unresolved)} identifier(s) unresolved by the static scan; "
                f"use the tool-assisted semantic check (mypy/ruff) for confirmation")
        return result

    @staticmethod
    def _is_outer(def_scope: str, ref_scope: str) -> bool:
        if def_scope == ref_scope:
            return False
        if def_scope == "<module>":
            return True
        return ref_scope.startswith(def_scope + ".")

    @classmethod
    def _resolve(cls, reference: dict[str, Any],
                 defined_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
        candidates = defined_by_name.get(reference["name"])
        if not candidates:
            return None
        scope = reference["scope"]
        best = None
        for record in candidates:
            if record["scope"] == scope:
                if record["line"] <= reference["line"]:
                    if best is None or record["line"] > best["line"]:
                        best = record
        if best is not None:
            return best
        for record in candidates:
            if cls._is_outer(record["scope"], scope):
                if best is None or len(record["scope"]) > len(best["scope"]):
                    best = record
        return best


class GenericSymbolAdapter:
    """Fallback heuristic adapter — everything is INFERRED, never VERIFIED."""

    name = "generic-heuristic"
    version = ADAPTER_VERSION

    def __init__(self, language_id: str) -> None:
        self.language_id = language_id

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> SymbolAnalysis:
        result = SymbolAnalysis(language_id=self.language_id, adapter=self.name,
                                adapter_version=self.version,
                                source_sha256=source_sha256, file=file, module=module)
        idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", source)
        counts: dict[str, int] = {}
        for token in idents:
            counts[token] = counts.get(token, 0) + 1
        for token, count in sorted(counts.items()):
            result.symbols.append({"name": token, "qualified_name": token,
                                   "kind": "identifier", "scope": "<module>",
                                   "line": None, "column": None,
                                   "exported": False, "occurrences": count})
        result.status = "UNVERIFIED"
        result.unsupported = [
            "no grammar/symbol oracle qualified for "
            f"{self.language_id!r} on this host; identifiers observed heuristically"]
        return result


class KarkainSymbolAdapter:
    """Karkain symbol analysis — BLOCKED while KCC cannot start.

    KCC is the authoritative compiler/lexer on this contract. Inventing a
    Karkain type checker or symbol table from spec reading alone would
    violate the no-fabrication rule, so this adapter refuses to emit symbols.
    """

    name = "karkain-kcc"
    version = ADAPTER_VERSION

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> SymbolAnalysis:
        result = SymbolAnalysis(language_id="karkain", adapter=self.name,
                                adapter_version=self.version,
                                source_sha256=source_sha256, file=file, module=module)
        result.status = "BLOCKED"
        result.diagnostics.append(Diagnostic(
            category="blocked",
            message="symbol analysis requires KCC; kcc.exe fails with 0xC0000135 on this host",
            severity="ERROR", source=self.name, tool_version="kcc v1.1.0 (unavailable)",
            evidence_classification="UNVERIFIED"))
        result.unsupported = [
            "KARK-BENCH: 0 instantiated tasks",
            "kcc.exe 0xC0000135 — symbol table unavailable; no parser substitution authorized"]
        return result


class SymbolAnalysisEngine:
    """Language-neutral sequencing of symbol adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {
            "python": PythonSymbolAdapter(),
            "karkain": KarkainSymbolAdapter(),
        }

    def analyze(self, source: str, *, language_id: str, file: str,
                module: str | None = None) -> SymbolAnalysis:
        module = module or file.rsplit(".", 1)[0].replace("/", ".")
        digest = source_sha256(source)
        adapter = self._adapters.get(language_id.lower())
        if adapter is None:
            adapter = GenericSymbolAdapter(language_id.lower())
        return adapter.analyze(source, source_sha256=digest, file=file, module=module)


def fingerprint_analysis(analysis: SymbolAnalysis) -> str:
    from taksh.dlcae.types import fingerprint

    return fingerprint(analysis.to_dict())
