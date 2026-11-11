"""Type relationship representation — language-neutral, discipline-preserving.

The TE does not impose one language's semantics on another. Each relation
carries a ``discipline`` tag: static | gradual | dynamic | inferred |
structural | unknown. Absent annotations are recorded as **explicit unknown**;
never as a guessed type. Compatibility queries return observations, not
verdicts — only the Validation Engine may promote observations to PASS.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from taksh.dlcae.semantic.contract import Diagnostic, source_sha256

ADAPTER_VERSION = "typesys/1.0.0"

# Minimal, documented compatibility matrix. Anything unlisted is UNVERIFIED.
_WIDENINGS = {
    ("int", "float"),
    ("bool", "int"),
    ("int", "complex"),
    ("float", "complex"),
}


@dataclass
class TypeAnalysis:
    language_id: str
    adapter: str
    adapter_version: str
    source_sha256: str
    file: str
    module: str
    status: str = "UNVERIFIED"          # PASS | FAIL | PARTIAL | BLOCKED | UNVERIFIED
    discipline: str = "unknown"
    relations: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.type_analysis/1",
            "language_id": self.language_id,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "source_sha256": self.source_sha256,
            "file": self.file,
            "module": self.module,
            "status": self.status,
            "discipline": self.discipline,
            "relations": self.relations,
            "unsupported": self.unsupported,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _unparse(annotation: ast.AST | None) -> dict[str, Any]:
    if annotation is None:
        return {"notion": "unknown", "explicit_unknown": True, "type": None,
                "detail": "absent annotation — explicitly unknown; never inferred silently"}
    try:
        text = ast.unparse(annotation)
    except Exception:  # noqa: BLE001
        return {"notion": "unknown", "explicit_unknown": True, "type": None,
                "detail": "annotation not printable"}
    subscripted = "[" in text
    return {
        "notion": "composite" if subscripted else "primitive",
        "explicit_unknown": False,
        "type": text,
        "detail": f"annotation `{text}` ({'composite' if subscripted else 'primitive'})",
    }


class PythonTypeAdapter:
    name = "python-ast-types"
    version = ADAPTER_VERSION

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> TypeAnalysis:
        result = TypeAnalysis(language_id="python", adapter=self.name,
                              adapter_version=self.version,
                              source_sha256=source_sha256, file=file, module=module,
                              discipline="gradual")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            result.status = "BLOCKED"
            result.unsupported = [f"syntax failure — type analysis cannot proceed: {exc.msg}"]
            return result

        def scope_of(stack):
            return ".".join(stack) or "<module>"

        def visit(node, stack):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = f"{stack}.{node.name}" if stack else node.name
                all_params = (list(node.args.posonlyargs) + list(node.args.args)
                              + ([node.args.vararg] if node.args.vararg else [])
                              + list(node.args.kwonlyargs)
                              + ([node.args.kwarg] if node.args.kwarg else []))
                signature_relation = {
                    "kind": "signature", "name": fn, "discipline": "gradual",
                    "parameters": [], "returns": None, "source": self.name,
                }
                for param in all_params:
                    if param is None or param.arg == "_":
                        continue
                    signature_relation["parameters"].append(
                        {"name": param.arg, **_unparse(param.annotation)})
                signature_relation["returns"] = _unparse(node.returns)
                result.relations.append(signature_relation)
                for param in all_params:
                    if param is None or param.arg == "_":
                        continue
                    parsed = _unparse(param.annotation)
                    result.relations.append({
                        "kind": "parameter", "name": f"{fn}.{param.arg}",
                        "scope": fn, "discipline": "gradual",
                        **parsed, "source": self.name,
                        "line": getattr(param, "lineno", 1)})
                if node.returns is not None:
                    parsed = _unparse(node.returns)
                    result.relations.append({
                        "kind": "return", "name": f"{fn} -> {parsed['type']}",
                        "scope": fn, "discipline": "gradual",
                        **parsed, "source": self.name,
                        "line": getattr(node, "lineno", 1)})
                for child in ast.iter_child_nodes(node):
                    visit(child, stack + [node.name])
                return
            if isinstance(node, ast.ClassDef):
                # type parameters: Generic[T] or plain bases kept as observations
                for base in node.bases:
                    base_text = None
                    try:
                        base_text = ast.unparse(base)
                    except Exception:  # noqa: BLE001
                        pass
                    if base_text and base_text.startswith("Generic["):
                        params_text = base_text[base_text.index("[") + 1:base_text.rindex("]")]
                        for raw in params_text.split(","):
                            raw = raw.strip()
                            if raw:
                                result.relations.append({
                                    "kind": "type_parameter",
                                    "name": f"{stack}.{node.name}.{raw}" if stack else f"{node.name}.{raw}",
                                    "scope": f"{stack}.{node.name}" if stack else node.name,
                                    "discipline": "gradual",
                                    "type": raw, "explicit_unknown": False,
                                    "detail": f"type parameter of generic class `{node.name}`",
                                    "line": getattr(node, "lineno", 1),
                                    "source": self.name})
                for child in ast.iter_child_nodes(node):
                    visit(child, stack + [node.name])
                return
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                scope = scope_of(stack)
                parsed = _unparse(node.annotation)
                result.relations.append({
                    "kind": "variable", "name": node.target.id, "scope": scope,
                    "discipline": "gradual", **parsed, "source": self.name,
                    "line": getattr(node, "lineno", 1)})
                for child in ast.iter_child_nodes(node):
                    visit(child, stack)
                return
            if isinstance(node, ast.Assign):
                # explicit unknown: unannotated assignment carries no type evidence
                scope = scope_of(stack)
                for target in node.targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name):
                            result.relations.append({
                                "kind": "variable", "name": child.id, "scope": scope,
                                "discipline": "dynamic",
                                "kind_detail": "explicit_unknown",
                                "type": None, "explicit_unknown": True,
                                "detail": "unannotated assignment — dynamic; explicitly unknown",
                                "line": getattr(child, "lineno", 1),
                                "source": self.name})
                for child in ast.iter_child_nodes(node):
                    visit(child, stack)
                return
            if isinstance(node, ast.arg):
                return
            for child in ast.iter_child_nodes(node):
                visit(child, stack)

        visit(tree, [])
        result.status = "PARTIAL"   # annotations are partial by nature; mypy can widen this
        return result


class KarkainTypeAdapter:
    """Karkain types — BLOCKED. No invented type checker, ever."""

    name = "karkain-kcc-types"
    version = ADAPTER_VERSION

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> TypeAnalysis:
        return TypeAnalysis(
            language_id="karkain", adapter=self.name,
            adapter_version=self.version, source_sha256=source_sha256,
            file=file, module=module, status="BLOCKED",
            discipline="static",
            unsupported=["kcc.exe 0xC0000135 — no type oracle; "
                         "Karkain semantics must not be inferred from spec reading"])


class GenericTypeAdapter:
    name = "generic-types"
    version = ADAPTER_VERSION

    def __init__(self, language_id: str) -> None:
        self.language_id = language_id

    def analyze(self, source: str, *, source_sha256: str,
                file: str, module: str) -> TypeAnalysis:
        return TypeAnalysis(language_id=self.language_id, adapter=self.name,
                            adapter_version=self.version,
                            source_sha256=source_sha256, file=file, module=module,
                            status="UNVERIFIED", discipline="unknown",
                            unsupported=[f"no type adapter qualified for {self.language_id!r}; "
                                         f"all types explicitly unknown"])


class TypeAnalysisEngine:
    def __init__(self) -> None:
        self._adapters = {"python": PythonTypeAdapter(), "karkain": KarkainTypeAdapter()}

    def analyze(self, source: str, *, language_id: str, file: str,
                module: str | None = None) -> TypeAnalysis:
        module = module or file.rsplit(".", 1)[0].replace("/", ".")
        digest = source_sha256(source)
        adapter = self._adapters.get(language_id.lower()) or GenericTypeAdapter(language_id.lower())
        return adapter.analyze(source, source_sha256=digest, file=file, module=module)


def types_compatible(declared: str | None, expected: str | None) -> dict[str, Any]:
    """Compatibility *observation* — never a verdict on source correctness.

    ``None`` means explicitly unknown: the answer is UNVERIFIED, always.
    """
    if declared is None or expected is None:
        return {"observation": "UNVERIFIED",
                "reason": "explicit unknown type — compatibility not established"}
    d, e = declared.strip(), expected.strip()
    if d == e:
        return {"observation": "IDENTICAL", "reason": f"`{d}` == `{e}`"}
    if (d, e) in _WIDENINGS:
        return {"observation": "COMPATIBLE_WIDENING", "reason": f"`{d}` widens to `{e}`"}
    return {"observation": "INCOMPATIBLE_OR_UNKNOWN",
            "reason": f"no compatibility rule for `{d}` vs `{e}` — observation only"}
