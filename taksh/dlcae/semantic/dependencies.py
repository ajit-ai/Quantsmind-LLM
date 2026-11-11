"""Dependency analysis — local, offline import graphs.

Builds file/module dependencies, import graphs, package relationships,
circular-dependency detection, missing-dependency detection (with installed
metadata via ``importlib.util.find_spec`` — never imports, never network),
dependency provenance/version identity, and impact analysis for changed
symbols/modules.

Nothing downloads, resolves remote registries, or executes code.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DependencyGraph:
    source_of: str                      # sha256 of the primary source
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.dlcae.dependency_graph/1",
            "source_of": self.source_of,
            "packages": {k: v for k, v in sorted(self.nodes.items())},
            "edges": self.edges,
            "cycles": self.cycles,
            "missing": self.missing,
            "diagnostics": self.diagnostics,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


def _stdlib_names() -> frozenset[str]:
    return frozenset(getattr(sys, "stdlib_module_names", set()))


def _installed_version(package: str) -> str | None:
    """Installed version via importlib.metadata — offline metadata only."""
    try:
        from importlib import metadata

        return metadata.version(package)
    except Exception:  # noqa: BLE001
        return None


def analyze_python_dependencies(sources: dict[str, str], *, workspace_root: Path | None = None) -> DependencyGraph:
    """Build the import graph over a set of Python sources.

    ``sources`` maps module name -> source text (synthetic fixtures or local
    project files only). ``workspace_root`` enables relative-import resolution
    against a local tree; absent root, relative imports are reported only.
    """
    graph = DependencyGraph(source_of=hash_of_sources(sources))
    stdlib = _stdlib_names()
    module_names = set(sources)
    for module, source in sorted(sources.items()):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            graph.diagnostics.append({"category": "syntax_error", "module": module,
                                      "message": f"{exc.msg} (line {exc.lineno})"})
            continue
        graph.nodes[module] = {"module": module, "kind": "module",
                               "source_sha256": _sha256_text(source)}
        for node in ast.walk(tree):
            imports = list(_iter_imports(node))
            for imported, level, lineno in imports:
                resolved = _resolve_import(imported, level, module, module_names,
                                           workspace_root)
                if resolved is None:
                    kind, available, version = _classify_external(imported, stdlib)
                    graph.nodes.setdefault(imported, {
                        "module": imported, "kind": kind,
                        "available": available, "version": _installed_version(imported),
                    })
                    graph.edges.append({"from": module, "to": imported,
                                        "kind": kind, "line": lineno,
                                        "relative_level": level})
                    if not available:
                        graph.missing.append({"module": imported, "imported_by": module,
                                              "line": lineno,
                                              "reason": "not found in stdlib, local modules, or installed distributions"})
                        graph.diagnostics.append({
                            "category": "missing_dependency", "module": module,
                            "line": lineno,
                            "message": f"missing dependency: {imported}"})
                else:
                    graph.edges.append({"from": module, "to": resolved,
                                        "kind": "local", "line": lineno,
                                        "relative_level": level})
    graph.cycles = _find_cycles(module_names, graph.edges)
    for cycle in graph.cycles:
        graph.diagnostics.append({"category": "circular_dependency",
                                  "message": f"circular import: {' -> '.join(cycle)}"})
    return graph


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_of_sources(sources: dict[str, str]) -> str:
    import hashlib
    import json

    blob = json.dumps({k: _sha256_text(v) for k, v in sorted(sources.items())},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _iter_imports(node: ast.AST):
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name, 0, getattr(node, "lineno", 1)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            module = node.module or ""
            if node.level:
                yield "." * node.level + module, node.level, getattr(node, "lineno", 1)
            else:
                yield module, 0, getattr(node, "lineno", 1)


def _resolve_import(imported: str, level: int, current_module: str,
                    module_names: set[str],
                    workspace_root: Path | None) -> str | None:
    if level == 0:
        top = imported.split(".")[0]
        return imported if imported in module_names else (top if top in module_names else None)
    # relative import: walk up `level` packages from current module
    parts = current_module.split(".")
    base = parts[:-level] if level > 1 else parts[:-1]
    if imported.startswith("."):
        imported = imported.lstrip(".")
    if imported:
        candidate = ".".join(base + imported.split("."))
    elif base:
        candidate = ".".join(base)
    else:
        candidate = imported
    return candidate if candidate in module_names else None


def _classify_external(imported: str, stdlib: frozenset[str]) -> tuple[str, bool, str | None]:
    top = imported.split(".")[0]
    if top in stdlib:
        return "stdlib", True, None
    spec = None
    try:
        spec = importlib.util.find_spec(top)
    except Exception:  # noqa: BLE001 — metadata lookup failure is data, never fatal
        spec = None
    if spec is not None:
        return "third_party_installed", True, _installed_version(top)
    return "third_party_missing", False, None


def _find_cycles(module_names: set[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {name: [] for name in module_names}
    for edge in edges:
        if edge["from"] in adjacency and edge["to"] in adjacency:
            adjacency[edge["from"]].append(edge["to"])
    color: dict[str, int] = {}   # 0=unvisited, 1=in-stack, 2=done
    stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for target in adjacency.get(node, ()):
            state = color.get(target, 0)
            if state == 0:
                dfs(target)
            elif state == 1:
                cycle = stack[stack.index(target):] + [target]
                if cycle not in cycles:
                    cycles.append(cycle)
        stack.pop()
        color[node] = 2

    for name in sorted(module_names):
        if color.get(name, 0) == 0:
            dfs(name)
    return cycles


def impact_analysis(graph: DependencyGraph, changed_modules: list[str]) -> dict[str, Any]:
    """Reverse-dependency impact: which modules depend on the changed ones."""
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        reverse.setdefault(edge["to"], []).append(edge["from"])
    affected: set[str] = set()
    queue = list(changed_modules)
    while queue:
        module = queue.pop(0)
        for dependent in reverse.get(module, ()):
            if dependent not in affected:
                affected.add(dependent)
                queue.append(dependent)
    return {
        "schema": "taksh.dlcae.impact_analysis/1",
        "changed": sorted(changed_modules),
        "affected": sorted(affected),
        "impact_count": len(affected),
        "note": "reverse-dependency impact over the local import graph; no remote resolution performed",
    }
