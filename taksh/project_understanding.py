"""TAKSH 1.4.0 — Project understanding primitives.

Repository discovery, file indexing, symbol indexing, dependency graphs,
and change-impact analysis. Read-only by default; never modifies the target
repository.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def discover_files(root: Path, *, exclude_patterns: list[str] | None = None) -> list[Path]:
    patterns = exclude_patterns or [".git", "node_modules", "__pycache__", "runs",
                                     "data", "taksh-hf-publication", ".pytest_cache",
                                     "dist", "build", ".venv"]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _matches_any(d, patterns)]
        for name in filenames:
            if not _matches_any(name, patterns):
                files.append(Path(dirpath) / name)
    return files


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def index_symbols(root: Path) -> dict[str, Any]:
    symbols: dict[str, list[dict[str, str]]] = defaultdict(list)
    imports: dict[str, set[str]] = defaultdict(set)
    files = discover_files(root)
    for fp in files:
        if fp.suffix != ".py":
            continue
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols[fp_relative(root, fp)].append({"type": type(node).__name__, "name": node.name,
                                                        "line": str(getattr(node, "lineno", 0))})
            elif isinstance(node, ast.Import):
                imports[fp_relative(root, fp)].update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports[fp_relative(root, fp)].add(node.module)
    return {
        "symbols": {k: v for k, v in symbols.items()},
        "imports": {k: sorted(v) for k, v in imports.items()},
        "file_count": len(files),
    }


def dependency_graph(root: Path) -> dict[str, list[str]]:
    # Simple module-level dependency graph built from import statements
    graph: dict[str, set[str]] = defaultdict(set)
    for fp in discover_files(root):
        if fp.suffix != ".py":
            continue
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    graph[fp_relative(root, fp) or fp.name].add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                graph[fp_relative(root, fp) or fp.name].add(node.module.split(".")[0])
    return {k: sorted(v) for k, v in graph.items()}


def fp_relative(root: Path, fp: Path) -> str:
    try:
        return fp.relative_to(root).as_posix()
    except ValueError:
        return fp.name


def project_summary(root: Path) -> dict[str, Any]:
    files = discover_files(root)
    py_files = [f for f in files if f.suffix == ".py"]
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    symbols = index_symbols(root)
    return {
        "root": str(root),
        "total_files": len(files),
        "python_files": len(py_files),
        "total_bytes": total_bytes,
        "symbols_indexed": sum(len(v) for v in symbols["symbols"].values()),
        "modules_with_imports": len(symbols["imports"]),
    }
