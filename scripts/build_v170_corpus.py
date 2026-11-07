"""Taksh v1.7.0 — Build a scaled cleared tokenizer corpus.

Collects project-owned Python source files plus optional external cleared
code directories into a larger training corpus (5M–50M tokens target).

Designed for v1.7.0 data pipeline scaling. All sources must be cleared
for distribution (project-owned or permissive-license verified).

Usage:
    python scripts/build_v170_corpus.py --out data/v170_corpus/taksh_code_v1.txt \
        --extra-dirs /path/to/cleared/code1 /path/to/cleared/code2

Output:
    data/v170_corpus/taksh_code_v1.txt         (plain text corpus)
    data/v170_corpus/taksh_code_v1.manifest.json  (provenance)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Project-owned source directories (cleared for distribution)
PROJECT_DIRS = [
    "taksh",
    "serving",
    "evaluation",
    "training",
    "scripts",
    "configs",
]

# Files to exclude
EXCLUDE_NAMES = {
    "__init__.py",
    "conftest.py",
}

DOC_SEP = "\n\n<|doc|>\n\n"


def collect_python_files(roots: List[Path], extra_dirs: List[str]) -> List[Path]:
    """Recursively collect .py files from project dirs and extra dirs."""
    files: List[Path] = []
    base = Path.cwd()

    for d in PROJECT_DIRS:
        p = base / d
        if p.exists():
            for f in sorted(p.rglob("*.py")):
                if f.name in EXCLUDE_NAMES or "__pycache__" in f.parts:
                    continue
                files.append(f)

    for extra in extra_dirs:
        p = Path(extra)
        if p.exists():
            for f in sorted(p.rglob("*.py")):
                if f.name in EXCLUDE_NAMES or "__pycache__" in f.parts:
                    continue
                files.append(f)
        else:
            print(f"WARNING: extra dir not found: {extra}", file=sys.stderr)

    return files


def build_corpus(files: List[Path], out_path: Path) -> Tuple[int, int, str]:
    """Write all source files to out_path separated by DOC_SEP.

    Returns (file_count, total_bytes, sha256_hex).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    hasher = hashlib.sha256()
    file_count = 0

    with out_path.open("w", encoding="utf-8") as out:
        for p in files:
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError) as exc:
                print(f"WARNING: skipping {p}: {exc}", file=sys.stderr)
                continue

            header = f"<|file|> {p}\n"
            out.write(header)
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            out.write(DOC_SEP)

            data = (header + text + DOC_SEP).encode("utf-8")
            hasher.update(data)
            total_bytes += len(data)
            file_count += 1

    return file_count, total_bytes, hasher.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Build scaled cleared tokenizer corpus for v1.7.0")
    p.add_argument("--out", required=True, help="output corpus file path")
    p.add_argument(
        "--extra-dirs", nargs="*", default=[],
        help="additional directories of cleared code to include",
    )
    args = p.parse_args()

    files = collect_python_files([Path.cwd()], args.extra_dirs)
    print(f"Collected {len(files)} Python source files")

    out_path = Path(args.out)
    file_count, total_bytes, sha256 = build_corpus(files, out_path)

    # Estimate token count (rough: ~4 chars per token for code)
    est_tokens = total_bytes // 4

    manifest = {
        "campaign": "TAKSH v1.7.0 — Scaled cleared corpus build",
        "created": datetime.now(timezone.utc).isoformat(),
        "project_dirs": PROJECT_DIRS,
        "extra_dirs": args.extra_dirs,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "estimated_tokens": est_tokens,
        "corpus_sha256": sha256,
        "output_file": str(out_path),
        "notes": "Corpus contains only cleared code. Scale target: 5M–50M tokens.",
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Files: {file_count}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Estimated tokens: {est_tokens:,}")
    print(f"Corpus SHA-256: {sha256}")
    print(f"Output: {out_path}")
    print(f"Manifest: {manifest_path}")

    if est_tokens < 5_000_000:
        print(f"\nNOTE: Estimated tokens ({est_tokens:,}) below v1.7.0 target (5M+).")
        print("Add more cleared code via --extra-dirs or accept smaller scale.")


if __name__ == "__main__":
    main()
