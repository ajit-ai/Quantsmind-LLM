"""Taksh v1.5.0 — Build a cleared tokenizer training corpus.

Collects the project's own Python source files (which the project owner
controls and can license-clear) into a single text file suitable for training
a BPE tokenizer. No external or uncleared training data is used.

Usage:
    python scripts/build_tokenizer_corpus.py --out data/tokenizer_corpus/taksh_code_v1.txt

The output file is a plain text corpus: one source file's contents per
document, separated by document boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Directories whose .py files are project-owned and cleared for tokenizer training.
SOURCE_DIRS = [
    "taksh",
    "serving",
    "evaluation",
    "training",
    "scripts",
    "configs",
]

# Files to exclude (generated, test-only fixtures, or build artifacts).
EXCLUDE_NAMES = {
    "__init__.py",  # usually empty or re-exports; skip to reduce duplication
    "conftest.py",
    "fake_generate_ids",  # not a real file but a pattern
}

# Document separator between source files.
DOC_SEP = "\n\n<|doc|>\n\n"


def collect_python_files(root: Path, dirs: List[str]) -> List[Path]:
    """Recursively collect .py files from the given source directories."""
    files: List[Path] = []
    for d in dirs:
        base = root / d
        if not base.exists():
            print(f"WARNING: source dir not found: {base}", file=sys.stderr)
            continue
        for p in sorted(base.rglob("*.py")):
            if p.name in EXCLUDE_NAMES:
                continue
            # Skip __pycache__ and any build output
            if "__pycache__" in p.parts:
                continue
            files.append(p)
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
            except UnicodeDecodeError:
                print(f"WARNING: skipping non-UTF-8 file: {p}", file=sys.stderr)
                continue
            except PermissionError:
                print(f"WARNING: skipping unreadable file: {p}", file=sys.stderr)
                continue

            rel = p.relative_to(p.anchor) if p.anchor else p
            header = f"<|file|> {rel}\n"
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
    p = argparse.ArgumentParser(description="Build a cleared tokenizer training corpus from project source")
    p.add_argument("--out", required=True, help="output corpus file path")
    p.add_argument("--dirs", nargs="*", default=SOURCE_DIRS, help="source directories to scan")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out)

    files = collect_python_files(root, args.dirs)
    print(f"Collected {len(files)} Python source files from {args.dirs}")

    file_count, total_bytes, sha256 = build_corpus(files, out_path)

    manifest = {
        "campaign": "TAKSH v1.5.0 — Tokenizer corpus build",
        "created": datetime.now(timezone.utc).isoformat(),
        "source_directories": args.dirs,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "corpus_sha256": sha256,
        "output_file": str(out_path),
        "notes": "Corpus contains only project-owned Python source code. Cleared for tokenizer training.",
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Files: {file_count}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Corpus SHA-256: {sha256}")
    print(f"Output: {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
