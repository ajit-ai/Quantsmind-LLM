"""Taksh v1.6.0 — Encode cleared tokenizer corpus to packed token bin files.

Encodes `data/tokenizer_corpus/taksh_code_v1.txt` (project-owned Python source,
cleared for distribution) using the v1.5.0 cleared tokenizer
(`tokenizer/taksh_code_bpe_v1.json`), then splits into train/val token streams.

Output:
  data/v160_tokens/train.tokens.bin + .meta.json
  data/v160_tokens/val.tokens.bin + .meta.json
  data/v160_tokens/manifest.json

Usage:
    python scripts/encode_v160_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS_PATH = Path("data/tokenizer_corpus/taksh_code_v1.txt")
TOKENIZER_PATH = Path("tokenizer/taksh_code_bpe_v1.json")
OUT_DIR = Path("data/v160_tokens")
TRAIN_RATIO = 0.90

# Document separator: <eos> token id 2 (same as the tokenizer's EOS)
DOC_SEP_TOKEN = 2


def load_documents(corpus_path: Path) -> list[str]:
    """Parse the corpus file into individual source-file documents.

    The corpus format (from scripts/build_tokenizer_corpus.py) is:
        <|file|> path/to/file.py
        <source code>
        <|doc|>
        <|file|> path/to/other.py
        <source code>
        <|doc|>

    Returns a list of document strings (source code without headers/separators).
    """
    text = corpus_path.read_text(encoding="utf-8")
    raw_docs = text.split("<|doc|>")
    documents = []
    for raw in raw_docs:
        raw = raw.strip()
        if not raw:
            continue
        # Strip the <|file|> header line
        lines = raw.split("\n", 1)
        if lines[0].startswith("<|file|>"):
            content = lines[1] if len(lines) > 1 else ""
        else:
            content = raw
        content = content.strip()
        if content:
            documents.append(content)
    return documents


def encode_documents(
    tok: Tokenizer, documents: list[str], max_tokens: int | None = None
) -> tuple[list[int], int]:
    """Encode all documents with <eos> separators between them.

    Returns (token_ids, num_documents_emitted).
    """
    ids: list[int] = []
    emitted = 0
    for doc in documents:
        if max_tokens is not None and len(ids) >= max_tokens:
            break
        doc_ids = tok.encode(doc).ids
        ids.extend(doc_ids)
        ids.append(DOC_SEP_TOKEN)  # document boundary
        emitted += 1
        if max_tokens is not None and len(ids) > max_tokens:
            ids = ids[: max_tokens]
            break
    return ids, emitted


def write_token_bin(ids: list[int], out_path: Path, meta: dict) -> None:
    """Write token ids as a binary file with sidecar metadata JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_id = max(ids) if ids else 0
    dtype = np.uint16 if max_id < 65536 else np.uint32
    np.asarray(ids, dtype=dtype).tofile(out_path)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  {out_path.name}: tokens={len(ids):,} dtype={np.dtype(dtype).name} "
        f"docs={meta.get('num_documents','?')}"
    )


def main() -> None:
    if not CORPUS_PATH.exists():
        print(f"ERROR: corpus not found: {CORPUS_PATH}", file=sys.stderr)
        print("Run scripts/build_tokenizer_corpus.py first.", file=sys.stderr)
        sys.exit(1)
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: tokenizer not found: {TOKENIZER_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"  vocab_size = {tok.get_vocab_size()}")

    print(f"Loading corpus: {CORPUS_PATH}")
    documents = load_documents(CORPUS_PATH)
    print(f"  documents = {len(documents)}")

    total_chars = sum(len(d) for d in documents)
    print(f"  total chars = {total_chars:,}")

    # Encode all documents
    all_ids, emitted = encode_documents(tok, documents)
    print(f"  encoded tokens = {len(all_ids):,} (emitted docs = {emitted})")

    # Split train/val (90/10) at document boundaries to avoid leakage
    n_train = int(emitted * TRAIN_RATIO)
    # Re-encode train and val separately for clean separation
    train_ids, _ = encode_documents(tok, documents[:n_train])
    val_ids, _ = encode_documents(tok, documents[n_train:])

    print(f"  train tokens = {len(train_ids):,}")
    print(f"  val tokens   = {len(val_ids):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_sha = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    tok_sha = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    train_meta = {
        "schema": "quantsmind.token_stream/1",
        "vocab_size": tok.get_vocab_size(),
        "num_tokens": len(train_ids),
        "num_documents": n_train,
        "dtype": "uint16",
        "corpus": str(CORPUS_PATH),
        "corpus_sha256": corpus_sha,
        "tokenizer": str(TOKENIZER_PATH),
        "tokenizer_sha256": tok_sha,
        "generated_at": now,
        "split": "train",
        "split_ratio": TRAIN_RATIO,
        "document_separator": "<eos>",
        "document_separator_id": DOC_SEP_TOKEN,
        "max_seq_len": 2048,
    }
    write_token_bin(train_ids, OUT_DIR / "train.tokens.bin", train_meta)

    val_meta = {
        "schema": "quantsmind.token_stream/1",
        "vocab_size": tok.get_vocab_size(),
        "num_tokens": len(val_ids),
        "num_documents": emitted - n_train,
        "dtype": "uint16",
        "corpus": str(CORPUS_PATH),
        "corpus_sha256": corpus_sha,
        "tokenizer": str(TOKENIZER_PATH),
        "tokenizer_sha256": tok_sha,
        "generated_at": now,
        "split": "val",
        "split_ratio": 1.0 - TRAIN_RATIO,
        "document_separator": "<eos>",
        "document_separator_id": DOC_SEP_TOKEN,
        "max_seq_len": 2048,
    }
    write_token_bin(val_ids, OUT_DIR / "val.tokens.bin", val_meta)

    manifest = {
        "campaign": "TAKSH v1.6.0 — Pretraining corpus encoding",
        "created": now,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_sha,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": tok_sha,
        "vocab_size": tok.get_vocab_size(),
        "total_documents": emitted,
        "train_documents": n_train,
        "val_documents": emitted - n_train,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "total_tokens": len(all_ids),
        "split_ratio": TRAIN_RATIO,
        "seq_len": 2048,
        "notes": "Encoded with cleared tokenizer (v1.5.0). Project-owned source only.",
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Train: {len(train_ids):,} tokens, {n_train} docs")
    print(f"Val:   {len(val_ids):,} tokens, {emitted - n_train} docs")


if __name__ == "__main__":
    main()
