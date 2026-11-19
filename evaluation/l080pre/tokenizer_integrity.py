"""TAKSH L0.8.0-PRE — B: tokenizer integrity verification.

Verifies the canonical tokenizer artifact, distinguishes Git-blob (LF,
canonical) from working-tree (CRLF) hashes, and confirms vocabulary parity
with the model configuration. Never modifies the tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

TOK_PATH = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
EXPECTED_COMMITTED = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
EXPECTED_WORKING = "554c8e7572d58923a0766f114b5f419b14e68bdff3a738803b30728ec3ffcd92"
EXPECTED_VOCAB = 5822


def committed_blob_hash() -> str:
    proc = subprocess.run(["git", "cat-file", "blob", "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO)
    return hashlib.sha256(proc.stdout).hexdigest()


def working_tree_hash() -> str:
    return hashlib.sha256(TOK_PATH.read_bytes()).hexdigest()


def verify() -> dict:
    committed = committed_blob_hash()
    working = working_tree_hash()
    vocabulary = None
    try:
        from tokenizers import Tokenizer

        vocabulary = Tokenizer.from_file(str(TOK_PATH)).get_vocab_size()
    except Exception as exc:  # noqa: BLE001
        vocabulary = f"unreadable: {exc}"
    vocab_ok = vocabulary == EXPECTED_VOCAB
    return {
        "schema": "taksh.l080pre_tokenizer_integrity/1",
        "generated_by": "evaluation/l080pre/tokenizer_integrity.py",
        "artifact_path": str(TOK_PATH.relative_to(REPO)),
        "hash_method": {
            "committed_git_blob_sha256": committed,
            "working_tree_sha256": working,
            "method_note": "committed Git blob (LF) is the canonical fingerprint; working-tree copy is CRLF on Windows and differs intentionally; both recorded, never conflated",
        },
        "committed_blob_matches_l0706_anchor": committed == EXPECTED_COMMITTED,
        "working_tree_matches_recorded_crlf_copy": working == EXPECTED_WORKING,
        "vocab_size": vocabulary,
        "vocab_size_expected": EXPECTED_VOCAB,
        "vocabulary_equal": vocab_ok,
        "artifact_modified_or_rewritten": False,
        "labels": {"committed_blob": "measured (git blob)", "working_tree": "measured (working tree)",
                   "vocab_size": "measured (tokenizers load)"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_tokenizer_integrity.json")
    args = parser.parse_args(argv)

    result = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not (result["committed_blob_matches_l0706_anchor"] and result["vocabulary_equal"]):
        print("TOKENIZER INVALID:", {k: v for k, v in result.items()})
        return 2
    print(f"tokenizer verified: committed={result['hash_method']['committed_git_blob_sha256'][:16]}… vocab={result['vocab_size']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
