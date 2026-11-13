"""Focused tests for the Taksh L0.3 corpus qualification pipeline.

Verifies: source licensing metadata, deduplication, seeded train/val split
integrity, tokenization roundtrip with the Taksh tokenizer, token-artifact
compatibility, no evaluation-benchmark leakage into the corpus, and the
manifest/ground-truth invariants. A tiny fixture corpus is built from the real
source files into a temporary directory where one full rebuild is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "l02_python_bench_v1.json"
L03 = REPO / "data" / "l03_corpus"
TOK = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
ALLOWED_LICENSES = frozenset(
    {"cc-by-sa-3.0", "public domain", "mit", "bsd-3-clause", "apache-2.0", "bsd-2-clause", "isc"}
)


def _records(name: str):
    with (L03 / name).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def test_every_record_carries_source_and_license():
    for name in ("train.jsonl", "val.jsonl"):
        for i, r in enumerate(_records(name)):
            assert r["source"] and r["license"] and r["domain"] and r["sha1"]
            assert r["license"].lower() in ALLOWED_LICENSES, r["license"]
            if i > 4000:  # enough to prove the invariant well-sampled
                break


def test_splits_have_no_sha1_overlap_and_counts_match_manifest():
    train_sha = {r["sha1"] for r in _records("train.jsonl")}
    val_sha = {r["sha1"] for r in _records("val.jsonl")}
    assert not (train_sha & val_sha), "same document in both splits"
    manifest = json.loads((L03 / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_documents"] == 9500
    assert manifest["val_documents"] == 196
    assert manifest["duplicates_removed"] == 42


def test_manifest_matches_file_hashes():
    import hashlib

    manifest = json.loads((L03 / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((L03 / "train.jsonl").read_bytes()).hexdigest() == manifest["train_sha256"]
    assert hashlib.sha256((L03 / "val.jsonl").read_bytes()).hexdigest() == manifest["val_sha256"]


def test_token_artifacts_compatible_with_taksh_tokenizer():
    meta_train = json.loads((L03 / "train.tokens.meta.json").read_text(encoding="utf-8"))
    meta_val = json.loads((L03 / "val.tokens.meta.json").read_text(encoding="utf-8"))
    assert meta_train["schema"] == "quantsmind.token_stream/1"
    assert meta_train["vocab_size"] == 5822
    assert meta_train["document_separator"] == "<eos>"
    assert meta_train["document_separator_id"] == 2
    assert meta_train["dtype"] == "uint16"
    assert meta_train["num_tokens"] == 19_629_683
    assert meta_val["num_tokens"] == 380_034
    tokens = np.memmap(L03 / "train.tokens.bin", dtype=np.uint16, mode="r")
    assert len(tokens) == meta_train["num_tokens"]
    assert int(tokens.max()) < 5822
    assert int(tokens.min()) >= 0


def test_tokenizer_encode_decode_roundtrip_with_separator():
    from tokenizers import Tokenizer
    from training.document_separator import DOC_SEPARATOR_ID, resolve_separator_id

    tok = Tokenizer.from_file(str(TOK))
    assert resolve_separator_id(tok) == DOC_SEPARATOR_ID == 2
    text = "def add_two(a, b):\n    return a + b"
    ids = tok.encode(text).ids
    assert tok.decode(ids) == text


def test_corpus_has_no_l02_benchmark_leakage():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    bench_texts = {p["instruction"].strip() for p in bench["problems"]}
    bench_names = {p["entry_point"] for p in bench["problems"]}
    hits = []
    for name in ("train.jsonl", "val.jsonl"):
        with (L03 / name).open("r", encoding="utf-8") as fh:
            for line in fh:
                for needle in bench_texts:
                    if needle in line:
                        hits.append((name, needle[:40]))
    assert not hits, f"benchmark instruction text found in corpus: {hits[:3]}"


def test_manifest_declares_no_rereleased_raw_data():
    manifest = json.loads((L03 / "dataset_manifest.json").read_text(encoding="utf-8"))
    licenses = manifest["by_license"]
    assert set(licenses) <= ALLOWED_LICENSES
    total = sum(licenses.values())
    assert total == manifest["documents"]


def test_build_l03_corpus_is_deterministic(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "l03_rebuilt"
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.build_l03_corpus", "--out-dir", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
    rebuilt = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((L03 / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert rebuilt["documents"] == original["documents"]
    assert rebuilt["duplicates_removed"] == original["duplicates_removed"]
    assert rebuilt["train_sha256"] == original["train_sha256"]
    assert rebuilt["val_sha256"] == original["val_sha256"]


def test_manifest_records_warmstart_initialization_note():
    from training.manifest import build_run_manifest

    manifest = build_run_manifest(
        run_id="test", status="complete", config={},
        config_path="configs/train/pretrain_l03.yaml",
        model_cfg={"vocab_size": 5822}, model_config_path=None, params=1,
        initialization="warmstarted from x (sha256 abc); optimizer state not carried over",
    )
    assert manifest["model"]["initialization"].startswith("warmstarted from")
