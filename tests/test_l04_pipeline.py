"""Focused tests for the TAKSH L0.4 pipeline.

Covers: frozen benchmark integrity, corpus license/provenance qualification,
code-token share, secret scanning, benchmark-leakage quarantine, near-duplicate
detection, split determinism, tokenizer coverage/specials, sandboxed execution,
warmstart compatibility, greedy determinism, and benchmark hash enforcement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "l02_python_bench_v1.json"
BENCH_SHA256 = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
L04 = REPO / "data" / "l04_code_corpus"
TOK = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
FROZEN_NOW = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"

import scripts.build_l04_corpus as B  # noqa: E402


def _records(name):
    with (L04 / name).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Frozen benchmark
# --------------------------------------------------------------------------- #
def test_frozen_benchmark_hash_is_unchanged():
    import hashlib

    assert hashlib.sha256(BENCH.read_bytes()).hexdigest() == BENCH_SHA256


def test_benchmark_sha256_flag_rejects_wrong_hash(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "evaluation.l02_runner",
         "--ckpt", "runs/v180_sft/checkpoint_last.pt", "--tag", "tmp_hashcheck_reject",
         "--bench-sha256", "0" * 64, "--limit", "1"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert rc.returncode != 0
    assert "benchmark hash mismatch" in rc.stderr


# --------------------------------------------------------------------------- #
# Corpus qualification
# --------------------------------------------------------------------------- #
def test_every_record_carries_source_license_domain_hash():
    allowed = {"cc-by-sa-3.0", "public domain", "mit", "bsd-3-clause", "apache-2.0"}
    for name in ("train.jsonl", "code_val.jsonl", "general_val.jsonl"):
        for i, r in enumerate(_records(name)):
            assert r["source"] and r["license"] and r["domain"] and r["sha1"]
            assert r["license"].lower() in allowed
            if i > 200:
                break


def test_code_token_share_measured_and_met():
    manifest = json.loads((L04 / "dataset_manifest.json").read_text(encoding="utf-8"))
    tok = manifest["tokenization"]
    assert tok["train_tokens"] == 3182359
    assert tok["code_token_share"] >= 0.49
    assert manifest["near_duplicates_removed"] == 1
    assert manifest["exact_duplicates_removed"] == 41
    assert manifest["benchmark_leaks_quarantined"] == 4
    assert manifest["secret_quarantined"] == 0


def test_corpus_has_no_benchmark_instruction_strings():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    instructions = [p["instruction"].strip().lower() for p in bench["problems"]]
    for name in ("train.jsonl", "code_val.jsonl", "general_val.jsonl"):
        with (L04 / name).open("r", encoding="utf-8") as fh:
            for line in fh:
                low = line.lower()
                for needle in instructions:
                    assert needle not in low


def test_secret_scanner_flags_synthetic_key():
    doc = {"text": "key = 'AKIAIOSFODNN7EXAMPLE' plus padding here to be long enough to matter for scanning",
           "source": "test", "license": "mit", "domain": "code"}
    assert B.scan_secrets([doc]), "AKIA-pattern private key was not flagged"


def test_benchmark_leakage_scanner_flags_instruction():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    probe_doc = {"text": "some prose\n" + bench["problems"][0]["instruction"],
                 "source": "test", "license": "mit", "domain": "code"}
    hits = B.scan_benchmark_leakage([probe_doc], BENCH)
    assert hits and hits[0][0] == 0


def test_near_duplicate_detection_drops_paraphrase():
    base = "def compute_total(items):\n    total = 0\n    for item in items:\n        total += item\n    return total"
    near = base + "\n"
    distinct = "class SimpleStack:\n    pass"
    docs = [
        {"text": base, "sha1": "a" * 40, "source": "t", "license": "mit", "domain": "code"},
        {"text": near, "sha1": "b" * 40, "source": "t", "license": "mit", "domain": "code"},
        {"text": distinct, "sha1": "c" * 40, "source": "t", "license": "mit", "domain": "code"},
    ]
    kept, dropped = B.near_duplicates(docs, set())
    assert dropped == 1
    assert [d["sha1"] for d in kept] == ["a" * 40, "c" * 40]


def test_splits_are_disjoint_and_deterministic():
    train_sha = {r["sha1"] for r in _records("train.jsonl")}
    code_val_sha = {r["sha1"] for r in _records("code_val.jsonl")}
    general_val_sha = {r["sha1"] for r in _records("general_val.jsonl")}
    assert not (train_sha & code_val_sha)
    assert not (train_sha & general_val_sha)


def test_token_artifacts_match_tokenizer():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK))
    assert tok.get_vocab_size() == 5822
    for name in ("train.tokens.meta.json",):
        meta = json.loads((L04 / name).read_text(encoding="utf-8"))
        assert meta["vocab_size"] == 5822
        assert meta["dtype"] == "uint16"
        assert meta["document_separator"] == "<eos>"
    arr = np.memmap(L04 / "train.tokens.bin", dtype=np.uint16, mode="r")
    assert int(arr.max()) < 5822


def test_tokenizer_report_has_expected_shape():
    report = json.loads((REPO / "evaluation" / "l04_tokenizer_report.json").read_text(encoding="utf-8"))
    for key in ("tokenizer", "programs", "key_token_ids", "indentation",
                "punctuation", "packing", "conclusions"):
        assert key in report
    assert report["indentation"]["1 level"]["n_tokens"] == 1
    assert all(p["decode_roundtrip_exact"] for p in report["programs"])


def test_diagnostic_report_has_required_sections():
    report = json.loads((REPO / "evaluation" / "l04_diagnostic_report.json").read_text(encoding="utf-8"))
    for key in ("tokenizer", "token_inventory", "indentation", "identifiers_l03_code",
                "documents_l03_code", "separators_l03_train", "mix_l03_train",
                "model_probes", "code_syntax_parse"):
        assert key in report


def test_sandbox_rejects_timeout_and_imports():
    from evaluation.l02_runner import ast_gate, run_in_sandbox

    ok, _ = ast_gate("import math\ndef f():\n    return math.floor(3.7)")
    assert ok
    ok, _ = ast_gate("import os\ndef f():\n    return 1")
    assert not ok
    problem = {"kind": "function", "entry_point": "spin",
               "tests": [{"name": "loop", "args": [], "expected": 1}]}
    out = run_in_sandbox("def spin():\n    while True:\n        pass", problem, timeout=2.0)
    assert out["ok"] is False and "timeout" in out["error"]


def test_warmstart_l03_checkpoint_loads_with_strict_compatibility():
    from serving.local_inference import torch_load_checkpoint

    ckpt = torch_load_checkpoint("runs/l03_pretrain/checkpoint_last.pt")
    assert ckpt.get("step") == 600
    assert len(ckpt["model"]) == 111


def test_overfit_checkpoint_greedy_determinism():
    from serving import local_assistant as la
    from serving.local_inference import generate_ids, load_model
    from tokenizers import Tokenizer

    model, _ = load_model("runs/l02_overfit/checkpoint_last.pt", "cpu")
    tok = Tokenizer.from_file(str(TOK))
    prompt = la.build_prompt(tok, "Write a Python function named add_two(a, b) that returns the sum of the two numbers a and b.")
    outs = []
    for _ in range(2):
        r = generate_ids(model, tok, prompt, max_new_tokens=48, temperature=0.0,
                         device="cpu", binding_repair=True)
        outs.append(r["text"])
    assert outs[0] == outs[1]
    assert "def add_two" in outs[0]
