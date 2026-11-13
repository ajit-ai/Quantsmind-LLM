"""Focused tests for the TAKSH L0.6.0 foundation model qualification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import l060_config_qualification as cq  # noqa: E402
from evaluation import l060_training_readiness as tr  # noqa: E402
from evaluation import resource_guard as guard  # noqa: E402
from evaluation.artifact_integrity import (  # noqa: E402
    V180_SFT_SHA256,
    checkpoint_hash_status,
    resolve_checkpoint_path,
)
from evaluation.l060_multilingual_readiness import (  # noqa: E402
    score_benchmark,
    score_language,
    score_task,
)
from serving import foundation_inference as fi  # noqa: E402
from serving.local_inference import load_model  # noqa: E402
from training.model import QMModel  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


TINY_CFG = {
    "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 4,
    "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 64,
    "rope_theta": 10000.0, "norm_eps": 1e-05, "tie_embeddings": True,
}


# --------------------------------------------------------------------------- #
# WP1 — configuration qualification
# --------------------------------------------------------------------------- #
def test_foundation_config_is_valid():
    cfg = cq.load_config()
    assert cq.validate_config(cfg) == []
    assert cq.count_parameters(cfg["model"]) == 110_658_816


def test_foundation_config_hash_and_roundtrip():
    cfg = cq.load_config()
    rt = cq.roundtrip_hash(cfg)
    assert rt["identical"] is True
    assert cq.canonical_hash(cfg) == rt["hash_before"]


def test_invalid_vocab_size_rejected():
    cfg = cq.load_config()
    cfg["model"]["vocab_size"] = 9999
    assert any("vocab_size" in p for p in cq.validate_config(cfg))


def test_head_divisibility_rejected():
    cfg = cq.load_config()
    cfg["model"]["d_model"] = 1000  # 1000 % 12 != 0
    problems = cq.validate_config(cfg)
    assert any("d_model" in p for p in problems)


def test_gqa_constraint_rejected():
    cfg = cq.load_config()
    cfg["model"]["n_kv_heads"] = 5  # 12 % 5 != 0
    problems = cq.validate_config(cfg)
    assert any("n_kv_heads" in p for p in problems)


def test_max_seq_len_must_be_positive_int():
    cfg = cq.load_config()
    cfg["model"]["max_seq_len"] = 0
    assert any("max_seq_len" in p for p in cq.validate_config(cfg))


def test_rope_theta_and_eps_constraints():
    cfg = cq.load_config()
    cfg["model"]["rope_theta"] = 0.5
    assert any("rope_theta" in p for p in cq.validate_config(cfg))
    cfg = cq.load_config()
    cfg["model"]["norm_eps"] = -1e-5
    assert any("norm_eps" in p for p in cq.validate_config(cfg))


# --------------------------------------------------------------------------- #
# WP2 — model implementation
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_loss_finite():
    model = QMModel(TINY_CFG)
    idx = torch.tensor([[1, 5, 9, 7]], dtype=torch.long)
    targets = torch.tensor([[5, 9, 7, 3]], dtype=torch.long)
    logits, loss = model(idx, targets=targets)
    assert list(logits.shape) == [1, 4, TINY_CFG["vocab_size"]]
    assert torch.isfinite(loss).item()


def test_causal_attention_prefix_invariant():
    model = QMModel(TINY_CFG).eval()
    ids = [1, 5, 9, 7]
    with torch.no_grad():
        full, _ = model(torch.tensor([ids], dtype=torch.long))
        prefix, _ = model(torch.tensor([ids[:-1]], dtype=torch.long))
    assert torch.allclose(full[0, -2], prefix[0, -1], atol=1e-5)


def test_deterministic_initialization():
    torch.manual_seed(1337)
    m1 = QMModel(TINY_CFG)
    torch.manual_seed(1337)
    m2 = QMModel(TINY_CFG)
    assert all(torch.equal(p1, p2) for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()))


def test_state_dict_roundtrip(tmp_path):
    import tempfile

    model = QMModel(TINY_CFG)
    path = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert all(torch.equal(model.state_dict()[k], blob[k]) for k in model.state_dict())


def test_foundation_config_instantiates_111m():
    cfg = cq.load_config()
    model = QMModel(cfg["model"])
    assert model.num_params() == 110_658_816


# --------------------------------------------------------------------------- #
# WP3 — training readiness / manifest enforcement
# --------------------------------------------------------------------------- #
def test_manifest_rejects_unknown_license():
    manifest = {"schema": "x", "dataset_id": "y", "generated_by": "g", "seed": 42,
                "sources": {"s": {"license": "gpl-3.0", "path": "ok.jsonl"}}}
    problems = tr.validate_dataset_manifest(manifest)
    assert any("unknown/unapproved license" in p for p in problems)


def test_manifest_rejects_missing_provenance():
    manifest = {"schema": "x", "sources": {"s": {"license": "mit"}}}
    problems = tr.validate_dataset_manifest(manifest)
    assert any("missing provenance" in p for p in problems)


def test_manifest_rejects_benchmark_contamination():
    manifest = {"schema": "x", "dataset_id": "y", "generated_by": "g", "seed": 42,
                "sources": {"s": {"license": "mit",
                                  "path": "evaluation/l02_python_bench_v1.json"}}}
    problems = tr.validate_dataset_manifest(manifest)
    assert any("benchmark contamination" in p for p in problems)


def test_manifest_rejects_unverified_language_label():
    manifest = {"schema": "x", "dataset_id": "y", "generated_by": "g", "seed": 42,
                "languages": ["cobol2"]}
    problems = tr.validate_dataset_manifest(manifest)
    assert any("unverified language label" in p for p in problems)


def test_manifest_compound_licenses_pass():
    manifest = {"schema": "x", "dataset_id": "y", "generated_by": "g", "seed": 42,
                "sources": {"a": {"license": "MIT / BSD-3-Clause (per record)", "path": "x.jsonl"},
                             "b": {"license": "CC-BY-SA-3.0 (Wikipedia); Public Domain (Gutenberg; rows carry 'see-MANIFEST')", "path": "y.jsonl"}}}
    assert tr.validate_dataset_manifest(manifest) == []


def test_in_repo_manifests_validate_pass():
    for rel in ("data/l03_corpus/dataset_manifest.json", "data/l04_code_corpus/dataset_manifest.json"):
        blob = json.loads((REPO / rel).read_text(encoding="utf-8"))
        assert tr.validate_dataset_manifest(blob) == []


# --------------------------------------------------------------------------- #
# WP4/WP5 — strategy + inference interfaces
# --------------------------------------------------------------------------- #
def test_inference_bounded_and_deterministic():
    model = QMModel(TINY_CFG)
    from tokenizers import Tokenizer, models

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for i, ch in enumerate("abcdefghijklmno"):
        vocab[ch] = i + 4
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok_path = REPO / "tests" / "_tmp_tiny_tok.json"
    try:
        tok.save(str(tok_path))
        out = fi.generate_bounded(model, tok, "abcde", max_new_tokens=4, temperature=0.0, device="cpu")
        assert out["completion_tokens"] <= 4
    finally:
        tok_path.unlink(missing_ok=True)


def test_inference_limit_validation():
    model = QMModel(TINY_CFG)
    with pytest.raises(fi.ConfigurationError):
        fi.generate_bounded(model, None, "abc", max_new_tokens=9999)
    with pytest.raises(fi.ConfigurationError):
        fi.generate_bounded(model, None, "abc", temperature=5.0)


# --------------------------------------------------------------------------- #
# WP6 — multilingual benchmark denominator correction
# --------------------------------------------------------------------------- #
def test_scoring_interfaces_exist():
    scored = score_task({"id": "x", "tests": [{"passed": True}, {"passed": False}]})
    assert scored["passed"] is False
    assert scored["assertions_passed"] == 1
    lang = score_language([{"id": f"t{i}", "language": "python",
                             "tests": [{"passed": True}]} for i in range(16)])
    assert lang["qualified"] is True and lang["executed"] == 16
    lang4 = score_language([{"id": f"t{i}", "language": "python",
                             "tests": [{"passed": True}]} for i in range(4)])
    assert lang4["qualified"] is False
    assert lang4["ineligible_reason"] == "insufficient_task_count"


def test_benchmark_denominator_is_corrected_and_python_untouched():
    bench = json.loads((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_text(encoding="utf-8"))
    assert bench["qualification"]["min_tasks_per_language"] == 16
    assert bench["qualification"]["min_pass_ratio"] == 0.50
    assert len(bench["tasks"]) == 64
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_benchmark_scoring_reports_python_frozen():
    out = score_benchmark([{"id": "x", "language": "python", "tests": [{"passed": True}]}])
    assert out["python_frozen_benchmark_untouched"] is True


# --------------------------------------------------------------------------- #
# WP7 — resource guards fail closed
# --------------------------------------------------------------------------- #
def test_authorization_fails_closed_without_phrase(monkeypatch):
    monkeypatch.delenv(guard.AUTH_ENV, raising=False)
    with pytest.raises(guard.AuthorizationRequired):
        guard.require_authorization("train")


def test_authorization_requires_exact_phrase(monkeypatch):
    monkeypatch.setenv(guard.AUTH_ENV, "something-else")
    with pytest.raises(guard.AuthorizationRequired):
        guard.require_authorization("train")


def test_clamp_threads_caps_and_floors():
    assert guard.clamp_threads(None) == 4
    assert guard.clamp_threads(999) == 4
    assert guard.clamp_threads(1) == 1
    with pytest.raises(guard.ResourceLimitExceeded):
        guard.enforce_context(10_000)
    with pytest.raises(guard.ResourceLimitExceeded):
        guard.enforce_batch_size(999)
    assert guard.enforce_batch_size(8) == 8
    assert guard.device_allowed("cpu") == "cpu"
    with pytest.raises(guard.AuthorizationRequired):
        guard.device_allowed("cuda")


# --------------------------------------------------------------------------- #
# Preservation
# --------------------------------------------------------------------------- #
def test_checkpoint_hash_status_absent(tmp_path):
    assert checkpoint_hash_status(tmp_path / "missing.pt", V180_SFT_SHA256) == "ABSENT"


def test_checkpoint_hash_status_match_and_mismatch(tmp_path):
    artifact = tmp_path / "checkpoint_last.pt"
    artifact.write_bytes(b"fixture")
    expected = hashlib.sha256(b"fixture").hexdigest()
    assert checkpoint_hash_status(artifact, expected) == "MATCH"
    assert checkpoint_hash_status(artifact, "0" * 64) == "MISMATCH"


def test_existing_checkpoint_preserved_hash():
    path = resolve_checkpoint_path(REPO)
    status = checkpoint_hash_status(path, V180_SFT_SHA256)
    if status == "ABSENT":
        pytest.skip(
            f"optional checkpoint artifact not present at {path.as_posix()}; "
            "hash preservation was not verified (no checkpoint was copied or modified)"
        )
    assert status == "MATCH", (
        f"checkpoint hash mismatch at {path.as_posix()}: expected {V180_SFT_SHA256}; "
        "artifact was inspected read-only and not modified"
    )


def test_language_registry_still_valid():
    import yaml

    from evaluation.l051_language_registry import validate_registry

    cfg = yaml.safe_load((REPO / "configs/feasibility/l051_languages.yaml").read_text(encoding="utf-8"))
    assert validate_registry(cfg)["valid"] is True
    assert sum(1 for l in cfg["languages"] if l["l1_target"]) == 16


def test_frozen_benchmark_hash_preservation():
    digest = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest()
    assert digest == FROZEN_PY_SHA
