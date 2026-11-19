"""Tests for M3 dataset preparation and model configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from data.prepare_m3_dataset import (
    LicenceGateError,
    check_split_isolation,
    prepare,
    verify_licences,
)

TOKENIZER = "tokenizer/qm_code_bpe_v0.json"


def record(source_id, repo, split, text=None):
    text = text or (f"# {source_id}\n" + "def f():\n    return 1\n" * 20)
    return {
        "text": text,
        "content_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "source_id": source_id,
        "repo": repo,
        "commit": "a" * 40,
        "spdx": "MIT",
        "attribution": "t",
        "file_path": "f.py",
        "language": "python",
        "retrieved_at": "2026-10-18",
        "split": split,
    }


def write_corpus(tmp_path, records):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


def write_licence_report(tmp_path, verified, needs_review, all_verified=True):
    path = tmp_path / "licence.json"
    path.write_text(json.dumps({
        "counts": {
            "checked": len(verified) + len(needs_review),
            "verified": len(verified), "rejected": len(needs_review),
            "needs_human_review": len(needs_review),
        },
        "verified_ids": list(verified),
        "needs_human_review_ids": list(needs_review),
        "all_verified": all_verified,
    }), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Licence gate
# --------------------------------------------------------------------------- #
def test_records_from_unreviewed_sources_are_withheld(tmp_path):
    records = [record("good", "org/a", "train"), record("unreviewed", "org/b", "train")]
    report = write_licence_report(tmp_path, ["good"], ["unreviewed"], all_verified=False)
    gate = verify_licences(records, report)
    assert [r["source_id"] for r in gate["authorised"]] == ["good"]
    assert [r["source_id"] for r in gate["withheld"]] == ["unreviewed"]
    assert gate["withheld_records"] == 1


def test_a_record_claiming_mit_is_not_enough_to_be_authorised(tmp_path):
    """A record's own spdx field is not evidence a human reviewed the licence."""
    records = [record("good", "org/a", "train"),
               dict(record("unreviewed", "org/b", "train"), spdx="MIT")]
    report = write_licence_report(tmp_path, ["good"], ["unreviewed"],
                                  all_verified=False)
    gate = verify_licences(records, report)
    assert [r["source_id"] for r in gate["authorised"]] == ["good"]
    assert [r["source_id"] for r in gate["withheld"]] == ["unreviewed"]


def test_missing_licence_report_blocks_everything(tmp_path):
    with pytest.raises(LicenceGateError, match="licence verification report not found"):
        verify_licences([record("good", "org/a", "train")], tmp_path / "absent")


def test_empty_verified_list_blocks(tmp_path):
    report = write_licence_report(tmp_path, [], [], all_verified=False)
    with pytest.raises(LicenceGateError, match="no verified sources"):
        verify_licences([record("good", "org/a", "train")], report)


# --------------------------------------------------------------------------- #
# Split isolation
# --------------------------------------------------------------------------- #
def test_isolation_detects_a_shared_repository():
    records = [record("a", "org/shared", "train"), record("b", "org/shared", "val")]
    report = check_split_isolation(records)
    assert report["ok"] is False
    assert "org/shared" in report["shared_repositories"]


def test_isolation_detects_a_shared_content_hash():
    text = "identical text"
    records = [record("a", "org/a", "train", text=text),
               record("b", "org/b", "val", text=text)]
    assert check_split_isolation(records)["ok"] is False


def test_isolation_passes_for_disjoint_repositories():
    records = [record("a", "org/a", "train"), record("b", "org/b", "val")]
    assert check_split_isolation(records)["ok"] is True


# --------------------------------------------------------------------------- #
# Preparation, including the filename regression
# --------------------------------------------------------------------------- #
def test_splits_get_distinct_files(tmp_path):
    """Regression: Path.with_suffix destroyed '.train'/'.val', so the val split
    silently overwrote the train split's artifacts."""
    corpus = write_corpus(tmp_path, [record("a", "org/a", "train"),
                                     record("b", "org/b", "val")])
    licence = write_licence_report(tmp_path, ["a", "b"], [])
    prefix = str(tmp_path / "ds")

    manifest = prepare(corpus, licence, TOKENIZER, prefix, encode=False)

    train, val = Path(f"{prefix}.train.jsonl"), Path(f"{prefix}.val.jsonl")
    assert train.exists() and val.exists() and train != val
    assert len(train.read_text(encoding="utf-8").splitlines()) == 1
    assert len(val.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest["splits"]["train"]["jsonl"] == str(train)


def test_token_artifact_records_the_jsonl_it_came_from(tmp_path):
    corpus = write_corpus(tmp_path, [record("a", "org/a", "train"),
                                     record("b", "org/b", "val")])
    licence = write_licence_report(tmp_path, ["a", "b"], [])
    prefix = str(tmp_path / "ds")

    manifest = prepare(corpus, licence, TOKENIZER, prefix, encode=True)

    for split in ("train", "val"):
        info = manifest["splits"][split]
        assert info["token_chain_matches_jsonl"] is True
        meta = json.loads(Path(info["token_meta"]).read_text(encoding="utf-8"))
        assert meta["corpus_sha256"] == info["jsonl_sha256"]
        assert meta["tokenizer"] == TOKENIZER


def test_preparation_refuses_leaking_splits(tmp_path):
    corpus = write_corpus(tmp_path, [record("a", "org/shared", "train"),
                                     record("b", "org/shared", "val")])
    licence = write_licence_report(tmp_path, ["a", "b"], [])
    with pytest.raises(ValueError, match="not isolated"):
        prepare(corpus, licence, TOKENIZER, str(tmp_path / "ds"), encode=False)


def test_preparation_refuses_an_empty_authorised_split(tmp_path):
    corpus = write_corpus(tmp_path, [record("bad", "org/b", "train")])
    licence = write_licence_report(tmp_path, ["good"], ["bad"], all_verified=False)
    with pytest.raises(ValueError, match="no licence-authorised records"):
        prepare(corpus, licence, TOKENIZER, str(tmp_path / "ds"), encode=False)


def test_holdout_split_is_not_emitted(tmp_path):
    corpus = write_corpus(tmp_path, [record("a", "org/a", "train"),
                                     record("b", "org/b", "val"),
                                     record("c", "org/c", "test")])
    licence = write_licence_report(tmp_path, ["a", "b", "c"], [])
    prefix = str(tmp_path / "ds")
    manifest = prepare(corpus, licence, TOKENIZER, prefix, encode=False)
    assert manifest["holdout"]["emitted"] is False
    assert not Path(f"{prefix}.test.jsonl").exists()


# --------------------------------------------------------------------------- #
# Shipped artifacts
# --------------------------------------------------------------------------- #
def test_shipped_m3_dataset_excludes_unresolved_licences():
    manifest = json.loads(
        Path("data/m3_code_verified.manifest.json").read_text(encoding="utf-8")
    )
    gate = manifest["licence_gate"]
    assert gate["withheld_records"] == 162
    assert set(gate["withheld_sources"]) == {"psf-black", "pypa-setuptools"}
    assert gate["unreviewed_allowed"] is False
    assert manifest["isolation"]["ok"] is True
    assert manifest["holdout"]["emitted"] is False


def test_shipped_train_split_contains_no_unresolved_source():
    train_path = Path("data/m3_code_verified.train.jsonl")
    if not train_path.exists():
        pytest.skip("M3 dataset artifacts are not part of the source release")
    verified = set(json.loads(
        Path("data/licence_verification.json").read_text(encoding="utf-8")
    )["verified_ids"])
    sources = {
        json.loads(line)["source_id"]
        for line in train_path
        .read_text(encoding="utf-8").splitlines() if line.strip()
    }
    assert sources <= verified, f"unreviewed sources leaked: {sources - verified}"


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
def load_model_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["model"]


def test_m3_configs_match_the_code_tokenizer_vocabulary():
    from tokenizers import Tokenizer

    vocab = Tokenizer.from_file(TOKENIZER).get_vocab_size()
    assert vocab == 5822
    for path in ("configs/model/code_m3_9m.yaml", "configs/model/code_m3_3m.yaml"):
        assert load_model_cfg(path)["vocab_size"] == vocab, path


def test_m3_training_configs_never_resolve_to_the_orphan_vocabulary():
    """code_30m.yaml declares 32768, which matches no tokenizer shipped here."""
    orphan = load_model_cfg("configs/model/code_30m.yaml")["vocab_size"]
    assert orphan == 32768
    for path in ("configs/train/pretrain_m3.yaml",
                 "configs/train/pretrain_m3_small.yaml"):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert load_model_cfg(cfg["run"]["model_config"])["vocab_size"] != orphan, path


def test_m3_model_parameter_counts_are_as_documented():
    from training.model import QMModel

    counts = {}
    for name in ("code_m3_9m", "code_m3_3m"):
        model = QMModel(load_model_cfg(f"configs/model/{name}.yaml"))
        counts[name] = sum(p.numel() for p in model.parameters())
    assert counts["code_m3_9m"] == 9_359_104
    assert counts["code_m3_3m"] == 2_853_280


def test_vocabulary_difference_explains_the_code_30m_parameter_count():
    """The orphan config's extra parameters are exactly its wider vocabulary."""
    from training.model import QMModel

    small = load_model_cfg("configs/model/code_m3_9m.yaml")
    corrected = sum(p.numel() for p in QMModel(small).parameters())
    orphan = dict(small, vocab_size=32768)
    orphan_count = sum(p.numel() for p in QMModel(orphan).parameters())
    # docs/VERSIONING.md records code_30m as 16,257,280 parameters.
    assert orphan_count == 16_257_280
    assert orphan_count - corrected == (32768 - small["vocab_size"]) * small["d_model"]


def test_m3_configs_keep_vocab_and_context_equal_to_the_contract():
    """Vocabulary and context length are correctness contracts, not knobs."""
    for path in ("configs/model/code_m3_9m.yaml", "configs/model/code_m3_3m.yaml"):
        cfg = load_model_cfg(path)
        assert cfg["vocab_size"] == 5822
        assert cfg["max_seq_len"] == 512


def test_m3_configs_are_internally_consistent():
    from training.model import QMModel

    for path in ("configs/model/code_m3_9m.yaml", "configs/model/code_m3_3m.yaml"):
        cfg = load_model_cfg(path)
        # These assertions live in Attention; check the model builds at all.
        QMModel(cfg)
        assert cfg["d_model"] % cfg["n_heads"] == 0
        assert cfg["n_heads"] % cfg["n_kv_heads"] == 0