"""Taksh v1.3.0 qualification tests: bounded Python pretraining.

Synthetic fixtures and a tiny model only. No network, no GPU, no real corpus.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from data.python_v1.pipeline import build_dataset
from training import objective
from training.authorization import AuthorizationError, authorize_dataset
from training.data_loader import DeterministicSampler, ManifestDataset, WindowStream
from training.determinism import set_determinism
from training.precision import PrecisionError, resolve_device, resolve_precision
from training.schedules import WarmupCosineSchedule
from training.sequence import SequenceBuilder, SequenceError
from training.train_config import (
    TRAIN_CONFIG_SCHEMA,
    TrainConfigError,
    load_train_config,
    validate_train_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_REL = "tokenizer/qm_code_bpe_v0.json"
TOKENIZER = REPO_ROOT / TOKENIZER_REL

TINY_MODEL = {
    "name": "v130-tiny",
    "vocab_size": 5822,
    "d_model": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 64,
    "max_seq_len": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1.0e-5,
    "tie_embeddings": True,
}


def _load_synthetic_builder():
    path = REPO_ROOT / "scripts" / "build_synthetic_corpus.py"
    spec = importlib.util.spec_from_file_location("build_synthetic_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model_config(root: Path) -> Path:
    path = root / "tiny_model.yaml"
    path.write_text(
        yaml.safe_dump({"model": TINY_MODEL}, sort_keys=True), encoding="utf-8"
    )
    return path


def _build_synthetic_corpus(root: Path, sources: int = 6, modules: int = 2) -> Path:
    """Build a tiny synthetic corpus and return its manifest path."""
    builder = _load_synthetic_builder()
    corpus_root = root / "corpus_src"
    corpus_root.mkdir(parents=True, exist_ok=True)
    entries = builder.build_corpus(corpus_root, sources, modules, 4242)
    contract = builder._contract(entries, 4242)
    contract_path = corpus_root / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(contract, sort_keys=True), encoding="utf-8"
    )
    (corpus_root / "benchmarks.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "quantsmind.benchmark_registry/1",
                "version": "synthetic",
                "suites": [
                    {
                        "id": "none_available",
                        "version": "0",
                        "availability": "unavailable",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    build_dataset(contract_path, corpus_root / "corpus")
    return corpus_root / "corpus" / "manifest.json"


def _train_config(
    root: Path,
    manifest: Path,
    *,
    mode: str = "synthetic",
    model_config: Path | None = None,
    max_steps: int = 4,
    **overrides,
) -> Path:
    """Write a bounded training configuration next to the data."""
    cfg = {
        "schema": TRAIN_CONFIG_SCHEMA,
        "mode": mode,
        "run": {
            "name": "v130-test",
            "model_config": "tiny_model.yaml",
            "seed": 7,
        },
        "data": {
            "manifest": "corpus_src/corpus/manifest.json",
            "tokenizer": TOKENIZER_REL,
            "context_length": 64,
            "sequence_policy": "document",
            "append_eos": True,
            "allow_cross_document": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr_peak": 3.0e-3,
            "warmup_steps": 1,
            "min_lr_ratio": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
        },
        "batch": {
            "micro_batch_size": 2,
            "grad_accum_steps": 2,
            "shuffle": True,
        },
        "precision": {"device": "cpu", "dtype": "float32"},
        "training": {
            "max_steps": max_steps,
            "max_tokens": 200000,
            "max_wall_seconds": 60,
            "checkpoint_every_steps": 2,
            "eval_every_steps": 2,
            "val_batches": 2,
            "output_dir": "out",
        },
        "limits": {"max_checkpoints": 2, "max_output_bytes": 200000000},
    }
    for section, values in overrides.items():
        cfg.setdefault(section, {})
        cfg[section].update(values)
    path = root / "train_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def synthetic_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("v130")
    manifest = _build_synthetic_corpus(root)
    model_config = _write_model_config(root)
    return {"root": root, "manifest": manifest, "model_config": model_config}


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


# --------------------------------------------------------------------------- #
# G1 - dataset authorization
# --------------------------------------------------------------------------- #
def test_g1_synthetic_manifest_is_refused_for_production(synthetic_corpus):
    with pytest.raises(AuthorizationError) as excinfo:
        authorize_dataset(
            synthetic_corpus["manifest"], mode="production", tokenizer_path=TOKENIZER
        )
    assert excinfo.value.reason == "synthetic_manifest_refused"


def test_g1_production_manifest_is_refused_in_synthetic_mode(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantsmind.python_dataset_manifest/1",
                "manifest_sha256": "sha256:0",
                "notes": {"corpus_class": "production", "corpus_ready": True},
                "sources": {"approved": ["x"]},
                "splits": {"ready": True},
                "tokenizer": {},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuthorizationError) as excinfo:
        authorize_dataset(manifest, mode="synthetic", tokenizer_path=TOKENIZER)
    assert excinfo.value.reason == "production_manifest_refused_in_synthetic_mode"


def test_g1_missing_manifest_is_refused(tmp_path):
    with pytest.raises(AuthorizationError) as excinfo:
        authorize_dataset(
            tmp_path / "nope.json", mode="synthetic", tokenizer_path=TOKENIZER
        )
    assert excinfo.value.reason == "manifest_missing"


def test_g1_synthetic_manifest_is_authorized_for_synthetic_mode(synthetic_corpus):
    authorization = authorize_dataset(
        synthetic_corpus["manifest"], mode="synthetic", tokenizer_path=TOKENIZER
    )
    assert authorization.corpus_class == "synthetic"
    assert authorization.corpus_ready is False
    assert authorization.dataset_id
    assert authorization.tokenizer_sha256


def test_g1_tokenizer_mismatch_is_refused(synthetic_corpus, tmp_path):
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(AuthorizationError) as excinfo:
        authorize_dataset(
            synthetic_corpus["manifest"], mode="synthetic", tokenizer_path=other
        )
    assert excinfo.value.reason == "tokenizer_mismatch"


def test_g1_tampered_artifact_is_refused(synthetic_corpus, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(synthetic_corpus["manifest"].parent, copy)
    path = copy / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AuthorizationError) as excinfo:
        authorize_dataset(
            copy / "manifest.json", mode="synthetic", tokenizer_path=TOKENIZER
        )
    assert excinfo.value.reason == "artifact_verification_failed"


# --------------------------------------------------------------------------- #
# G2 - tokenizer and control-token safety
# --------------------------------------------------------------------------- #
def test_g2_reserved_literal_in_source_is_refused(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    with pytest.raises(SequenceError) as excinfo:
        builder.encode('token = "<|user|>"\n')
    assert excinfo.value.reason == "reserved_literal_in_source"


def test_g2_eos_literal_in_source_is_refused(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    with pytest.raises(SequenceError) as excinfo:
        builder.encode('token = "<eos>"\n')
    assert excinfo.value.reason == "reserved_literal_in_source"


def test_g2_control_id_is_refused_without_authorisation(tokenizer):
    from training.tokenizer_contract import build_contract_tokenizer

    derived = build_contract_tokenizer(TOKENIZER)
    builder = SequenceBuilder(
        derived, context_length=32, model_vocab=5828, control_token_floor=5822
    )
    builder.authorized_control_ids = set()
    builder.reserved_literals = ()
    with pytest.raises(SequenceError) as excinfo:
        builder.encode('token = "<|user|>"\n')
    assert excinfo.value.reason == "unauthorized_control_token"


def test_g2_authorised_control_id_is_accepted(tokenizer):
    from training.tokenizer_contract import build_contract_tokenizer

    derived = build_contract_tokenizer(TOKENIZER)
    builder = SequenceBuilder(
        derived,
        context_length=32,
        model_vocab=5828,
        control_token_floor=5822,
        authorized_control_ids=[5823],
        reserved_literals=(),
    )
    ids = builder.encode('token = "<|user|>"\n')
    assert 5823 in ids


def test_g2_id_beyond_model_vocab_is_refused(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=100)
    with pytest.raises(SequenceError) as excinfo:
        builder.encode("value = 1\n")
    assert excinfo.value.reason == "token_id_beyond_model_vocab"


def test_g2_builder_inserts_its_own_end_of_document_token(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    ids = builder.encode_document("value = 1\n")
    assert ids[-1] == builder.eos_id
    assert builder.eos_id not in builder.encode("value = 1\n")


# --------------------------------------------------------------------------- #
# G3 - sequence correctness
# --------------------------------------------------------------------------- #
def test_g3_labels_are_shifted_by_one(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    ids = builder.encode_document("def f():\n    return 1\n")
    inputs, labels = objective.shift_for_causal(torch.tensor(ids), builder.pad_id)
    assert torch.equal(inputs, torch.tensor(ids[:-1]))
    assert torch.equal(
        labels[labels != objective.IGNORE_INDEX], torch.tensor(ids[1:])
    )


LONG_DOC = (
    "def f():\n    return 1\n"
    "def g():\n    return 2\n"
    "def h():\n    return 3\n"
    "def i():\n    return 4\n"
)


def test_g3_padding_positions_are_not_supervised(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=10, model_vocab=5822)
    windows = builder.build_windows([("s", "a.py", LONG_DOC)])
    batch = builder.to_batch(windows)
    padded = batch.labels == objective.IGNORE_INDEX
    assert padded.any()
    assert batch.input_ids.shape == batch.labels.shape == batch.attention_mask.shape
    assert int(batch.attention_mask.sum()) < int(batch.attention_mask.numel())


def test_g3_document_policy_never_mixes_documents(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=64, model_vocab=5822)
    documents = [
        ("s1", "a.py", "def alpha():\n    return 1\n"),
        ("s2", "b.py", "def beta():\n    return 2\n"),
    ]
    windows = builder.build_windows(documents)
    for window, count, cross in windows:
        assert count == 1
        assert cross is False


def test_g3_packed_policy_requires_explicit_consent(tokenizer):
    with pytest.raises(SequenceError) as excinfo:
        SequenceBuilder(tokenizer, context_length=64, policy="packed")
    assert excinfo.value.reason == "cross_document_packing_not_authorised"


def test_g3_packed_policy_is_marked_cross_document(tokenizer):
    builder = SequenceBuilder(
        tokenizer,
        context_length=64,
        policy="packed",
        allow_cross_document=True,
        model_vocab=5822,
    )
    documents = [
        ("s1", "a.py", "def alpha():\n    return 1\n"),
        ("s2", "b.py", "def beta():\n    return 2\n"),
    ]
    windows = builder.build_windows(documents)
    assert windows
    assert all(cross is True for _, _, cross in windows)


def test_g3_token_accounting_is_recorded(tokenizer):
    builder = SequenceBuilder(tokenizer, context_length=64, model_vocab=5822)
    builder.encode_document("def f():\n    return 1\n")
    stats = builder.statistics()
    assert stats["documents_encoded"] == 1
    assert stats["tokens_encoded"] > 0
    assert stats["policy"] == "document"


# --------------------------------------------------------------------------- #
# G4 - causal objective correctness
# --------------------------------------------------------------------------- #
def _tiny_model(seed: int = 0):
    from training.model import QMModel

    set_determinism(seed)
    return QMModel(TINY_MODEL)


def test_g4_loss_matches_a_manual_cross_entropy(tokenizer):
    model = _tiny_model(1)
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    windows = builder.build_windows([("s", "a.py", "def f():\n    return 1\n")])
    batch = builder.to_batch(windows)
    with torch.no_grad():
        logits, _ = model(batch.input_ids)
        summed, count = objective.token_mean_loss(logits, batch.labels)
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            batch.labels.reshape(-1),
            ignore_index=objective.IGNORE_INDEX,
            reduction="sum",
        )
    assert count > 0
    assert torch.allclose(summed, manual, atol=1e-5)


def test_g4_padding_is_excluded_from_the_loss(tokenizer):
    model = _tiny_model(2)
    builder = SequenceBuilder(tokenizer, context_length=10, model_vocab=5822)
    windows = builder.build_windows([("s", "a.py", LONG_DOC)])
    batch = builder.to_batch(windows)
    with torch.no_grad():
        logits, _ = model(batch.input_ids)
        summed, count = objective.token_mean_loss(logits, batch.labels)
    supervised = int((batch.labels != objective.IGNORE_INDEX).sum())
    assert count == supervised
    assert count < batch.labels.numel()
    assert int((batch.attention_mask == 0).sum()) > 0


def test_g4_future_tokens_cannot_influence_earlier_logits(tokenizer):
    model = _tiny_model(3)
    model.eval()
    ids = torch.randint(0, 500, (1, 16))
    with torch.no_grad():
        base = model(ids)[0]
        changed = ids.clone()
        changed[0, -3:] = 7
        other = model(changed)[0]
    assert torch.allclose(base[0, :-3], other[0, :-3], atol=1e-5)
    assert not torch.allclose(base[0, -1], other[0, -1])


def test_g4_loss_is_token_mean_over_the_window():
    assert objective.accumulation_weights([10, 30]) == [0.25, 0.75]
    assert objective.mean_loss(4.0, 2) == 2.0
    assert objective.mean_loss(4.0, 0) == 0.0


def test_g4_padding_suffix_cannot_leak_into_valid_positions(tokenizer):
    """Suffix padding is safe under causal masking (no padding attention mask)."""
    model = _tiny_model(4)
    model.eval()
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    windows = builder.build_windows([("s", "a.py", "def f():\n    return 1\n")])
    batch = builder.to_batch(windows)
    valid = int(batch.attention_mask[0].sum())
    with torch.no_grad():
        padded_logits = model(batch.input_ids)[0]
        unpadded = model(batch.input_ids[:, :valid])[0]
    assert torch.allclose(
        padded_logits[0, :valid], unpadded[0, :valid], atol=1e-5
    )


# --------------------------------------------------------------------------- #
# G5 - optimizer correctness
# --------------------------------------------------------------------------- #
def test_g5_accumulation_equals_one_large_batch(tokenizer):
    """Two microbatches accumulated must equal a single batch of both."""
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    documents = [
        ("s1", "a.py", "def alpha():\n    return 1\n"),
        ("s2", "b.py", "def beta():\n    return 2\n"),
        ("s3", "c.py", "def gamma():\n    return 3\n"),
        ("s4", "d.py", "def delta():\n    return 4\n"),
    ]
    windows = builder.build_windows(documents)
    model = _tiny_model(5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = builder.to_batch(windows[:2])
    second = builder.to_batch(windows[2:])
    counts = [first.supervised_tokens, second.supervised_tokens]
    weights = objective.accumulation_weights(counts)
    optimizer.zero_grad(set_to_none=True)
    for batch, weight in zip((first, second), weights):
        logits, _ = model(batch.input_ids)
        summed, count = objective.token_mean_loss(logits, batch.labels)
        (summed / count * weight).backward()
    accumulated = [p.grad.detach().clone() for p in model.parameters()]

    model2 = _tiny_model(5)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    optimizer2.zero_grad(set_to_none=True)
    whole = builder.to_batch(windows)
    logits, _ = model2(whole.input_ids)
    summed, count = objective.token_mean_loss(logits, whole.labels)
    (summed / count).backward()
    single = [p.grad.detach().clone() for p in model2.parameters()]
    for left, right in zip(accumulated, single):
        assert torch.allclose(left, right, atol=1e-6)


def test_g5_gradient_clipping_bounds_the_norm(tokenizer):
    model = _tiny_model(6)
    builder = SequenceBuilder(tokenizer, context_length=32, model_vocab=5822)
    windows = builder.build_windows([("s", "a.py", "def f():\n    return 1\n")])
    batch = builder.to_batch(windows)
    logits, _ = model(batch.input_ids)
    summed, count = objective.token_mean_loss(logits, batch.labels)
    (summed / count).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    assert float(norm) > 0
    after = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
    )
    assert float(after) <= 0.5 + 1e-4


def test_g5_scheduler_warms_up_then_decays():
    schedule = WarmupCosineSchedule(
        peak_lr=1.0e-3, warmup_steps=4, total_steps=20, min_ratio=0.1
    )
    assert schedule.lr_at(0) == pytest.approx(1.0e-3 / 4)
    assert schedule.lr_at(3) == pytest.approx(1.0e-3)
    assert schedule.lr_at(4) == pytest.approx(1.0e-3)
    assert schedule.lr_at(19) < schedule.lr_at(4)
    assert schedule.lr_at(19) >= 1.0e-4 * 0.99


def test_g5_scheduler_state_roundtrips():
    schedule = WarmupCosineSchedule(
        peak_lr=2.0e-3, warmup_steps=2, total_steps=10, min_ratio=0.2
    )
    state = schedule.state_dict()
    other = WarmupCosineSchedule(peak_lr=1.0, warmup_steps=0, total_steps=1)
    other.load_state_dict(state)
    assert other.lr_at(3) == pytest.approx(schedule.lr_at(3))
    with pytest.raises(ValueError):
        other.load_state_dict({"type": "something_else"})


def test_g5_optimizer_step_counting_is_explicit():
    model = _tiny_model(7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = optimizer.state_dict()["state"]
    builder = SequenceBuilder(
        __import__("tokenizers").Tokenizer.from_file(str(TOKENIZER)),
        context_length=32,
        model_vocab=5822,
    )
    windows = builder.build_windows([("s", "a.py", "def f():\n    return 1\n")])
    batch = builder.to_batch(windows)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.input_ids)
        summed, count = objective.token_mean_loss(logits, batch.labels)
        (summed / count).backward()
        optimizer.step()
    after = optimizer.state_dict()["state"]
    assert len(after) == len(list(model.parameters()))
    assert all(int(state["step"].item()) == 3 for state in after.values())


# --------------------------------------------------------------------------- #
# G6 - deterministic sampling
# --------------------------------------------------------------------------- #
def test_g6_sampler_order_is_reproducible():
    first = DeterministicSampler(50, seed=11).order(0)
    second = DeterministicSampler(50, seed=11).order(0)
    assert first == second
    assert sorted(first) == list(range(50))


def test_g6_different_epochs_and_seeds_differ():
    assert DeterministicSampler(50, seed=11).order(0) != DeterministicSampler(
        50, seed=11
    ).order(1)
    assert DeterministicSampler(50, seed=11).order(0) != DeterministicSampler(
        50, seed=12
    ).order(0)


def test_g6_resume_replays_the_same_order():
    sampler = DeterministicSampler(20, seed=3)
    order = sampler.order(2)
    resumed = DeterministicSampler(20, seed=3)
    state = resumed.state_dict()
    resumed.load_state_dict(state)
    assert resumed.order(2) == order


def test_g6_window_stream_positions_are_stable(synthetic_corpus):
    dataset = ManifestDataset(synthetic_corpus["manifest"], "train", require_ready=False)
    builder = SequenceBuilder(
        __import__("tokenizers").Tokenizer.from_file(str(TOKENIZER)),
        context_length=64,
        model_vocab=5822,
    )
    windows = builder.build_windows(
        [(r["source_id"], r["file_path"], r["text"]) for r in dataset]
    )
    sampler = DeterministicSampler(len(windows), seed=5)
    stream = WindowStream(
        windows,
        sampler=sampler,
        batch_builder=builder.to_batch,
        micro_batch_size=2,
        grad_accum_steps=2,
    )
    first = [b.supervised_tokens for _, b in stream.microbatches(0)]
    second = [b.supervised_tokens for _, b in stream.microbatches(0)]
    assert first == second
    resumed = [b.supervised_tokens for _, b in stream.microbatches(0, start_position=2)]
    assert resumed == first[2:]


# --------------------------------------------------------------------------- #
# G8 - resource safety (declared before G7 for readability of the trainer tests)
# --------------------------------------------------------------------------- #
def test_g8_half_precision_on_cpu_is_refused():
    device = resolve_device("cpu")
    with pytest.raises(PrecisionError):
        resolve_precision(device, "float16")
    with pytest.raises(PrecisionError):
        resolve_precision(device, "bfloat16")
    assert resolve_precision(device, "float32") == torch.float32


def test_g8_unknown_device_is_refused():
    with pytest.raises(PrecisionError):
        resolve_device("tpu")


def test_g8_cuda_request_without_cuda_is_refused():
    import torch as _torch

    if _torch.cuda.is_available():
        pytest.skip("this host has CUDA; the refusal path is not reachable")
    with pytest.raises(PrecisionError):
        resolve_device("cuda")


def test_g8_invalid_configurations_are_rejected(tmp_path):
    with pytest.raises(TrainConfigError):
        load_train_config(tmp_path / "missing.yaml")
    cfg = {
        "schema": TRAIN_CONFIG_SCHEMA,
        "mode": "production",
        "run": {"name": "x", "model_config": "m.yaml", "seed": 1},
        "data": {
            "manifest": "m.json",
            "tokenizer": "t.json",
            "context_length": 8,
            "sequence_policy": "document",
        },
        "optimizer": {"type": "adamw", "lr_peak": 1e-3, "grad_clip": 1.0},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
        "precision": {"device": "cpu", "dtype": "float32"},
        "training": {
            "max_steps": 1,
            "max_tokens": 10,
            "max_wall_seconds": 1.0,
            "output_dir": "out",
        },
        "limits": {"max_checkpoints": 1, "max_output_bytes": 10},
    }
    assert validate_train_config(cfg) == []
    absolute = json.loads(json.dumps(cfg))
    absolute["data"]["manifest"] = "C:/absolute/manifest.json"
    assert any("absolute" in problem for problem in validate_train_config(absolute))
    bad_mode = json.loads(json.dumps(cfg))
    bad_mode["mode"] = "production-with-synthetic-data"
    assert any("mode must be" in problem for problem in validate_train_config(bad_mode))
    unbounded = json.loads(json.dumps(cfg))
    unbounded["training"]["max_steps"] = 0
    assert any("max_steps" in problem for problem in validate_train_config(unbounded))
    bad_dtype = json.loads(json.dumps(cfg))
    bad_dtype["precision"]["dtype"] = "int8"
    assert any("dtype" in problem for problem in validate_train_config(bad_dtype))


def test_g8_unbounded_training_is_refused_by_the_config_validator(tmp_path):
    cfg = {
        "schema": TRAIN_CONFIG_SCHEMA,
        "mode": "synthetic",
        "run": {"name": "x", "model_config": "m.yaml", "seed": 1},
        "data": {
            "manifest": "m.json",
            "tokenizer": "t.json",
            "context_length": 8,
            "sequence_policy": "document",
        },
        "optimizer": {"type": "adamw", "lr_peak": 1e-3, "grad_clip": 1.0},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
        "precision": {"device": "cpu", "dtype": "float32"},
        "training": {
            "max_steps": 10,
            "max_tokens": 0,
            "max_wall_seconds": 1.0,
            "output_dir": "out",
        },
        "limits": {"max_checkpoints": 1, "max_output_bytes": 10},
    }
    assert any("max_tokens" in problem for problem in validate_train_config(cfg))


# --------------------------------------------------------------------------- #
# G7 - checkpoint and resume
# --------------------------------------------------------------------------- #
def _run(config_path, **kwargs):
    from training.train import run_training

    return run_training(config_path, **kwargs)


def test_g7_checkpoint_records_the_full_resume_state(synthetic_corpus, tmp_path):
    config = _train_config(
        synthetic_corpus["root"],
        synthetic_corpus["manifest"],
        model_config=synthetic_corpus["model_config"],
        max_steps=2,
    )
    _run(config)
    checkpoint = torch.load(
        synthetic_corpus["root"] / "out" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["schema"] == "quantsmind.checkpoint/2"
    for key in (
        "model",
        "optimizer",
        "step",
        "scheduler",
        "rng_state",
        "sampler_state",
        "consumed_tokens",
        "dataset_identity",
        "tokenizer_identity",
        "train_config_sha256",
        "mode",
        "synthetic",
    ):
        assert key in checkpoint, key
    provenance = checkpoint["provenance"]
    assert provenance["contract_id"]
    assert provenance["synthetic"] is True
    assert provenance["mode"] == "synthetic"
    assert provenance["dataset_manifest_sha256"]
    assert provenance["train_config_sha256"]


def test_g7_resume_matches_an_uninterrupted_run(synthetic_corpus, tmp_path):
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    _write_model_config(full_dir)
    corpus = _build_synthetic_corpus(full_dir, sources=4, modules=1)
    config = _train_config(
        full_dir, corpus, model_config=full_dir / "tiny_model.yaml", max_steps=4
    )
    _run(config)
    uninterrupted = torch.load(
        full_dir / "out" / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    _write_model_config(split_dir)
    corpus2 = _build_synthetic_corpus(split_dir, sources=4, modules=1)
    config2 = _train_config(
        split_dir, corpus2, model_config=split_dir / "tiny_model.yaml", max_steps=4
    )
    _run(config2, max_steps_override=2)
    _run(
        config2,
        resume_from=str(split_dir / "out" / "checkpoint_last.pt"),
        max_steps_override=4,
    )
    resumed = torch.load(
        split_dir / "out" / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    assert resumed["step"] == uninterrupted["step"]
    for left, right in zip(
        uninterrupted["model"].values(), resumed["model"].values()
    ):
        assert torch.allclose(left, right, atol=1e-6)


def test_g7_resume_under_a_different_configuration_is_refused(synthetic_corpus):
    config = _train_config(
        synthetic_corpus["root"],
        synthetic_corpus["manifest"],
        model_config=synthetic_corpus["model_config"],
        max_steps=2,
    )
    _run(config)
    other = _train_config(
        synthetic_corpus["root"],
        synthetic_corpus["manifest"],
        model_config=synthetic_corpus["model_config"],
        max_steps=2,
        optimizer={"lr_peak": 9.0e-3},
    )
    with pytest.raises(TrainConfigError) as excinfo:
        _run(other, resume_from=str(synthetic_corpus["root"] / "out" / "checkpoint_last.pt"))
    assert "different training configuration" in str(excinfo.value)


def test_g7_resume_against_a_different_dataset_is_refused(synthetic_corpus, tmp_path):
    config = _train_config(
        synthetic_corpus["root"],
        synthetic_corpus["manifest"],
        model_config=synthetic_corpus["model_config"],
        max_steps=2,
    )
    _run(config)
    other_root = tmp_path / "other"
    other_root.mkdir()
    _write_model_config(other_root)
    other_manifest = _build_synthetic_corpus(other_root, sources=4, modules=1)
    other = _train_config(
        other_root, other_manifest, model_config=other_root / "tiny_model.yaml", max_steps=2
    )
    with pytest.raises(TrainConfigError) as excinfo:
        _run(other, resume_from=str(synthetic_corpus["root"] / "out" / "checkpoint_last.pt"))
    assert "different dataset" in str(excinfo.value)


def test_g7_corrupt_checkpoint_is_refused(synthetic_corpus, tmp_path):
    config = _train_config(
        synthetic_corpus["root"],
        synthetic_corpus["manifest"],
        model_config=synthetic_corpus["model_config"],
        max_steps=2,
    )
    _run(config)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        _run(config, resume_from=str(corrupt))


# --------------------------------------------------------------------------- #
# G9 - bounded synthetic learning demonstration
# --------------------------------------------------------------------------- #
def test_g9_bounded_run_reduces_the_loss(synthetic_corpus, tmp_path):
    run_root = tmp_path / "learn"
    run_root.mkdir()
    _write_model_config(run_root)
    corpus = _build_synthetic_corpus(run_root, sources=6, modules=2)
    config = _train_config(
        run_root, corpus, model_config=run_root / "tiny_model.yaml", max_steps=8
    )
    manifest = _run(config)
    metrics = manifest["training"]["metrics"]
    assert metrics["final_train_loss"] < metrics["initial_train_loss"]
    assert metrics["initial_train_loss"] > 0
    evaluation = json.loads(
        (run_root / "out" / "evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["synthetic"] is True
    assert evaluation["corpus_ready"] is False
    assert "does not establish" in evaluation["evidence_scope"]


def test_g9_run_manifest_and_telemetry_are_written(synthetic_corpus, tmp_path):
    run_root = tmp_path / "telemetry"
    run_root.mkdir()
    _write_model_config(run_root)
    corpus = _build_synthetic_corpus(run_root, sources=4, modules=1)
    config = _train_config(
        run_root, corpus, model_config=run_root / "tiny_model.yaml", max_steps=2
    )
    _run(config)
    run_manifest = json.loads(
        (run_root / "out" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["schema"] == "quantsmind.run_manifest/1"
    assert run_manifest["training"]["mode"] == "synthetic"
    assert run_manifest["checkpoint"]["format"] == "quantsmind.checkpoint/2"
    log_lines = [
        json.loads(line)
        for line in (run_root / "out" / "loss_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in log_lines]
    assert "run_start" in events
    assert "step" in events
    assert "run_end" in events


def test_g9_max_steps_limit_is_enforced(synthetic_corpus, tmp_path):
    run_root = tmp_path / "bounded"
    run_root.mkdir()
    _write_model_config(run_root)
    corpus = _build_synthetic_corpus(run_root, sources=4, modules=1)
    config = _train_config(
        run_root, corpus, model_config=run_root / "tiny_model.yaml", max_steps=2
    )
    manifest = _run(config)
    assert manifest["training"]["steps_completed"] == 2
    assert manifest["status"] == "complete"


def test_g9_token_budget_limit_is_enforced(synthetic_corpus, tmp_path):
    run_root = tmp_path / "tokens"
    run_root.mkdir()
    _write_model_config(run_root)
    corpus = _build_synthetic_corpus(run_root, sources=4, modules=1)
    config = _train_config(
        run_root,
        corpus,
        model_config=run_root / "tiny_model.yaml",
        max_steps=50,
        training={"max_tokens": 200},
    )
    manifest = _run(config)
    assert manifest["training"]["tokens_processed"] >= 200
    assert manifest["training"]["steps_completed"] < 50
    assert manifest["training"]["stopping_criteria"]["stop_reason"] == "max_tokens"


def test_g10_v110_checkpoint_schema_still_validates(synthetic_corpus):
    from training.checkpoint_schema import SCHEMA_V2, validate_provenance

    assert SCHEMA_V2 == "quantsmind.checkpoint/2"
    problems = validate_provenance({"contract_id": "x"})
    assert problems
    assert any("missing field" in problem for problem in problems)
