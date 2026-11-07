"""Tests for training-loop correctness: B1/B2/B3/B4/B6 and loss retention.

Everything here is synthetic. No model is trained, no checkpoint from the
repository is loaded for weight values, and no network is touched.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from training.checkpoint import (
    CheckpointIncompatible,
    init_from_checkpoint,
    load_model_weights,
)
from training.document_separator import (
    DOC_SEPARATOR_NAME,
    SeparatorUnavailable,
    encode_records,
    resolve_separator_id,
)
from training.loss_log import LossLogger, read_log
from training.sft import SFTBatcher


class TinyModel(torch.nn.Module):
    """Two-parameter model: enough to test compatibility without training."""

    def __init__(self, width: int = 4):
        super().__init__()
        self.a = torch.nn.Linear(width, width, bias=True)
        self.b = torch.nn.Linear(width, width, bias=False)


# --------------------------------------------------------------------------- #
# A1 / B1 / B2 -- SFT dataset consumption
# --------------------------------------------------------------------------- #
def test_legacy_batch_pattern_never_left_the_first_micro_bs():
    """Establishes the actual defect rather than assuming the 0.5% figure."""
    dataset = list(range(442))
    micro_bs = 2
    seen = set()
    for _ in range(200):  # 200 steps x 2 accum = 400 example slots
        batch = [dataset[i % len(dataset)] for i in range(micro_bs)]
        seen.update(batch)
    assert seen == {0, 1}, "legacy pattern must be shown to be broken"
    assert len(seen) / len(dataset) < 0.01


def test_batcher_covers_the_whole_dataset():
    dataset_size = 442
    batcher = SFTBatcher(dataset_size, batch_size=2, seed=7)
    for _ in range(221):  # exactly one epoch worth of micro-batches
        batcher.next_indices()
    assert batcher.unique_coverage == 1.0
    assert batcher.consumed == dataset_size
    assert batcher.seen == set(range(dataset_size))


def test_batcher_cycles_epochs_and_reshuffles():
    batcher = SFTBatcher(8, batch_size=2, seed=3)
    first = [tuple(batcher.next_indices()) for _ in range(4)]
    second = [tuple(batcher.next_indices()) for _ in range(4)]
    assert sorted(i for b in first for i in b) == list(range(8))
    assert sorted(i for b in second for i in b) == list(range(8))
    assert first != second, "a new epoch must reshuffle"
    assert batcher.epochs_completed >= 1
    assert batcher.refills >= 2


def test_batcher_refills_mid_batch_without_dropping_examples():
    batcher = SFTBatcher(5, batch_size=4, seed=1)
    indices = []
    for _ in range(3):
        indices.extend(batcher.next_indices())
    assert len(indices) == 12
    assert len(set(indices)) == 5, "every example must appear once in a full cycle"
    assert set(indices) == set(range(5))


def test_batcher_is_deterministic_for_a_fixed_seed():
    a = SFTBatcher(20, batch_size=3, seed=42)
    b = SFTBatcher(20, batch_size=3, seed=42)
    assert [a.next_indices() for _ in range(6)] == [
        b.next_indices() for _ in range(6)
    ]
    c = SFTBatcher(20, batch_size=3, seed=43)
    assert [c.next_indices() for _ in range(6)] != [a.next_indices() for _ in range(6)]


def test_batcher_rejects_empty_dataset():
    with pytest.raises(ValueError):
        SFTBatcher(0, batch_size=2)


def test_batcher_coverage_report_is_observable():
    batcher = SFTBatcher(10, batch_size=2, seed=0)
    for _ in range(5):
        batcher.next_indices()
    report = batcher.coverage_report()
    assert report["dataset_size"] == 10
    assert report["consumed_examples"] == 10
    assert report["unique_examples_seen"] == 10
    assert report["unique_coverage"] == 1.0
    assert report["epochs_consumed"] == 1.0


def test_partial_schedule_reports_partial_coverage():
    batcher = SFTBatcher(100, batch_size=2, seed=0)
    for _ in range(5):
        batcher.next_indices()
    report = batcher.coverage_report()
    assert report["unique_coverage"] == pytest.approx(0.10)
    assert report["unique_coverage"] < 1.0


def test_sft_module_imports_random():
    """B2: SFTBatcher uses random.Random but the module never imported random."""
    import random as random_mod
    import training.sft as sft_mod

    assert sft_mod.random is random_mod


# --------------------------------------------------------------------------- #
# A2 / B3 -- checkpoint compatibility
# --------------------------------------------------------------------------- #
def test_compatible_checkpoint_loads_cleanly():
    src, dst = TinyModel(), TinyModel()
    report = load_model_weights(dst, src.state_dict(), source="compatible")
    assert report.ok
    assert not report.missing and not report.unexpected and not report.mismatched
    for name, tensor in src.state_dict().items():
        assert torch.equal(dst.state_dict()[name], tensor)


def test_missing_required_parameters_are_rejected_by_default():
    src, dst = TinyModel(), TinyModel()
    partial = {k: v for k, v in src.state_dict().items() if k != "b.weight"}
    with pytest.raises(CheckpointIncompatible) as exc:
        load_model_weights(dst, partial, source="missing-keys")
    assert "b.weight" in str(exc.value)
    assert "random" in str(exc.value).lower()


def test_missing_parameters_are_allowed_only_with_explicit_partial_flag():
    src, dst = TinyModel(), TinyModel()
    partial = {k: v for k, v in src.state_dict().items() if k != "b.weight"}
    report = load_model_weights(dst, partial, allow_partial=True, source="partial")
    assert report.partial is True
    assert report.missing == ["b.weight"]
    assert "b.weight" not in report.unexpected
    # The provided weights really were applied.
    assert torch.equal(dst.state_dict()["a.weight"], src.state_dict()["a.weight"])


def test_unexpected_parameters_are_rejected():
    src, dst = TinyModel(), TinyModel()
    state = dict(src.state_dict())
    state["mystery.weight"] = torch.zeros(2, 2)
    with pytest.raises(CheckpointIncompatible) as exc:
        load_model_weights(dst, state, source="extra-keys")
    assert "mystery.weight" in str(exc.value)


def test_shape_mismatch_is_rejected_even_in_partial_mode():
    src = TinyModel()
    dst = TinyModel()
    state = dict(src.state_dict())
    state["a.weight"] = torch.zeros(9, 9)  # right name, wrong geometry
    with pytest.raises(CheckpointIncompatible) as exc:
        load_model_weights(dst, state, allow_partial=True, source="bad-shape")
    assert "MISMATCH" in str(exc.value)
    assert "never tolerated" in str(exc.value)


def test_checkpoint_file_without_model_key_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"step": 3, "optimizer": {}}, path)
    with pytest.raises(CheckpointIncompatible) as exc:
        init_from_checkpoint(TinyModel(), str(path))
    assert "no 'model' entry" in str(exc.value)


def test_missing_checkpoint_file_is_reported_clearly(tmp_path):
    with pytest.raises(CheckpointIncompatible) as exc:
        init_from_checkpoint(TinyModel(), str(tmp_path / "absent.pt"))
    assert "not found" in str(exc.value)


def test_bare_state_dict_is_accepted(tmp_path):
    src = TinyModel()
    path = tmp_path / "bare.pt"
    torch.save(src.state_dict(), path)
    dst = TinyModel()
    report = init_from_checkpoint(dst, str(path))
    assert report.ok


def test_report_is_serialisable():
    src, dst = TinyModel(), TinyModel()
    state = dict(src.state_dict())
    state.pop("b.weight")
    report = load_model_weights(dst, state, allow_partial=True)
    blob = json.dumps(report.as_dict())
    assert "b.weight" in blob


# --------------------------------------------------------------------------- #
# A3 / B6 -- document boundaries
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    """Minimal tokenizer stand-in: maps text to a deterministic id list."""

    def __init__(self, special_ids=None):
        # Explicit empty dict must stay empty; only None means "use defaults".
        self.special_ids = (
            {DOC_SEPARATOR_NAME: 2} if special_ids is None else special_ids
        )
        self._vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def token_to_id(self, name):
        return self.special_ids.get(name)

    def encode(self, text):
        if not text:
            return type("Enc", (), {"ids": []})()
        ids = [10 + (ord(c) % 50) for c in text]
        return type("Enc", (), {"ids": ids})()


def test_adjacent_documents_are_separated_exactly_once():
    tok = FakeTokenizer()
    ids, emitted = encode_records(tok, ["alpha", "beta", "gamma"])
    sep = resolve_separator_id(tok)
    assert emitted == 3
    assert ids.count(sep) == 2, "n documents need n-1 separators"
    assert ids[0] != sep
    assert ids[-1] != sep


def test_no_duplicate_or_empty_records_create_double_separators():
    tok = FakeTokenizer()
    sep = resolve_separator_id(tok)
    ids, emitted = encode_records(tok, ["alpha", "", "beta"])
    assert emitted == 2, "an empty record must not count as a document"
    assert ids.count(sep) == 1
    assert sep + sep not in [
        tuple(ids[i : i + 2]) for i in range(len(ids) - 1)
    ]


def test_token_budget_truncates_at_document_granularity():
    """A budget cut must never split a document in half."""
    tok = FakeTokenizer()
    first_doc = encode_records(tok, ["one"])[0]
    both_docs = encode_records(tok, ["one", "two"])[0]
    # A budget with room for the first document plus a separator, but not for
    # the second document's content.
    budget = len(first_doc) + 1
    assert budget < len(both_docs)
    ids, emitted = encode_records(tok, ["one", "two"], max_tokens=budget)
    assert emitted == 1, "the second document must be dropped whole, not half-emitted"
    assert ids == first_doc


def test_single_document_larger_than_the_budget_is_truncated():
    tok = FakeTokenizer()
    doc = "x" * 50
    ids, emitted = encode_records(tok, [doc], max_tokens=10)
    assert emitted == 1
    assert len(ids) == 10, "with nothing whole to fall back on, the budget wins"


def test_budget_never_leaves_a_trailing_separator():
    tok = FakeTokenizer()
    ids, _ = encode_records(tok, ["aa", "bb", "cc"], max_tokens=7)
    sep = resolve_separator_id(tok)
    assert ids[-1] != sep


def test_tokenizer_without_separator_is_refused_not_faked():
    tok = FakeTokenizer(special_ids={})
    with pytest.raises(SeparatorUnavailable) as exc:
        resolve_separator_id(tok)
    assert DOC_SEPARATOR_NAME in str(exc.value)


def test_repository_tokenizer_defines_the_separator():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    assert resolve_separator_id(tok) == tok.token_to_id(DOC_SEPARATOR_NAME)
    assert resolve_separator_id(tok) == 2


def test_encoder_writes_separator_metadata(tmp_path):
    """The meta file must state the separator so the stream is self-describing."""
    import sys

    import numpy as np
    from tokenizers import Tokenizer

    from data.encode_dataset import iter_records, main as encode_main

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"text": "def a():\\n    return 1\\n"}\n'
        '{"text": "def b():\\n    return 2\\n"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "stream.bin"
    argv = sys.argv
    sys.argv = [
        "encode_dataset",
        "--corpus", str(corpus),
        "--tokenizer", "tokenizer/qm_code_bpe_v0.json",
        "--out", str(out),
    ]
    try:
        encode_main()
    finally:
        sys.argv = argv

    meta = json.loads((tmp_path / "stream.meta.json").read_text(encoding="utf-8"))
    assert meta["document_separator"] == DOC_SEPARATOR_NAME
    assert meta["num_documents"] == 2
    ids = np.fromfile(out, dtype=np.uint16)
    assert int((ids == 2).sum()) == 1
    assert list(iter_records(corpus))


def test_encoder_records_the_oversized_document_policy(tmp_path):
    import sys

    from data.encode_dataset import main as encode_main

    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "a"}\n', encoding="utf-8")
    argv = sys.argv
    sys.argv = [
        "encode_dataset",
        "--corpus", str(corpus),
        "--tokenizer", "tokenizer/qm_code_bpe_v0.json",
        "--out", str(tmp_path / "s.bin"),
    ]
    try:
        encode_main()
    finally:
        sys.argv = argv
    meta = json.loads((tmp_path / "s.meta.json").read_text(encoding="utf-8"))
    assert "never truncated" in meta["oversized_document_policy"]


# --------------------------------------------------------------------------- #
# A5 -- loss log retention
# --------------------------------------------------------------------------- #
def test_loss_log_persists_synthetic_values(tmp_path):
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(
        path, run_id="runA", run_kind="pretrain", dataset="d.bin", config={"a": 1}
    )
    logger.log_header(total_steps=3)
    logger.log_step(0, 9.5, lr=1e-3, ema_loss=9.4)
    logger.log_step(1, 4.25, lr=1e-3, ema_loss=6.7)
    logger.close(status="complete")

    records = read_log(path)
    assert [r["event"] for r in records] == ["run_start", "step", "step", "run_end"]
    losses = [r["loss"] for r in records if r["event"] == "step"]
    assert losses == [9.5, 4.25]
    assert all(r["run_id"] == "runA" for r in records)
    assert records[0]["dataset"] == "d.bin"
    assert records[0]["config_fingerprint"]
    assert records[-1]["status"] == "complete"


def test_loss_log_appends_rather_than_overwriting(tmp_path):
    """A second run must not destroy the first run's evidence."""
    path = tmp_path / "loss_log.jsonl"
    first = LossLogger(path, run_id="run1", config={"x": 1})
    first.log_step(0, 1.0)
    first.close()

    second = LossLogger(path, run_id="run2", config={"x": 2})
    second.log_step(0, 2.0)
    second.close()

    records = read_log(path)
    run_ids = {r["run_id"] for r in records}
    assert run_ids == {"run1", "run2"}
    assert [r["loss"] for r in records if r["event"] == "step"] == [1.0, 2.0]


def test_loss_log_marks_interrupted_runs(tmp_path):
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path, run_id="runX")
    logger.log_step(0, 5.0)
    with pytest.raises(RuntimeError):
        with logger:
            logger.log_step(1, 4.0)
            raise RuntimeError("simulated crash")
    records = read_log(path)
    assert records[-1]["event"] == "run_end"
    assert records[-1]["status"] == "interrupted"
    # Everything logged before the crash survives.
    assert [r["loss"] for r in records if r["event"] == "step"] == [5.0, 4.0]


def test_loss_log_records_are_flushed_immediately(tmp_path):
    """An interrupted run must still have its records on disk."""
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path, run_id="runF")
    logger.log_step(0, 3.3)
    # Read without closing: the record must already be there.
    assert [r["loss"] for r in read_log(path) if r["event"] == "step"] == [3.3]
    logger.close()


def test_validation_loss_is_tagged_separately(tmp_path):
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path, run_id="runV")
    logger.log_validation(10, 4.44)
    logger.close()
    rec = [r for r in read_log(path) if r["event"] == "step"][0]
    assert rec["phase"] == "validation"
    assert rec["loss"] == 4.44


def test_loss_log_open_failure_surfaces(tmp_path):
    """A log that cannot be opened must stop the run, not vanish silently."""
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        LossLogger(blocker / "loss_log.jsonl", run_id="runE")
    assert "cannot be opened" in str(exc.value)


def test_loss_log_write_failure_surfaces(tmp_path, monkeypatch):
    """A failure mid-run must surface rather than silently dropping records."""
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path, run_id="runE")
    logger.log_step(0, 1.0)

    boom = OSError("disk full")

    def explode(*args, **kwargs):
        raise boom

    monkeypatch.setattr(logger._fh, "write", explode)
    with pytest.raises(RuntimeError) as exc:
        logger.log_step(1, 2.0)
    assert "loss log write failed" in str(exc.value)


def test_disabled_logger_writes_nothing(tmp_path):
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path, run_id="off", enabled=False)
    logger.log_step(0, 1.0)
    logger.close()
    assert not path.exists()


# --------------------------------------------------------------------------- #
# A4 / B4 -- configuration validation and accounting
# --------------------------------------------------------------------------- #
def _valid_cfg():
    return {
        "run": {"model_config": "configs/model/code_tiny.yaml", "seed": 42},
        "data": {"dataset_tag": "corpus-v0", "seq_len": 512},
        "batch": {"micro_batch_size": 2, "grad_accum_steps": 2},
        "optimizer": {"lr_peak": 1e-3},
        "training": {"max_steps": 10},
    }


def test_valid_config_has_no_problems():
    from training.pretrain import validate_train_config

    assert validate_train_config(_valid_cfg()) == []


def test_missing_max_steps_is_reported_actionably():
    """B4: pretrain_pilot.yaml defined only max_tokens, so training raised KeyError."""
    from training.pretrain import validate_train_config

    cfg = _valid_cfg()
    cfg["training"] = {"max_tokens": 1.2e10}
    problems = validate_train_config(cfg)
    assert any("max_steps" in p for p in problems)
    assert any("max_tokens" in p for p in problems), "message should name the wrong key"


def test_auto_micro_batch_is_reported():
    from training.pretrain import validate_train_config

    cfg = _valid_cfg()
    cfg["batch"]["micro_batch_size"] = "auto"
    problems = validate_train_config(cfg)
    assert any("micro_batch_size" in p and "auto" in p for p in problems)


def test_grad_accum_under_precision_is_flagged_as_unread():
    """B4: grad_accum_steps lived under `precision` and was never read."""
    from training.pretrain import unused_config_keys, validate_train_config

    cfg = _valid_cfg()
    cfg["precision"] = {"dtype": "bfloat16", "grad_accum_steps": "auto"}
    assert "precision.grad_accum_steps" in unused_config_keys(cfg)
    # And the value under `batch` is the one that must be a real int.
    assert validate_train_config(cfg) == []


def test_missing_token_source_is_reported():
    from training.pretrain import validate_train_config

    cfg = _valid_cfg()
    cfg["data"] = {"seq_len": 512}
    problems = validate_train_config(cfg)
    assert any("tokens_path" in p for p in problems)


def test_declared_dead_keys_are_reported():
    from training.pretrain import unused_config_keys

    cfg = _valid_cfg()
    cfg["data"]["pack_documents"] = True
    cfg["watchdog"] = {"loss_spike_sigma": 3.0, "spike_action": "rollback"}
    found = unused_config_keys(cfg)
    assert "data.pack_documents" in found
    assert "watchdog.loss_spike_sigma" in found
    assert "watchdog.spike_action" in found


def test_implemented_keys_are_no_longer_reported_as_dead():
    """M3 implemented the validation loop, so these keys now have an effect.

    This test previously asserted the opposite: that ``eval_every_steps`` was
    reported as unread. That was true when the loop had no validation pass. It
    is now false, and a config that sets it really does trigger validation, so
    reporting it as dead would be wrong.
    """
    from training.pretrain import unused_config_keys

    cfg = _valid_cfg()
    cfg["training"]["eval_every_steps"] = 500
    found = unused_config_keys(cfg)
    assert "training.eval_every_steps" not in found


@pytest.mark.parametrize(
    "name",
    [
        "code_smoke.yaml",
        "sft_smoke.yaml",
        "code_local.yaml",
        "local_15m.yaml",
        "corporate_ft.yaml",
        "pretrain_pilot.yaml",
        "smoke_toy.yaml",
    ],
)
def test_shipped_text_configs_are_runnable(name):
    """B4: every config the text loop consumes must validate cleanly."""
    from pathlib import Path

    from training.pretrain import load_yaml, unused_config_keys, validate_train_config

    root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(str(root / "configs" / "train" / name))
    assert validate_train_config(cfg) == []
    assert unused_config_keys(cfg) == []


def test_checkpoint_metadata_carries_progress(tmp_path):
    """A checkpoint must identify its configuration and progress."""
    from training.pretrain import save_checkpoint

    model = TinyModel()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model,
        optim,
        7,
        {"run": {"name": "unit-test"}},
        extra={"consumed_examples": 128, "dataset_size": 442},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "quantsmind.checkpoint/1"
    assert payload["step"] == 7
    assert payload["consumed_examples"] == 128
    assert payload["dataset_size"] == 442
    assert payload["config"]["run"]["name"] == "unit-test"