"""End-to-end training smoke test (Workstream C).

Drives the **real** entry points -- ``training.pretrain.main`` and
``training.sft.main`` -- over a synthetic corpus encoded by the real
``data.encode_dataset`` pipeline, with the real tokenizer, the real loss log and
the real checkpoint format. Nothing is mocked; only data and model size shrink.

No network, no downloads, no production corpus. Every artifact is written under
pytest's temporary directory.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from data.encode_dataset import main as encode_main
from training.loss_log import read_log
from training.sft import main as sft_main

CODE_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"
VOCAB = 5822
SEQ_LEN = 64


def _run(argv, module_main):
    import sys

    saved = sys.argv
    sys.argv = ["smoke", *argv]
    try:
        return module_main()
    finally:
        sys.argv = saved


def _model_cfg_dict():
    return {
        "name": "qm-smoke",
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 64,
        "max_seq_len": 128,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    }


def _write_model_config(path: Path):
    path.write_text(yaml.safe_dump({"model": _model_cfg_dict()}), encoding="utf-8")
    return path


def _synthetic_corpus(path: Path, docs=40):
    with path.open("w", encoding="utf-8") as f:
        for i in range(docs):
            block = (
                f'def helper_{i}(value):\n'
                f'    """Docstring for helper {i}."""\n'
                f"    return value + {i}\n"
            )
            f.write(json.dumps({"text": block, "id": i}) + "\n")
    return path


@pytest.fixture(scope="module")
def encoded_corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke_data")
    corpus = _synthetic_corpus(tmp / "corpus.jsonl")
    out = tmp / "corpus.tokens.bin"
    _run(
        ["--corpus", str(corpus), "--tokenizer", CODE_TOKENIZER, "--out", str(out)],
        encode_main,
    )
    assert out.exists()
    return tmp, corpus, out


def _pretrain_cfg(path, model_cfg, tokens, steps=3):
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "smoke-pretrain", "model_config": str(model_cfg),
                        "seed": 42},
                "data": {"tokens_path": str(tokens), "seq_len": SEQ_LEN},
                "batch": {"micro_batch_size": 2, "grad_accum_steps": 1},
                "optimizer": {"lr_peak": 1e-3, "warmup_steps": 1, "grad_clip": 1.0},
                "training": {"max_steps": steps, "checkpoint_every_steps": steps,
                             "log_every_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pretrain_end_to_end(tmp_path, encoded_corpus):
    """Config -> data -> forward/backward -> optimizer -> checkpoint -> log."""
    import training.pretrain as pretrain

    data_dir, _corpus, tokens = encoded_corpus
    model_cfg = _write_model_config(data_dir / "model.yaml")
    cfg = _pretrain_cfg(tmp_path / "pretrain.yaml", model_cfg, tokens)
    out_dir = tmp_path / "runs" / "pretrain"

    # main() returns None by design; success is evidenced by the artifacts below.
    _run(["--config", str(cfg), "--out-dir", str(out_dir)], pretrain.main)

    records = read_log(out_dir / "loss_log.jsonl")
    header = [r for r in records if r["event"] == "run_start"]
    steps = [r for r in records if r["event"] == "step"]
    end = [r for r in records if r["event"] == "run_end"]

    assert len(header) == 1
    assert header[0]["run_kind"] == "pretrain"
    assert header[0]["model_config"] == str(model_cfg)
    assert header[0]["dataset"] == str(tokens)
    assert len(steps) >= 3, "every step must be logged"
    for record in steps:
        assert isinstance(record["loss"], float)
        assert record["loss"] == record["loss"], "loss must not be NaN"
        assert record["tokens_per_second"] > 0
    assert end and end[0]["status"] == "complete"
    assert end[0]["optimizer_steps_completed"] == 3
    assert end[0]["total_tokens_processed"] == 3 * 2 * SEQ_LEN

    ckpt_path = out_dir / "checkpoint_last.pt"
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "quantsmind.checkpoint/1"
    assert payload["step"] == 3
    assert payload["seq_len"] == SEQ_LEN
    assert payload["tokens_per_step"] == 2 * SEQ_LEN
    assert payload["optimizer_steps_completed"] == 3

    from training.checkpoint import init_from_checkpoint
    from training.model import QMModel

    model = QMModel(_model_cfg_dict())
    report = init_from_checkpoint(model, str(ckpt_path))
    assert report.ok, report.describe()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload["model"][name])


def test_gradients_reach_the_weights(tmp_path, encoded_corpus):
    """Real forward/backward/optimizer on a fixed batch; loss must fall."""
    from training.data import TokenDataset
    from training.model import QMModel

    _data_dir, _corpus, tokens = encoded_corpus
    torch.manual_seed(0)
    model = QMModel(_model_cfg_dict())
    dataset = TokenDataset(str(tokens), seed=0)
    x, y = dataset.batch(2, SEQ_LEN)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(15):
        optim.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        assert torch.isfinite(loss), "loss went non-finite"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"


def test_pretrain_resume_appends_a_new_run(tmp_path, encoded_corpus):
    """Resume reloads weights and starts a distinct, appended run."""
    import training.pretrain as pretrain

    data_dir, _corpus, tokens = encoded_corpus
    model_cfg = _write_model_config(data_dir / "model.yaml")
    cfg = _pretrain_cfg(tmp_path / "resume.yaml", model_cfg, tokens, steps=2)
    out_dir = tmp_path / "runs" / "resume"
    argv = ["--config", str(cfg), "--out-dir", str(out_dir)]

    _run(argv, pretrain.main)
    first = read_log(out_dir / "loss_log.jsonl")
    first_runs = {r["run_id"] for r in first}

    _run([*argv, "--resume"], pretrain.main)
    second = read_log(out_dir / "loss_log.jsonl")

    assert len(second) > len(first)
    assert first_runs <= {r["run_id"] for r in second}, "history must be preserved"
    assert len({r["run_id"] for r in second}) == 2


def _sft_cfg(path, model_cfg, steps=3, accum=2, micro=2):
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "smoke-sft", "model_config": str(model_cfg),
                        "seed": 42},
                "data": {"seq_len": 256},
                "batch": {"micro_batch_size": micro, "grad_accum_steps": accum},
                "optimizer": {"lr_peak": 1e-3, "warmup_steps": 1, "grad_clip": 1.0},
                "training": {"max_steps": steps, "checkpoint_every_steps": steps,
                             "log_every_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def _sft_pairs(path, n=12):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "instruction": f"Write helper {i}",
                        "response": f"def helper_{i}(v):\n    return v + {i}\n",
                    }
                )
                + "\n"
            )
    return path


def test_sft_end_to_end(tmp_path):
    """SFT path: dataset -> masked labels -> training -> coverage artifact."""
    pairs = _sft_pairs(tmp_path / "pairs.jsonl", n=12)
    model_cfg = _write_model_config(tmp_path / "sft_model.yaml")
    cfg = _sft_cfg(tmp_path / "sft.yaml", model_cfg)
    out_dir = tmp_path / "runs" / "sft"

    _run(
        ["--config", str(cfg), "--sft-data", str(pairs), "--out-dir", str(out_dir)],
        sft_main,
    )

    coverage = json.loads((out_dir / "sft_coverage.json").read_text(encoding="utf-8"))
    assert coverage["eligible_examples"] == 12
    assert coverage["dataset_size"] == 12
    assert coverage["examples_per_step"] == 4
    assert coverage["consumed_examples"] == 3 * 4
    assert coverage["unique_coverage"] == 1.0
    assert coverage["optimizer_steps_completed"] == 3

    records = read_log(out_dir / "loss_log.jsonl")
    start = [r for r in records if r["event"] == "run_start"]
    assert start and start[0]["run_kind"] == "sft"
    assert {r["run_id"] for r in records} == {start[0]["run_id"]}
    assert {r.get("phase") for r in records if r["event"] == "step"} == {"train"}
    assert [r for r in records if r["event"] == "run_end"][0]["status"] == "complete"

    payload = torch.load(
        out_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    assert payload["consumed_examples"] == 12
    assert payload["dataset_size"] == 12
    assert payload["coverage"]["unique_coverage"] == 1.0


def test_sft_partial_schedule_reports_partial_coverage(tmp_path):
    """A schedule that cannot cover the dataset must say so, not claim full."""
    pairs = _sft_pairs(tmp_path / "pairs.jsonl", n=12)
    model_cfg = _write_model_config(tmp_path / "m.yaml")
    cfg = _sft_cfg(tmp_path / "c.yaml", model_cfg, steps=1)
    out_dir = tmp_path / "runs" / "partial"

    _run(
        ["--config", str(cfg), "--sft-data", str(pairs), "--out-dir", str(out_dir)],
        sft_main,
    )

    coverage = json.loads((out_dir / "sft_coverage.json").read_text(encoding="utf-8"))
    assert coverage["consumed_examples"] == 4
    assert coverage["unique_coverage"] < 1.0
    assert coverage["epochs_consumed"] < 1.0


def test_sft_preserves_every_example_across_two_epochs(tmp_path):
    """The B1 regression, end to end: full coverage over a full schedule."""
    pairs = _sft_pairs(tmp_path / "pairs.jsonl", n=12)
    model_cfg = _write_model_config(tmp_path / "m.yaml")
    # 12 examples, 4 per step -> 3 steps per epoch, 6 steps = 2 epochs.
    cfg = _sft_cfg(tmp_path / "c.yaml", model_cfg, steps=6)
    out_dir = tmp_path / "runs" / "epochs"

    _run(
        ["--config", str(cfg), "--sft-data", str(pairs), "--out-dir", str(out_dir)],
        sft_main,
    )

    coverage = json.loads((out_dir / "sft_coverage.json").read_text(encoding="utf-8"))
    assert coverage["consumed_examples"] == 24
    assert coverage["unique_examples_seen"] == 12
    assert coverage["unique_coverage"] == 1.0
    assert coverage["epochs_consumed"] == 2.0
    assert coverage["epochs_completed"] >= 1