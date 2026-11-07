"""End-to-end M3 test: train, checkpoint, reload in a FRESH process, generate.

The fresh-process requirement is the point. A checkpoint reloaded in the same
interpreter can pass through module-level state that a real user would never
have, so this test launches a new interpreter to load the checkpoint and
generate from it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def build_fixture(tmp_path, vocab=64, tokens=8192):
    """A tiny but complete, licence-gated world to train on."""
    import hashlib

    import numpy as np

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    model_dir = tmp_path / "configs"
    model_dir.mkdir(exist_ok=True)

    def make_jsonl(name, repo, split, n=6):
        records = []
        for i in range(n):
            text = f"# {repo} file {i}\ndef f{i}():\n    return {i}\n" + "x = 1\n" * 40
            records.append({
                "text": text,
                "content_sha1": hashlib.sha1(text.encode()).hexdigest(),
                "source_id": repo.replace("/", "-"),
                "repo": repo, "commit": "a" * 40, "spdx": "MIT",
                "attribution": "t", "file_path": f"{i}.py",
                "language": "python", "retrieved_at": "2026-10-18",
                "split": split,
            })
        path = data_dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records),
                        encoding="utf-8")
        return path

    corpus = data_dir / "corpus.jsonl"
    parts = [
        make_jsonl("tr.jsonl", "org/train", "train"),
        make_jsonl("va.jsonl", "org/val", "val"),
    ]
    corpus.write_text(
        "".join(p.read_text(encoding="utf-8") for p in parts), encoding="utf-8"
    )

    licence = tmp_path / "licence.json"
    licence.write_text(json.dumps({
        "counts": {"checked": 2, "verified": 2, "rejected": 0,
                   "needs_human_review": 0},
        "verified_ids": ["org-train", "org-val"], "needs_human_review_ids": [],
        "all_verified": True,
    }), encoding="utf-8")

    from data.prepare_m3_dataset import prepare

    manifest = prepare(corpus, licence, "tokenizer/qm_bpe_v0.json",
                       str(data_dir / "ds"), encode=True)
    assert manifest["isolation"]["ok"] is True

    # qm_bpe_v0 has vocab 49152; the model must match the tokenizer it uses.
    from tokenizers import Tokenizer

    real_vocab = Tokenizer.from_file("tokenizer/qm_bpe_v0.json").get_vocab_size()
    model_cfg = {
        "name": "e2e", "vocab_size": real_vocab, "d_model": 32, "n_layers": 2,
        "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 32,
        "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True,
    }
    model_path = model_dir / "model.yaml"
    # Written with a real YAML dumper on purpose: formatting `1e-05` by hand
    # yields the STRING "1e-05" in YAML, which reaches RMSNorm as a string.
    model_path.write_text(
        yaml.safe_dump({"model": model_cfg}, sort_keys=False), encoding="utf-8"
    )

    train_cfg = f"""
run:
  name: e2e-m3
  model_config: {model_path.as_posix()}
  seed: 42
data:
  tokens_path: {(data_dir / 'ds.train.tokens.bin').as_posix()}
  val_tokens_path: {(data_dir / 'ds.val.tokens.bin').as_posix()}
  tokenizer: tokenizer/qm_bpe_v0.json
  dataset_manifest: {(data_dir / 'ds.manifest.json').as_posix()}
  seq_len: 32
  val_seed: 99
optimizer:
  lr_peak: 1.0e-3
  warmup_steps: 1
  min_lr_ratio: 0.1
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  grad_clip: 1.0
precision:
  dtype: float32
batch:
  micro_batch_size: 1
  grad_accum_steps: 1
training:
  max_steps: 3
  checkpoint_every_steps: 2
  log_every_steps: 1
  eval_every_steps: 1
  val_batches: 2
watchdog:
  nan_check_every_steps: 1
logging:
  tracker: none
  run_manifest_path: {(tmp_path / 'run_manifest.json').as_posix()}
"""
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(train_cfg, encoding="utf-8")
    return cfg_path, model_cfg, tmp_path


def run_training(cfg_path, out_dir, *extra):
    return subprocess.run(
        [sys.executable, "-m", "training.pretrain", "--config", str(cfg_path),
         "--out-dir", str(out_dir), *extra],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )


@pytest.mark.slow
def test_train_checkpoint_reload_and_generate(tmp_path):
    cfg_path, model_cfg, _ = build_fixture(tmp_path)
    out_dir = tmp_path / "run"

    first = run_training(cfg_path, out_dir)
    assert first.returncode == 0, first.stdout + first.stderr

    ckpt = out_dir / "checkpoint_last.pt"
    assert ckpt.exists(), "training did not produce a checkpoint"

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["success"] is True
    assert manifest["training"]["steps_completed"] == 3
    assert manifest["training"]["tokens_processed"] == 3 * 32
    assert manifest["checkpoint"]["sha256"]
    assert manifest["validation"]["configured"] is True
    assert manifest["holdout_policy"].startswith("The M2 holdout")

    # Validation actually ran and produced a finite number.
    val_losses = [
        entry["val_loss"] for entry in manifest["training"]["history"]
        if entry.get("val_loss") is not None
    ]
    assert val_losses, "no validation loss was recorded"
    assert all(v == v for v in val_losses), "validation loss was NaN"

    # A best-validation checkpoint should exist once validation improved.
    assert (out_dir / "checkpoint_best.pt").exists()

    # ---- fresh process: reload and generate ----
    record_path = tmp_path / "generation.json"
    gen = subprocess.run(
        [sys.executable, "-m", "training.generate", "--ckpt", str(ckpt),
         "--prompt", "def add(a, b):", "--max-new-tokens", "8", "--greedy",
         "--tokenizer", "tokenizer/qm_bpe_v0.json",
         "--record", str(record_path)],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert gen.returncode == 0, gen.stdout + gen.stderr

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["checkpoint"]["parameters"] == sum(
        p.numel() for p in __import__("training.model", fromlist=["QMModel"])
        .QMModel(model_cfg).parameters()
    )
    assert record["checkpoint"]["checkpoint_step"] == 3
    assert record["result"]["completion_tokens"] == 8
    assert "No hosted or external language model" in record["provenance"]


@pytest.mark.slow
def test_resume_continues_from_saved_state(tmp_path):
    cfg_path, _model_cfg, _ = build_fixture(tmp_path)
    out_dir = tmp_path / "run"

    first = run_training(cfg_path, out_dir, "--max-steps", "2")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "resumed from" not in first.stdout

    resumed = run_training(cfg_path, out_dir, "--max-steps", "4", "--resume")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "resumed from" in resumed.stdout
    assert "RNG state restored" in resumed.stdout

    import torch

    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location="cpu",
                      weights_only=False)
    assert ckpt["step"] == 4
    # 4 steps x 32 tokens, with no double counting of the resumed span.
    assert ckpt["tokens_processed"] == 4 * 32


@pytest.mark.slow
def test_resume_without_a_checkpoint_fails_loudly(tmp_path):
    cfg_path, _model_cfg, _ = build_fixture(tmp_path)
    result = run_training(cfg_path, tmp_path / "empty", "--resume")
    assert result.returncode != 0
    assert "no checkpoint exists" in (result.stdout + result.stderr)