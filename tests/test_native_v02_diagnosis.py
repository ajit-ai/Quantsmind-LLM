"""Tests for TAKSH Native v0.2 generation diagnosis utilities.

Covers prompt variant rendering, diagnostic capture schema/preservation, decoding
config handling, and compatibility with Native v0.1 baseline artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.native_v02 import diagnose  # noqa: E402
from evaluation.native_v01.tasks import get_tasks  # noqa: E402
from serving.foundation_inference import load_foundation  # noqa: E402


def test_subset_covers_required_types():
    tasks = diagnose.get_subset_tasks()
    assert [t.task_id for t in tasks] == list(diagnose.SUBSET_TASK_IDS)
    cats = {t.category for t in tasks}
    kinds = {t.kind for t in tasks}
    assert "basic_function_generation" in cats
    assert "arithmetic_string_manipulation" in cats
    assert "lists_dicts_iteration" in cats
    assert "function_completion" in kinds or "function_completion" in cats
    assert "instruction_following" in cats
    assert "code_correction_debugging" in cats


def test_prompt_variants_render_without_dropping_task():
    task = next(t for t in get_tasks() if t.task_id == "NV01-BASIC-001")
    for v in diagnose.PROMPT_VARIANTS:
        text, meta = diagnose.render_prompt(task, v)
        assert isinstance(text, str) and text
        assert meta["variant"] == v
        assert meta["task_id"] == task.task_id


def test_finalize_recovered_sft_prompt_with_tokenizer():
    task = next(t for t in get_tasks() if t.task_id == "NV01-BASIC-001")
    text, meta = diagnose.render_prompt(task, "recovered_sft_format")
    tok_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
    assert tok_path.is_file()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    final = diagnose.finalize_prompt_for_tokenizer(text, meta, tok)
    assert "Complete the following Python function" in final or "<bos>" in final
    # prompt half should not include assistant response content
    assert "a + b" not in final


def test_diagnostic_generate_captures_raw_and_repair_separately(tmp_path):
    # tiny synthetic model/tokenizer for fast schema test
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from training.model import QMModel

    vocab = {"[PAD]": 0, "<bos>": 1, "<eos>": 2, "[UNK]": 3}
    for i in range(4, 40):
        vocab[f"t{i}"] = i
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    cfg = {"name": "tiny", "vocab_size": 40, "d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 64, "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True}
    torch.manual_seed(0)
    model = QMModel(cfg).eval()
    diag = diagnose.diagnostic_generate(model, tok, "<bos> t4 t5", max_new_tokens=4, temperature=0.0, seed=1, device="cpu", capture_steps=3)
    assert "generated_token_ids" in diag
    assert "decoded_before_repair" in diag and "decoded_after_repair" in diag
    assert "raw_text" in diag and "text" in diag
    assert diag["generated_token_count"] == len(diag["generated_token_ids"])
    assert "step_diagnostics" in diag and len(diag["step_diagnostics"]) <= 3
    fs = diag["first_step"]
    assert "eos" in fs and "rank" in fs["eos"] and "prob" in fs["eos"]
    assert "top5" in fs and isinstance(fs["top5"], list)
    # consistency: raw token ids decode to decoded_before_repair; runner also records this field
    assert tok.decode(diag["generated_token_ids"]) == diag["decoded_before_repair"]
    assert "repair" in diag and "enabled" in diag["repair"] and "status" in diag["repair"]


def test_summarize_basic():
    exps = [
        {"variant": "v01_benchmark", "decoding": "greedy_v01_baseline", "immediate_eos_greedy": True, "finish_reason": "stop", "generated_token_count": 0, "decoded_before_repair": "", "raw_ids_match_decoded_before_repair": True},
        {"variant": "raw_completion_prefix", "decoding": "temp_0.7", "immediate_eos_greedy": False, "finish_reason": "length", "generated_token_count": 2, "decoded_before_repair": "x", "raw_ids_match_decoded_before_repair": True},
    ]
    s = diagnose.summarize(exps)
    assert s["experiment_count"] == 2
    assert s["zero_completion_tokens_rate"] == 0.5
    assert "by_variant" in s and "v01_benchmark" in s["by_variant"]


def test_v01_baseline_artifacts_preserved():
    assert (REPO / "evaluation/native_v01/baseline_results.json").is_file()
    assert (REPO / "docs/audit/TAKSH-NATIVE-V0.1-BASELINE.md").is_file()
    # v0.1 task inventory intact
    tasks = get_tasks()
    assert len(tasks) == 20
