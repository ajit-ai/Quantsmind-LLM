"""Focused tests for the TAKSH L0.2 capability-recovery pipeline.

Covers: training target construction, tokenizer contract, BOS/EOS handling,
label shifting and loss masking, checkpoint loading path, benchmark execution
isolation and scoring correctness, train/eval split integrity, and the L0.1
application regression surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "evaluation" / "l02_python_bench_v1.json"
SFT_DATA = REPO / "data" / "l02_sft" / "sft.jsonl"
TOK_PATH = REPO / "tokenizer" / "taksh_code_bpe_v1.json"


def _bench():
    return json.loads(BENCH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Tokenizer contract / BOS-EOS
# --------------------------------------------------------------------------- #
def test_bos_and_eos_are_registered_single_tokens():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_PATH))
    assert tok.token_to_id("<bos>") == 1
    assert tok.token_to_id("<eos>") == 2


def test_contract_is_v1_and_render_starts_with_bos():
    from tokenizers import Tokenizer
    from training.chat_template import SYSTEM_PROMPT, render_for
    from training.tokenizer_contract import CONTRACT_V1, detect_contract

    tok = Tokenizer.from_file(str(TOK_PATH))
    assert detect_contract(tok) == CONTRACT_V1
    prompt = render_for(tok, SYSTEM_PROMPT, "Write a function.")
    assert prompt.startswith("<bos>")
    assert prompt.count("<bos>") == 1
    assert "<eos>" not in prompt


# --------------------------------------------------------------------------- #
# Training target construction / label shifting / loss masking
# --------------------------------------------------------------------------- #
def _toy_pairs(path: Path):
    pairs = [
        {"id": "t0", "task": "t", "instruction": "Write a Python function named add_one(x) that returns x + 1.",
         "input": "", "response": "def add_one(x):\n    return x + 1"},
        {"id": "t1", "task": "t", "instruction": "Write a Python function named twice(y) that returns y * 2.",
         "input": "", "response": "def twice(y):\n    return y * 2"},
    ]
    path.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")


def test_sft_targets_shift_by_one_and_mask_prompt(tmp_path):
    from tokenizers import Tokenizer
    from training.sft import SFTPairDataset
    from training.chat_template import SYSTEM_PROMPT, render_prompt, render_completion
    from training.tokenizer_contract import CONTRACT_V1

    pairs = tmp_path / "pairs.jsonl"
    _toy_pairs(pairs)
    tok = Tokenizer.from_file(str(TOK_PATH))
    ds = SFTPairDataset(str(pairs), tok, 512)
    assert len(ds) == 2
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == y.shape
        prompt = render_prompt(SYSTEM_PROMPT, ds.pairs[i]["instruction"], contract_id=CONTRACT_V1)
        full = render_completion(SYSTEM_PROMPT, ds.pairs[i]["instruction"], ds.pairs[i]["response"], contract_id=CONTRACT_V1)
        ids = tok.encode(full).ids
        assert ids[0] == tok.token_to_id("<bos>")
        assert ids[-1] == tok.token_to_id("<eos>")
        prompt_len = len(tok.encode(prompt).ids) - 1
        assert x.tolist() == ids[:-1]
        # Prompt region is masked; supervised region carries ids shifted by one.
        assert (y[:prompt_len] == -100).all()
        assert (y[prompt_len:] != -100).all()
        assert y[prompt_len:].tolist() == ids[prompt_len + 1:]
        # First supervised position predicts the first response token.
        assert y[prompt_len] == tok.encode(ds.pairs[i]["response"]).ids[0]


def test_sft_eos_is_supervised(tmp_path):
    from tokenizers import Tokenizer
    from training.sft import SFTPairDataset

    pairs = tmp_path / "pairs.jsonl"
    _toy_pairs(pairs)
    tok = Tokenizer.from_file(str(TOK_PATH))
    ds = SFTPairDataset(str(pairs), tok, 512)
    _, y = ds[0]
    assert (y == tok.token_to_id("<eos>")).any()


# --------------------------------------------------------------------------- #
# Benchmark integrity
# --------------------------------------------------------------------------- #
def test_benchmark_has_30_problems_split_10_10_10():
    bench = _bench()
    problems = bench["problems"]
    assert len(problems) == 30
    counts = {}
    for p in problems:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    assert counts == {"A": 10, "B": 10, "C": 10}
    ids = [p["id"] for p in problems]
    assert len(set(ids)) == 30
    for p in problems:
        assert p["tests"], f"{p['id']} has no tests"
        assert p["entry_point"]


def test_benchmark_instructions_are_unique():
    instructions = [p["instruction"] for p in _bench()["problems"]]
    assert len(set(instructions)) == len(instructions)


# --------------------------------------------------------------------------- #
# Benchmark execution isolation and scoring correctness
# --------------------------------------------------------------------------- #
def test_ast_gate_rejects_dangerous_code():
    from evaluation.l02_runner import ast_gate

    ok, _ = ast_gate("def f():\n    return 1")
    assert ok
    ok, reason = ast_gate("import os\ndef f():\n    return os.getcwd()")
    assert not ok
    ok, reason = ast_gate("def f():\n    return eval('1+1')")
    assert not ok
    ok, _ = ast_gate("def f():\n    return open('../../secret').read()")
    assert not ok
    ok, _ = ast_gate("def f():\n    return open('C:/secret.txt').read()")
    assert not ok


def test_sandbox_executes_benign_code_and_reports():
    from evaluation.l02_runner import run_in_sandbox

    problem = {
        "kind": "function",
        "entry_point": "double_it",
        "tests": [
            {"name": "basic", "args": [2], "expected": 4},
            {"name": "zero", "args": [0], "expected": 0},
        ],
    }
    out = run_in_sandbox("def double_it(x):\n    return x * 2", problem, timeout=5.0)
    assert out["ok"] is True
    assert all(t["passed"] for t in out["results"])


def test_sandbox_times_out_on_infinite_loop():
    from evaluation.l02_runner import run_in_sandbox

    problem = {
        "kind": "function",
        "entry_point": "spin",
        "tests": [{"name": "loop", "args": [], "expected": 1}],
    }
    out = run_in_sandbox("def spin():\n    while True:\n        pass", problem, timeout=2.0)
    assert out["ok"] is False
    assert "timeout" in out["error"]


def test_score_problem_statuses():
    from evaluation.l02_runner import score_problem

    problem = next(p for p in _bench()["problems"] if p["id"] == "A01")
    degenerate = score_problem(problem, "   )")
    assert degenerate["status"] == "degenerate"
    bad_syntax = score_problem(problem, "def compute_sum(numbers)\n    return sum(numbers)")
    assert bad_syntax["status"] == "syntax_error"
    correct = score_problem(problem, "def compute_sum(numbers):\n    return sum(numbers)")
    assert correct["status"] == "correct" and correct["passed"]
    wrong = score_problem(problem, "def compute_sum(numbers):\n    return len(numbers)")
    assert wrong["status"] == "wrong_answer"


# --------------------------------------------------------------------------- #
# Train/eval split integrity
# --------------------------------------------------------------------------- #
def test_l02_sft_has_no_benchmark_leakage():
    bench = _bench()
    bench_instructions = {p["instruction"].strip() for p in bench["problems"]}
    bench_entries = {p["entry_point"] for p in bench["problems"]}
    rows = [json.loads(l) for l in SFT_DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    instructions = [r["instruction"].strip() for r in rows]
    assert len(set(instructions)) == len(instructions), "duplicate SFT instructions"
    for r in rows:
        assert r["instruction"].strip() not in bench_instructions
        defined = r["response"].split("(")[0].removeprefix("def ")
        assert defined not in bench_entries


# --------------------------------------------------------------------------- #
# Regression tests for the 2026-11-09 SFT config-resolution fixes
# --------------------------------------------------------------------------- #
def test_sft_tokenizer_comes_from_config_not_hardcoded():
    from training.sft import resolve_sft_tokenizer_path

    cfg = {"data": {"tokenizer": "tokenizer/taksh_code_bpe_v1.json"}}
    assert resolve_sft_tokenizer_path(cfg) == "tokenizer/taksh_code_bpe_v1.json"
    assert resolve_sft_tokenizer_path({"data": {}}) == "tokenizer/qm_code_bpe_v0.json"
    assert resolve_sft_tokenizer_path({}) == "tokenizer/qm_code_bpe_v0.json"


def test_sft_seq_len_accepts_seq_len_or_max_length():
    from training.sft import resolve_sft_seq_len

    assert resolve_sft_seq_len({"data": {"seq_len": 256}}) == 256
    assert resolve_sft_seq_len({"data": {"max_length": 512}}) == 512
    assert resolve_sft_seq_len({"data": {}}) == 2048
    assert resolve_sft_seq_len({"data": {"max_length": 512, "seq_len": 128}}) == 128


def test_model_config_path_accepts_model_or_base_model_config():
    from training.sft import resolve_model_config_path

    assert resolve_model_config_path({"run": {"model_config": "m.yaml"}}) == "m.yaml"
    assert resolve_model_config_path({"run": {"base_model_config": "b.yaml"}}) == "b.yaml"
    assert (
        resolve_model_config_path({"run": {"model_config": "m.yaml", "base_model_config": "b.yaml"}})
        == "m.yaml"
    )


# --------------------------------------------------------------------------- #
# L0.1 regression surface
# --------------------------------------------------------------------------- #
def test_l01_application_still_loopback_only():
    from serving import local_assistant as la

    assert la.SERVER_NAME == "127.0.0.1"
    with pytest.raises(ValueError):
        la.validate_host("0.0.0.0")
    prompt = la.build_prompt(
        __import__("tokenizers").Tokenizer.from_file(str(TOK_PATH)), "Test prompt."
    )
    assert prompt.startswith("<bos>")
