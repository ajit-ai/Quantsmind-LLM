"""Regression tests for evaluation false-pass guards.

Background: an empty held-out split used to score a *perfect* perplexity of 1.0.
`compute_code_perplexity` divided ``sum([])`` by ``max(1, 0)``, got ``0.0``, and
``exp(0.0) == 1.0``. Two committed reports carry that value. These tests exist so
missing evaluation data can never again read as a passing score.

Each test builds a deliberately undersized token file, so none of them requires a
trained checkpoint or a GPU.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluation.eval_code import (
    MBPP_PROBLEMS,
    SUBSET_SIZE,
    _unmeasured_subset,
    compute_code_perplexity,
)
from evaluation.evaluate_pilot import heldout_ppl, summarize


def _write_tokens(path, n_tokens, vocab=5822):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    ids.tofile(path)
    return path


@pytest.fixture
def workdir(tmp_path):
    """Function-scoped temp dir for building undersized token files."""
    return tmp_path


class _StubModel:
    """Minimal stand-in that returns a fixed loss, so no weights are needed."""

    def __init__(self, loss=1.2345):
        self._loss = loss
        self.calls = 0

    def __call__(self, idx, targets=None):
        self.calls += 1
        return None, torch.tensor(self._loss)


# --------------------------------------------------------------------------
# Empty / undersized held-out split must never produce a number
# --------------------------------------------------------------------------


def test_empty_split_reports_unavailable_not_perfect(workdir):
    """The original defect: 0 losses -> mean 0.0 -> exp(0) == perplexity 1.0."""
    tokens = _write_tokens(workdir / "tiny.tokens.bin", 100_244)

    result = compute_code_perplexity(_StubModel(), str(tokens), seq_len=2048)

    assert result["ok"] is False
    assert result["perplexity"] is None
    assert result["nll"] is None
    assert result["windows"] == 0
    assert "error" in result


def test_empty_split_never_yields_perplexity_one(workdir):
    """Directly encode the failure mode that produced the retracted reports."""
    tokens = _write_tokens(workdir / "tiny.tokens.bin", 50_000)
    result = compute_code_perplexity(_StubModel(), str(tokens), seq_len=2048)
    assert result["perplexity"] != 1.0
    assert result["ok"] is False


def test_split_shorter_than_one_window_is_rejected(workdir):
    """1% tail of the 100,244-token code corpus is 1,003 tokens < 2,049 needed."""
    tokens = _write_tokens(workdir / "code.tokens.bin", 100_244)
    result = compute_code_perplexity(_StubModel(), str(tokens), seq_len=2048)
    assert result["val_tokens"] < 2049
    assert result["ok"] is False


def test_zero_length_token_file_fails_explicitly(workdir):
    """A zero-length file must be an explicit error, never a silent perfect score.

    `np.memmap` refuses to map an empty file, so this raises before any scoring
    happens. That is the required direction: an explicit failure, not a pass. This
    test pins the behaviour so it cannot regress into a fabricated number.
    """
    tokens = _write_tokens(workdir / "empty.tokens.bin", 0)
    with pytest.raises(ValueError):
        compute_code_perplexity(_StubModel(), str(tokens), seq_len=2048)


def test_measurable_split_does_score(workdir):
    """Guard against over-correcting: a large enough split must still produce a number.

    500,000 tokens at a 1% tail leaves 5,000 held-out tokens, which fits two
    2,048-token windows at offsets split and split+2048.
    """
    tokens = _write_tokens(workdir / "big.tokens.bin", 500_000)
    model = _StubModel(loss=1.5)
    result = compute_code_perplexity(model, str(tokens), seq_len=2048, max_windows=2)

    assert result["ok"] is True
    assert result["windows"] == 2
    assert result["nll"] == pytest.approx(1.5, abs=1e-3)
    assert result["perplexity"] > 1.0
    assert model.calls == 2


def test_heldout_pilot_returns_none_on_empty_split(workdir):
    """`evaluation.evaluate_pilot.heldout_ppl` must return (None, None), not (0.0, 1.0)."""
    tokens = _write_tokens(workdir / "pilot.tokens.bin", 100_244)
    nll, ppl = heldout_ppl(_StubModel(), str(tokens), seq_len=2048)
    assert nll is None
    assert ppl is None


# --------------------------------------------------------------------------
# Seed-set summary must not report a perfect 0.0 mean NLL for an empty set
# --------------------------------------------------------------------------


def test_summarize_empty_rows_is_not_zero():
    """`sum([])/max(1, 0)` produced a perfect mean NLL of 0.0."""
    result = summarize([])
    assert result["overall_mean_nll"] is None
    assert result["ok"] is False
    assert result["n_items"] == 0
    assert "error" in result


def test_summarize_non_empty_still_works():
    rows = [
        {"category": "a", "nll_per_token": 1.0},
        {"category": "a", "nll_per_token": 3.0},
        {"category": "b", "nll_per_token": 2.0},
    ]
    result = summarize(rows)
    assert result["ok"] is True
    assert result["n_items"] == 3
    assert result["overall_mean_nll"] == pytest.approx(2.0)
    assert result["by_category"] == {"a": 2.0, "b": 2.0}


# --------------------------------------------------------------------------
# Benchmark subsets must never read as promotion-eligible scores
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suite", ["humaneval", "mbpp"])
def test_subset_stamp_is_unmeasured_and_not_promotion_eligible(suite):
    stamp = _unmeasured_subset(suite)
    assert stamp["status"] == "unmeasured"
    assert stamp["promotion_eligible"] is False
    assert stamp["published_problems"] > stamp["problems_evaluated"]


@pytest.mark.parametrize("suite", ["humaneval", "mbpp"])
def test_subset_is_strictly_smaller_than_published_benchmark(suite):
    """A pass rate over 5 problems must not be comparable to a 164/500 benchmark."""
    assert SUBSET_SIZE[suite] < 164
    assert SUBSET_SIZE["humaneval"] <= 5
    assert SUBSET_SIZE["mbpp"] <= 5


def test_no_full_suite_benchmark_claims_exist_in_reports():
    """Committed eval reports must not contain a benchmark pass rate.

    HumanEval and MBPP have never been run, so any report claiming a rate for them
    is fabricated. Perplexity *may* appear -- including ``null`` with ``ok: False``,
    which is the guard working correctly.

    The false-pass signature is a *numeric* perplexity accompanied by
    ``windows == 0``: a number produced without evaluating anything. That combination
    requires a retraction notice.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    reports = sorted(root.glob("runs/*/eval_report.json"))
    if not reports:
        pytest.skip("historical runs/ eval reports are local-only and not published")

    for report_path in reports:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        for suite in ("humaneval", "mbpp"):
            assert suite not in data, f"{report_path} claims a {suite} result"

        ppl = data.get("perplexity")
        if ppl is None:
            continue

        numeric = isinstance(ppl.get("perplexity"), (int, float))
        if numeric and ppl.get("windows") == 0:
            note = report_path.with_name("eval_report.RETRACTED.md")
            assert note.exists(), (
                f"{report_path} reports a numeric perplexity from 0 evaluated "
                f"windows but carries no retraction notice"
            )
        elif numeric:
            assert ppl.get("ok") is True, (
                f"{report_path} reports perplexity {ppl.get('perplexity')} "
                f"without ok=True"
            )


def test_retraction_notices_exist_for_false_pass_artifacts():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for run_dir in ("code_pretrain", "code_smoke"):
        json_path = root / "runs" / run_dir / "eval_report.json"
        note_path = root / "runs" / run_dir / "eval_report.RETRACTED.md"
        if not json_path.exists():
            pytest.skip("historical runs/ retraction artifacts are local-only and not published")
        assert json_path.exists(), "original artifact must be preserved"
        assert note_path.exists(), "false-pass artifact must carry a retraction notice"

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["perplexity"]["windows"] == 0
        assert note_path.read_text(encoding="utf-8").strip().startswith("# RETRACTED")