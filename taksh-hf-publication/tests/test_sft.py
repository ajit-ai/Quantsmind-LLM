import json

import pytest

torch = pytest.importorskip("torch")

from training.sft import (
    SFTPairDataset,
    SYSTEM_PROMPT,
    TEMPLATE,
    collate_sft,
    find_quarantine,
)

SFT_SAMPLES = [
    {"instruction": "Write a function to add two numbers", "response": "def add(a, b):\n    return a + b"},
    {"instruction": "Explain what a decorator is", "response": "A decorator is a function that wraps another function to extend its behavior."},
    {"instruction": "Fix the bug: def divide(a, b): return a / b", "response": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b"},
]


@pytest.fixture(scope="module")
def sft_jsonl(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sft")
    path = tmp / "sft_data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in SFT_SAMPLES:
            f.write(json.dumps(item) + "\n")
    return str(path)


def test_sft_template_format():
    text = TEMPLATE.format(system="test", instruction="do something", response="result")
    assert "test" in text
    assert "do something" in text
    assert "result" in text


def test_sft_system_prompt():
    assert "Python" in SYSTEM_PROMPT
    assert "JavaScript" in SYSTEM_PROMPT or "TypeScript" in SYSTEM_PROMPT


def test_collate_sft_shapes():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([2, 3, 4])
    x2 = torch.tensor([5, 6])
    y2 = torch.tensor([6, 7])
    bx, by = collate_sft([(x, y), (x2, y2)])
    assert bx.shape[0] == 2
    assert bx.shape[1] == 3
    assert by.shape == bx.shape
    assert (by[1, 2] == -100).item()


# --------------------------------------------------------------------------
# Quarantine guard
#
# data/code_sft_v0.jsonl is quarantined: 423 of 442 responses are fixed boilerplate
# and the 19 real pairs come from a faulty regex extractor. A checkpoint trained on
# it supports no capability claim, so the trainer must refuse the dataset outright.
# --------------------------------------------------------------------------


def test_shipped_sft_corpus_is_quarantined():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    dataset = root / "data" / "code_sft_v0.jsonl"
    marker = root / "data" / "code_sft_v0.quarantine.json"

    assert dataset.exists(), "original SFT corpus must be preserved, not deleted"
    assert marker.exists(), "quarantine marker must ship with the corpus"

    meta = json.loads(marker.read_text(encoding="utf-8"))
    assert meta["quarantined"] is True
    assert meta["status"] == "unusable_for_capability_claims"


@pytest.fixture
def workdir(tmp_path):
    """Function-scoped temp dir for quarantine marker files."""
    return tmp_path


def test_find_quarantine_detects_marker(workdir):
    dataset = workdir / "pairs.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    assert find_quarantine(str(dataset)) is None

    marker = workdir / "pairs.quarantine.json"
    marker.write_text(json.dumps({"quarantined": True, "reason": "boilerplate"}), encoding="utf-8")
    assert find_quarantine(str(dataset)) == marker


def test_find_quarantine_ignores_clean_dataset(workdir):
    dataset = workdir / "clean.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    (workdir / "clean.json").write_text("{}", encoding="utf-8")
    assert find_quarantine(str(dataset)) is None


def test_quarantine_marker_records_the_measured_defects():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    meta = json.loads((root / "data" / "code_sft_v0.quarantine.json").read_text(encoding="utf-8"))

    measured = meta["measured"]
    assert measured["total_pairs"] == 442
    assert measured["content_free_boilerplate_pairs"] == 423
    assert measured["genuine_supervision_pairs"] == 0
    assert sum(measured["by_task"].values()) == measured["total_pairs"]
