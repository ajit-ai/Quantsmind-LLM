import json

import pytest

torch = pytest.importorskip("torch")

from training.sft import SFTPairDataset, SYSTEM_PROMPT, TEMPLATE, collate_sft

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
