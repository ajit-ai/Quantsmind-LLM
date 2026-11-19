import pytest

tokenizers = pytest.importorskip("tokenizers")

from tokenizer.train_tokenizer import SPECIAL_TOKENS, build_tokenizer, sanity_report, train_tokenizer

CODE_SAMPLES = [
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "const debounce = (fn, delay) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n};",
    "interface User {\n  id: number;\n  name: string;\n  email: string;\n}",
    "class DataProcessor {\n  private cache = new Map();\n\n  process(data: any[]) {\n    return data.filter(x => x.valid);\n  }\n}",
    "import asyncio\n\nasync def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:\n        return await session.get(url)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("code_tok")
    corpus = tmp / "code_corpus.txt"
    lines = CODE_SAMPLES * 100
    corpus.write_text("\n".join(lines), encoding="utf-8")
    out = tmp / "code_tokenizer.json"
    return train_tokenizer(str(corpus), str(out), vocab_size=1000)


def test_code_special_tokens_present(tok):
    vocab = tok.get_vocab()
    for s in SPECIAL_TOKENS:
        assert s in vocab


def test_code_roundtrip(tok):
    for s in CODE_SAMPLES:
        assert tok.decode(tok.encode(s).ids) == s


def test_code_no_unknown_ids(tok):
    unk_id = tok.token_to_id("<unk>")
    for s in CODE_SAMPLES:
        assert all(i != unk_id for i in tok.encode(s).ids)


def test_code_sanity_report(tok):
    report = sanity_report(tok, CODE_SAMPLES)
    assert report["all_roundtrip_ok"]
    assert report["vocab_size"] <= 1000


def test_code_token_efficiency(tok):
    for s in CODE_SAMPLES:
        ids = tok.encode(s).ids
        assert len(ids) < len(s) * 0.8
