import pytest

tokenizers = pytest.importorskip("tokenizers")

from tokenizer.train_tokenizer import SPECIAL_TOKENS, build_tokenizer, sanity_report, train_tokenizer

SAMPLES = [
    "AAPL closed at 187.43 (+1.2%) on 2026-08-21.",
    "The NIFTY50 index fell 245 points amid rate concerns.",
    "Revenue was Rs.1,234 crore vs Rs.987 crore last quarter.",
    "Hindi check: bazaar aaj girawat hai.",
]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tok")
    corpus = tmp / "corpus.txt"
    lines = SAMPLES * 50
    corpus.write_text("\n".join(lines), encoding="utf-8")
    out = tmp / "tokenizer.json"
    return train_tokenizer(str(corpus), str(out), vocab_size=500)


def test_special_tokens_present(tok):
    vocab = tok.get_vocab()
    for s in SPECIAL_TOKENS:
        assert s in vocab


def test_roundtrip(tok):
    for s in SAMPLES:
        assert tok.decode(tok.encode(s).ids) == s


def test_no_unknown_ids(tok):
    unk_id = tok.token_to_id("<unk>")
    for s in SAMPLES:
        assert all(i != unk_id for i in tok.encode(s).ids)


def test_sanity_report_all_ok(tok):
    report = sanity_report(tok, SAMPLES)
    assert report["all_roundtrip_ok"]
    assert report["vocab_size"] <= 500
