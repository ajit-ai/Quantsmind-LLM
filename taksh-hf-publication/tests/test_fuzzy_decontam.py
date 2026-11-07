from data.filters import Decontaminator, MinHashDedup, word_shingles


def test_word_shingles():
    s = word_shingles("The quick brown fox jumps over the lazy dog", k=4)
    assert "quick brown fox jumps" in s
    assert len(s) == 6


def test_minhash_catches_near_duplicate():
    d = MinHashDedup(num_perms=64, bands=16, threshold=0.8)
    para = (
        "The Reserve Bank of India announced the repo rate decision today after the "
        "monetary policy committee meeting concluded in Mumbai with all members present. "
    )
    base = para * 3
    assert d.add(base) is True
    near = base + " Markets reacted calmly to the announcement."
    assert d.add(near) is False


def test_minhash_allows_unique():
    d = MinHashDedup()
    a = "Equity markets rallied as IT stocks surged on strong deal wins and favorable currency movement."
    b = "Agriculture output depends heavily on the monsoon season across most Indian states every year."
    assert d.add(a) is True
    assert d.add(b) is True


def test_decontaminator_catches_eval_text():
    eval_q = "What is the expense ratio in a mutual fund and how does it affect returns?"
    eval_ref = "Annual fund operating costs as a percentage of AUM deducted from returns."
    d = Decontaminator([eval_q, eval_ref])
    contaminated = "Some preamble text. " + eval_ref + " More trailing text here."
    clean = "Completely unrelated content about cricket scores and weather patterns in Kerala."
    assert d.is_contaminated(contaminated) is True
    assert d.is_contaminated(clean) is False
