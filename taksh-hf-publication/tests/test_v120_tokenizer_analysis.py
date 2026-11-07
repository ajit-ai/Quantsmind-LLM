"""v1.2.0 tokenizer suitability tests (measured, not asserted by construction)."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import tokenizer_analysis as ta

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"

SAMPLES = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "class Account:\n    def __init__(self, balance=0):\n        self.balance = balance\n",
    "import json\n\ndef load(path):\n    with open(path) as handle:\n        return json.load(handle)\n",
    "CONSTANTS = {'alpha': 1, 'beta': 2}\n\ndef total():\n    return sum(CONSTANTS.values())\n",
]


def test_measurements_are_produced_for_python_samples():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    metrics = ta.analyse_texts(tokenizer, SAMPLES, source_label="fixture")
    assert metrics["samples"] == 4
    assert metrics["tokens"] > 0
    assert metrics["characters_per_token"] > 0
    assert metrics["tokens_per_character"] > 0
    assert metrics["bytes_per_token"] > 1.0
    assert 0.0 <= metrics["identifier_fragmentation"] <= 1.0
    assert 0.0 <= metrics["keywords_single_token"] <= 1.0
    assert 0.0 <= metrics["operators_single_token"] <= 1.0
    assert metrics["lossless_roundtrip_failures"] == 0
    assert metrics["lossless_round_trip_failure_rate"] == 0.0
    assert metrics["unknown_token_rate"] == 0.0


def test_roundtrip_is_exact_for_every_sample():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    for text in SAMPLES:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
        assert (
            tokenizer.decode(
                tokenizer.encode(text).ids, skip_special_tokens=False
            )
            == text
        )


def test_reserved_literal_text_is_a_decoding_policy_not_a_loss():
    """A literal special-token spelling is decoded away by the library default.

    The artifact is lossless: with special-token skipping disabled the text is
    reproduced exactly. The difference is the library's documented decoding
    policy, and it is a corpus-side injection risk, not a tokenizer defect.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    text = 'tok = tokenizer.token_to_id("<eos>")\nvalue = 1\n'
    ids = tokenizer.encode(text).ids
    assert tokenizer.decode(ids) != text
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    result = ta.classify_roundtrip_failures(tokenizer, [text])
    assert result["policy_failures"] == 1
    assert result["lossless_failures"] == 0
    assert result["special_token_drop"] == 1
    assert result["genuine_loss"] == 0


def test_control_marker_literal_encodes_to_the_control_id():
    from tokenizers import Tokenizer

    from training.tokenizer_contract import build_contract_tokenizer, contract_ids

    derived = build_contract_tokenizer(TOKENIZER)
    ids = contract_ids(derived)
    encoded = derived.encode('marker = "<|user|>"\n').ids
    assert ids["<|user|>"] in encoded
    text = 'marker = "<|user|>"\n'
    assert derived.decode(encoded, skip_special_tokens=False) == text


def test_reserved_literal_incidence_is_measured():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    metrics = ta.analyse_texts(
        tokenizer, ['x = "<eos>"\n', "y = 1\n"], source_label="fixture"
    )
    assert metrics["reserved_literal_files"] == 1
    assert metrics["reserved_literal_occurrences"] == 1


def test_lossless_failure_drives_the_migration_verdict():
    verdict, reasons = ta.decide(
        {
            "samples": 100,
            "bytes": 100_000,
            "identifier_fragmentation": 0.2,
            "characters_per_token": 3.5,
            "lossless_round_trip_failure_rate": 0.05,
            "policy_round_trip_failure_rate": 0.05,
        }
    )
    assert verdict == "TOKENIZER_MIGRATION_RECOMMENDED"
    assert any("lossless" in reason for reason in reasons)


def test_policy_failure_alone_does_not_force_migration():
    verdict, reasons = ta.decide(
        {
            "samples": 100,
            "bytes": 100_000,
            "identifier_fragmentation": 0.2,
            "characters_per_token": 2.9,
            "lossless_round_trip_failure_rate": 0.0,
            "policy_round_trip_failure_rate": 0.14,
        }
    )
    assert verdict == "TOKENIZER_ACCEPTED_WITH_LIMITATIONS"
    assert any("injection risk" in reason for reason in reasons)


def test_control_token_derivation_does_not_change_fertility():
    from tokenizers import Tokenizer

    base = Tokenizer.from_file(str(TOKENIZER))
    derived = ta.build_contract_tokenizer(TOKENIZER)
    before = ta.analyse_texts(base, SAMPLES)
    after = ta.analyse_texts(derived, SAMPLES)
    assert after["tokens_per_character"] == pytest.approx(
        before["tokens_per_character"], rel=1e-9
    )
    assert after["bytes_per_token"] == pytest.approx(before["bytes_per_token"], rel=1e-9)
    assert derived.get_vocab_size() == base.get_vocab_size() + 6


def test_report_states_a_decision_with_reasons():
    report = ta.build_report(TOKENIZER, SAMPLES, label="fixture")
    verdict = report["decision"]["verdict"]
    assert verdict in (
        "TOKENIZER_ACCEPTED",
        "TOKENIZER_ACCEPTED_WITH_LIMITATIONS",
        "TOKENIZER_MIGRATION_RECOMMENDED",
        "INSUFFICIENT_EVIDENCE",
    )
    assert report["decision"]["reasons"]
    assert report["tokenizer"]["sha256"]
    assert report["baseline"]["byte_level_fallback_bytes_per_token"] == 1.0
    assert report["limitations"]


def test_insufficient_evidence_is_reported_not_guessed():
    verdict, reasons = ta.decide({"samples": 2, "bytes": 100, "identifier_fragmentation": 0.0,
                                  "characters_per_token": 9.0})
    assert verdict == "INSUFFICIENT_EVIDENCE"
    assert reasons


def test_migration_is_recommended_when_identifiers_fragment():
    verdict, reasons = ta.decide(
        {
            "samples": 100,
            "bytes": 100_000,
            "identifier_fragmentation": 0.9,
            "characters_per_token": 2.0,
            "lossless_round_trip_failure_rate": 0.0,
            "policy_round_trip_failure_rate": 0.0,
        }
    )
    assert verdict == "TOKENIZER_MIGRATION_RECOMMENDED"
    assert any("fragmentation" in reason for reason in reasons)


def test_migration_is_recommended_when_round_trip_fails_often():
    verdict, reasons = ta.decide(
        {
            "samples": 100,
            "bytes": 100_000,
            "identifier_fragmentation": 0.2,
            "characters_per_token": 3.5,
            "lossless_round_trip_failure_rate": 0.14,
            "policy_round_trip_failure_rate": 0.14,
        }
    )
    assert verdict == "TOKENIZER_MIGRATION_RECOMMENDED"
    assert any("lossless" in reason for reason in reasons)


def test_roundtrip_failures_are_attributed_to_registered_specials():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    text = 'tok = tokenizer.token_to_id("<eos>")\nvalue = 1\n'
    result = ta.classify_roundtrip_failures(tokenizer, [text])
    assert result["policy_failures"] == 1
    assert result["lossless_failures"] == 0
    assert result["special_token_drop"] == 1


def test_local_source_collection_is_bounded(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    for i in range(5):
        (root / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")
    texts = ta.collect_python_sources(root, max_files=3, max_bytes=10_000)
    assert len(texts) == 3
