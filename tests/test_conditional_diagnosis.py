"""Focused tests for Increment 12 conditional diagnostics. No training."""

from __future__ import annotations

import pytest

from evaluation.conditional import (
    PAIRED_CASES,
    distribution_divergence,
    swap_case,
    token_cross_entropy,
    token_probability,
    token_rank,
)


def test_paired_cases_differ_only_in_identifiers():
    for _pair, _op, _instruction, sig_a, sig_b, _prefix, ident_a, ident_b in PAIRED_CASES:
        assert sig_a != sig_b
        # Identifiers appear in their own signature side.
        assert ident_a in sig_a and ident_b in sig_b
        # One-identifier swap maps A onto B exactly for single-param pairs;
        # for two-param pairs only the named identifier changes.
        swapped = swap_case(sig_a, ident_a, ident_b)
        assert ident_b in swapped and ident_a not in swapped


def test_probability_rank_and_ce_math():
    probs = [0.1, 0.7, 0.2]
    assert token_probability(probs, 1) == 0.7
    assert token_rank(probs, 1) == 1
    assert token_rank(probs, 0) == 3
    assert token_cross_entropy(probs, 1) == pytest.approx(-__import__("math").log(0.7))
    assert token_probability(probs, 99) == 0.0


def test_distribution_divergence_extremes():
    same = distribution_divergence([0.5, 0.5], [0.5, 0.5])
    assert same["tv_distance"] == 0.0 and same["top_token_agree"] is True
    disjoint = distribution_divergence([1.0, 0.0], [0.0, 1.0])
    assert disjoint["tv_distance"] == 1.0 and disjoint["top_token_agree"] is False


def test_swap_consistency_round_trip():
    for _pair, _op, _instruction, sig_a, sig_b, _prefix, ident_a, ident_b in PAIRED_CASES:
        assert swap_case(swap_case(sig_a, ident_a, ident_b), ident_b, ident_a) == sig_a
