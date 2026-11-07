import pytest

from data.filters import ExactDedup, hashlib_key, quality_check, scrub_pii
from evaluation.load_eval import load_eval_set


def test_scrub_pii_aadhaar_pan():
    text = "Customer Aadhaar 1234 5678 9012 and PAN ABCDE1234F on file."
    out = scrub_pii(text)
    assert "[AADHAAR]" in out and "[PAN]" in out
    assert "1234" not in out.replace("[AADHAAR]", "")


def test_scrub_pii_phone_email():
    out = scrub_pii("Call 9876543210 or mail a.b@x.co for KYC.")
    assert "[PHONE]" in out and "[EMAIL]" in out


def test_quality_check_bounds():
    ok, reason = quality_check("short", min_chars=200)
    assert not ok and reason == "length"
    ok, reason = quality_check("1234567890" * 30, min_chars=200)
    assert not ok and reason == "alpha_ratio"
    ok, reason = quality_check("The Indian market rose today. " * 10, min_chars=200)
    assert ok


def test_exact_dedup():
    d = ExactDedup()
    assert d.add("hello world") is True
    assert d.add("hello world") is False
    assert d.add("hello worlds") is True
    assert len(d) == 2


def test_hashlib_key_stable():
    assert hashlib_key(" abc ") == hashlib_key("abc")


def test_eval_seed_loads():
    items = load_eval_set()
    assert len(items) >= 24
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    arithmetic = [i for i in items if i["category"] == "arithmetic"]
    assert all(i["requires_tool"] for i in arithmetic)
