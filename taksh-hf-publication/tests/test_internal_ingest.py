import pytest

from data.ingest_internal import scrub_secrets, scan_for_secrets


def test_scrub_secrets_api_key():
    text = 'api_key = "sk-1234567890abcdef"'
    result = scrub_secrets(text)
    assert "sk-1234567890abcdef" not in result
    assert "[REDACTED_SECRET]" in result


def test_scrub_secrets_private_key():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA..."
    result = scrub_secrets(text)
    assert "[REDACTED_PRIVATE_KEY]" in result


def test_scrub_secrets_aws_key():
    text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
    result = scrub_secrets(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert "[REDACTED_AWS_KEY]" in result


def test_scan_secrets_finds_api_key():
    text = 'password = "supersecret123"'
    findings = scan_for_secrets(text)
    assert len(findings) > 0


def test_scan_secrets_clean_code():
    text = "def add(a, b):\n    return a + b"
    findings = scan_for_secrets(text)
    assert len(findings) == 0


def test_scrub_secrets_github_token():
    text = "token = ghp_1234567890abcdefghijklmnopqrstuvwxyz"
    result = scrub_secrets(text)
    assert "ghp_1234567890" not in result
