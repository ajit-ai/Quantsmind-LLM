"""Offline tests for the M1 code-corpus builder (data/build_code_corpus_v1.py).

Network-free: HTTP is replaced by ``MockTransport``, a fake serving
GitHub-shaped payloads from dictionaries. Live acquisition is done by the CLI
and is deliberately not exercised here.

Coverage maps to the M1 Phase 1 requirements: allowlisting, licence gating
before download, content validation, deduplication, deterministic splits,
leakage checks, size/disk limits, transport errors and retries, resume and
partial-file recovery, and honest ``complete`` / ``truncated`` / ``failed``
manifest states.
"""

import json
import urllib.error

import pytest
import yaml

from data.build_code_corpus_v1 import (
    GITHUB_API,
    GITHUB_RAW,
    CorpusBuilder,
    ResponseTooLarge,
    Transport,
    content_split,
    main,
)

REPO = "psf/requests"
COMMIT = "a" * 40
PINNED_SPDX = "Apache-2.0"


def py_source(name="mod", n=6):
    """Valid Python source, comfortably above the 200-byte minimum."""
    lines = [
        '"""Module providing small arithmetic helpers for testing purposes."""',
        "",
        "import os",
        "",
    ]
    for i in range(n):
        lines += [
            f"def {name}_{i}(value):",
            f'    """Return value scaled by {i} plus a small offset."""',
            f"    return value * {i} + {i} + len(os.sep)",
            "",
        ]
    return "\n".join(lines).encode("utf-8")


def tree_entry(path, size):
    return {"path": path, "type": "blob", "size": size}


def make_cfg(sources=None, **overrides):
    cfg = {
        "policy": {
            "allowed_spdx": ["mit", "apache-2.0", "bsd-3-clause"],
            "require_pinned_commit": True,
            "verify_license_at_runtime": True,
            "extensions": [".py"],
            "min_file_bytes": 200,
            "max_file_bytes": 262144,
            "min_alpha_ratio": 0.55,
            "require_python_parses": True,
        },
        "limits": {
            "max_repos": 5,
            "max_files": 100,
            "max_total_bytes": 10_000_000,
            "max_retries_per_request": 2,
            "request_timeout_seconds": 5,
        },
        "dedup": {
            "exact": True,
            "minhash": True,
            "minhash_perms": 64,
            "minhash_bands": 16,
            "minhash_threshold": 0.8,
        },
        "splits": {
            "method": "content_hash",
            "train": 0.98,
            "val": 0.01,
            "test": 0.01,
        },
        "sources": sources
        if sources is not None
        else [
            {
                "id": "psf-requests",
                "repo": REPO,
                "commit": COMMIT,
                "spdx": PINNED_SPDX,
                "language": "python",
                "attribution": "Copyright the requests authors; Apache-2.0",
                "retrieved": "2026-10-14",
            }
        ],
    }
    cfg.update(overrides)
    return cfg


class MockTransport:
    """Serves GitHub API and raw-content payloads from dictionaries.

    ``repo_meta``: repo -> API metadata dict (licence at .license.spdx_id)
    ``trees``: repo -> list of tree nodes
    ``blobs``: (repo, path) -> bytes, or an Exception instance to raise
    """

    def __init__(self, repo_meta=None, trees=None, blobs=None):
        self.repo_meta = repo_meta or {}
        self.trees = trees or {}
        self.blobs = blobs or {}
        self.calls = []
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "too_large": 0}

    def get(self, url, max_bytes=None):
        self.calls.append(url)
        self.stats["requests"] += 1
        if "/git/trees/" in url:
            repo = url.split("/repos/")[1].split("/git/trees/")[0]
            payload = self.trees[repo]
            if isinstance(payload, Exception):
                raise payload
            return json.dumps({"tree": payload}).encode("utf-8")
        if url.startswith(f"{GITHUB_API}/repos/"):
            repo = url[len(f"{GITHUB_API}/repos/") :]
            payload = self.repo_meta[repo]
            if isinstance(payload, Exception):
                raise payload
            return json.dumps(payload).encode("utf-8")
        if url.startswith("https://raw.githubusercontent.com/"):
            parts = url[len("https://raw.githubusercontent.com/") :].split("/", 3)
            repo = f"{parts[0]}/{parts[1]}"
            path = parts[3]
            payload = self.blobs[(repo, path)]
            if isinstance(payload, Exception):
                raise payload
            if max_bytes is not None and len(payload) > max_bytes:
                raise ResponseTooLarge(f"body exceeds cap {max_bytes}")
            return payload
        raise AssertionError(f"unexpected URL: {url}")

    def raw_calls(self):
        return [u for u in self.calls if u.startswith("https://raw.githubusercontent.com/")]


def meta(spdx="Apache-2.0"):
    return {"license": {"spdx_id": spdx}}


def licence_evidence(source_id, commit, spdx="Apache-2.0", **extra):
    """Minimal accepted verification evidence for one source.

    The builder's licence gate is mandatory: a source with no verification
    result is blocked. Tests that are not about licensing must supply evidence.
    """
    entry = {
        "id": source_id,
        "repo": REPO,
        "commit": commit,
        "result": "verified",
        "detected_spdx": spdx.lower(),
        "derived_exclusion_paths": [],
        "conflicting_subtrees": {},
        "notice_obligations": [],
    }
    entry.update(extra)
    return entry


def licence_index_for(cfg, result="verified", **per_source):
    """Index verification entries by source id, as the loader produces."""
    return {
        s["id"]: licence_evidence(
            s["id"],
            s["commit"],
            s.get("spdx", "Apache-2.0"),
            result=per_source.get(s["id"], result),
        )
        for s in cfg["sources"]
    }


def build(tmp_path, cfg, transport, out="corpus.jsonl", licence_index=None, **kwargs):
    if licence_index is None:
        licence_index = licence_index_for(cfg)
    builder = CorpusBuilder(
        cfg,
        transport,
        out_path=tmp_path / out,
        manifest_path=tmp_path / (out + ".manifest.json"),
        licence_index=licence_index,
        **kwargs,
    )
    return builder, builder.run()


def good_repo(**over):
    """One repo, three distinct acceptable files."""
    blobs = {}
    tree = []
    for i in range(3):
        path = f"src/module_{i}.py"
        data = py_source(name=f"module_{i}")
        blobs[(REPO, path)] = data
        tree.append(tree_entry(path, len(data)))
    blobs.update(over.get("blobs", {}))
    tree += over.get("tree", [])
    return MockTransport(
        repo_meta={REPO: over.get("meta", meta())},
        trees={REPO: over.get("tree_payload", tree)},
        blobs=blobs,
    )


# --------------------------------------------------------------------------- #
# 1. Allowlisting / discovery refusal
# --------------------------------------------------------------------------- #
def test_no_sources_configured_is_a_failed_run(tmp_path):
    _, manifest = build(tmp_path, make_cfg(sources=[]), MockTransport())
    assert manifest["status"] == "failed"
    assert manifest["complete"] is False
    assert manifest["counts"]["kept"] == 0


def test_held_back_sources_are_not_processed(tmp_path):
    cfg = make_cfg(
        sources=[
            {
                "id": "psf-requests",
                "repo": REPO,
                "commit": COMMIT,
                "spdx": PINNED_SPDX,
                "language": "python",
                "attribution": "x",
                "retrieved": "2026-10-14",
                "phase1_default": False,
            }
        ]
    )
    transport = good_repo()
    builder, manifest = build(tmp_path, cfg, transport)
    assert transport.calls == []
    assert manifest["status"] == "failed"


# --------------------------------------------------------------------------- #
# 2. Licence verification strictly precedes any file download
# --------------------------------------------------------------------------- #
def test_licence_is_checked_before_any_raw_download(tmp_path):
    transport = good_repo()
    _, manifest = build(tmp_path, make_cfg(), transport)
    first_raw = min(
        i for i, u in enumerate(transport.calls) if u.startswith("https://raw")
    )
    meta_call = next(
        i for i, u in enumerate(transport.calls) if u == f"{GITHUB_API}/repos/{REPO}"
    )
    assert meta_call < first_raw
    assert manifest["status"] == "complete"


def test_licence_mismatch_aborts_before_download(tmp_path):
    transport = good_repo(meta=meta("MIT"))
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert transport.raw_calls() == []
    assert manifest["sources"][0]["status"] == "rejected_licence"
    assert "drift" in manifest["sources"][0]["error"]
    assert manifest["status"] == "failed"


# --------------------------------------------------------------------------- #
# 3. Missing / ambiguous / unapproved licences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spdx", [None, "NOASSERTION", "GPL-3.0", "AGPL-3.0", "UNLICENSED"]
)
def test_unacceptable_live_licence_is_rejected(tmp_path, spdx):
    transport = good_repo(meta=meta(spdx) if spdx else {"license": None})
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert transport.raw_calls() == []
    assert manifest["sources"][0]["status"] == "rejected_licence"
    assert manifest["status"] == "failed"


def test_allowlist_is_enforced_against_pinned_value(tmp_path):
    src = make_cfg()["sources"][0]
    src["spdx"] = "MIT"
    cfg = make_cfg(sources=[src])
    transport = good_repo(meta=meta("MIT"))
    _, manifest = build(tmp_path, cfg, transport)
    assert manifest["sources"][0]["status"] == "ok"
    assert manifest["licensing"]["by_spdx"] == {"MIT": 3}


# --------------------------------------------------------------------------- #
# 4. Acceptance, attribution and licence metadata preservation
# --------------------------------------------------------------------------- #
def test_valid_python_is_accepted_with_full_provenance(tmp_path):
    transport = good_repo()
    _, manifest = build(tmp_path, make_cfg(), transport)
    rows = [
        json.loads(line)
        for line in (tmp_path / "corpus.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert row["repo"] == REPO
        assert row["commit"] == COMMIT
        assert row["spdx"] == PINNED_SPDX
        assert row["attribution"]
        assert row["file_path"].endswith(".py")
        assert row["retrieved_at"] == "2026-10-14"
        assert row["content_sha1"]
        assert row["split"] in {"train", "val", "test"}
        compile(row["text"], "<test>", "exec")
    assert manifest["counts"]["kept"] == 3
    assert manifest["counts"]["by_source"] == {"psf-requests": 3}
    assert manifest["licensing"]["attribution_recorded_per_chunk"] is True
    assert len(manifest["content_hashes"]) == 3


# --------------------------------------------------------------------------- #
# 5. Content validation and rejection reasons
# --------------------------------------------------------------------------- #
def test_rejection_reasons_are_recorded(tmp_path):
    bad = py_source(name="bad")
    syntax = b"def broken(:\n    pass\n" + b"# padding\n" * 60
    noncode = b"1234567890" * 40
    generated = py_source(name="gen").replace(
        b"import os", b"# @generated - do not edit\nimport os", 1
    )
    empty = b""
    blobs = {
        (REPO, "src/syntax_error.py"): syntax,
        (REPO, "src/not_code.py"): noncode,
        (REPO, "src/generated.py"): generated,
        (REPO, "src/empty.py"): empty,
        (REPO, "tests/test_thing.py"): bad,
        (REPO, "src/huge.py"): b"x" * 10,
    }
    tree = [
        tree_entry("src/syntax_error.py", len(syntax)),
        tree_entry("src/not_code.py", len(noncode)),
        tree_entry("src/generated.py", len(generated)),
        tree_entry("src/empty.py", len(empty)),
        tree_entry("tests/test_thing.py", len(bad)),
        tree_entry("src/huge.py", 999_999_999),
    ]
    transport = MockTransport(
        repo_meta={REPO: meta()}, trees={REPO: tree}, blobs=blobs
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    rejected = manifest["counts"]["rejected"]
    assert rejected["syntax_error"] == 1
    assert rejected["low_alpha_ratio"] == 1
    assert rejected["generated_marker"] == 1
    assert rejected["too_small"] == 1
    assert rejected["path_excluded"] == 1
    assert rejected["too_large"] == 1
    assert manifest["sources"][0]["files_skipped_path"] == 1
    assert manifest["sources"][0]["files_skipped_oversize"] == 1
    # Oversized and excluded files are decided from the listing: never fetched.
    fetched = " ".join(transport.raw_calls())
    assert "huge.py" not in fetched
    assert "test_thing.py" not in fetched
    assert manifest["status"] == "failed"


def test_binary_file_is_rejected_as_not_utf8(tmp_path):
    blob = bytes(range(256)) * 4
    transport = MockTransport(
        repo_meta={REPO: meta()},
        trees={REPO: [tree_entry("src/blob.py", len(blob))]},
        blobs={(REPO, "src/blob.py"): blob},
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["rejected"]["not_utf8"] == 1


# --------------------------------------------------------------------------- #
# 6. Deduplication: exact and near-duplicate
# --------------------------------------------------------------------------- #
def test_exact_duplicate_is_rejected(tmp_path):
    same = py_source(name="dup")
    transport = MockTransport(
        repo_meta={REPO: meta()},
        trees={
            REPO: [
                tree_entry("src/a.py", len(same)),
                tree_entry("src/b.py", len(same)),
            ]
        },
        blobs={(REPO, "src/a.py"): same, (REPO, "src/b.py"): same},
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["kept"] == 1
    assert manifest["counts"]["rejected"]["exact_duplicate"] == 1
    assert manifest["counts"]["deduplicated"] == 1


def test_near_duplicate_is_rejected(tmp_path):
    base = py_source(name="near", n=30)
    variant = base.replace(b"value * 7", b"value * 8", 1)
    assert variant != base
    transport = MockTransport(
        repo_meta={REPO: meta()},
        trees={
            REPO: [
                tree_entry("src/a.py", len(base)),
                tree_entry("src/b.py", len(variant)),
            ]
        },
        blobs={(REPO, "src/a.py"): base, (REPO, "src/b.py"): variant},
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["kept"] == 1
    assert manifest["counts"]["rejected"]["near_duplicate"] == 1


# --------------------------------------------------------------------------- #
# 7. Deterministic splits and leakage checks
# --------------------------------------------------------------------------- #
def test_splits_are_deterministic_across_runs(tmp_path):
    cfg = make_cfg()
    _, first = build(tmp_path, cfg, good_repo(), out="a.jsonl")
    _, second = build(tmp_path, cfg, good_repo(), out="b.jsonl")
    rows_a = (tmp_path / "a.jsonl").read_text("utf-8")
    rows_b = (tmp_path / "b.jsonl").read_text("utf-8")
    assert rows_a == rows_b
    assert first["splits"] == second["splits"]
    assert (
        first["corpus"]["config_fingerprint"] == second["corpus"]["config_fingerprint"]
    )


def test_config_fingerprint_changes_when_policy_changes(tmp_path):
    cfg_a = make_cfg()
    cfg_b = make_cfg()
    cfg_b["splits"]["train"] = 0.9
    builder_a, _ = build(tmp_path, cfg_a, good_repo(), out="a.jsonl")
    builder_b, _ = build(tmp_path, cfg_b, good_repo(), out="b.jsonl")
    assert builder_a.config_fingerprint() != builder_b.config_fingerprint()


def test_content_split_is_a_pure_function_of_content():
    splits_cfg = {"train": 0.98, "val": 0.01, "test": 0.01}
    text = "def f():\n    return 1\n"
    assert content_split(text, splits_cfg) == content_split(text, splits_cfg)
    assert content_split(text, splits_cfg) in {"train", "val", "test"}


def test_leakage_check_reports_zero_cross_split_duplicates(tmp_path):
    transport = good_repo()
    _, manifest = build(tmp_path, make_cfg(), transport)
    leak = manifest["leakage_check"]
    assert leak["exact_duplicate_chunks_in_multiple_splits"] == 0
    assert leak["near_duplicate_pairs"] == 0
    assert leak["near_duplicate_pairs_crossing_splits"] == 0
    assert "limitations" in leak


# --------------------------------------------------------------------------- #
# 8. Download and disk limits
# --------------------------------------------------------------------------- #
def test_max_files_limit_truncates_run(tmp_path):
    _, manifest = build(
        tmp_path, make_cfg(limits={**make_cfg()["limits"], "max_files": 2}), good_repo()
    )
    assert manifest["counts"]["kept"] == 2
    assert manifest["truncated"] is True
    assert manifest["truncated_reasons"] == {"max_files": 1}
    assert manifest["limits_reached"] is True
    assert manifest["status"] == "truncated"


def test_max_total_bytes_limit_truncates_run(tmp_path):
    limits = {**make_cfg()["limits"], "max_total_bytes": 700}
    _, manifest = build(tmp_path, make_cfg(limits=limits), good_repo())
    assert manifest["truncated"] is True
    assert manifest["truncated_reasons"] == {"max_total_bytes": 1}
    assert manifest["status"] == "truncated"


def test_max_repos_limit_limits_sources(tmp_path):
    second = {
        "id": "urllib3",
        "repo": "urllib3/urllib3",
        "commit": "b" * 40,
        "spdx": "MIT",
        "language": "python",
        "attribution": "x",
        "retrieved": "2026-10-14",
    }
    cfg = make_cfg(sources=make_cfg()["sources"] + [second])
    cfg["limits"]["max_repos"] = 1
    transport = good_repo()
    _, manifest = build(tmp_path, cfg, transport)
    assert len(manifest["sources"]) == 1
    assert "urllib3" not in " ".join(transport.calls)


# --------------------------------------------------------------------------- #
# 9. HTTP errors, rate limiting, retries, malformed responses
# --------------------------------------------------------------------------- #
def test_one_bad_file_does_not_abort_the_source(tmp_path):
    transport = good_repo(
        blobs={(REPO, "src/module_1.py"): urllib.error.HTTPError(
            "https://x", 404, "not found", {}, None
        )}
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["kept"] == 2
    assert manifest["counts"]["rejected"]["fetch_error"] == 1
    assert manifest["sources"][0]["status"] == "ok"
    assert "module_1.py" in manifest["sources"][0]["error"]


def test_tree_failure_marks_source_and_run_failed(tmp_path):
    transport = MockTransport(
        repo_meta={REPO: meta()},
        trees={REPO: urllib.error.HTTPError("https://x", 429, "rate limited", {}, None)},
        blobs={},
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["sources"][0]["status"] == "failed"
    assert manifest["status"] == "failed"
    assert "HTTPError" in manifest["sources"][0]["error"]


def test_malformed_tree_payload_marks_run_failed(tmp_path):
    transport = MockTransport(
        repo_meta={REPO: meta()}, trees={REPO: []}, blobs={}
    )
    transport.trees[REPO] = None  # json() -> {"tree": None}
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["kept"] == 0
    assert manifest["status"] == "failed"


# --------------------------------------------------------------------------- #
# 10. Transport internals: caps, deadlines, retries
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, body, headers=None, chunk=None):
        self._body = body
        self._pos = 0
        self.headers = headers or {}
        self._chunk = chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        if self._chunk is not None:
            n = min(n, self._chunk)
        out = self._body[self._pos : self._pos + n]
        self._pos += len(out)
        return out


def test_content_length_over_cap_is_refused_before_reading():
    t = Transport()
    with pytest.raises(ResponseTooLarge):
        t._read_capped(FakeResponse(b"x" * 50, {"Content-Length": "50"}), max_bytes=10)


def test_body_over_cap_is_refused_during_read():
    t = Transport()
    with pytest.raises(ResponseTooLarge):
        t._read_capped(FakeResponse(b"x" * 100), max_bytes=10)


def test_total_transfer_deadline_is_enforced():
    t = Transport(deadline=0.0)
    with pytest.raises(TimeoutError):
        t._read_capped(FakeResponse(b"x" * 100), max_bytes=None)


def test_transport_retries_transient_failures(monkeypatch):
    import data.build_code_corpus_v1 as mod

    attempts = {"n": 0}

    def flaky_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.URLError("temporary failure")
        return FakeResponse(b"payload")

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    t = Transport(max_retries=3)
    assert t.get("https://example.invalid/x") == b"payload"
    assert attempts["n"] == 2
    assert t.stats["retries"] == 1


def test_transport_does_not_retry_404(monkeypatch):
    import data.build_code_corpus_v1 as mod

    attempts = {"n": 0}

    def not_found(req, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "gone", {}, None)

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setattr("urllib.request.urlopen", not_found)
    t = Transport(max_retries=3)
    with pytest.raises(urllib.error.HTTPError):
        t.get("https://example.invalid/missing")
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- #
# 11. Resume / idempotency / partial-file recovery
# --------------------------------------------------------------------------- #
def test_resume_continues_after_interruption(tmp_path):
    cfg = make_cfg(limits={**make_cfg()["limits"], "max_files": 2})
    out = tmp_path / "corpus.jsonl"
    _, first = build(tmp_path, cfg, good_repo(), out="corpus.jsonl")
    assert first["status"] == "truncated"
    assert len(out.read_text("utf-8").splitlines()) == 2

    # Simulate a run killed mid-write: a partial final JSON line.
    with out.open("ab") as f:
        f.write(b'{"text": "partial record with no closing brace')

    cfg2 = make_cfg(limits={**make_cfg()["limits"], "max_files": 5})
    _, second = build(tmp_path, cfg2, good_repo(), out="corpus.jsonl", resume=True)

    text = out.read_text("utf-8")
    rows = [json.loads(line) for line in text.splitlines()]  # all valid JSON
    assert "partial record" not in text
    assert len(rows) == 3
    assert second["counts"]["resumed"] == 2
    assert second["status"] == "complete"
    # The two re-fetched files are recognised as already collected.
    assert second["counts"]["rejected"]["exact_duplicate"] == 2


def test_rerun_without_resume_is_idempotent(tmp_path):
    cfg = make_cfg()
    build(tmp_path, cfg, good_repo(), out="corpus.jsonl")
    first_bytes = (tmp_path / "corpus.jsonl").read_bytes()
    _, manifest = build(tmp_path, cfg, good_repo(), out="corpus.jsonl")
    assert (tmp_path / "corpus.jsonl").read_bytes() == first_bytes
    assert manifest["status"] == "complete"


# --------------------------------------------------------------------------- #
# 12. Honest manifest states; no false success after partial acquisition
# --------------------------------------------------------------------------- #
def test_status_complete(tmp_path):
    _, manifest = build(tmp_path, make_cfg(), good_repo())
    assert manifest["status"] == "complete"
    assert manifest["complete"] is True
    assert all(s["state"] == "complete" for s in manifest["sources"])


def test_status_truncated(tmp_path):
    cfg = make_cfg(limits={**make_cfg()["limits"], "max_files": 1})
    _, manifest = build(tmp_path, cfg, good_repo())
    assert manifest["status"] == "truncated"
    assert manifest["complete"] is False
    assert manifest["sources"][0]["state"] == "truncated"


def test_status_failed_when_source_fails_even_with_kept_rows(tmp_path):
    cfg = make_cfg()
    cfg["sources"].append(
        {
            "id": "urllib3",
            "repo": "urllib3/urllib3",
            "commit": "b" * 40,
            "spdx": "MIT",
            "language": "python",
            "attribution": "x",
            "retrieved": "2026-10-14",
        }
    )
    transport = good_repo()
    transport.trees["urllib3/urllib3"] = urllib.error.HTTPError(
        "https://x", 500, "server error", {}, None
    )
    _, manifest = build(tmp_path, cfg, transport)
    assert manifest["counts"]["kept"] == 3
    assert manifest["status"] == "failed"
    assert manifest["complete"] is False
    assert manifest["sources_failed"] == 1


def test_empty_corpus_is_failed_not_complete(tmp_path):
    transport = MockTransport(
        repo_meta={REPO: meta()},
        trees={REPO: [tree_entry("src/empty.py", 0)]},
        blobs={(REPO, "src/empty.py"): b""},
    )
    _, manifest = build(tmp_path, make_cfg(), transport)
    assert manifest["counts"]["kept"] == 0
    assert manifest["status"] == "failed"
    assert manifest["complete"] is False


# --------------------------------------------------------------------------- #
# 13. CLI: exit codes and preservation of a previous manifest
# --------------------------------------------------------------------------- #
def _write_cfg(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _write_verification(tmp_path, cfg, result="verified", **per_source):
    """Write a licence verification artifact for the configured sources."""
    payload = {
        "schema": "quantsmind.licence_verification/1",
        "generated_at": "2026-10-15T00:00:00+00:00",
        "sources": [
            {
                "id": s["id"],
                "repo": s["repo"],
                "commit": s["commit"],
                "result": per_source.get(s["id"], result),
                "detected_spdx": s["spdx"].lower(),
                "derived_exclusion_paths": [],
                "conflicting_subtrees": {},
            }
            for s in cfg["sources"]
        ],
    }
    path = tmp_path / "licence_verification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _main_args(tmp_path, cfg_path, out="c.jsonl", manifest="m.json", verification=None):
    args = [
        "--config", str(cfg_path),
        "--out", str(tmp_path / out),
        "--manifest", str(tmp_path / manifest),
    ]
    if verification is not None:
        args += ["--licence-verification", str(verification)]
    return args


def test_main_returns_zero_on_complete_run(tmp_path, monkeypatch):
    import data.build_code_corpus_v1 as mod

    cfg = make_cfg()
    cfg_path = _write_cfg(tmp_path, cfg)
    verification = _write_verification(tmp_path, cfg)
    monkeypatch.setattr(mod, "Transport", lambda **kw: good_repo())
    rc = main(_main_args(tmp_path, cfg_path, verification=verification))
    assert rc == 0
    written = json.loads((tmp_path / "m.json").read_text("utf-8"))
    assert written["status"] == "complete"


def test_main_returns_one_when_no_eligible_samples(tmp_path, monkeypatch):
    import data.build_code_corpus_v1 as mod

    cfg = make_cfg()
    cfg_path = _write_cfg(tmp_path, cfg)
    verification = _write_verification(tmp_path, cfg)
    empty = MockTransport(
        repo_meta={REPO: meta()},
        trees={REPO: [tree_entry("src/empty.py", 0)]},
        blobs={(REPO, "src/empty.py"): b""},
    )
    monkeypatch.setattr(mod, "Transport", lambda **kw: empty)
    rc = main(_main_args(tmp_path, cfg_path, verification=verification))
    assert rc == 1


def test_main_preserves_previous_manifest_when_run_fails(tmp_path, monkeypatch):
    import data.build_code_corpus_v1 as mod

    cfg = make_cfg()
    cfg_path = _write_cfg(tmp_path, cfg)
    verification = _write_verification(tmp_path, cfg)
    monkeypatch.setattr(mod, "Transport", lambda **kw: good_repo(meta=meta("MIT")))
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text('{"previous": "complete"}', encoding="utf-8")
    rc = main(_main_args(tmp_path, cfg_path, verification=verification))
    assert rc == 1
    stale = tmp_path / "m.json.stale"
    assert stale.read_text("utf-8") == '{"previous": "complete"}'
    assert json.loads(manifest_path.read_text("utf-8"))["status"] == "failed"
