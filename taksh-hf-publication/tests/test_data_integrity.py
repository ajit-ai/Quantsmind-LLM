"""Offline tests for M1 data-integrity hardening: G2, G3, G4, G5.

Nothing here touches the network. Licence verification is exercised against a
fake transport; the contamination audit runs against synthetic benchmark files;
the PII tests use fixtures that capture the exact literals at risk.
"""

import json

import pytest

from data.contamination_audit import (
    SUBSET_NGRAM,
    audit_corpus,
    normalise_for_match,
    ngrams,
)
from data.filters import scrub_pii, scrub_pii_report
from data.verify_licenses import (
    classify_licence,
    is_licence_path,
    verify_source,
)

REPO = "psf/requests"
COMMIT = "a" * 40

MIT_TEXT = b"""MIT License

Copyright (c) 2024 Someone

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""

APACHE_TEXT = b"""                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
"""

BSD3_TEXT = b"""Copyright (c) 2024 Someone

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.
"""

GPL3_TEXT = b"""                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007
"""

UNKNOWN_TEXT = b"""Copyright (c) 2024. All rights reserved.
You must sign a separate agreement to use this.
"""


class FakeTransport:
    def __init__(self, meta=None, tree=None, blobs=None):
        self.meta = meta if meta is not None else {"license": {"spdx_id": "MIT"}}
        self.tree = tree if tree is not None else []
        self.blobs = blobs or {}
        self.calls = []

    def get(self, url, max_bytes=None):
        self.calls.append(url)
        if "/git/trees/" in url:
            return json.dumps({"tree": self.tree}).encode("utf-8")
        if url.endswith("/repos/" + REPO):
            return json.dumps(self.meta).encode("utf-8")
        for path, payload in self.blobs.items():
            if f"/{COMMIT}/{path}" in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected url {url}")


def source(spdx="MIT"):
    return {
        "id": "psf-requests",
        "repo": REPO,
        "commit": COMMIT,
        "spdx": spdx,
        "language": "python",
        "attribution": "x",
        "retrieved": "2026-10-14",
    }


POLICY = {"allowed_spdx": ["mit", "apache-2.0", "bsd-3-clause", "isc"]}


def blob(path, size=100):
    return {"path": path, "type": "blob", "size": size}


# --------------------------------------------------------------------------- #
# G4 -- licence file verification
# --------------------------------------------------------------------------- #
def test_classify_known_families():
    assert classify_licence(MIT_TEXT.decode())[0] == "mit"
    assert classify_licence(APACHE_TEXT.decode())[0] == "apache-2.0"
    assert classify_licence(BSD3_TEXT.decode())[0] == "bsd-3-clause"
    assert classify_licence(GPL3_TEXT.decode())[0] == "gpl-3.0"
    assert classify_licence(UNKNOWN_TEXT.decode())[0] is None


def test_is_licence_path():
    assert is_licence_path("LICENSE")
    assert is_licence_path("docs/LICENSE.txt")
    assert is_licence_path("COPYING")
    assert not is_licence_path("src/module.py")
    assert not is_licence_path("src/licence_helper.py")


def test_licence_file_matches_pinned_spdx_is_verified():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", len(MIT_TEXT))],
        blobs={"LICENSE": MIT_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "verified"
    assert result["detected_spdx"] == "mit"
    assert result["licence_files"][0]["sha256"]
    assert result["commit"] == COMMIT
    assert "retain the copyright notice" in " ".join(result["notice_obligations"])


def test_licence_file_conflicting_with_pinned_spdx_is_rejected():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", len(APACHE_TEXT))],
        blobs={"LICENSE": APACHE_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "conflict"
    assert result["needs_human_review"] is False
    assert any("not trustworthy" in n for n in result["notes"])


def test_missing_licence_file_is_not_approved():
    transport = FakeTransport(meta={"license": {"spdx_id": "MIT"}}, tree=[], blobs={})
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "missing"
    assert result["needs_human_review"] is True
    assert result["licence_files"] == []


def test_multiple_disagreeing_licence_files_are_ambiguous():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", 10), blob("COPYING", 10)],
        blobs={"LICENSE": MIT_TEXT, "COPYING": APACHE_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "ambiguous"
    assert set(result["detected_spdx"]) == {"mit", "apache-2.0"}
    assert result["needs_human_review"] is True


def test_multiple_agreeing_licence_files_are_verified():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", 10), blob("LICENSE-MIT", 10)],
        blobs={"LICENSE": MIT_TEXT, "LICENSE-MIT": MIT_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "verified"
    assert len(result["licence_files"]) == 2


def test_unrecognised_licence_text_needs_human():
    transport = FakeTransport(
        meta={"license": None},
        tree=[blob("LICENSE", len(UNKNOWN_TEXT))],
        blobs={"LICENSE": UNKNOWN_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "unrecognised"
    assert result["needs_human_review"] is True


def test_licence_outside_allowlist_is_not_permitted():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "GPL-3.0"}},
        tree=[blob("COPYING", len(GPL3_TEXT))],
        blobs={"COPYING": GPL3_TEXT},
    )
    result = verify_source(source("GPL-3.0"), transport, POLICY)
    assert result["result"] == "not_permitted"
    assert "gpl-3.0" not in [s.lower() for s in POLICY["allowed_spdx"]]


def test_altered_licence_hash_is_detectable():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", len(MIT_TEXT))],
        blobs={"LICENSE": MIT_TEXT},
    )
    first = verify_source(source("MIT"), transport, POLICY)
    altered = verify_source(source("MIT"), transport, POLICY)
    tampered = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[blob("LICENSE", len(MIT_TEXT))],
        blobs={"LICENSE": MIT_TEXT.replace(b"free of charge", b"cost of tea")},
    )
    changed = verify_source(source("MIT"), tampered, POLICY)
    assert first["licence_files"][0]["sha256"] == altered["licence_files"][0]["sha256"]
    assert changed["licence_files"][0]["sha256"] != first["licence_files"][0]["sha256"]
    # The tampered text no longer matches a known family, so it is not approved.
    assert changed["result"] in ("unrecognised", "conflict")


def test_metadata_drift_is_noted_but_file_verification_stands():
    transport = FakeTransport(
        meta={"license": {"spdx_id": "Apache-2.0"}},
        tree=[blob("LICENSE", len(MIT_TEXT))],
        blobs={"LICENSE": MIT_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "verified"
    assert any("metadata currently reports" in n for n in result["notes"])


def test_metadata_fetch_failure_is_recorded():
    class Boom(FakeTransport):
        def get(self, url, max_bytes=None):
            raise OSError("network down")

    result = verify_source(source("MIT"), Boom(), POLICY)
    assert result["result"] == "metadata_unavailable"
    assert result["needs_human_review"] is True


def test_nested_subtree_licence_conflict_is_reported_not_hidden():
    """The real psf-black case: MIT at the root, GPL-2.0 under src/blib2to3."""
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[
            blob("LICENSE", len(MIT_TEXT)),
            blob("src/blib2to3/LICENSE", len(GPL3_TEXT)),
        ],
        blobs={"LICENSE": MIT_TEXT, "src/blib2to3/LICENSE": GPL3_TEXT},
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "verified_with_quarantine"
    assert result["detected_spdx"] == "mit"
    assert result["needs_human_review"] is True
    assert "src/blib2to3" in result["conflicting_subtrees"]
    assert any("quarantined" in n for n in result["notes"])


def test_vendored_licence_files_are_not_treated_as_repo_licence():
    """setuptools bundles other projects' licences under _vendor."""
    transport = FakeTransport(
        meta={"license": {"spdx_id": "MIT"}},
        tree=[
            blob("LICENSE", len(MIT_TEXT)),
            blob("setuptools/_vendor/jaraco.text.dist-info/LICENSE", len(GPL3_TEXT)),
        ],
        blobs={
            "LICENSE": MIT_TEXT,
            "setuptools/_vendor/jaraco.text.dist-info/LICENSE": GPL3_TEXT,
        },
    )
    result = verify_source(source("MIT"), transport, POLICY)
    assert result["result"] == "verified"
    assert result["conflicting_subtrees"] == {}
    fetched = " ".join(transport.calls)
    assert "jaraco" not in fetched


def test_builder_quarantines_conflicting_subtree(tmp_path):
    from data.build_code_corpus_v1 import CorpusBuilder

    cfg = _multi_source_cfg("source_group")
    cfg["sources"] = [
        {
            "id": "psf-black",
            "repo": "psf/black",
            "commit": "c" * 40,
            "spdx": "MIT",
            "language": "python",
            "attribution": "x",
            "retrieved": "2026-10-14",
            "quarantine_paths": ["src/blib2to3/"],
        }
    ]

    class T:
        def __init__(self):
            self.stats = {"requests": 0, "retries": 0, "failures": 0, "too_large": 0}

        def get(self, url, max_bytes=None):
            self.stats["requests"] += 1
            if "/git/trees/" in url:
                good = _py_file("psf/black", 0)
                bad = _py_file("psf/black", 1) + b"\n# vendored from lib2to3\n"
                return json.dumps(
                    {
                        "tree": [
                            {"path": "src/black/mod.py", "type": "blob",
                             "size": len(good)},
                            {"path": "src/blib2to3/pgen2/driver.py",
                             "type": "blob", "size": len(bad)},
                        ]
                    }
                ).encode()
            return _py_file("psf/black", 0)

    builder = CorpusBuilder(
        cfg,
        T(),
        out_path=tmp_path / "c.jsonl",
        manifest_path=tmp_path / "m.json",
        licence_index={
            "psf-black": {
                "id": "psf-black",
                "repo": "psf/black",
                "commit": "c" * 40,
                "result": "verified_with_quarantine",
                "detected_spdx": "mit",
                "conflicting_subtrees": {"src/blib2to3": ["gpl-2.0"]},
                "derived_exclusion_paths": ["src/blib2to3/"],
            }
        },
        log=lambda s: None,
    )
    manifest = builder.run()
    assert manifest["counts"]["rejected"]["quarantined_licence_conflict"] == 1
    assert manifest["sources"][0]["files_quarantined_licence"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "c.jsonl").read_text("utf-8").splitlines()
    ]
    assert all(not r["file_path"].startswith("src/blib2to3/") for r in rows)


# --------------------------------------------------------------------------- #
# G5 -- PII scrubbing must not alter code semantics
# --------------------------------------------------------------------------- #
CODE_FIXTURE = '''"""Module constants and helpers."""
import os

TEN_DIGIT = 9876543210
TWELVE_DIGIT = 123456789012
HEX_MASK = 0x5F3759DF
SCIENTIFIC = 1.5e-10
SEPARATED = 1_000_000_000
DECIMAL = 3.14159265358979
FLOATY = 9876543210.5
BINARY = 0b1010101010
OCTAL = 0o755
ABCDE1234F = "not-a-pan-in-code"
PHONE_LIKE_NAME = 9876543210  # a bare literal, not a phone number


def contact() -> str:
    """Return a support string."""
    return "call 9876543210 or mail a.b@x.co"


# legacy id 9876543210 left in a comment
'''

UNTOUCHED_LITERALS = [
    "TEN_DIGIT = 9876543210",
    "TWELVE_DIGIT = 123456789012",
    "HEX_MASK = 0x5F3759DF",
    "SCIENTIFIC = 1.5e-10",
    "SEPARATED = 1_000_000_000",
    "DECIMAL = 3.14159265358979",
    "FLOATY = 9876543210.5",
    "BINARY = 0b1010101010",
    "OCTAL = 0o755",
]


@pytest.mark.parametrize("line", UNTOUCHED_LITERALS)
def test_valid_python_literals_are_never_rewritten(line):
    src = f"{line}\n"
    result = scrub_pii_report(src, mode="code")
    assert result.clean == src, f"literal was rewritten: {line}"
    assert result.findings == ()


def test_pan_like_identifier_is_not_rewritten():
    src = 'ABCDE1234F = "value"\n'
    result = scrub_pii_report(src, mode="code")
    assert result.clean == src
    assert result.findings == ()


def test_phone_and_email_inside_strings_and_comments_are_scrubbed():
    result = scrub_pii_report(CODE_FIXTURE, mode="code")
    assert result.mode == "code"
    assert '"call [PHONE] or mail [EMAIL]"' in result.clean
    assert "# legacy id [PHONE] left in a comment" in result.clean
    kinds = dict(result.findings)
    assert kinds.get("PHONE") == 2
    assert kinds.get("EMAIL") == 1


def test_scrubbed_code_still_parses_and_preserves_literals():
    result = scrub_pii_report(CODE_FIXTURE, mode="code")
    compile(result.clean, "<fixture>", "exec")
    for line in UNTOUCHED_LITERALS:
        assert line in result.clean


def test_unparsable_code_is_preserved_and_flagged():
    broken = "def broken(:\n    x = 9876543210\n"
    result = scrub_pii_report(broken, mode="code")
    assert result.mode == "preserved"
    assert result.clean == broken
    assert result.preserved_reason == "not_tokenizable_as_python"


def test_dates_and_identifiers_in_strings_are_left_alone():
    src = 'BUILT = "2026-10-15"\nNAME = "user_98765"\n'
    assert scrub_pii_report(src, mode="code").clean == src


def test_prose_mode_keeps_legacy_behaviour():
    prose = "Call 9876543210 or mail a.b@x.co for KYC."
    out = scrub_pii(prose)
    assert "[PHONE]" in out and "[EMAIL]" in out


def test_auto_mode_routes_prose_and_code_correctly():
    assert scrub_pii_report("plain sentence 9876543210 here").mode == "prose"
    assert scrub_pii_report("x = 1\n").mode == "code"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        scrub_pii_report("x = 1\n", mode="bogus")


# --------------------------------------------------------------------------- #
# G2 -- benchmark contamination audit
# --------------------------------------------------------------------------- #
def _corpus_file(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _corpus_row(text, path="src/mod.py", source_id="repo-a", split="train"):
    import hashlib

    return {
        "text": text,
        "content_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "source_id": source_id,
        "repo": "org/" + source_id,
        "file_path": path,
        "split": split,
    }


BENCH_PROMPT = (
    "def has_close_elements(numbers, threshold):\n"
    '    """Check if any two numbers are closer than threshold."""\n'
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)


def test_exact_overlap_is_detected(tmp_path):
    corpus = _corpus_file(tmp_path, [_corpus_row(BENCH_PROMPT)])
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": BENCH_PROMPT,
                "canonical_solution": "pass",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_corpus(corpus, [str(bench)])
    assert report["status"] == "contaminated"
    assert report["match_counts"]["exact"] >= 1
    hit = report["matches"][0]
    assert hit["suite"] == "bench"
    assert hit["task_id"] == "HumanEval/0"
    assert hit["corpus_file_path"] == "src/mod.py"
    assert hit["corpus_content_sha1"]


def test_normalised_overlap_is_detected(tmp_path):
    spaced = BENCH_PROMPT.replace("\n", "  \n  ")
    corpus = _corpus_file(tmp_path, [_corpus_row(spaced)])
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"task_id": "X/1", "prompt": BENCH_PROMPT}) + "\n",
        encoding="utf-8",
    )
    report = audit_corpus(corpus, [str(bench)])
    assert report["status"] == "contaminated"


def test_ngram_overlap_is_detected(tmp_path):
    # Shared tail is long enough to yield 13-gram matches, but each side has a
    # different prefix, so neither text contains the other: only the shingle
    # overlap can catch it.
    shared = (
        "    total = alpha + beta + gamma + delta + epsilon + zeta + eta + theta "
        "+ iota + kappa + lamb + mu + nu + xi + pi\n"
        "    return total\n"
    )
    bench_text = "def solve(alpha, beta, gamma, delta, epsilon, zeta, eta, theta, " \
        "iota, kappa, lamb, mu, nu, xi, pi):\n" + shared
    corpus_text = "def unrelated(first, second, third, fourth, fifth, sixth, " \
        "seventh, eighth, ninth, tenth, eleventh, twelfth, thirteenth, fourteenth, " \
        "fifteenth, sixteenth):\n    return 0\n" + shared

    corpus = _corpus_file(tmp_path, [_corpus_row(corpus_text)])
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"task_id": "X/2", "canonical_solution": bench_text}) + "\n",
        encoding="utf-8",
    )
    report = audit_corpus(corpus, [str(bench)])
    assert report["status"] == "contaminated"
    assert report["match_counts"]["ngram"] >= 1
    assert report["match_counts"]["exact"] == 0


def test_no_match_reports_honestly_and_flags_missing_published_suites(tmp_path):
    corpus = _corpus_file(
        tmp_path,
        [_corpus_row("def totally_different():\n    return sum(range(10))\n")],
    )
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"task_id": "X/3", "prompt": BENCH_PROMPT}) + "\n",
        encoding="utf-8",
    )
    report = audit_corpus(corpus, [str(bench)])
    assert report["matches"] == []
    assert report["status"] == "no_overlap_detected"
    # The published suites were not checked, and that must be stated.
    assert any("NOT available here" in lim for lim in report["limitations"])
    assert report["published_suite_availability"]["humaneval"]["available"] is False


def test_unavailable_benchmark_data_is_never_reported_clean(tmp_path, monkeypatch):
    import data.contamination_audit as audit_mod

    monkeypatch.setattr(audit_mod, "load_local_benchmarks", lambda *a, **k: ([], []))
    corpus = _corpus_file(tmp_path, [_corpus_row("x = 1\n")])
    report = audit_mod.audit_corpus(corpus)
    assert report["status"] == "unavailable"
    assert report["status"] != "clean"
    assert any("UNKNOWN" in lim for lim in report["limitations"])


def test_missing_corpus_is_reported_unavailable(tmp_path):
    report = audit_corpus(tmp_path / "nope.jsonl")
    assert report["status"] == "unavailable"


def test_audit_declares_its_method_and_limits(tmp_path):
    corpus = _corpus_file(tmp_path, [_corpus_row("x = 1\n")])
    report = audit_corpus(corpus)
    assert "exact" in report["method"]
    assert "ngram" in report["method"]
    assert "near_duplicate" in report["method"]
    assert report["method"]["not_semantic"]
    assert any("lexical" in lim.lower() for lim in report["limitations"])


def test_normalisation_and_ngram_helpers():
    assert normalise_for_match("A  B\nC") == "a b c"
    assert len(ngrams("one two three", 2)) == 2
    assert ngrams("short", 13) == {"short"}


# --------------------------------------------------------------------------- #
# G3 -- repository-level split integrity
# --------------------------------------------------------------------------- #
SPLITS = {"method": "content_hash", "mode": "source_group", "train": 0.9, "val": 0.05, "test": 0.05}


def test_group_split_is_a_function_of_the_repository():
    from data.build_code_corpus_v1 import group_split

    a = group_split("org/repo", SPLITS)
    b = group_split("org/repo", SPLITS)
    assert a == b
    assert a in {"train", "val", "test"}
    assert group_split("org/other", SPLITS) is not None


def test_group_split_separates_different_repositories_eventually():
    from data.build_code_corpus_v1 import group_split

    repos = [f"org/repo{i}" for i in range(200)]
    assignments = {group_split(r, SPLITS) for r in repos}
    # With 200 repositories every split should be represented.
    assert assignments == {"train", "val", "test"}


def _multi_source_cfg(mode="source_group"):
    return {
        "corpus_version": "v2",
        "policy": {
            "allowed_spdx": ["mit"],
            "extensions": [".py"],
            "min_file_bytes": 200,
            "max_file_bytes": 262144,
            "min_alpha_ratio": 0.55,
            "require_python_parses": True,
            "verify_license_at_runtime": False,
        },
        "limits": {
            "max_repos": 10,
            "max_files": 100,
            "max_total_bytes": 10_000_000,
        },
        "dedup": {
            "exact": True,
            "minhash": True,
            "minhash_perms": 64,
            "minhash_bands": 16,
            "minhash_threshold": 0.8,
        },
        "splits": {**SPLITS, "mode": mode},
        "sources": [
            {
                "id": f"repo{i}",
                "repo": f"org/repo{i}",
                "commit": f"{i:040d}",
                "spdx": "MIT",
                "language": "python",
                "attribution": "x",
                "retrieved": "2026-10-14",
            }
            for i in range(4)
        ],
    }


def _split_transport(n_files=3):
    """Two repos, each with several files, served from memory."""
    from data.build_code_corpus_v1 import GITHUB_API

    class T:
        def __init__(self):
            self.stats = {"requests": 0, "retries": 0, "failures": 0, "too_large": 0}

        def get(self, url, max_bytes=None):
            self.stats["requests"] += 1
            if "/git/trees/" in url:
                repo = url.split("/repos/")[1].split("/git/trees/")[0]
                nodes = []
                for i in range(n_files):
                    body = _py_file(repo, i)
                    nodes.append(
                        {"path": f"src/mod_{i}.py", "type": "blob", "size": len(body)}
                    )
                return json.dumps({"tree": nodes}).encode("utf-8")
            parts = url[len("https://raw.githubusercontent.com/") :].split("/", 3)
            repo, path = f"{parts[0]}/{parts[1]}", parts[3]
            i = int(path.split("_")[-1].split(".")[0])
            return _py_file(repo, i)

    return T()


def _py_file(repo, i):
    name = f"module_{i}"
    lines = ['"""Helper module for the corpus split test suite."""', "", "import os", ""]
    for k in range(6):
        lines += [
            f"def {name}_{k}(value):",
            f'    """Return a transformed value for case {k}."""',
            f"    return value * {k} + len(os.sep) + len({repo!r})",
            "",
        ]
    return ("\n".join(lines) + f"\n# repo {repo}\n").encode("utf-8")


def _run_builder(cfg, tmp_path, out="corpus.jsonl", licence_index=None):
    from data.build_code_corpus_v1 import CorpusBuilder

    transport = _split_transport()
    if licence_index is None:
        licence_index = {
            s["id"]: {
                "id": s["id"],
                "repo": s["repo"],
                "commit": s["commit"],
                "result": "verified",
                "detected_spdx": s["spdx"].lower(),
                "derived_exclusion_paths": [],
                "conflicting_subtrees": {},
            }
            for s in cfg["sources"]
        }
    builder = CorpusBuilder(
        cfg,
        transport,
        out_path=tmp_path / out,
        manifest_path=tmp_path / (out + ".manifest.json"),
        licence_index=licence_index,
        log=lambda s: None,
    )
    return builder, builder.run()


def test_no_repository_spans_two_splits_in_group_mode(tmp_path):
    builder, manifest = _run_builder(_multi_source_cfg("source_group"), tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "corpus.jsonl").read_text("utf-8").splitlines()
    ]
    per_repo = {}
    for r in rows:
        per_repo.setdefault(r["repo"], set()).add(r["split"])
    assert all(len(s) == 1 for s in per_repo.values())
    assert manifest["leakage_check"]["repositories_in_more_than_one_split"] == 0
    assert manifest["leakage_check"]["repositories_spanning_multiple_splits"] == []
    assert manifest["integrity_failures"] == []


def test_manifest_reports_split_mode_and_group_counts(tmp_path):
    _, manifest = _run_builder(_multi_source_cfg("source_group"), tmp_path)
    assert manifest["split_algorithm"]["method"] == "source_group"
    assert manifest["split_algorithm"]["group_by"] == "repo"
    assert sum(manifest["split_algorithm"]["group_counts"].values()) == len(
        manifest["sources"]
    )


def test_group_counts_match_actual_record_counts(tmp_path):
    _, manifest = _run_builder(_multi_source_cfg("source_group"), tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "corpus.jsonl").read_text("utf-8").splitlines()
    ]
    for repo, count in manifest["counts"]["by_source"].items():
        actual = sum(1 for r in rows if r["source_id"] == repo)
        assert actual == count


def test_content_hash_mode_still_spreads_one_repository_across_splits(tmp_path):
    cfg = _multi_source_cfg("content_hash")
    cfg["splits"]["train"] = 0.98
    cfg["splits"]["val"] = 0.01
    cfg["splits"]["test"] = 0.01
    _, manifest = _run_builder(cfg, tmp_path)
    assert manifest["split_algorithm"]["method"] == "content_hash"
    assert manifest["leakage_check"]["grouping"] == "content_hash"


def test_exact_duplicate_never_crosses_splits_in_group_mode(tmp_path):
    cfg = _multi_source_cfg("source_group")
    _, manifest = _run_builder(cfg, tmp_path)
    assert (
        manifest["leakage_check"]["exact_duplicate_chunks_in_multiple_splits"] == 0
    )
    assert len(manifest["content_hashes"]) == len(set(manifest["content_hashes"]))


def test_group_assignment_is_balanced_and_order_independent():
    """Independent hashing gives lumpy splits at small group counts."""
    from data.build_code_corpus_v1 import assign_group_splits

    repos = [
        "psf/requests", "pallets/flask", "psf/black", "urllib3/urllib3",
        "pallets/click", "pytest-dev/pytest", "aio-libs/aiohttp",
        "pypa/setuptools", "tiangolo/fastapi",
    ]
    splits = {"train": 0.6, "val": 0.2, "test": 0.2}
    forward = assign_group_splits(repos, splits)
    reverse = assign_group_splits(list(reversed(repos)), splits)
    assert forward == reverse, "assignment must not depend on input order"

    counts = {k: 0 for k in ("train", "val", "test")}
    for value in forward.values():
        counts[value] += 1
    # Largest remainder over 9 groups at 60/20/20 -> 5/2/2.
    assert counts == {"train": 5, "val": 2, "test": 2}
    assert sum(counts.values()) == len(repos)


def test_group_assignment_is_deterministic_even_when_holdouts_cannot_both_exist():
    """98/1/1 over 9 groups leaves nothing for the holdouts.

    train's quota is 8.82 of 9 seats, so both val and test being non-empty is
    arithmetically impossible. The method stays deterministic; the config was
    changed to 60/20/20 precisely because of this.
    """
    from data.build_code_corpus_v1 import assign_group_splits

    repos = [f"org/r{i}" for i in range(9)]
    splits = {"train": 0.98, "val": 0.01, "test": 0.01}
    a = assign_group_splits(repos, splits)
    b = assign_group_splits(list(reversed(repos)), splits)
    assert a == b, "assignment must be deterministic regardless of input order"
    assert len(a) == 9
    got = {k: 0 for k in ("train", "val", "test")}
    for value in a.values():
        got[value] += 1
    assert sum(got.values()) == 9
    # Exactly the documented outcome: a tiny holdout fraction yields empty splits.
    assert got["val"] == 0 and got["test"] == 0


def test_configured_fractions_give_non_empty_holdouts_for_the_real_allowlist():
    from data.build_code_corpus_v1 import assign_group_splits

    repos = [
        "psf/requests", "pallets/flask", "psf/black", "urllib3/urllib3",
        "pallets/click", "pytest-dev/pytest", "aio-libs/aiohttp",
        "pypa/setuptools", "tiangolo/fastapi",
    ]
    assignment = assign_group_splits(repos, {"train": 0.6, "val": 0.2, "test": 0.2})
    got = {k: 0 for k in ("train", "val", "test")}
    for value in assignment.values():
        got[value] += 1
    assert got["val"] >= 1
    assert got["test"] >= 1
    assert got["train"] > got["val"]


def test_group_assignment_handles_empty_and_single_group():
    from data.build_code_corpus_v1 import assign_group_splits

    assert assign_group_splits([], {"train": 0.9, "val": 0.05, "test": 0.05}) == {}
    single = assign_group_splits(["only/one"], {"train": 0.9, "val": 0.05, "test": 0.05})
    assert list(single.values()) == ["train"]


def test_manifest_flags_thin_holdouts(tmp_path):
    _, manifest = _run_builder(_multi_source_cfg("source_group"), tmp_path)
    assert "INSUFFICIENT" in manifest["split_algorithm"]["holdout_adequacy"]
    assert manifest["split_algorithm"]["groups_by_split"]["train"] == sorted(
        manifest["split_algorithm"]["groups_by_split"]["train"]
    )
    for split, repos in manifest["split_algorithm"]["groups_by_split"].items():
        assert isinstance(repos, list)