"""Mandatory licence enforcement: a source cannot enter a corpus without evidence.

Covers the gate itself (B1), the agreed quarantine decisions (B2), nested-licence
detection (B3) and enforcement at the point of inclusion (B4).

All offline: no network, no GitHub quota, and no touching of the baseline corpus
artifacts in ``data/``.
"""

import json

import pytest

from data.build_code_corpus_v1 import (
    LicenceGateError,
    CorpusBuilder,
    licence_decision,
    load_licence_verification,
)
from data.licence_paths import normalise_prefix, path_within
from data.verify_licenses import derive_exclusions, subtree_of

REPO = "org/example"
COMMIT = "b" * 40


def source(**over):
    base = {
        "id": "example",
        "repo": REPO,
        "commit": COMMIT,
        "spdx": "MIT",
        "language": "python",
        "attribution": "Copyright Example",
        "retrieved": "2026-10-15",
    }
    base.update(over)
    return base


def evidence(result="verified", **over):
    entry = {
        "id": "example",
        "repo": REPO,
        "commit": COMMIT,
        "result": result,
        "detected_spdx": "mit",
        "derived_exclusion_paths": [],
        "conflicting_subtrees": {},
        "notice_obligations": ["retain the copyright notice"],
    }
    entry.update(over)
    return entry


# --------------------------------------------------------------------------- #
# B1 -- the gate is blocking
# --------------------------------------------------------------------------- #
def test_no_evidence_blocks_the_source():
    decision = licence_decision(source(), None, {})
    assert decision["allowed"] is False
    assert "no licence verification evidence" in decision["reason"]


@pytest.mark.parametrize(
    "result",
    ["failed", "unavailable", "missing", "ambiguous", "conflict",
     "not_permitted", "unrecognised", None],
)
def test_every_non_verified_result_is_blocked(result):
    decision = licence_decision(source(), evidence(result=result), {})
    assert decision["allowed"] is False, f"{result} must block inclusion"


def test_verified_source_is_allowed():
    decision = licence_decision(source(), evidence(), {})
    assert decision["allowed"] is True
    assert decision["verification_result"] == "verified"


def test_verified_with_quarantine_is_allowed_when_subtrees_are_excluded():
    ev = evidence(
        result="verified_with_quarantine",
        conflicting_subtrees={"thirdparty": ["gpl-3.0"]},
        derived_exclusion_paths=["thirdparty/"],
    )
    decision = licence_decision(source(), ev, {})
    assert decision["allowed"] is True
    assert decision["exclusion_paths"] == ["thirdparty/"]


def test_verified_with_quarantine_is_blocked_when_subtree_is_not_excluded():
    """A conflict that is not actually excluded must not pass the gate."""
    ev = evidence(
        result="verified_with_quarantine",
        conflicting_subtrees={"thirdparty": ["gpl-3.0"]},
        derived_exclusion_paths=[],
    )
    decision = licence_decision(source(), ev, {})
    assert decision["allowed"] is False
    assert "not excluded" in decision["reason"]


def test_missing_verification_artifact_is_an_error(tmp_path):
    with pytest.raises(LicenceGateError) as exc:
        load_licence_verification(
            str(tmp_path / "absent.json"), commit_by_source={"example": COMMIT}
        )
    assert "not found" in str(exc.value)
    assert "verify_licenses" in str(exc.value)


def test_malformed_verification_artifact_is_an_error(tmp_path):
    path = tmp_path / "v.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LicenceGateError):
        load_licence_verification(str(path), commit_by_source={})


def test_verification_for_a_different_revision_is_rejected(tmp_path):
    """Evidence must correspond to the revision actually being built."""
    path = tmp_path / "v.json"
    path.write_text(
        json.dumps(
            {"sources": [
                {"id": "example", "repo": REPO, "commit": "old", "result": "verified"}
            ]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(LicenceGateError) as exc:
        load_licence_verification(str(path), commit_by_source={"example": COMMIT})
    assert "different revision" in str(exc.value)


def test_loader_indexes_by_source_id(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"sources": [evidence()]}), encoding="utf-8")
    loaded = load_licence_verification(str(path), commit_by_source={})
    assert loaded["by_source"]["example"]["result"] == "verified"


# --------------------------------------------------------------------------- #
# B2 -- the agreed quarantine decisions
# --------------------------------------------------------------------------- #
def test_black_and_setuptools_exclusions_are_configured():
    """The project decisions must be present in the shipped configuration."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(
        (root / "configs" / "data" / "approved_code_sources.yaml").read_text(
            encoding="utf-8"
        )
    )
    by_id = {s["id"]: s for s in cfg["sources"]}

    black = by_id["psf-black"]
    assert black["commit"] == "11cff1312bf96e02bce9a467d0b5a2dd65300642"
    assert "src/blib2to3/" in black.get("quarantine_paths", [])

    setuptools = by_id["pypa-setuptools"]
    assert setuptools["commit"] == "4e612f9e44ecfad130d0b2a27a2dce0740599b4b"
    assert (
        "setuptools/config/_validate_pyproject/" in setuptools.get("quarantine_paths", [])
    )


def test_exclusion_covers_descendants():
    prefixes = ["src/blib2to3/"]
    assert path_within("src/blib2to3/pgen2/driver.py", prefixes)
    assert path_within("src/blib2to3", prefixes)
    assert not path_within("src/black/mod.py", prefixes)
    # A sibling directory sharing a name prefix must not be swept in.
    assert not path_within("src/blib2to3_extra/mod.py", prefixes)


def test_no_prefixes_means_nothing_is_excluded():
    assert not path_within("anything/at/all.py", [])
    assert not path_within("anything/at/all.py", None)


def test_normalise_prefix_is_idempotent():
    assert normalise_prefix("a/b") == "a/b/"
    assert normalise_prefix("a/b/") == "a/b/"


# --------------------------------------------------------------------------- #
# B3 -- nested licence detection
# --------------------------------------------------------------------------- #
def test_subtree_of_isolates_the_directory():
    assert subtree_of("LICENSE") == ""
    assert subtree_of("src/blib2to3/LICENSE") == "src/blib2to3"
    assert subtree_of("a/b/c/NOTICE") == "a/b/c"


def test_conflicting_subtree_is_excluded():
    prefixes, rows = derive_exclusions({"thirdparty": {"gpl-3.0"}}, root_family="mit")
    assert prefixes == ["thirdparty/"]
    assert rows[0]["excluded"] is True
    assert rows[0]["rule"] == "conflicting_family"
    assert rows[0]["reason"]


def test_unclassifiable_subtree_is_excluded():
    prefixes, rows = derive_exclusions({"weird": {None}}, root_family="mit")
    assert prefixes == ["weird/"]
    assert rows[0]["rule"] == "unknown_text"


def test_matching_subtree_is_not_excluded():
    prefixes, rows = derive_exclusions({"sub": {"mit"}}, root_family="mit")
    assert prefixes == []
    assert rows[0]["excluded"] is False
    assert rows[0]["rule"] == "matching_family"


def test_subtree_without_a_licence_file_is_not_excluded_by_detection():
    """Absence of a licence file is not a licence claim, and is not inferred here."""
    prefixes, _ = derive_exclusions({}, root_family="mit")
    assert prefixes == []


def test_root_is_never_excluded_by_nested_detection():
    """A root-level conflict is the caller's problem, not a nested exclusion."""
    prefixes, rows = derive_exclusions(
        {"": {"gpl-3.0"}, "sub": {"gpl-3.0"}}, root_family="gpl-3.0"
    )
    assert prefixes == [], "the root must never appear as an exclusion prefix"
    assert all(r["subtree"] for r in rows), "no row should describe the root"


def test_nested_mismatch_against_root_is_still_excluded():
    prefixes, _ = derive_exclusions(
        {"": {"gpl-3.0"}, "sub": {"mit"}}, root_family="gpl-3.0"
    )
    assert prefixes == ["sub/"]


def test_derivation_is_deterministic():
    first, _ = derive_exclusions({"b": {"gpl-3.0"}, "a": {"mit"}}, root_family="mit")
    second, _ = derive_exclusions({"a": {"mit"}, "b": {"gpl-3.0"}}, root_family="mit")
    assert first == second == ["b/"]


# --------------------------------------------------------------------------- #
# B4 -- enforcement at the point of inclusion
# --------------------------------------------------------------------------- #
class _Transport:
    """Serves a small tree so the builder can decide inclusion."""

    def __init__(self, paths):
        self.paths = paths
        self.raw_requests = []
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "too_large": 0}

    def get(self, url, max_bytes=None):
        self.stats["requests"] += 1
        if "/git/trees/" in url:
            return json.dumps(
                {
                    "tree": [
                        {"path": p, "type": "blob", "size": len(_body(i))}
                        for i, p in enumerate(self.paths)
                    ]
                }
            ).encode()
        if url.startswith("https://raw.githubusercontent.com/"):
            self.raw_requests.append(url)
            index = int(url.rsplit("_", 1)[-1].split(".")[0])
            return _body(index)
        raise AssertionError(url)


def _body(index):
    lines = ['"""Example module for the licence gate tests."""', "", "import os", ""]
    for k in range(5):
        lines += [
            f"def fn_{k}(value):",
            f'    """Return a value for case {k}."""',
            f"    return value * {k} + len(os.sep)",
            "",
        ]
    return ("\n".join(lines) + f"\n# variant {index}\n").encode()


def _cfg(**over):
    cfg = {
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
            "max_repos": 5,
            "max_files": 100,
            "max_total_bytes": 5_000_000,
        },
        "dedup": {
            "exact": True,
            "minhash": True,
            "minhash_perms": 64,
            "minhash_bands": 16,
            "minhash_threshold": 0.8,
        },
        "splits": {"method": "content_hash", "train": 1.0, "val": 0.0, "test": 0.0},
        "sources": [source()],
    }
    cfg.update(over)
    return cfg


def _run(tmp_path, cfg, paths, licence_index, transport=None):
    builder = CorpusBuilder(
        cfg,
        transport or _Transport(paths),
        out_path=tmp_path / "corpus.jsonl",
        manifest_path=tmp_path / "corpus.manifest.json",
        licence_index=licence_index,
        log=lambda s: None,
    )
    return builder, builder.run()


def _rows(tmp_path):
    shard = tmp_path / "corpus.jsonl"
    if not shard.exists():
        return []
    return [
        json.loads(line)
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_blocked_source_contributes_no_records_and_fails_the_run(tmp_path):
    """A source that failed verification must not appear in the corpus at all."""
    transport = _Transport(["src/mod_0.py", "src/mod_1.py"])
    _, manifest = _run(
        tmp_path,
        _cfg(),
        ["src/mod_0.py", "src/mod_1.py"],
        {"example": evidence(result="failed")},
        transport=transport,
    )
    assert _rows(tmp_path) == []
    assert manifest["status"] == "failed"
    assert manifest["counts"]["kept"] == 0
    assert manifest["sources"][0]["status"] == "rejected_licence"
    assert manifest["sources"][0]["error"]
    # Nothing was even fetched for the blocked source.
    assert transport.raw_requests == []


def test_missing_evidence_blocks_even_a_perfectly_good_source(tmp_path):
    _, manifest = _run(tmp_path, _cfg(), ["src/mod_0.py"], {})
    assert _rows(tmp_path) == []
    assert manifest["sources"][0]["status"] == "rejected_licence"
    assert "no licence verification evidence" in manifest["sources"][0]["error"]


def test_verified_source_is_included(tmp_path):
    _, manifest = _run(
        tmp_path, _cfg(), ["src/mod_0.py"], {"example": evidence()}
    )
    assert len(_rows(tmp_path)) == 1
    assert manifest["status"] == "complete"


def test_excluded_subtree_files_are_never_fetched_or_written(tmp_path):
    """Exclusion happens at the tree stage, before any transfer."""
    # "embedded" is deliberately not a directory the generic path filter already
    # excludes, so this exercises the licence gate rather than the vendor filter.
    paths = ["src/mod_0.py", "embedded/lib_0.py", "embedded/lib_1.py"]
    ev = evidence(
        result="verified_with_quarantine",
        conflicting_subtrees={"embedded": ["gpl-3.0"]},
        derived_exclusion_paths=["embedded/"],
    )
    transport = _Transport(paths)
    _, manifest = _run(
        tmp_path, _cfg(), paths, {"example": ev}, transport=transport
    )
    rows = _rows(tmp_path)
    assert [r["file_path"] for r in rows] == ["src/mod_0.py"]
    assert "embedded" not in " ".join(transport.raw_requests)
    assert manifest["counts"]["rejected"]["quarantined_licence_conflict"] == 2


def test_config_quarantine_alone_is_enforced(tmp_path):
    """The agreed project decisions work even without derived evidence."""
    paths = ["src/mod_0.py", "src/blib2to3/lib_0.py"]
    cfg = _cfg(sources=[source(quarantine_paths=["src/blib2to3/"])])
    transport = _Transport(paths)
    _, manifest = _run(
        tmp_path, cfg, paths, {"example": evidence()}, transport=transport
    )
    assert [r["file_path"] for r in _rows(tmp_path)] == ["src/mod_0.py"]
    assert manifest["counts"]["rejected"]["quarantined_licence_conflict"] == 1


def test_unexcluded_conflict_blocks_the_source_entirely(tmp_path):
    """If a conflict is declared but not excluded, nothing from it may ship."""
    paths = ["src/mod_0.py", "embedded/lib_0.py"]
    ev = evidence(
        result="verified_with_quarantine",
        conflicting_subtrees={"embedded": ["gpl-3.0"]},
        derived_exclusion_paths=[],
    )
    transport = _Transport(paths)
    _, manifest = _run(
        tmp_path, _cfg(), paths, {"example": ev}, transport=transport
    )
    assert _rows(tmp_path) == []
    assert manifest["sources"][0]["status"] == "rejected_licence"
    assert "not excluded" in manifest["sources"][0]["error"]
    assert transport.raw_requests == []


def test_manifest_records_the_gate_decision_per_source(tmp_path):
    ev = evidence(
        result="verified_with_quarantine",
        conflicting_subtrees={"embedded": ["gpl-3.0"]},
        derived_exclusion_paths=["embedded/"],
    )
    paths = ["src/mod_0.py", "embedded/lib_0.py"]
    _, manifest = _run(tmp_path, _cfg(), paths, {"example": ev})
    licensing = manifest["licensing"]
    assert licensing["policy_version"]
    entry = licensing["per_source"][0]
    assert entry["id"] == "example"
    assert entry["verification_result"] == "verified_with_quarantine"
    assert entry["allowed"] is True
    assert entry["exclusion_paths"] == ["embedded/"]
    assert "derived_from_licence_files" in entry["exclusion_sources"]
    assert entry["files_excluded_by_licence"] == 1


def test_baseline_corpus_artifacts_are_not_touched(tmp_path):
    """The gate must not read, rewrite or regenerate the shipped corpora."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    baseline = root / "data" / "code_corpus_v1.jsonl"
    if not baseline.exists():
        pytest.skip("dataset artifacts are not part of the source release")
    before = baseline.stat().st_mtime_ns
    size_before = baseline.stat().st_size

    _run(tmp_path, _cfg(), ["src/mod_0.py"], {"example": evidence()})

    assert baseline.exists()
    assert baseline.stat().st_size == size_before
    assert baseline.stat().st_mtime_ns == before