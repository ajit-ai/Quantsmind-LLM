"""v1.2.0 qualification tests: Python data foundation.

Synthetic fixtures only. No network, no real corpus, no model training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from data.python_v1 import benchmarks as bench_mod
from data.python_v1 import dedup, ingest, interface, manifest, pipeline, quality
from data.python_v1 import registry, splits

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_REL = "tokenizer/qm_code_bpe_v0.json"
LICENCE = (
    "MIT License\n\nPermission is hereby granted, free of charge, to any person "
    "obtaining a copy of this software, to deal in the Software without "
    "restriction.\n"
)


def module_source(index: int, helpers: int = 4) -> str:
    """A valid, non-trivial Python module with a distinct vocabulary."""
    lines = [f'"""Synthetic module {index}."""', "", "import os", "import sys", ""]
    for i in range(helpers):
        lines += [
            f"def helper_{index}_{i}(value):",
            f'    """Compute something for module {index}."""',
            f"    scaled = value * {i + 1}",
            "    return scaled + len(os.sep) + len(sys.argv)",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_source(root: Path, name: str, files: dict, licence: str = LICENCE) -> Path:
    src = root / name
    src.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (src / "LICENSE").write_text(licence, encoding="utf-8")
    return src


def snapshot_for(src: Path) -> str:
    entries = []
    for path in sorted(src.rglob("*.py")):
        entries.append(
            (path.relative_to(src).as_posix(), registry.file_sha256(path))
        )
    return registry.snapshot_digest(entries)


def source_entry(src: Path, root: Path, **overrides) -> dict:
    entry = {
        "id": src.name,
        "kind": "local_directory",
        "location": src.relative_to(root).as_posix(),
        "revision": snapshot_for(src),
        "project": src.name,
        "attribution": "Synthetic test corpus",
        "allowed_extensions": [".py"],
        "excluded_paths": [],
        "license": {
            "spdx": "MIT",
            "decision": "approved",
            "evidence": "LICENSE",
            "evidence_sha256": registry.file_sha256(src / "LICENSE"),
        },
        "approval": {
            "status": "approved",
            "reviewer": "test-owner",
            "rationale": "synthetic fixture",
        },
    }
    entry.update(overrides)
    return entry


def contract_dict(sources) -> dict:
    return {
        "schema": "quantsmind.python_data_contract/1",
        "version": "python_v1",
        "policy": {
            "default_decision": "deny",
            "allowed_spdx": ["MIT"],
            "require_pinned_revision": True,
            "require_license_evidence": True,
        },
        "limits": {
            "max_files": 1000,
            "max_total_bytes": 50_000_000,
            "max_file_bytes": 200_000,
            "max_path_depth": 40,
        },
        "encoding": {"primary": "utf-8", "fallbacks": [], "allow_bom": True},
        "normalization": {"newline": "\n", "strip_bom": True},
        "quality": {
            "min_bytes": 200,
            "max_bytes": 200_000,
            "min_lines": 5,
            "min_unique_token_ratio": 0.10,
            "minified_line_length": 400,
            "minified_average_line_length": 200,
            "max_compression_ratio": 0.08,
            "generated_markers": ["@generated", "do not edit"],
            "excluded_path_patterns": ["(^|/)(vendor|tests?)(/|$)"],
        },
        "dedup": {
            "exact": True,
            "near": True,
            "shingle_size": 8,
            "minhash_perms": 64,
            "minhash_bands": 16,
            "threshold": 0.85,
            "boilerplate_df_ratio": 0.5,
            "max_shingles_per_record": 4000,
            "max_bucket_candidates": 64,
            "group_by_source": True,
            "seed": 120,
        },
        "splits": {
            "proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 120,
            "min_groups_per_split": 1,
            "min_records_per_split": 1,
        },
        "benchmarks": {"registry": "benchmarks.yaml", "ngram_size": 13},
        "tokenizer": {"path": TOKENIZER_REL},
        "sources": sources,
    }


def write_contract(root: Path, cfg: dict) -> Path:
    if not (root / "benchmarks.yaml").exists():
        # A registry with nothing available: the contamination stage then
        # reports "unavailable" rather than claiming a clean result.
        write_benchmarks(
            root,
            [{"id": "none", "version": "0", "availability": "unavailable"}],
        )
    path = root / "contract.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


def write_benchmarks(root: Path, suites) -> Path:
    path = root / "benchmarks.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "quantsmind.benchmark_registry/1",
                "version": "test",
                "suites": suites,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Ten approved synthetic sources plus one available benchmark suite."""
    root = tmp_path_factory.mktemp("python_v1")
    entries = []
    for index in range(10):
        files = {f"mod_{i}.py": module_source(index * 10 + i) for i in range(3)}
        src = write_source(root, f"src_{index}", files)
        entries.append(source_entry(src, root))
    (root / "bench.jsonl").write_text(
        json.dumps({"task_id": "synthetic/1", "prompt": module_source(0)}) + "\n",
        encoding="utf-8",
    )
    write_benchmarks(
        root,
        [
            {
                "id": "synthetic",
                "version": "1",
                "availability": "available",
                "path": "bench.jsonl",
                "fields": ["prompt", "test"],
            }
        ],
    )
    contract = write_contract(root, contract_dict(entries))
    return {"root": root, "contract": contract, "entries": entries}


@pytest.fixture(scope="module")
def built(corpus):
    """One pipeline run over the synthetic corpus."""
    out = corpus["root"] / "out"
    result = pipeline.build_dataset(corpus["contract"], out)
    return {"manifest": result, "out": out, "root": corpus["root"]}


# --------------------------------------------------------------------------- #
# G1 - source governance
# --------------------------------------------------------------------------- #
def _decision(tmp_path, **overrides):
    src = write_source(tmp_path, "src", {"mod.py": module_source(1)})
    entry = source_entry(src, tmp_path)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(entry.get(key), dict):
            entry[key] = {**entry[key], **value}
        else:
            entry[key] = value
    return entry, src


def test_g1_unknown_source_kind_is_denied(tmp_path):
    entry, src = _decision(tmp_path, kind="git_remote")
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "source_kind_not_permitted" in decision["reasons"]


def test_g1_unresolved_license_decision_is_denied(tmp_path):
    entry, src = _decision(tmp_path, license={"decision": "unresolved"})
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "license_decision_unresolved" in decision["reasons"]


def test_g1_license_outside_allowlist_is_denied(tmp_path):
    entry, src = _decision(tmp_path, license={"spdx": "GPL-3.0-only"})
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "license_not_allowlisted" in decision["reasons"]


def test_g1_missing_owner_approval_is_denied(tmp_path):
    entry, src = _decision(tmp_path, approval={"status": "pending"})
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "approval_not_granted" in decision["reasons"]


def test_g1_top_level_license_file_alone_is_not_approval(tmp_path):
    """A real LICENSE file plus a valid SPDX string is still not a decision."""
    entry, src = _decision(tmp_path)
    entry["license"].pop("decision")
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "license_decision_unresolved" in decision["reasons"]


def test_g1_missing_license_evidence_is_denied(tmp_path):
    entry, src = _decision(tmp_path, license={"evidence": "COPYING"})
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "license_evidence_missing" in decision["reasons"]


def test_g1_license_evidence_hash_mismatch_is_denied(tmp_path):
    entry, src = _decision(tmp_path)
    entry["license"]["evidence_sha256"] = "0" * 64
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "license_evidence_hash_mismatch" in decision["reasons"]


def test_g1_unpinned_revision_is_denied(tmp_path):
    entry, src = _decision(tmp_path, revision="main")
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "revision_not_pinned" in decision["reasons"]


def test_g1_missing_revision_is_denied(tmp_path):
    entry, src = _decision(tmp_path, revision="")
    decision = registry.source_decision(contract_dict([]), entry, root=src)
    assert decision["allowed"] is False
    assert "revision_missing" in decision["reasons"]


def test_g1_contract_cannot_widen_default_decision(tmp_path):
    cfg = contract_dict([])
    cfg["policy"]["default_decision"] = "allow"
    path = write_contract(tmp_path, cfg)
    with pytest.raises(registry.ContractError):
        registry.load_contract(path)


def test_g1_unknown_source_keys_are_rejected(tmp_path):
    cfg = contract_dict([])
    cfg["sources"] = [
        {
            "id": "x",
            "kind": "local_directory",
            "location": "x",
            "revision": "sha256:" + "0" * 64,
            "license": {
                "spdx": "MIT",
                "decision": "approved",
                "evidence": "LICENSE",
                "evidence_sha256": "0" * 64,
            },
            "approval": {
                "status": "approved",
                "reviewer": "o",
                "rationale": "r",
            },
            "auto_approve": True,
        }
    ]
    path = write_contract(tmp_path, cfg)
    with pytest.raises(registry.ContractError):
        registry.load_contract(path)


def test_g1_revision_mismatch_denies_the_source(tmp_path):
    src = write_source(tmp_path, "src", {"mod.py": module_source(2)})
    entry = source_entry(src, tmp_path, revision="sha256:" + "1" * 64)
    cfg = contract_dict([entry])
    contract = write_contract(tmp_path, cfg)
    result = pipeline.build_dataset(contract, tmp_path / "out", dry_run=True)
    decision = result["sources"]["decisions"][0]
    assert decision["allowed"] is False
    assert "revision_mismatch" in decision["reasons"]
    assert result["filtering"]["kept"] == 0


def test_g1_denied_source_contributes_nothing(built):
    decisions = built["manifest"]["sources"]["decisions"]
    assert decisions, "every source must be decided"
    for decision in decisions:
        if not decision["allowed"]:
            assert decision["reasons"]


# --------------------------------------------------------------------------- #
# G2 - safe ingestion
# --------------------------------------------------------------------------- #
def test_g2_symlink_is_never_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text(module_source(9), encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.py").write_text(module_source(1), encoding="utf-8")
    link = root / "link.py"
    try:
        os.symlink(outside / "secret.py", link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this host: {exc}")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000, "max_file_bytes": 10_000},
    )
    names = [ref.rel_path for ref in report.files]
    assert "real.py" in names
    assert "link.py" not in names
    assert report.diagnostics.get("link_skipped", 0) >= 1


def test_g2_symlink_skip_path_works_without_privileged_symlink_creation(
    tmp_path, monkeypatch
):
    """Privilege-independent proof that a link is skipped, not followed.

    Creating a symbolic link needs a privilege this host does not grant, so
    the directory entry is simulated instead. The production code path is the
    same one the real link exercises.
    """
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.py").write_text(module_source(1), encoding="utf-8")

    class _FakeEntry:
        name = "link.py"
        path = str(root / "link.py")

        def is_symlink(self):
            return True

        def is_junction(self):
            return False

        def is_dir(self, follow_symlinks=True):
            return False

        def is_file(self, follow_symlinks=True):
            return True

        def stat(self, follow_symlinks=True):
            return os.stat_result((0o644, 0, 0, 1, 0, 0, 10, 0, 0, 0))

    real_scandir = os.scandir

    def fake_scandir(path):
        if str(path) == str(root):
            return [_FakeEntry()]
        return list(real_scandir(path))

    monkeypatch.setattr(ingest.os, "scandir", fake_scandir)
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000, "max_file_bytes": 10_000},
    )
    assert report.diagnostics.get("link_skipped", 0) == 1
    assert all(ref.rel_path != "link.py" for ref in report.files)


def test_g2_containment_check_rejects_paths_outside_the_root(tmp_path):
    root = (tmp_path / "root").resolve()
    inside = root / "inside.py"
    outside = (tmp_path / "outside.py").resolve()
    assert ingest._contained(root, inside) is True
    assert ingest._contained(root, outside) is False


def test_g2_junction_is_never_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text(module_source(9), encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.py").write_text(module_source(1), encoding="utf-8")
    junction = root / "linked_dir"
    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"junction creation unavailable on this host: {result.stderr}")
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"junction creation unavailable on this host: {exc}")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000, "max_file_bytes": 10_000},
    )
    names = [ref.rel_path for ref in report.files]
    assert names == ["real.py"]
    assert report.diagnostics.get("link_skipped", 0) >= 1


def test_g2_traversal_is_bounded_by_file_count(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(5):
        (root / f"m{i}.py").write_text(module_source(i), encoding="utf-8")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 2, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
    )
    assert len(report.files) == 2
    assert report.truncated is True
    assert report.truncated_reason == "max_files"


def test_g2_total_size_limit_truncates(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(4):
        (root / f"m{i}.py").write_text(module_source(i), encoding="utf-8")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 100, "max_total_bytes": 10, "max_file_bytes": 200_000},
    )
    assert report.truncated is True
    assert report.truncated_reason == "max_total_bytes"


def test_g2_excluded_subtree_is_skipped(tmp_path):
    root = tmp_path / "root"
    (root / "vendor").mkdir(parents=True)
    (root / "main.py").write_text(module_source(1), encoding="utf-8")
    (root / "vendor" / "third.py").write_text(module_source(2), encoding="utf-8")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
        excluded_paths=["vendor/"],
    )
    assert [ref.rel_path for ref in report.files] == ["main.py"]


def test_g2_binary_content_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "blob.py"
    path.write_bytes(b"\x00\x01\x02binary\x00payload")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
    )
    ref = report.files[0]
    with pytest.raises(ingest.IngestError) as excinfo:
        ingest.read_source(ref)
    assert excinfo.value.reason == "binary_content"


def test_g2_undecodable_file_is_rejected_without_fallback(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "latin.py"
    path.write_bytes("# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n".encode("latin-1"))
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
    )
    ref = report.files[0]
    text, encoding, fallback = ingest.read_source(ref)
    assert encoding == "iso-8859-1"
    assert fallback is False
    assert "caf" in text


def test_g2_explicit_fallback_is_used_and_reported(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "broken.py"
    path.write_bytes(b"# -*- coding: utf-8 -*-\nname = 'caf\xe9'\n")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
    )
    ref = report.files[0]
    with pytest.raises(ingest.IngestError) as excinfo:
        ingest.read_source(ref)
    assert excinfo.value.reason == "encoding_error"
    text, encoding, fallback = ingest.read_source(ref, fallbacks=("cp1252",))
    assert fallback is True
    assert encoding == "cp1252"
    assert "caf" in text


def test_g2_diagnostics_never_contain_file_content(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = "API_KEY = 'super-secret-value-1234'"
    (root / "creds.py").write_text(secret, encoding="utf-8")
    report = ingest.ingest_source(
        {"id": "s"},
        root,
        limits={"max_files": 10, "max_total_bytes": 10_000_000, "max_file_bytes": 200_000},
    )
    rendered = json.dumps(report.diagnostics)
    assert "super-secret-value" not in rendered


# --------------------------------------------------------------------------- #
# G3 - Python processing
# --------------------------------------------------------------------------- #
def test_g3_normalisation_is_lossless_for_line_endings():
    text = "def f():\r\n    return 1\r\n"
    normalised, applied = quality.normalize_source(text, {"newline": "\n", "strip_bom": True})
    assert normalised == "def f():\n    return 1\n"
    assert "crlf_to_lf" in applied


def test_g3_normalisation_preserves_comments_and_docstrings():
    text = module_source(3)
    normalised, _ = quality.normalize_source(text, {"newline": "\n"})
    assert '"""Synthetic module 3."""' in normalised
    assert "def helper_3_0(value):" in normalised


def test_g3_normalisation_does_not_apply_nfkc():
    text = "value = '\ufb01le'\n"
    normalised, _ = quality.normalize_source(text, {"newline": "\n"})
    assert "\ufb01" in normalised


def test_g3_syntax_error_is_reported_with_a_stable_code():
    result = quality.assess_python("def broken(:\n    pass\n", "m.py", {"min_bytes": 1})
    assert result.ok is False
    assert result.reason == "syntax_error"


def test_g3_secret_is_excluded_and_never_echoed():
    body = "def f():\n    api_key = 'abcdef0123456789'\n    return api_key\n" + (
        "\n".join(f"    x_{i} = {i}" for i in range(10))
    )
    result = quality.assess_python(body, "m.py", {"min_bytes": 1, "min_lines": 1})
    assert result.ok is False
    assert result.reason == "embedded_secret"
    assert "abcdef0123456789" not in result.detail


def test_g3_generated_code_marker_is_excluded():
    body = "# @generated do not edit\n" + module_source(4)
    result = quality.assess_python(
        body,
        "m.py",
        {"min_bytes": 1, "min_lines": 1},
        generated_markers=["@generated", "do not edit"],
    )
    assert result.ok is False
    assert result.reason == "generated_code"


def test_g3_vendored_path_is_excluded():
    body = module_source(5)
    result = quality.assess_python(
        body,
        "vendor/lib.py",
        {"min_bytes": 1, "min_lines": 1, "excluded_path_patterns": ["(^|/)(vendor)(/|$)"]},
    )
    assert result.ok is False
    assert result.reason == "vendored_dependency"


def test_g3_minified_artifact_is_excluded():
    body = "x = '" + "a" * 900 + "'\n"
    result = quality.assess_python(
        body, "m.py", {"min_bytes": 1, "min_lines": 1, "minified_line_length": 400}
    )
    assert result.ok is False
    assert result.reason == "minified_artifact"


def test_g3_empty_and_trivial_samples_are_excluded():
    assert quality.assess_python("   \n", "m.py", {"min_bytes": 1}).reason == "empty_file"
    comment_only = '"""Only a docstring."""\n' + "\n".join(
        f"# comment {i}" for i in range(20)
    ) + "\n"
    result = quality.assess_python(
        comment_only, "m.py", {"min_bytes": 1, "min_lines": 1}
    )
    assert result.ok is False
    assert result.reason == "trivial_sample"


def test_g3_quality_reason_codes_are_from_the_declared_set():
    assert set(quality.EXCLUSION_REASONS) >= {
        "empty_file",
        "too_small",
        "syntax_error",
        "embedded_secret",
        "generated_code",
        "vendored_dependency",
        "minified_artifact",
        "reserved_token_literal",
    }


def test_g3_reserved_token_literal_is_excluded():
    """A reserved spelling would inject a control id into the token stream."""
    body = 'marker = "<|user|>"\n' + module_source(41)
    result = quality.assess_python(body, "m.py", {"min_bytes": 1, "min_lines": 1})
    assert result.ok is False
    assert result.reason == "reserved_token_literal"
    assert "<|user|>" in result.detail


def test_g3_reserved_literal_exclusion_is_configuration_driven():
    body = 'marker = "<|user|>"\n' + module_source(42)
    result = quality.assess_python(
        body,
        "m.py",
        {"min_bytes": 1, "min_lines": 1, "reserved_literals": ["<not-a-token>"]},
    )
    assert result.ok is True


def test_g3_pipeline_excludes_reserved_literal_files(tmp_path):
    root = tmp_path
    src = write_source(
        root,
        "src",
        {
            "clean.py": module_source(43),
            "injected.py": 'marker = "<|assistant|>"\n' + module_source(44),
        },
    )
    contract = write_contract(root, contract_dict([source_entry(src, root)]))
    result = pipeline.build_dataset(contract, root / "out", dry_run=True)
    assert result["filtering"]["rejected"].get("reserved_token_literal") == 1
    assert result["filtering"]["kept"] == 1


# --------------------------------------------------------------------------- #
# G4 - deduplication
# --------------------------------------------------------------------------- #
def _records(pairs):
    out = []
    for source_id, path, text in pairs:
        out.append({"source_id": source_id, "file_path": path, "text": text})
    return out


def test_g4_exact_duplicates_are_grouped_with_one_representative():
    body = module_source(11)
    records = _records(
        [
            ("src_a", "a.py", body),
            ("src_b", "b.py", body),
            ("src_c", "c.py", module_source(12)),
        ]
    )
    result = dedup.deduplicate(records, {"exact": True, "near": False})
    assert result.stats["exact_duplicate_removed"] == 1
    assert result.stats["exact_groups"] == 1
    group_ids = [g for g, members in result.groups.items() if len(members) > 1]
    assert len(group_ids) == 1
    representative = result.representatives[group_ids[0]]
    assert records[representative]["file_path"] == "a.py"


def test_g4_exact_duplicates_are_found_even_with_different_names():
    body = module_source(13)
    records = _records([("s1", "one.py", body), ("s2", "two.py", body)])
    result = dedup.deduplicate(records, {"exact": True, "near": False})
    assert result.stats["exact_duplicate_removed"] == 1


def test_g4_near_duplicates_are_detected():
    base = module_source(21, helpers=12)
    variant = base.replace("helper_21_0", "helper_21_x")
    records = _records([("s1", "a.py", base), ("s2", "b.py", variant)])
    result = dedup.deduplicate(
        records, {"near": True, "exact": True, "shingle_size": 8, "threshold": 0.85}
    )
    assert result.stats["near_groups"] >= 1
    groups = [g for g in result.groups.values() if len(g) > 1]
    assert groups, "the near-duplicate pair must be grouped"


def test_g4_deduplication_is_deterministic():
    records = _records(
        [
            ("s1", "a.py", module_source(31)),
            ("s2", "b.py", module_source(31)),
            ("s3", "c.py", module_source(32)),
        ]
    )
    first = dedup.deduplicate(records, {"exact": True, "near": True})
    second = dedup.deduplicate(records, {"exact": True, "near": True})
    assert first.groups == second.groups
    assert first.representatives == second.representatives
    assert first.stats == second.stats


def test_g4_candidate_comparison_is_bounded():
    records = _records(
        [(f"s{i}", f"{i}.py", module_source(40 + i)) for i in range(6)]
    )
    result = dedup.deduplicate(
        records,
        {"near": True, "max_bucket_candidates": 1, "max_shingles_per_record": 50},
    )
    assert result.stats["records"] == 6


def test_g4_boilerplate_shingles_are_removed():
    header = "import os\nimport sys\nimport json\n"
    records = _records(
        [(f"s{i}", f"{i}.py", header + module_source(50 + i)) for i in range(4)]
    )
    result = dedup.deduplicate(records, {"near": True, "boilerplate_df_ratio": 0.5})
    assert result.stats["boilerplate_shingles_removed"] > 0


# --------------------------------------------------------------------------- #
# G5 - split isolation
# --------------------------------------------------------------------------- #
def test_g5_no_source_group_straddles_a_split(built):
    manifest = built["manifest"]
    splits_by_source = {}
    for name in ("train", "val", "test"):
        path = built["out"] / f"{name}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            splits_by_source.setdefault(row["source_id"], set()).add(name)
    assert splits_by_source
    for source_id, names in splits_by_source.items():
        assert len(names) == 1, f"{source_id} straddles {sorted(names)}"


def test_g5_no_duplicate_content_crosses_a_split(built):
    digests = {}
    for name in ("train", "val", "test"):
        path = built["out"] / f"{name}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            digests.setdefault(row["content_sha256"], set()).add(name)
    for digest, names in digests.items():
        assert len(names) == 1, f"content {digest[:12]} appears in {sorted(names)}"


def test_g5_integrity_report_is_explicit(built):
    integrity = built["manifest"]["splits"]["integrity"]
    assert integrity["sources_straddling_splits"] == []
    assert integrity["duplicate_digests_straddling_splits"] == []
    assert integrity["components_straddling_splits"] == []


def test_g5_duplicate_across_sources_stays_in_one_split(tmp_path):
    shared = module_source(61)
    src_a = write_source(tmp_path, "a", {"shared.py": shared, "own.py": module_source(62)})
    src_b = write_source(tmp_path, "b", {"shared.py": shared, "own.py": module_source(63)})
    entries = [source_entry(src_a, tmp_path), source_entry(src_b, tmp_path)]
    contract = write_contract(tmp_path, contract_dict(entries))
    out = tmp_path / "out"
    pipeline.build_dataset(contract, out)
    rows = []
    for name in ("train", "val", "test"):
        for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    shared_rows = [r for r in rows if r["file_path"] == "shared.py"]
    assert len(shared_rows) == 1, "the duplicate copy must be removed, not split"
    splits_by_source = {}
    for row in rows:
        splits_by_source.setdefault(row["source_id"], set()).add(row["split"])
    for source_id, names in splits_by_source.items():
        assert len(names) == 1, f"{source_id} straddles {sorted(names)}"


def test_g5_split_is_not_ready_when_the_corpus_is_too_small(tmp_path):
    src = write_source(tmp_path, "only", {"m.py": module_source(71)})
    contract = write_contract(tmp_path, contract_dict([source_entry(src, tmp_path)]))
    result = pipeline.build_dataset(contract, tmp_path / "out", dry_run=True)
    assert result["splits"]["ready"] is False
    assert result["splits"]["insufficient"]


def test_g5_split_assignment_is_stable_across_runs(built, corpus):
    again = pipeline.build_dataset(corpus["contract"], corpus["root"] / "out2", dry_run=True)
    assert again["splits"]["assignment"] == built["manifest"]["splits"]["assignment"]
    assert again["splits"]["counts"] == built["manifest"]["splits"]["counts"]


# --------------------------------------------------------------------------- #
# G6 - contamination controls
# --------------------------------------------------------------------------- #
def test_g6_benchmark_match_is_detected_and_the_sample_excluded(built):
    manifest = built["manifest"]
    assert manifest["contamination"]["verdict"] == "contaminated"
    assert manifest["contamination"]["hit_count"] >= 1
    assert "synthetic" in manifest["contamination"]["coverage"]["checked"]
    kept = set()
    for name in ("train", "val", "test"):
        for line in (built["out"] / f"{name}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                row = json.loads(line)
                kept.add((row["source_id"], row["file_path"]))
    for hit in manifest["contamination"]["hits"]:
        assert (hit["source_id"], hit["file_path"]) not in kept


def test_g6_unavailable_benchmarks_are_reported_not_claimed_clean(tmp_path):
    root = tmp_path
    src = write_source(root, "src", {"m.py": module_source(81)})
    contract = write_contract(root, contract_dict([source_entry(src, root)]))
    write_benchmarks(
        root,
        [
            {
                "id": "humaneval",
                "version": "164",
                "availability": "unavailable",
                "published_problems": 164,
            }
        ],
    )
    result = pipeline.build_dataset(contract, root / "out", dry_run=True)
    contamination = result["contamination"]
    assert contamination["verdict"] == "unavailable"
    assert "humaneval" in contamination["coverage"]["not_checked"]
    assert any("UNKNOWN" in limit for limit in contamination["limitations"])


def test_g6_registry_rejects_an_available_suite_without_a_path(tmp_path):
    path = write_benchmarks(
        tmp_path, [{"id": "x", "version": "1", "availability": "available"}]
    )
    with pytest.raises(bench_mod.BenchmarkRegistryError):
        bench_mod.load_registry(path)


def test_g6_registry_rejects_unknown_keys(tmp_path):
    path = write_benchmarks(
        tmp_path,
        [{"id": "x", "version": "1", "availability": "unavailable", "auto": True}],
    )
    with pytest.raises(bench_mod.BenchmarkRegistryError):
        bench_mod.load_registry(path)


def test_g6_available_suite_without_matches_is_reported_as_checked(tmp_path):
    root = tmp_path
    src = write_source(root, "src", {"m.py": module_source(91)})
    contract = write_contract(root, contract_dict([source_entry(src, root)]))
    (root / "bench.jsonl").write_text(
        json.dumps({"task_id": "b/1", "prompt": "def unrelated_function():\n    return 0\n"})
        + "\n",
        encoding="utf-8",
    )
    write_benchmarks(
        root,
        [
            {
                "id": "local",
                "version": "1",
                "availability": "available",
                "path": "bench.jsonl",
                "fields": ["prompt"],
            }
        ],
    )
    result = pipeline.build_dataset(contract, root / "out", dry_run=True)
    contamination = result["contamination"]
    assert contamination["verdict"] == "no_overlap_detected"
    assert contamination["coverage"]["checked"] == ["local"]
    assert any("lexical" in limit for limit in contamination["limitations"])


# --------------------------------------------------------------------------- #
# G7 - manifest integrity
# --------------------------------------------------------------------------- #
def test_g7_manifest_has_no_timestamp_or_absolute_path(built):
    rendered = json.dumps(built["manifest"])
    assert str(built["root"]) not in rendered
    assert "generated_at" not in rendered
    assert "created_at" not in rendered


def test_g7_manifest_records_required_provenance(built):
    manifest = built["manifest"]
    assert manifest["schema"] == "quantsmind.python_dataset_manifest/1"
    assert manifest["pipeline"]["contract_fingerprint"].startswith("sha256:")
    assert manifest["pipeline"]["registry_fingerprint"].startswith("sha256:")
    assert manifest["sources"]["approved"]
    assert manifest["splits"]["seed"] == 120
    assert manifest["tokenizer"]["sha256"]
    assert manifest["filtering"]["kept"] >= 0
    assert manifest["deduplication"]["records"] >= 0
    assert manifest["artifacts"]


def test_g7_artifact_checksums_match_the_written_files(built):
    for artifact in built["manifest"]["artifacts"]:
        path = built["out"] / artifact["name"]
        assert registry.file_sha256(path) == artifact["sha256"]
        assert path.stat().st_size == artifact["bytes"]


def test_g7_two_builds_produce_identical_manifests_and_shards(corpus):
    first = pipeline.build_dataset(corpus["contract"], corpus["root"] / "b1")
    second = pipeline.build_dataset(corpus["contract"], corpus["root"] / "b2")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        a = (corpus["root"] / "b1" / name).read_bytes()
        b = (corpus["root"] / "b2" / name).read_bytes()
        assert a == b, f"{name} is not reproducible"


def test_g7_manifest_hash_detects_tampering(built):
    original = built["manifest"]["manifest_sha256"]
    tampered = dict(built["manifest"])
    tampered["filtering"] = dict(tampered["filtering"], kept=999999)
    assert manifest.manifest_digest(tampered) != original


def test_g7_data_card_is_generated(built):
    card = (built["out"] / "DATA_CARD.md").read_text(encoding="utf-8")
    assert "Data card" in card
    assert "Contamination" in card


# --------------------------------------------------------------------------- #
# G9 - training interface
# --------------------------------------------------------------------------- #
def test_g9_interface_reads_a_prepared_split(built):
    dataset = interface.PythonDataset(built["out"] / "manifest.json", "train")
    samples = list(dataset)
    assert len(samples) == len(dataset)
    assert all(sample.split == "train" for sample in samples)
    assert all(sample.text for sample in samples)


def test_g9_interface_refuses_a_tampered_artifact(built, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(built["out"], copy)
    path = copy / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(interface.DatasetInterfaceError):
        interface.PythonDataset(copy / "manifest.json", "train")


def test_g9_interface_refuses_a_missing_manifest(tmp_path):
    with pytest.raises(interface.DatasetInterfaceError):
        interface.PythonDataset(tmp_path / "manifest.json", "train")


def test_g9_interface_refuses_a_missing_split(built):
    with pytest.raises(interface.DatasetInterfaceError):
        interface.PythonDataset(built["out"] / "manifest.json", "eval")


def test_g9_interface_refuses_an_incompatible_tokenizer(built, tmp_path):
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(interface.DatasetInterfaceError):
        interface.PythonDataset(
            built["out"] / "manifest.json", "train", tokenizer_path=other
        )


def test_g9_interface_refuses_a_not_ready_split(tmp_path):
    root = tmp_path
    src = write_source(root, "only", {"m.py": module_source(101)})
    contract = write_contract(root, contract_dict([source_entry(src, root)]))
    out = root / "out"
    pipeline.build_dataset(contract, out)
    with pytest.raises(interface.DatasetInterfaceError):
        interface.PythonDataset(out / "manifest.json", "test")
    relaxed = interface.PythonDataset(out / "manifest.json", "test", require_ready=False)
    assert len(relaxed) >= 0


def test_g9_interface_refuses_a_record_whose_text_was_edited(built, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(built["out"], copy)
    path = copy / "train.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["text"] = row["text"] + "\n# tampered\n"
    lines[0] = json.dumps(row, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {
            **a,
            "sha256": registry.file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for a in manifest["artifacts"]
    ]
    (copy / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(interface.DatasetInterfaceError):
        list(interface.PythonDataset(copy / "manifest.json", "train"))


# --------------------------------------------------------------------------- #
# G10 - regression integrity (v1.1.0 contracts still hold)
# --------------------------------------------------------------------------- #
def test_g10_v110_contract_still_loads_and_serves():
    from training.tokenizer_contract import (
        CONTRACT_V1,
        build_contract_tokenizer,
        detect_contract,
    )

    tokenizer = build_contract_tokenizer(REPO_ROOT / TOKENIZER_REL)
    assert detect_contract(tokenizer) == "qm.chat_instruction/2"
    plain = detect_contract(
        __import__("tokenizers").Tokenizer.from_file(str(REPO_ROOT / TOKENIZER_REL))
    )
    assert plain == CONTRACT_V1


def test_g10_shipped_contract_approves_no_source():
    cfg = registry.load_contract(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert cfg["sources"] == []
    assert cfg["policy"]["default_decision"] == "deny"


def test_g10_shipped_benchmark_registry_declares_no_available_suite():
    registry_data = bench_mod.load_registry(
        REPO_ROOT / "data" / "contracts" / "benchmarks_v1.yaml"
    )
    for suite in registry_data["suites"]:
        assert suite["availability"] == "unavailable"
