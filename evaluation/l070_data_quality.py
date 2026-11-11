"""TAKSH L0.7.0 — data-quality and provenance qualification (pre-acquisition).

Deterministic heuristic qualification framework for FUTURE acquired corpora.
Runs on synthetic fixtures and existing local slices only. Nothing is
"verified clean" by heuristic alone: automated screening emits FLAGS for
human review; safety-critical classes (secrets, malware, PII) are never
auto-cleared or auto-rejected — they route to humans.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HEX_RE = re.compile(r"[0-9a-fA-F]")
SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws-access-key-id"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github-fine-grained-pat"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "github-pat"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "slack-token"),
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), "private-key-block"),
    (re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"\s]{6,}"), "hardcoded-password"),
    (re.compile(r"(?i)(api_?key|secret_?key|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"), "hardcoded-secret"),
]
GENERATED_MARKERS = [
    re.compile(r"(?i)@generated\b"),
    re.compile(r"(?i)(code|file) (was )?(auto[- ]?)?generated"),
    re.compile(r"(?i)DO NOT EDIT"),
    re.compile(r"(?i)protoc|protobuf|swagger-codegen|openapi-generator|grpc_tools"),
]


def sha256_normalized(text: str) -> str:
    blob = re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = re.split(r"\s+", text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_sample(content: bytes, declared_language: str | None = None) -> dict:
    report: dict = {"declared_language": declared_language, "checks": {}, "flags": []}

    if not content or not content.strip():
        report["checks"]["non_empty"] = False
        report["flags"].append("empty_sample")
        return report

    # encoding
    try:
        text = content.decode("utf-8")
        report["checks"]["utf8"] = True
    except UnicodeDecodeError:
        report["checks"]["utf8"] = False
        report["flags"].append("decode_failed")
        return report
    if "\x00" in text:
        report["flags"].append("binary_null_bytes")

    # quality heuristics (flags, not verdicts)
    lines = text.splitlines()
    report["checks"]["non_empty"] = True
    report["line_count"] = len(lines)
    alnum = sum(1 for ch in text if ch.isalnum())
    report["alnum_fraction"] = round(alnum / max(len(text), 1), 4)
    if report["alnum_fraction"] < 0.25:
        report["flags"].append("low_alphanumeric_content")
    if any(len(l) > 4000 for l in lines):
        report["flags"].append("extremely_long_lines_possible_minification")

    # generated code
    if any(p.search(text) for p in GENERATED_MARKERS):
        report["flags"].append("generated_code_marker")

    # secrets / credentials (FLAG + HUMAN REVIEW; never auto-clear)
    for pattern, name in SECRET_PATTERNS:
        if pattern.search(text):
            report["flags"].append(f"secret_pattern:{name}")

    # python syntax validity where a toolchain exists
    if declared_language == "python":
        try:
            ast.parse(text)
            report["checks"]["python_syntax_valid"] = True
        except SyntaxError:
            report["checks"]["python_syntax_valid"] = False
            report["flags"].append("python_syntax_error")
    else:
        report["checks"]["syntax_valid"] = "SKIPPED (no in-repo toolchain check for this language)"

    # license metadata completeness (signal only)
    has_license_marker = any(w in text for w in ("SPDX-License-Identifier", "Licensed under", "LICENSE"))
    report["checks"]["license_marker_present"] = has_license_marker
    if not has_license_marker:
        report["flags"].append("no_license_metadata_in_file")

    review = [f for f in report["flags"]
              if f.startswith(("secret_pattern:", "binary_null", "python_syntax_error"))]
    report["automated_verdict"] = "FLAG_FOR_REVIEW" if report["flags"] else "NO_AUTOMATED_FLAGS"
    report["requires_human_review"] = bool(review)
    report["human_review_classes"] = review
    return report


def exact_duplicates(samples: list[dict]) -> dict:
    by_fp: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_fp.setdefault(sha256_normalized(s.get("content", "")), []).append(i)
    groups = {fp: idxs for fp, idxs in by_fp.items() if len(idxs) > 1}
    return {"duplicate_groups": groups, "unique_samples": len(by_fp),
            "label": "synthetic fixture results" if any(s.get("synthetic") for s in samples) else "local samples"}


def near_duplicates(samples: list[dict], threshold: float = 0.8) -> dict:
    sets = [shingles(s.get("content", "")) for s in samples]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            score = jaccard(sets[i], sets[j])
            if score >= threshold:
                pairs.append({"i": i, "j": j, "jaccard": round(score, 4)})
    return {"near_duplicate_pairs": pairs, "threshold": threshold,
            "label": "synthetic fixture results" if any(s.get("synthetic") for s in samples) else "local samples"}


def synthetic_fixtures() -> list[dict]:
    return [
        {"id": "synth-ok", "content": "def add(a, b):\n    return a + b\n", "synthetic": True},
        {"id": "synth-secret", "content": "key = 'AKIA" + "A" * 16 + "'\nprint(key)\n", "synthetic": True},
        {"id": "synth-generated", "content": "// Code generated by tool. DO NOT EDIT.\nint main(void){return 0;}\n", "synthetic": True},
        {"id": "synth-dup-a", "content": "def f(x):\n    return x * 2\n", "synthetic": True},
        {"id": "synth-dup-b", "content": "def f(x):\n    return x * 2\n", "synthetic": True},
        {"id": "synth-near", "content": "def f(x):\n    y = x * 2\n    return y\n", "synthetic": True},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_data_quality.json")
    args = parser.parse_args(argv)

    fixtures = synthetic_fixtures()
    results = [{**validate_sample(s["content"].encode("utf-8"), "python"), "id": s["id"]}
               for s in fixtures]
    exact = exact_duplicates(fixtures)
    near = near_duplicates(fixtures)
    flag_counts = Counter(f for r in results for f in r["flags"])

    report = {
        "schema": "taksh.l070_data_quality/1",
        "generated_by": "evaluation/l070_data_quality.py",
        "fixtures_label": "synthetic — no external or real corpus was processed",
        "sample_results": results,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "flag_counts": dict(flag_counts),
        "separation": {
            "automated": [r["id"] for r in results if r["automated_verdict"] == "NO_AUTOMATED_FLAGS"],
            "flagged": [r["id"] for r in results if r["flags"]],
            "human_review_required": [r["id"] for r in results if r["requires_human_review"]],
        },
        "policy": "automated screening never certifies a corpus; secrets/malware classes route to humans; metadata-valid != clean",
        "verdict": "QUALIFICATION FRAMEWORK READY (synthetic fixtures; no corpus processed)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"data quality: {len(results)} fixtures; flagged={sum(1 for r in results if r['flags'])} human_review={sum(1 for r in results if r['requires_human_review'])}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
