"""TAKSH L0.7.1 — structural parsing and quality validation.

Classifies each sample into the L0.7.1 quality states. Toolchains are used
only where they already exist in-repo and are safe (Python's ast). Everything
else is honestly NOT_CHECKED — never guessed. Automated screening flags;
humans adjudicate.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l070_data_quality import (  # noqa: E402
    GENERATED_MARKERS,
    SECRET_PATTERNS,
)

STATES = ["VALIDATED", "VALIDATED_WITH_WARNINGS", "NO_AUTOMATED_FLAGS",
          "HUMAN_REVIEW_REQUIRED", "INVALID", "BLOCKED", "NOT_CHECKED"]


def classify_sample(text: str, language: str) -> dict:
    flags: list[str] = []
    checks: dict = {"non_empty": True, "encoding": True, "syntax": "NOT_CHECKED"}

    if not text.strip():
        return {"quality_status": "INVALID", "flags": ["empty_sample"], "checks": {"non_empty": False}}

    alnum = sum(1 for ch in text if ch.isalnum())
    alnum_fraction = alnum / max(len(text), 1)
    if alnum_fraction < 0.20:
        flags.append("low_alphanumeric_content")
    if any(len(line) > 4000 for line in text.splitlines()):
        flags.append("extremely_long_lines")
    if any(p.search(text) for p in GENERATED_MARKERS):
        flags.append("generated_code_marker")

    secret_flags = [name for pattern, name in SECRET_PATTERNS if pattern.search(text)]
    flags += [f"secret_pattern:{name}" for name in secret_flags]

    if language == "python":
        probe = text.replace("\ufeff", "")          # encoding artifact; never valid Python
        lines = probe.split("\n")
        separator_lines = sum(1 for l in lines if l.strip() == "<|doc|>")
        try:
            ast.parse(probe)
            checks["syntax"] = "VALIDATED"
            if separator_lines:
                flags.append("multi_document_bundle")
        except SyntaxError:
            if separator_lines:
                segments, current = [], []
                for l in lines:
                    if l.strip() == "<|doc|>":
                        segments.append("\n".join(current))
                        current = []
                    else:
                        current.append(l)
                segments.append("\n".join(current))
                failures = 0
                for seg in segments:
                    if not seg.strip():
                        continue
                    try:
                        ast.parse(seg)
                    except SyntaxError:
                        failures += 1
                checks["bundle_segments"] = len([s for s in segments if s.strip()])
                flags.append("multi_document_bundle")
                if failures:
                    checks["syntax"] = "FAILED"
                    flags.append(f"python_syntax_error_x{failures}")
                else:
                    checks["syntax"] = "VALIDATED (bundle)"
            else:
                checks["syntax"] = "FAILED"
                flags.append("python_syntax_error")
        if "\ufeff" in text:
            flags.append("interior_bom_removed_for_parse")
    else:
        checks["syntax"] = "NOT_CHECKED (no in-repo toolchain for this language)"

    has_license_marker = any(w in text for w in ("SPDX-License-Identifier", "Licensed under", "LICENSE"))
    checks["license_marker_present"] = has_license_marker
    if not has_license_marker:
        flags.append("no_license_metadata_in_file")

    if secret_flags:
        return {"quality_status": "HUMAN_REVIEW_REQUIRED", "flags": flags, "checks": checks}
    if checks.get("syntax") == "FAILED":
        return {"quality_status": "INVALID", "flags": flags, "checks": checks}
    if flags:
        return {"quality_status": "VALIDATED_WITH_WARNINGS", "flags": flags, "checks": checks}
    if isinstance(checks.get("syntax"), str) and checks["syntax"].startswith("VALIDATED"):
        return {"quality_status": "VALIDATED", "flags": flags, "checks": checks}
    return {"quality_status": "NO_AUTOMATED_FLAGS",
            "flags": ["syntax not checked for this language"], "checks": checks}


def parser_adapter_status(language: str) -> dict:
    if language == "python":
        return {"language": "python", "adapter": "ast.parse", "available": True}
    if language in ("c", "cpp", "java", "go", "rust", "csharp", "javascript", "typescript",
                    "kotlin", "swift", "dart", "php", "ruby", "bash", "sql"):
        return {"language": language, "adapter": None,
                "available": False,
                "limitation": "no in-repo toolchain pinned; syntax validation skipped (NOT_CHECKED)"}
    return {"language": language, "adapter": None, "available": False,
            "limitation": "no adapter defined; NOT_CHECKED"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_quality.json")
    args = parser.parse_args(argv)

    fixtures = [
        ("ok", "# SPDX-License-Identifier: Apache-2.0\ndef add(a, b):\n    return a + b\n", "python"),
        ("bad_syntax", "def broken(:\n", "python"),
        ("secret", "key = 'AKIA" + "A" * 16 + "'\n", "python"),
        ("generated", "// Code generated. DO NOT EDIT.\nint x;\n", "c"),
        ("minified", "x" * 5000, "javascript"),
    ]
    results = [{"fixture": name, "language": lang, **classify_sample(text, lang)}
               for name, text, lang in fixtures]

    report = {
        "schema": "taksh.l071_quality/1",
        "generated_by": "evaluation/l071_quality.py",
        "classification_states": STATES,
        "fixtures_label": "synthetic",
        "fixture_results": results,
        "parser_adapters": [parser_adapter_status(l) for l in ["python", "c", "java", "haskell", "karkain"]],
        "policy": "NO_AUTOMATED_FLAGS is not a quality pass; secrets route to humans; syntax NOT_CHECKED when no toolchain exists",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"quality framework ready; {len(results)} synthetic fixtures classified")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
