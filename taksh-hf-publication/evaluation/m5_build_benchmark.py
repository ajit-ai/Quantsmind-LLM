"""Build the frozen M5 benchmark.

Two provenance groups, so no example claims a licence or an oracle it lacks:

**repository-authored** -- function specifications written for this repository,
each paired with an authored test. The oracle is trustworthy precisely because
this repository wrote both the specification and the assertions.

**corpus-derived** -- tasks extracted from licence-verified source files.
Provenance is genuine (``source_id`` and ``license_id`` are the real ones).
These deliberately carry **no functional oracle**: upstream text is not a test.

A synthetic gold answer is never generated to fill a gap. Where no trustworthy
oracle exists, the validator marks the example ``requires_human_review`` and it
is excluded from automated denominators.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m5_schema import (
    SCHEMA,
    SCORING_VERSION,
    is_scorable,
    stable_id,
    manifest_hash,
    validate_dataset,
)

BENCHMARK_ID = "m5-benchmark-v1"
REPO_AUTHORED_SOURCE = "quantsmind-authored"
REPO_AUTHORED_LICENSE = "CC0-1.0"

FENCE_PROMPT = " Output only a single fenced Python code block."

CODE_TASKS = [
    ("reverse_string", "reverse_string that takes one string and returns it "
     "reversed"),
    ("is_palindrome", "is_palindrome that returns True when the input reads "
     "the same forwards and backwards, ignoring case"),
    ("fizzbuzz_n", "fizzbuzz_n returning a list of strings for 1..n where "
     "multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of 15 "
     "are 'FizzBuzz', and other numbers are their decimal string"),
    ("sum_list", "sum_list that returns the sum of all numbers in a list and 0 "
     "for an empty list"),
    ("count_vowels", "count_vowels that returns how many vowels (a, e, i, o, u) "
     "appear in a string, counting case insensitively"),
    ("flatten", "flatten that takes a list which may contain nested lists of "
     "arbitrary depth and returns a flat list preserving order"),
    ("second_largest", "second_largest that returns the second largest distinct "
     "value in a list of numbers, or None when there is no such value"),
    ("factorial", "factorial returning n! for a non-negative integer n, with "
     "factorial(0) == 1"),
]

CHECKLIST_TASKS = [
    ("instruction_following",
     "Respond with the single word ACKNOWLEDGED and no other text of any kind.",
     [{"type": "equals", "value": "ACKNOWLEDGED"},
      {"type": "max_lines", "value": 1}]),
    ("instruction_following",
     "Produce a bulleted list of exactly three prime numbers greater than 10. "
     "Use Markdown bullets and add nothing else.",
     [{"type": "max_lines", "value": 3},
      {"type": "line_starts_with", "value": "-"}]),
    ("instruction_following",
     "Produce a numbered list of exactly four items and add nothing else.",
     [{"type": "max_lines", "value": 4},
      {"type": "line_starts_with", "value": "1"}]),
    ("structured_output",
     "Emit a JSON object describing a cache entry. It must contain precisely: "
     "key (string), ttl_seconds (integer), warm (boolean). Output the JSON "
     "alone, without backticks or commentary.",
     [{"type": "json_keys", "keys": ["key", "ttl_seconds", "warm"]},
      {"type": "no_fences"}]),
    ("structured_output",
     "Output a Markdown table with the columns `service` and `port` and two "
     "rows: api on 8080 and web on 80. Output only the table.",
     [{"type": "non_empty"}, {"type": "no_fences"}]),
]

EXPLANATION_TASKS = [
    ("code_explanation",
     "In one short paragraph, explain what this function does and why the "
     "closure variable is needed:\n\n```python\ndef make_counter():\n"
     "    count = 0\n    def increment():\n        nonlocal count\n"
     "        count += 1\n        return count\n    return increment\n```",
     "describes a closure holding mutable state across calls", "python"),
    ("code_explanation",
     "In one short paragraph, explain why this loop may return the wrong "
     "index:\n\n```python\nfor i in range(len(xs)):\n    if xs[i] == target:\n"
     "        break\nreturn i\n```",
     "identifies that i is used after the loop and that an empty input returns "
     "0 rather than -1", "python"),
    ("sql_generation",
     "Write SQL selecting id and email from the users table where status is "
     "'active', ordered by email. Output only the SQL.",
     "a SELECT with WHERE and ORDER BY", "sql"),
    ("sql_generation",
     "Write SQL returning the customer_id and the number of orders for every "
     "customer with more than one order, most orders first. Output only the SQL.",
     "a GROUP BY with HAVING COUNT(*) > 1", "sql"),
]


def _provenance(origin, evidence, group, oracle, extra=None):
    payload = {
        "origin": origin,
        "license_evidence": evidence,
        "group": group,
        "oracle": oracle,
    }
    if extra:
        payload.update(extra)
    return payload


def _fit_prompt_to_context(prompt, tokenizer, budget_tokens):
    """Trim a prompt to a token budget, keeping its tail.

    Character limits are wrong here: this benchmark is compared across models
    with different tokenizers, so a prompt that exceeds one model's context
    cannot be run at all. Truncating by *tokens* against the smallest context in
    the comparison keeps every task evaluable by every candidate.

    The code block is closed after trimming so the prompt stays well-formed.
    """
    ids = tokenizer.encode(prompt).ids
    if len(ids) <= budget_tokens:
        return prompt, len(ids)
    tail = tokenizer.decode(ids[-budget_tokens:])
    # Re-close the fence if trimming cut into the code block.
    return tail.rstrip() + ("\n```" if "```" not in tail[-8:] else ""), budget_tokens


def _benchmark_tokenizer(tokenizer_path="tokenizer/qm_code_bpe_v0.json"):
    """The tokenizer used to bound prompts for context-budget purposes.

    This is the smallest-context model in the comparison, so bounding against
    it keeps every example runnable by every candidate.
    """
    path = Path(tokenizer_path)
    if not path.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def repo_authored_records():
    """Function specs with authored oracles, plus format and structure tasks."""
    records = []
    for test_key, spec in CODE_TASKS:
        prompt = f"Write a Python function called {spec}." + FENCE_PROMPT
        records.append({
            "id": stable_id("code_generation", prompt),
            "task_type": "code_generation",
            "prompt": prompt,
            "expected_behavior": "defines the named function with the stated contract",
            "test_specification": test_key,
            "language": "python",
            "source_id": REPO_AUTHORED_SOURCE,
            "license_id": REPO_AUTHORED_LICENSE,
            "split": "eval",
            "provenance": _provenance(
                "authored for this repository",
                f"{REPO_AUTHORED_LICENSE} authored fixture",
                "repository_authored",
                "authored assertions in evaluation/code_check.SAFE_TESTS",
            ),
            "executable": True,
        })

    for task_type, prompt, checklist in CHECKLIST_TASKS:
        records.append({
            "id": stable_id(task_type, prompt),
            "task_type": task_type,
            "prompt": prompt,
            "expected_behavior": "satisfies every machine-checkable constraint",
            "checklist": checklist,
            "language": "text",
            "source_id": REPO_AUTHORED_SOURCE,
            "license_id": REPO_AUTHORED_LICENSE,
            "split": "eval",
            "provenance": _provenance(
                "authored for this repository",
                f"{REPO_AUTHORED_LICENSE} authored fixture",
                "repository_authored", "deterministic checklist",
            ),
            "executable": False,
        })

    for task_type, prompt, expectation, language in EXPLANATION_TASKS:
        records.append({
            "id": stable_id(task_type, prompt),
            "task_type": task_type,
            "prompt": prompt,
            "expected_behavior": expectation,
            "language": language,
            "source_id": REPO_AUTHORED_SOURCE,
            "license_id": REPO_AUTHORED_LICENSE,
            "split": "eval",
            "provenance": _provenance(
                "authored for this repository",
                f"{REPO_AUTHORED_LICENSE} authored fixture",
                "repository_authored",
                "none: correctness needs human judgement",
            ),
            "executable": False,
        })
    return records


_DEF_RE = re.compile(r"^([ \t]*)(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)


def corpus_derived_records(corpus_path, verified_sources, max_per_source=2,
                           tokenizer=None, prompt_budget=384):
    """Tasks extracted from licence-verified source files.

    Genuine provenance: the source id and licence are the real ones. No
    functional oracle is attached, because upstream text is not a test.
    """
    records = []
    path = Path(corpus_path)
    if not path.exists():
        return records
    per_source = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        source = raw.get("source_id")
        if source not in verified_sources or per_source.get(source, 0) >= max_per_source:
            continue
        text = raw.get("text") or ""
        if raw.get("language") != "python" or len(text) < 800:
            continue
        lines = text.splitlines(keepends=True)
        if len(lines) < 60:
            continue
        origin = (f"{raw.get('repo')}@{str(raw.get('commit'))[:12]}"
                  f":{raw.get('file_path')}")
        evidence = (f"licence_verification.json:verified_ids contains {source!r}")

        cut = len(lines) // 2
        tail = "".join(lines[:cut]).rstrip()[-1500:]
        continuation = "".join(lines[cut:cut + 20]).rstrip()
        if len(continuation) >= 200:
            task = "code_completion"
            prompt = ("Continue this source file. Write the next lines exactly "
                      "as they would appear in the file.\n\n```python\n"
                      + tail + "\n```")
            if tokenizer is not None:
                prompt, _tokens = _fit_prompt_to_context(
                    prompt, tokenizer, prompt_budget)
            records.append({
                "id": stable_id(task, tail),
                "task_type": task,
                "prompt": prompt,
                "expected_behavior": "continues the file in the same style",
                "reference_answer": continuation,
                "language": "python",
                "source_id": source,
                "license_id": raw.get("spdx"),
                "split": "eval",
                "provenance": _provenance(
                    f"extracted from {origin}", evidence, "corpus_derived",
                    "none: similarity to upstream text is not correctness",
                    {"content_sha1": raw.get("content_sha1")},
                ),
                "executable": False,
            })
            per_source[source] = per_source.get(source, 0) + 1

        for match in list(_DEF_RE.finditer(text))[:1]:
            start = match.start()
            end = text.find("\n", start)
            if end == -1:
                break
            signature = text[start:end].rstrip()
            try:
                tree = ast.parse(text[start:start + 3000])
            except (SyntaxError, ValueError):
                break
            if not tree.body:
                break
            doc = (ast.get_docstring(tree.body[0], clean=True) or "").strip()
            if len(doc) < 60:
                break
            task = "documentation_generation"
            prompt = ("Write a Python docstring for this function. Describe "
                      "what it does, its parameters and what it returns."
                      "\n\n```python\n" + signature + "\n    ...\n```")
            records.append({
                "id": stable_id(task, signature + doc),
                "task_type": task,
                "prompt": prompt,
                "expected_behavior": "produces a docstring covering purpose, "
                                     "parameters and return value",
                "language": "python",
                "source_id": source,
                "license_id": raw.get("spdx"),
                "split": "eval",
                "provenance": _provenance(
                    f"signature from {origin}", evidence, "corpus_derived",
                    "none: docstring quality needs human judgement",
                ),
                "executable": False,
            })
            per_source[source] = per_source.get(source, 0) + 1
    return records


def build(out_path="evaluation/m5_benchmark.jsonl", corpus_path=None,
          licence_report="data/licence_verification.json", now=None):
    """Assemble, validate, leak-check and freeze the benchmark."""
    from evaluation.m5_leakage import default_reference_paths, leakage_report

    licence = json.loads(Path(licence_report).read_text(encoding="utf-8"))
    verified = set(licence.get("verified_ids") or [])
    if not verified:
        raise ValueError(
            f"{licence_report} lists no verified sources; the benchmark cannot "
            "be provenance-gated without them"
        )
    authorised = verified | {REPO_AUTHORED_SOURCE}

    records = repo_authored_records()
    records += corpus_derived_records(
        corpus_path or "data/code_corpus_v2.jsonl", verified,
        tokenizer=_benchmark_tokenizer(), prompt_budget=384)

    # Cross-version leakage is removed before anything is written. An example
    # that was used to guide an earlier milestone's work is not a held-out
    # measurement for this one, however legitimately it was authored.
    from evaluation.m5_leakage import default_reference_paths, leakage_report

    probe = Path(str(out_path) + ".probe")
    probe.parent.mkdir(parents=True, exist_ok=True)
    with probe.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pre = leakage_report(probe, default_reference_paths())
    probe.unlink(missing_ok=True)

    leaking = {
        example
        for finding in pre["references"]
        for example in finding["overlapping_examples"]
    }
    kept = [r for r in records if r["id"] not in leaking]
    excluded = [
        {"id": r["id"], "reason": "cross_version_leakage",
         "detail": "prompt matches a previously used evaluation prompt"}
        for r in records if r["id"] in leaking
    ]
    records = kept

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = validate_dataset(out, verified_sources=authorised)
    report["by_task"] = dict(Counter(r["task_type"] for r in records))
    report["by_split"] = dict(Counter(r["split"] for r in records))
    report["by_source_group"] = dict(Counter(
        r["provenance"]["group"] for r in records))
    report["by_language"] = dict(Counter(
        str(r.get("language")) for r in records))
    report["scorable_examples"] = sum(1 for r in records if is_scorable(r))
    report["manifest_hash"] = manifest_hash(records)
    report["excluded_for_leakage"] = excluded

    leakage = leakage_report(out, default_reference_paths())
    report["leakage"] = leakage

    manifest = {
        "schema": SCHEMA,
        "scoring_version": SCORING_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "generated_at": now or datetime.now(timezone.utc)
        .replace(microsecond=0).isoformat(),
        "path": str(out),
        "manifest_hash": report["manifest_hash"],
        "examples": len(records),
        "scorable": report["scorable_examples"],
        "requires_human_review": report["requires_human_review"],
        "validation": {k: v for k, v in report.items()
                       if k not in ("manifest_hash", "leakage")},
        "leakage": leakage,
        "licence_sources": {
            "verified_corpus_sources": sorted(verified),
            "repository_authored": [REPO_AUTHORED_SOURCE],
            "policy": (
                "A corpus example is authorised only by "
                "licence_verification.json:verified_ids. A record's own license "
                "string is never accepted as authorisation. The single "
                "repository-authored source is declared here explicitly."
            ),
        },
        "authored_sources": {
            REPO_AUTHORED_SOURCE: {
                "origin": "this repository",
                "license_id": REPO_AUTHORED_LICENSE,
                "redistribution": "permitted (CC0)",
            }
        },
        "excluded_from_automated_scoring": (
            "examples without a trustworthy oracle are counted in "
            "requires_human_review and never scored as failures"
        ),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    from collections import Counter

    parser = argparse.ArgumentParser(description="Build the frozen M5 benchmark")
    parser.add_argument("--out", default="evaluation/m5_benchmark.jsonl")
    parser.add_argument("--corpus", default="data/code_corpus_v2.jsonl")
    parser.add_argument("--licence", default="data/licence_verification.json")
    args = parser.parse_args(argv)

    try:
        manifest = build(args.out, args.corpus, args.licence)
    except (ValueError, FileNotFoundError) as exc:
        print(f"benchmark build refused: {exc}")
        return 1

    validation = manifest["validation"]
    leakage = manifest["leakage"]
    print(f"benchmark: {manifest['benchmark_id']}")
    print(f"  examples           : {manifest['examples']}")
    print(f"  scorable (automatic): {manifest['scorable']}")
    print(f"  human review only  : {manifest['requires_human_review']}")
    print(f"  by task            : {validation['by_task']}")
    print(f"  by source group    : {validation['by_source_group']}")
    print(f"  manifest hash      : {manifest['manifest_hash']}")
    print(f"  leakage overlaps   : {leakage['total_overlapping_examples']}")
    print(f"  unresolved risks   : "
          f"{len(leakage['unresolved_risks'])} shared source(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())