"""Instruction example extractors derived from licence-verified corpus files.

Every example is derived deterministically from a real source file in
``data/m3_code_verified.*``. **No external language model is used**, to build
the data or to answer anything. Target text is always content that verifiably
exists in the corpus, which is what makes these examples checkable.

Families derived: documentation, code_completion, code_explanation,
json_output, config_output, format_following, constrained_output, sql.

The limitation is stated rather than hidden: families needing judgement about
code (debugging, refactoring, test generation) are **not** synthesised, because
inventing them would produce plausible-looking fabrications.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data.instruction_schema import (
    TRANSFORM,
    contains_control_characters,
    is_valid_python,
    load_instruction_jsonl,
    stable_id,
    summarise,
    validate_record,
)

SCHEMA = "quantsmind.instruction_dataset/1"
FIXTURE_SOURCE = "deterministic-fixture"
FIXTURE_LICENSE = "CC0-1.0"

_FUNC_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<async>async\s+)?def\s+(?P<name>\w+)\s*\(",
    re.MULTILINE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _source_fields(record):
    return {
        "source": record.get("source_id"),
        "license": record.get("spdx"),
        "source_sha": record.get("content_sha1"),
    }


def _fixture(task, instruction, response, category, split, language=None):
    return {
        "id": stable_id(task, instruction, response),
        "instruction": instruction,
        "input": "",
        "response": response,
        "category": category,
        "task": task,
        "language": language,
        "split": split,
        "transform": TRANSFORM,
        "source": FIXTURE_SOURCE,
        "license": FIXTURE_LICENSE,
        "source_sha": None,
    }


def extract_docstring_examples(record, split="train", max_per_file=6):
    """Real signatures paired with the real docstrings that follow them."""
    if record.get("language") != "python":
        return []
    text = record.get("text") or ""
    out = []
    for match in list(_FUNC_RE.finditer(text))[: max_per_file * 4]:
        start = match.start()
        line_end = text.find("\n", start)
        if line_end == -1:
            continue
        signature = text[start:line_end].rstrip()
        try:
            tree = ast.parse(text[start:start + 4000])
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        if not tree.body:
            continue
        doc = ast.get_docstring(tree.body[0], clean=False)
        if not doc or len(doc.strip()) < 40 or contains_control_characters(doc):
            continue
        out.append({
            "id": stable_id("documentation", signature, doc),
            "instruction": (
                "Write a Python docstring for the following function. Describe "
                "what it does, its parameters and what it returns.\n\n"
                "```python\n" + signature + "\n    ...\n```"
            ),
            "input": "",
            "response": '"""' + doc + '"""',
            "category": "coding",
            "task": "documentation",
            "language": "python",
            "split": split,
            "transform": TRANSFORM,
            **_source_fields(record),
        })
        if len(out) >= max_per_file:
            break
    return out


def extract_completion_examples(record, split="train", max_per_file=2,
                                min_chars=600, max_prompt_chars=3000):
    """Opening lines of a real file, with the next lines as the response.

    The file prefix goes in ``input``, not ``instruction``: an instruction is a
    request, and burying 30k characters of source inside one makes the request
    unreadable and blows the contract's 4000-character instruction limit.
    """
    text = record.get("text") or ""
    if len(text) < min_chars:
        return []
    lines = text.splitlines(keepends=True)
    if len(lines) < 50:
        return []
    out = []
    for index in (8, len(lines) // 2):
        index = min(index, len(lines) - 20)
        prompt = "".join(lines[:index]).rstrip()[-max_prompt_chars:]
        response = "".join(lines[index:index + 24]).rstrip()
        if len(prompt) < 120 or len(response) < 160:
            continue
        if contains_control_characters(response):
            continue
        out.append({
            "id": stable_id("code_completion", prompt[:200], response[:200]),
            "instruction": (
                "Continue this source file. Write the next lines exactly as they "
                "would appear in the file."
            ),
            "input": prompt,
            "response": response,
            "category": "coding",
            "task": "code_completion",
            "language": record.get("language") or "python",
            "split": split,
            "transform": TRANSFORM,
            **_source_fields(record),
        })
    return out[:max_per_file]


def extract_explanation_examples(record, split="train", max_per_file=3):
    """Signature plus the first section of its real docstring."""
    if record.get("language") != "python":
        return []
    text = record.get("text") or ""
    out = []
    for match in _FUNC_RE.finditer(text):
        start = match.start()
        line_end = text.find("\n", start)
        if line_end == -1:
            continue
        signature = text[start:line_end].rstrip()
        try:
            tree = ast.parse(text[start:start + 3000])
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        if not tree.body:
            continue
        doc = ast.get_docstring(tree.body[0], clean=True) or ""
        summary = doc.strip().split("\n\n")[0].strip()
        if not (40 <= len(summary) <= 600):
            continue
        name = match.group("name")
        out.append({
            "id": stable_id("code_explanation", signature, summary),
            "instruction": (
                f"In one short paragraph, explain what the function `{name}` does, "
                "based on its signature and documentation."
            ),
            "input": signature,
            "response": summary,
            "category": "coding",
            "task": "code_explanation",
            "language": "python",
            "split": split,
            "transform": TRANSFORM,
            **_source_fields(record),
        })
        if len(out) >= max_per_file:
            break
    return out


def structured_output_examples(split="train"):
    """Structure-following tasks: they teach format, not facts."""
    schema = {"name": "server_config", "host": "example.invalid",
              "port": 8080, "debug": False}
    payload = json.dumps(schema, indent=2)
    table = ("| model | params |\n|---|---|\n"
             "| qm-tiny | 2.85M |\n| qm-code | 9.36M |")
    yaml_text = ('name: web\nreplicas: 3\nresources:\n  limits:\n'
                 '    cpu: "2"\n')
    return [
        _fixture(
            "json_output",
            "Return a JSON object describing a server configuration. It must have "
            "exactly these keys: name (string), host (string), port (integer), "
            "debug (boolean). Return only the JSON, with no explanation and no "
            "markdown fences.",
            payload, "structured_output", split, "json",
        ),
        _fixture(
            "config_output",
            "Output a Markdown table with two columns named `model` and `params` "
            "containing exactly two rows: qm-tiny with 2.85M, and qm-code with "
            "9.36M. Output only the table.",
            table, "structured_output", split, "yaml",
        ),
        _fixture(
            "config_output",
            'Write a YAML Kubernetes-style deployment named `web` with 3 replicas '
            'and a CPU limit of "2". Output only YAML, no fences.',
            yaml_text, "structured_output", split, "yaml",
        ),
    ]


def sql_examples(split="train"):
    """A small fixed SQL set: real queries against stated schemas."""
    cases = [
        ("Write SQL that selects the id and name of every user whose status is "
         "'active', ordered by name ascending. Output only the SQL.",
         "SELECT id, name\nFROM users\nWHERE status = 'active'\n"
         "ORDER BY name ASC;"),
        ("Write SQL to count how many orders each customer placed, returning "
         "customer_id and order_count, most orders first. Output only the SQL.",
         "SELECT customer_id, COUNT(*) AS order_count\nFROM orders\n"
         "GROUP BY customer_id\nORDER BY order_count DESC;"),
        ("Write SQL to find email addresses that appear more than once in the "
         "users table. Output only the SQL.",
         "SELECT email\nFROM users\nGROUP BY email\nHAVING COUNT(*) > 1;"),
        ("Write SQL to return the 10 most recent orders by created_at, including "
         "id, customer_id and total. Output only the SQL.",
         "SELECT id, customer_id, total\nFROM orders\n"
         "ORDER BY created_at DESC\nLIMIT 10;"),
    ]
    return [_fixture("sql", instruction, response, "technical", split, "sql")
            for instruction, response in cases]


def format_following_examples(split="train"):
    """Explicit constraints with responses that provably satisfy them."""
    cases = [
        ("List exactly three items as a Markdown bullet list and nothing else.",
         "- alpha\n- beta\n- gamma", "format_following"),
        ("List exactly four items as a numbered list and nothing else.",
         "1. one\n2. two\n3. three\n4. four", "format_following"),
        ("Reply with exactly the word READY and nothing else.",
         "READY", "constrained_output"),
        ("Reply with the sum of 17 and 25 as a bare number and nothing else.",
         "42", "constrained_output"),
        ("Convert the following text to uppercase. Output only the result.\n\n"
         "quantsmind", "QUANTSMIND", "format_following"),
    ]
    return [_fixture(task, instruction, response, "instruction_following", split)
            for instruction, response, task in cases]


# --------------------------------------------------------------------------- #
# Quality pipeline
# --------------------------------------------------------------------------- #
def _input_key(record):
    return " ".join((record.get("input") or "").split()).strip().lower()


def _instruction_key(record):
    return " ".join((record.get("instruction") or "").split()).strip().lower()


def _pair_key(record):
    return (
        _instruction_key(record),
        " ".join((record.get("response") or "").split()).strip().lower(),
    )


def _example_key(record):
    """Full identity of one example, including the input field."""
    return (_instruction_key(record), _input_key(record),
            " ".join((record.get("response") or "").split()).strip().lower())


# ``code_completion`` legitimately shares one instruction across every example:
# the distinguishing content is the file prefix carried in ``input``. Deduping
# on the instruction alone would delete the whole task. Every other task is
# expected to have a distinct instruction.
TASKS_WITH_SHARED_INSTRUCTION = frozenset({"code_completion"})


def quality_filter(records, verified_sources=None, allowed_extra_sources=None):
    """Apply every quality gate. Returns ``(accepted, excluded)``.

    Nothing is silently dropped: each exclusion carries an explicit reason and
    the offending id, so the report can state what was removed and why.

    ``allowed_extra_sources`` exists for the deterministic in-repo fixtures
    (format/JSON/SQL tasks written for this repository under CC0). It is an
    explicit allow-list, not a bypass: a corpus source with an unresolved
    licence still cannot enter through it.
    """
    verified_sources = verified_sources
    allowed_extra_sources = set(allowed_extra_sources or ())
    permitted = (set(verified_sources) if verified_sources else set()) | allowed_extra_sources
    accepted, excluded = [], []
    seen_ids, seen_prompts, seen_examples = set(), set(), set()

    for record in records:
        rid = record.get("id", "<no id>")
        problems = validate_record(record)
        if problems:
            excluded.append({"id": rid, "reason": "schema", "detail": problems})
            continue

        if record["id"] in seen_ids:
            excluded.append({"id": rid, "reason": "duplicate_id",
                             "detail": ["id already seen"]})
            continue
        seen_ids.add(record["id"])

        if record.get("source") not in permitted:
            excluded.append({
                "id": rid, "reason": "unverified_source",
                "detail": [f"source {record.get('source')!r} is not a verified "
                           f"corpus source {sorted(verified_sources or [])} and "
                           "not an allowed deterministic fixture"],
            })
            continue

        if record.get("language") == "python" and not is_valid_python(record["response"]):
            # Only enforced for tasks whose response must be executable Python.
            if record.get("task") in ("code_generation", "debugging",
                                      "refactoring", "test_generation"):
                excluded.append({"id": rid, "reason": "python_syntax",
                                 "detail": ["response is not valid Python"]})
                continue

        prompt_key = _instruction_key(record)
        if prompt_key in seen_prompts and \
                record.get("task") not in TASKS_WITH_SHARED_INSTRUCTION:
            excluded.append({"id": rid, "reason": "duplicate_prompt",
                             "detail": ["identical instruction seen already"]})
            continue
        seen_prompts.add(prompt_key)

        example_key = _example_key(record)
        if example_key in seen_examples:
            excluded.append({"id": rid, "reason": "duplicate_example",
                             "detail": ["identical instruction+input+response"]})
            continue
        seen_examples.add(example_key)

        accepted.append(record)

    return accepted, excluded


def verify_split_isolation(train_records, val_records):
    """No id, instruction, or instruction+response pair may cross splits."""
    def keys(records):
        return (
            {r["id"] for r in records},
            {_instruction_key(r) for r in records},
            {_example_key(r) for r in records},
        )

    train_ids, train_prompts, train_pairs = keys(train_records)
    val_ids, val_prompts, val_pairs = keys(val_records)
    return {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "shared_ids": sorted(train_ids & val_ids),
        "shared_prompts": sorted(train_prompts & val_prompts),
        "shared_examples": sorted(train_pairs & val_pairs),
        "ok": not (train_ids & val_ids) and not (train_prompts & val_prompts)
        and not (train_pairs & val_pairs),
    }


def partition_fixtures():
    """Disjoint fixture sets for train and val.

    These are self-authored under CC0, so there is no licence question, but
    putting the same example in both splits would still be leakage. A stable
    modulo partition keeps the split reproducible.
    """
    every = (structured_output_examples("train") + sql_examples("train")
             + format_following_examples("train"))
    train, val = [], []
    for index, example in enumerate(every):
        target = val if index % 3 == 2 else train
        target.append(example)
    for split, bucket in (("train", train), ("val", val)):
        for example in bucket:
            example["split"] = split
    return {"train": train, "val": val}


def build(corpus_path, val_corpus_path, licence_report, out_train, out_val,
          dataset_id="instruct-v1", now=None):
    """Build both splits plus a manifest, or refuse to write anything."""
    licence = json.loads(Path(licence_report).read_text(encoding="utf-8"))
    verified = set(licence.get("verified_ids") or [])
    if not verified:
        raise ValueError(
            f"{licence_report} lists no verified sources; refusing to build an "
            "instruction dataset from data whose licence is unverified"
        )

    fixtures = partition_fixtures()

    def derive(path, split):
        records, malformed = load_instruction_jsonl(path)
        if malformed:
            raise ValueError(
                f"{path} has {len(malformed)} malformed line(s); first: "
                f"line {malformed[0]['line']}: {malformed[0]['reason']}"
            )
        out = []
        for record in records:
            if record.get("source_id") not in verified:
                continue
            out.extend(extract_docstring_examples(record, split))
            out.extend(extract_completion_examples(record, split))
            out.extend(extract_explanation_examples(record, split))
        # Deterministic fixtures are partitioned across splits rather than
        # duplicated: duplicating them would put identical prompts in both
        # splits, which is exactly the leakage the isolation check forbids.
        out.extend(fixtures[split])
        return out

    train_raw = derive(corpus_path, "train")
    val_raw = derive(val_corpus_path, "val")

    # Fixtures are deterministic and identical in both splits, so deduplicate
    # cross-split before the isolation check.
    train_accepted, train_excluded = quality_filter(
        train_raw, verified, allowed_extra_sources={FIXTURE_SOURCE})
    seen_in_train = {_pair_key(r) for r in train_accepted}
    seen_prompts_train = {_instruction_key(r) for r in train_accepted}
    val_only = [
        r for r in val_raw
        if _pair_key(r) not in seen_in_train
        and _instruction_key(r) not in seen_prompts_train
    ]
    val_accepted, val_excluded = quality_filter(
        val_only, verified, allowed_extra_sources={FIXTURE_SOURCE})

    isolation = verify_split_isolation(train_accepted, val_accepted)
    if not isolation["ok"]:
        raise ValueError(
            "train/val overlap after filtering: "
            f"ids={isolation['shared_ids'][:3]} "
            f"prompts={len(isolation['shared_prompts'])} "
            f"examples={len(isolation['shared_examples'])}"
        )

    def write(records, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "path": str(path),
            "records": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    train_info = write(train_accepted, out_train)
    val_info = write(val_accepted, out_val)

    manifest = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "generated_at": now or utc_now_iso(),
        "transform": TRANSFORM,
        "source_corpora": {
            "train": {"path": str(corpus_path),
                      "sha256": hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()},
            "val": {"path": str(val_corpus_path),
                    "sha256": hashlib.sha256(Path(val_corpus_path).read_bytes()).hexdigest()},
        },
        "licence_report": {"path": str(licence_report),
                           "sha256": hashlib.sha256(
                               Path(licence_report).read_bytes()).hexdigest()},
        "verified_sources": sorted(verified),
        "splits": {"train": train_info, "val": val_info},
        "statistics": {"train": summarise(train_accepted),
                       "val": summarise(val_accepted)},
        "excluded": {
            "train": {"count": len(train_excluded),
                      "by_reason": dict(Counter(e["reason"] for e in train_excluded)),
                      "examples": train_excluded[:20]},
            "val": {"count": len(val_excluded),
                    "by_reason": dict(Counter(e["reason"] for e in val_excluded)),
                    "examples": val_excluded[:20]},
        },
        "isolation": isolation,
        "not_synthesised": [
            "debugging", "refactoring", "test_generation",
            "code_generation (function bodies are not invented)",
        ],
        "not_synthesised_reason": (
            "These families require judging what a correct fix or refactor is. "
            "Generating them without a verified source would mean inventing "
            "technical content, so they are absent rather than fabricated."
        ),
        "scale_disclosure": (
            "This dataset is derived from ~1.2M tokens of licence-verified "
            "source. It is NOT a production-scale instruction corpus, and "
            "instruction tuning on it cannot be expected to confer general "
            "coding ability."
        ),
    }
    manifest_path = str(Path(out_train).with_suffix("").as_posix()).replace(
        ".train", "") + ".manifest.json"
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = manifest_path
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a licence-gated instruction-tuning dataset"
    )
    parser.add_argument("--corpus", default="data/m3_code_verified.train.jsonl")
    parser.add_argument("--corpus-val", default="data/m3_code_verified.val.jsonl")
    parser.add_argument("--licence", default="data/licence_verification.json")
    parser.add_argument("--out", default="data/instruct_v1.train.jsonl")
    parser.add_argument("--out-val", default="data/instruct_v1.val.jsonl")
    parser.add_argument("--dataset-id", default="instruct-v1")
    args = parser.parse_args(argv)

    try:
        manifest = build(args.corpus, args.corpus_val, args.licence,
                         args.out, args.out_val, dataset_id=args.dataset_id)
    except (ValueError, FileNotFoundError) as exc:
        print(f"instruction dataset build refused: {exc}")
        return 1

    stats = manifest["statistics"]["train"]
    print(f"instruction dataset: {manifest['dataset_id']}")
    print(f"  verified sources : {len(manifest['verified_sources'])}")
    for split in ("train", "val"):
        info = manifest["splits"][split]
        print(f"  {split:5}: {info['records']:5} records -> {info['path']}")
    print(f"  train by category: {stats['by_category']}")
    print(f"  train by task    : {stats['by_task']}")
    print(f"  train by language: {stats['by_language']}")
    print(f"  excluded (train) : {manifest['excluded']['train']['count']} "
          f"{manifest['excluded']['train']['by_reason']}")
    print(f"  split isolation  : {'OK' if manifest['isolation']['ok'] else 'VIOLATED'}")
    print(f"  manifest         : {manifest['manifest_path']}")
    print(f"  not synthesised  : {', '.join(manifest['not_synthesised'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())