"""Build the M7 source-derived, licence-gated supervised coding dataset.

What this module does
---------------------
It derives instruction examples deterministically from the licence-verified
corpus and partitions them by **repository**, so that no repository, file, or
example crosses a partition:

* ``train``  -- five verified repositories (same source set as M3's train split)
* ``val``    -- one half of aiohttp's files, for SFT early stopping
* ``dev``    -- the other half of aiohttp's files, for development evaluation
* ``locked`` -- psf-requests, held out from every training and development step

The split is by source file, not by random example: two examples from the same
file can share imports, helper names, and near-duplicate code, so random example
splits leak. This is the same principle M3 used, extended to four partitions.

What it deliberately does not do
--------------------------------
* It does not invent debugging, refactoring, or test-generation examples. Those
  require judging what a correct fix is; fabricating them would produce
  plausible-looking non-evidence.
* It does not call an external language model. Every response is text that
  exists in the corpus.
* It does not put the reference body in a code-generation prompt. The prompt is
  the real signature and docstring; the response is the real implementation.

Functional oracles
------------------
A small set of pure ``psf-requests`` functions has hand-specified *inputs* and
reference-derived *expected outputs*: each expected value is produced by running
the licensed reference implementation in a sandbox at build time. Those tasks
are the only ones that can support a functional (level 3) claim. The set is
deliberately small and reported with its size, not presented as broad coverage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data.build_instruction_dataset import (
    format_following_examples,
    sql_examples,
    structured_output_examples,
)
from data.instruction_schema import (
    contains_control_characters,
    is_valid_python,
    stable_id,
    summarise,
    validate_record,
)

SCHEMA = "quantsmind.m7_dataset/1"
DATASET_ID = "m7-code-supervised-v0"
TRANSFORM = "qm.m7.source-derived/1"

FIXTURE_SOURCE = "deterministic-fixture"
FIXTURE_LICENSE = "CC0-1.0"

TRAIN_SOURCES = (
    "pallets-click",
    "pallets-flask",
    "pytest-dev-pytest",
    "tiangolo-fastapi",
    "urllib3-urllib3",
)
VAL_SOURCE = "aio-libs-aiohttp"
LOCKED_SOURCE = "psf-requests"

# Caps keep the SFT budget bounded and the task mix balanced. They are applied
# after deterministic ordering, never by random sampling.
CAPS = {
    "code_generation": 200,
    "code_completion": 100,
    "documentation": 100,
    "code_explanation": 100,
    "sql": 4,
    "json_output": 3,
    "config_output": 3,
    "format_following": 5,
    "constrained_output": 2,
}
HELDOUT_CAP = 60

# Input cases for the functional probes. Expected outputs are NOT written here:
# they are computed by executing the licensed reference function at build time.
# Only functions that are pure, network-free, and stand alone under the stated
# support imports are included. Each entry is (positional args, keyword args).
ORACLE_SPECS = (
    {
        "name": "dotted_netmask",
        "file": "utils.py",
        "support": "import socket\nimport struct\n",
        "cases": [([8], {}), ([16], {}), ([24], {}), ([32], {})],
    },
    {
        "name": "is_ipv4_address",
        "file": "utils.py",
        "support": "import socket\n",
        "cases": [(["127.0.0.1"], {}), (["256.0.0.1"], {}),
                  (["example.com"], {}), ([""], {})],
    },
    {
        "name": "urldefragauth",
        "file": "utils.py",
        "support": "from urllib.parse import urlparse, urlunparse\n",
        "cases": [(["http://user:pass@example.com/path#frag"], {}),
                  (["http://example.com"], {}),
                  (["https://example.com/a?b=1#x"], {})],
    },
    {
        "name": "get_auth_from_url",
        "file": "utils.py",
        "support": "from urllib.parse import urlparse, unquote\n",
        "cases": [(["http://user:pass@example.com"], {}),
                  (["http://example.com"], {}),
                  (["http://user%40name:p%40ss@example.com"], {})],
    },
    {
        "name": "to_key_val_list",
        "file": "utils.py",
        "support": ("from __future__ import annotations\n"
                    "from collections.abc import Mapping as _SupportsItems\n"),
        "cases": [([ [("a", 1), ("b", 2)] ], {}), ([{"a": 1}], {}),
                  ([None], {}), (["string"], {})],
    },
    {
        "name": "from_key_val_list",
        "file": "utils.py",
        "support": ("from __future__ import annotations\n"
                    "from collections import OrderedDict\n"),
        "cases": [([ [("a", 1), ("b", 2)] ], {}), ([{"a": 1}], {}),
                  ([None], {}), (["string"], {})],
    },
    {
        "name": "unquote_unreserved",
        "file": "utils.py",
        "support": ("from __future__ import annotations\n"
                    "UNRESERVED_SET = set('abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~')\n"
                    "class InvalidURL(Exception):\n    pass\n"),
        "cases": [(["http://example.com/%7Euser"], {}),
                  (["http://example.com/a%20b"], {}),
                  (["http://example.com/%zz"], {})],
    },
    {
        "name": "iter_slices",
        "file": "utils.py",
        "support": "from __future__ import annotations\n",
        "cases": [(["abcdef", 2], {}), (["abc", None], {}),
                  (["", 3], {}), (["abcdef", 0], {})],
    },
)

_HARNESS = r'''
import json
import sys
from collections import OrderedDict

def _jsonable(value):
    if isinstance(value, OrderedDict):
        return {"__ordered__": [[_jsonable(k), _jsonable(v)]
                                for k, v in value.items()]}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, bytes):
        return {"__bytes__": value.decode("latin-1")}
    return value

def _run(fn, args, kwargs):
    import inspect
    result = fn(*args, **kwargs)
    if inspect.isgenerator(result):
        result = list(result)
    return _jsonable(result)

CASES = __CASES__
out = []
for args, kwargs in CASES:
    try:
        out.append({"status": "ok", "value": _run(__TARGET__, args, kwargs)})
    except Exception as exc:
        out.append({"status": "raised", "type": type(exc).__name__,
                    "message": str(exc)[:200]})
print(json.dumps(out, sort_keys=True))
'''


@dataclass(frozen=True)
class FunctionInfo:
    record: dict
    name: str
    signature: str
    doc: str
    source: str
    has_doctest: bool
    lineno: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_fields(record) -> dict:
    return {
        "source": record.get("source_id"),
        "license": record.get("spdx"),
        "source_sha": record.get("content_sha1"),
    }


def _partition_for_source(source: str, file_path: str) -> str | None:
    if source in TRAIN_SOURCES:
        return "train"
    if source == VAL_SOURCE:
        bucket = int(hashlib.sha256(file_path.encode("utf-8")).hexdigest(), 16)
        return "val" if bucket % 2 == 0 else "dev"
    if source == LOCKED_SOURCE:
        return "locked"
    return None


def iter_functions(record, min_doc_chars=40) -> list[FunctionInfo]:
    """Top-level, undecorated functions with a substantive body.

    ``min_doc_chars=0`` is used only when locating the held-out functional
    probes: a short or empty docstring does not stop a function from having a
    checkable reference behaviour.
    """
    if record.get("language") != "python":
        return []
    text = record.get("text") or ""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    lines = text.splitlines(keepends=True)
    out: dict[str, FunctionInfo] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.decorator_list:
            continue
        doc = ast.get_docstring(node, clean=False) or ""
        if len(doc.strip()) < min_doc_chars or contains_control_characters(doc):
            continue
        body = [
            item for item in node.body
            if not (isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str))
        ]
        if not body or all(isinstance(item, ast.Pass) for item in body):
            continue
        try:
            source = ast.get_source_segment(text, node)
        except (ValueError, IndexError):
            continue
        if not source:
            continue
        source = textwrap.dedent(source).strip("\n")
        if not is_valid_python(source):
            continue
        header_end = min(item.lineno for item in node.body) - 1
        if header_end <= node.lineno - 1:
            # One-line definition: the header shares its line with the body.
            header_end = node.lineno
        signature = "".join(lines[node.lineno - 1:header_end]).rstrip()
        if not signature.lstrip().startswith(("def ", "async def ")):
            continue
        info = FunctionInfo(
            record=record,
            name=node.name,
            signature=signature,
            doc=doc,
            source=source,
            has_doctest=">>>" in doc,
            lineno=node.lineno,
        )
        # Overloads share a name; the last definition with a real body wins.
        out[node.name] = info
    return list(out.values())


def _spec_block(info: FunctionInfo) -> str:
    """The prompt half: real signature plus the real docstring."""
    body = info.signature + "\n"
    if info.doc.strip():
        body += '    """' + info.doc + '"""\n'
    body += "    ...\n"
    return "```python\n" + body + "```"


def _generation_example(info: FunctionInfo, partition: str, split: str,
                        oracle_name: str | None = None) -> dict:
    instruction = (
        "Implement the Python function described below. Return the complete "
        "function definition, including its signature and docstring. Do not add "
        "tests, examples, or explanation."
    )
    input_text = _spec_block(info)
    record = {
        "id": stable_id("code_generation", instruction, info.source),
        "instruction": instruction,
        "input": input_text,
        "response": info.source,
        "category": "coding",
        "task": "code_generation",
        "language": "python",
        "split": split,
        "partition": partition,
        "transform": TRANSFORM,
        **_source_fields(info.record),
    }
    if oracle_name:
        record["oracle"] = {"kind": "reference_execution", "function": oracle_name}
    return record


def _documentation_example(info: FunctionInfo, partition: str, split: str) -> dict:
    instruction = (
        "Write a Python docstring for the function below. Describe what it does, "
        "its parameters, and what it returns. Output only the docstring."
    )
    input_text = _spec_block(info)
    return {
        "id": stable_id("documentation", input_text, info.doc),
        "instruction": instruction,
        "input": input_text,
        "response": '"""' + info.doc + '"""',
        "category": "coding",
        "task": "documentation",
        "language": "python",
        "split": split,
        "partition": partition,
        "transform": TRANSFORM,
        **_source_fields(info.record),
    }


def _explanation_example(info: FunctionInfo, partition: str, split: str) -> dict:
    summary = info.doc.strip().split("\n\n")[0].strip()
    if not 40 <= len(summary) <= 600:
        return {}
    instruction = (
        f"In one short paragraph, explain what the function `{info.name}` does, "
        "based on its signature and documentation."
    )
    return {
        "id": stable_id("code_explanation", info.signature, summary),
        "instruction": instruction,
        "input": info.signature,
        "response": summary,
        "category": "coding",
        "task": "code_explanation",
        "language": "python",
        "split": split,
        "partition": partition,
        "transform": TRANSFORM,
        **_source_fields(info.record),
    }


def _file_completions(infos, partition: str, split: str) -> list[dict]:
    """File-level prefixes, once per source file, with a short target.

    A completion target must fit the training context. The M4 extractor used 24
    lines, which is often 600+ tokens and was rejected by the formatter. Twelve
    lines keeps the task trainable without truncating the target.
    """
    out = []
    seen_files = set()
    for info in infos:
        record = info.record
        path = record.get("file_path", "")
        if path in seen_files:
            continue
        seen_files.add(path)
        text = record.get("text") or ""
        lines = text.splitlines(keepends=True)
        if len(lines) < 50 or len(text) < 600:
            continue
        for index in (8, len(lines) // 2):
            index = min(index, len(lines) - 14)
            prompt = "".join(lines[:index]).rstrip()[-3000:]
            response = "".join(lines[index:index + 12]).rstrip()
            if len(prompt) < 120 or len(response) < 80:
                continue
            if contains_control_characters(response):
                continue
            out.append({
                "id": stable_id("code_completion", prompt[-200:], response[:200]),
                "instruction": (
                    "Continue this source file. Write the next lines exactly as "
                    "they would appear in the file."
                ),
                "input": prompt,
                "response": response,
                "category": "coding",
                "task": "code_completion",
                "language": record.get("language") or "python",
                "split": split,
                "partition": partition,
                "transform": TRANSFORM,
                **_source_fields(record),
            })
    return out


def _fixture_partitions() -> dict[str, list[dict]]:
    """Disjoint fixture cases per partition (self-authored, CC0)."""
    structured = structured_output_examples("train")
    sql = sql_examples("train")
    fmt = format_following_examples("train")
    assignment = {
        "train": ([structured[0], structured[1]], sql[:2], fmt[:3]),
        "dev": ([], [sql[2]], [fmt[3]]),
        "locked": ([structured[2]], [sql[3]], [fmt[4]]),
    }
    out: dict[str, list[dict]] = {}
    for partition, (s, q, f) in assignment.items():
        split = {"train": "train", "dev": "val", "locked": "test"}[partition]
        records = s + q + f
        for record in records:
            record["split"] = split
            record["partition"] = partition
        out[partition] = records
    return out


def _candidate_functions(records) -> dict[str, list[FunctionInfo]]:
    by_partition: dict[str, list[FunctionInfo]] = defaultdict(list)
    for record in records:
        partition = _partition_for_source(record["source_id"],
                                          record.get("file_path", ""))
        if partition is None:
            continue
        for info in iter_functions(record):
            by_partition[partition].append(info)
    for partition in by_partition:
        by_partition[partition].sort(
            key=lambda i: (i.record["source_id"], i.record.get("file_path", ""),
                           i.lineno, i.name)
        )
    return by_partition


def _interleave(infos):
    """Round-robin across repositories so a per-task cap cannot starve a source."""
    groups: dict[str, list[FunctionInfo]] = {}
    for info in infos:
        groups.setdefault(info.record["source_id"], []).append(info)
    order = list(groups)
    out = []
    while any(groups[source] for source in order):
        for source in order:
            if groups[source]:
                out.append(groups[source].pop(0))
    return out


def _fits(record, tokenizer, max_length=512):
    """Whether the example is trainable without discarding anything essential."""
    from training.instruction_format import FormattingError, encode_example

    try:
        encoded = encode_example(
            tokenizer, record["instruction"], record["response"],
            record.get("input", "") or "", max_length=max_length,
        )
    except FormattingError as exc:
        return False, str(exc)
    if encoded["truncated"] and record["task"] != "code_completion":
        return False, (
            "prompt would be truncated; a generation/explanation prompt must be "
            "seen in full")
    return True, None


def _build_partition(infos, partition, split, tokenizer=None):
    """Deterministic, capped task extraction. Returns (records, excluded)."""
    accepted, excluded = [], []
    per_task = Counter()
    seen_keys = set()
    seen_responses = set()

    def add(record, reason=None):
        if not record:
            return
        task = record["task"]
        if tokenizer is not None:
            fits, why = _fits(record, tokenizer)
            if not fits:
                excluded.append({"id": record.get("id"), "reason": "does_not_fit",
                                 "detail": [why]})
                return
        if per_task[task] >= (CAPS[task] if partition == "train" else HELDOUT_CAP):
            excluded.append({"id": record.get("id"), "reason": "cap",
                             "detail": [f"{task} cap reached"]})
            return
        key = (task, record.get("instruction"), record.get("input"))
        response_key = hashlib.sha256(
            (record.get("response") or "").encode("utf-8")).hexdigest()
        if key in seen_keys or response_key in seen_responses:
            excluded.append({"id": record.get("id"), "reason": "duplicate",
                             "detail": ["task+prompt or response already used"]})
            return
        seen_keys.add(key)
        seen_responses.add(response_key)
        accepted.append(record)
        per_task[task] += 1

    ordered = _interleave(infos)
    for info in ordered:
        # Held-out partitions keep doctest-bearing functions: their examples are
        # close to executable behaviour, and they never enter training.
        if partition == "train" and info.has_doctest:
            excluded.append({"id": None, "reason": "doctest_reserved_for_eval",
                             "detail": [f"{info.name} in {info.record.get('file_path')}"]})
            continue
        add(_generation_example(info, partition, split))
        add(_documentation_example(info, partition, split))
        add(_explanation_example(info, partition, split))
        if partition == "train" and per_task["code_generation"] >= CAPS["code_generation"]:
            break
    for record in _file_completions(infos, partition, split):
        add(record)
    return accepted, excluded


def _select_oracles(records):
    """Reference-derive expected outputs for the curated functional probes."""
    selected = []
    for spec in ORACLE_SPECS:
        match = None
        for record in records:
            if record.get("source_id") != LOCKED_SOURCE:
                continue
            if not record.get("file_path", "").endswith(spec["file"]):
                continue
            for info in iter_functions(record, min_doc_chars=0):
                if info.name == spec["name"]:
                    match = info
                    break
            if match:
                break
        if match is None:
            raise ValueError(
                f"functional probe {spec['name']!r} not found in {LOCKED_SOURCE}; "
                "refusing to invent a replacement"
            )
        selected.append((spec, match))
    return selected


def _run_reference(info: FunctionInfo, support: str, cases) -> list[dict]:
    script = (
        support + "\n" + info.source + "\n"
        + _HARNESS.replace("__TARGET__", info.name)
        .replace("__CASES__", repr(cases))
    )
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"})
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True, text=True, timeout=30, cwd=tmp, env=env,
        )
    if proc.returncode != 0:
        raise ValueError(
            f"reference execution failed for {info.name}: "
            f"{(proc.stderr or proc.stdout).strip()[-300:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def build_oracles(records, locked_records):
    """Build oracle records and add any missing functional probe tasks.

    The append is explicit: a probe whose function has a too-short docstring
    would otherwise have no locked task, and the evaluation would silently drop
    it. Missing probes are an error, not a smaller denominator.
    """
    by_id = {record["id"]: record for record in locked_records}
    oracles, added = [], []
    for spec, info in _select_oracles(records):
        results = _run_reference(info, spec["support"], spec["cases"])
        example = _generation_example(info, "locked", "test",
                                      oracle_name=spec["name"])
        task = by_id.get(example["id"])
        if task is None:
            problems = validate_record(example)
            if problems:
                raise ValueError(
                    f"functional probe {spec['name']!r} produced an invalid "
                    f"task: {problems}")
            locked_records.append(example)
            by_id[example["id"]] = example
            added.append(example)
            task = example
        task["oracle"] = {"kind": "reference_execution", "function": spec["name"]}
        oracles.append({
            "example_id": task["id"],
            "function": spec["name"],
            "support": spec["support"],
            "cases": [
                {"args": args, "kwargs": kwargs, "expected": result}
                for (args, kwargs), result in zip(spec["cases"], results)
            ],
            "source": info.record.get("source_id"),
            "license": info.record.get("spdx"),
            "source_sha": info.record.get("content_sha1"),
            "provenance": (
                "Expected outputs were produced by executing the licensed "
                "reference implementation in a subprocess at dataset build "
                "time. Inputs are hand-specified; expected values are not."
            ),
        })
    return oracles, added


def _isolation(partitions) -> dict:
    reports = {}
    names = sorted(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = partitions[left], partitions[right]
            def keys(rs):
                return {
                    "ids": {r["id"] for r in rs},
                    "examples": {(r["task"], r["instruction"], r.get("input", ""),
                                  r["response"]) for r in rs},
                    "responses": {hashlib.sha256(
                        r["response"].encode("utf-8")).hexdigest() for r in rs},
                }
            ka, kb = keys(a), keys(b)
            reports[f"{left}__{right}"] = {
                "shared_ids": len(ka["ids"] & kb["ids"]),
                "shared_examples": len(ka["examples"] & kb["examples"]),
                "shared_responses": len(ka["responses"] & kb["responses"]),
                "ok": not (ka["ids"] & kb["ids"])
                and not (ka["examples"] & kb["examples"])
                and not (ka["responses"] & kb["responses"]),
            }
    return {"pairs": reports, "ok": all(r["ok"] for r in reports.values())}


def _quality(records, verified, partition):
    accepted, excluded = [], []
    for record in records:
        problems = validate_record(record)
        if problems:
            excluded.append({"id": record.get("id"), "reason": "schema",
                             "detail": problems})
            continue
        if record.get("source") not in verified and record.get("source") != FIXTURE_SOURCE:
            excluded.append({"id": record["id"], "reason": "unverified_source",
                             "detail": [str(record.get("source"))]})
            continue
        if record.get("partition") != partition:
            excluded.append({"id": record["id"], "reason": "partition_mismatch",
                             "detail": [str(record.get("partition"))]})
            continue
        if record.get("task") == "code_generation" \
                and record.get("language") == "python" \
                and not is_valid_python(record["response"]):
            excluded.append({"id": record["id"], "reason": "python_syntax",
                             "detail": ["response does not parse"]})
            continue
        accepted.append(record)
    return accepted, excluded


def build(corpus_path="data/code_corpus_v2.jsonl",
          licence_report="data/licence_verification.json",
          out_dir="data/m7", dataset_id=DATASET_ID,
          tokenizer_path="tokenizer/qm_code_bpe_v0.json",
          now=None) -> dict:
    corpus_path = Path(corpus_path)
    licence = json.loads(Path(licence_report).read_text(encoding="utf-8"))
    verified = set(licence.get("verified_ids") or [])
    if not verified:
        raise ValueError("licence report lists no verified sources")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    records = [json.loads(line) for line in
               corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_source = Counter(r.get("source_id") for r in records)
    withheld_present = {source: count for source, count in by_source.items()
                        if source not in verified}
    if not TRAIN_SOURCES[0] in by_source:
        raise ValueError(
            f"{corpus_path} does not contain the expected verified sources; "
            "refusing to build a dataset whose provenance cannot be checked"
        )

    by_partition = _candidate_functions(records)
    fixtures = _fixture_partitions()
    split_of = {"train": "train", "val": "val", "dev": "val", "locked": "test"}

    partitions: dict[str, list[dict]] = {}
    exclusions: dict[str, list[dict]] = {}
    for partition in ("train", "val", "dev", "locked"):
        derived, excluded = _build_partition(
            by_partition.get(partition, []), partition, split_of[partition],
            tokenizer=tokenizer)
        derived.extend(fixtures.get(partition, []))
        accepted, quality_excluded = _quality(derived, verified, partition)
        partitions[partition] = accepted
        exclusions[partition] = excluded + quality_excluded

    if not partitions["locked"]:
        raise ValueError("locked partition is empty; evaluation would be vacuous")

    oracles, oracle_added = build_oracles(records, partitions["locked"])
    exclusions["locked"].extend(
        {"id": r["id"], "reason": "added_functional_probe",
         "detail": [r["oracle"]["function"]]} for r in oracle_added)

    isolation = _isolation(partitions)
    if not isolation["ok"]:
        raise ValueError(f"partition isolation failed: {isolation['pairs']}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for partition in ("train", "val", "dev", "locked"):
        path = out_dir / f"{dataset_id}.{partition}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(partitions[partition], key=lambda r: r["id"]):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[partition] = path

    oracles_path = out_dir / f"{dataset_id}.oracles.json"
    oracles_path.write_text(json.dumps({
        "schema": "quantsmind.m7_oracles/1",
        "dataset_id": dataset_id,
        "generated_at": now or utc_now_iso(),
        "locked_source": LOCKED_SOURCE,
        "oracles": oracles,
        "limitations": [
            "Eight pure functions from one held-out repository are not broad "
            "functional coverage.",
            "Expected values come from the reference implementation; a model "
            "that copies the reference passes by construction, which is why "
            "the reference itself is never in a prompt.",
            "No debugging, refactoring, or test-generation oracle exists, "
            "because the corpus contains no verified expected behaviour for them.",
        ],
    }, indent=2), encoding="utf-8")

    def info(partition):
        rs = partitions[partition]
        return {
            "path": str(paths[partition].as_posix()),
            "sha256": _sha256(paths[partition]),
            "records": len(rs),
            "sources": sorted({r["source"] for r in rs}),
            "tasks": dict(Counter(r["task"] for r in rs)),
            "licenses": dict(Counter(r["license"] for r in rs)),
            "statistics": summarise(rs),
        }

    manifest = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "generated_at": now or utc_now_iso(),
        "transform": TRANSFORM,
        "source_corpus": {"path": str(corpus_path.as_posix()),
                          "sha256": _sha256(corpus_path)},
        "licence_report": {"path": str(Path(licence_report).as_posix()),
                           "sha256": _sha256(Path(licence_report))},
        "tokenizer": {"path": str(Path(tokenizer_path).as_posix()),
                      "sha256": _sha256(Path(tokenizer_path)),
                      "max_length": 512},
        "trainability": {
            "policy": ("Every accepted example was encoded with the M7 tokenizer "
                       "at build time. Generation/explanation prompts must fit "
                       "without truncation; code-completion prompts may be "
                       "left-truncated by design, but targets are never "
                       "truncated."),
        },        "licence_gate": {
            "verified_sources": sorted(verified),
            "withheld_sources": withheld_present,
            "withheld_records": sum(withheld_present.values()),
            "policy": ("Only sources in licence_verification.json:verified_ids "
                       "enter any partition. Records from other sources are "
                       "counted and withheld."),
            "unreviewed_allowed": False,
        },
        "partitions": {p: info(p) for p in ("train", "val", "dev", "locked")},
        "partition_policy": {
            "train": list(TRAIN_SOURCES),
            "val": [VAL_SOURCE + " (even file-hash bucket)"],
            "dev": [VAL_SOURCE + " (odd file-hash bucket)"],
            "locked": [LOCKED_SOURCE],
            "unit": "source repository and source file, never individual examples",
        },
        "isolation": isolation,
        "excluded": {
            p: {"count": len(v),
                "by_reason": dict(Counter(x["reason"] for x in v)),
                "examples": v[:10]}
            for p, v in exclusions.items()
        },
        "oracles": {
            "path": str(oracles_path.as_posix()),
            "sha256": _sha256(oracles_path),
            "count": len(oracles),
            "functions": [o["function"] for o in oracles],
        },
        "unavailable_tasks": ["debugging", "refactoring", "test_generation"],
        "unavailable_reason": (
            "The corpus contains no verified expected behaviour (tests or "
            "reference fixes) for these tasks. They are absent rather than "
            "fabricated, so no capability claim about them is possible."
        ),
        "not_synthesised_reason": (
            "Every response is text from a licensed corpus file or a CC0 "
            "fixture written for this repository; no external model was used."
        ),
        "scale_disclosure": (
            "Derived from ~1.6M tokens across seven licence-verified "
            "repositories. This is a pilot dataset, not a production corpus."
        ),
        "coverage": {
            "corpus_records": len(records),
            "functions_with_docstring": sum(len(v) for v in by_partition.values()),
            "oracle_functions": len(oracles),
        },
    }
    manifest_path = out_dir / f"{dataset_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.as_posix())
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the M7 licence-gated supervised coding dataset")
    parser.add_argument("--corpus", default="data/code_corpus_v2.jsonl")
    parser.add_argument("--licence", default="data/licence_verification.json")
    parser.add_argument("--out-dir", default="data/m7")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args(argv)

    try:
        manifest = build(args.corpus, args.licence, args.out_dir, args.dataset_id)
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"M7 dataset build refused: {exc}")
        return 1

    print(f"M7 dataset: {manifest['dataset_id']}")
    for name, part in manifest["partitions"].items():
        print(f"  {name:7} {part['records']:4} records  {part['tasks']}")
    print(f"  oracles {manifest['oracles']['count']} "
          f"({', '.join(manifest['oracles']['functions'])})")
    print(f"  manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
