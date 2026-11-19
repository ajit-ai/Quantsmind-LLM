"""M5 benchmark contract and dataset validation.

A benchmark example is only useful if it can be audited: who wrote it, under
what licence, what it measures, and whether a trustworthy oracle exists. This
module defines that contract and validates a file against it.

The rule that shapes everything here: **a task without a trustworthy oracle is
not scored.** It is marked ``requires_human_review`` or excluded, and reported
with a denominator. Manufacturing a reference answer to fill a missing field
would make the benchmark a measure of the author's imagination.

A record's own ``license`` string is never accepted as authorisation. The
verified-source registry (``data/licence_verification.json``) decides, which is
the same rule M3 and M4 applied.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "quantsmind.benchmark/1"
SCORING_VERSION = "quantsmind.m5_scoring/1"

TASK_TYPES = (
    "code_generation",
    "code_completion",
    "debugging",
    "refactoring",
    "code_explanation",
    "unit_test_generation",
    "documentation_generation",
    "sql_generation",
    "software_architecture",
    "devops_config",
    "structured_output",
    "instruction_following",
)

SPLITS = ("eval", "dev", "train")

# Which metrics apply to which task. Recording this up front is what stops a
# single blended "score" from being invented later.
TASK_METRICS = {
    "code_generation": ("syntax", "functional", "compliance"),
    "code_completion": ("syntax", "functional", "compliance"),
    "debugging": ("syntax", "functional", "compliance"),
    "refactoring": ("syntax", "functional", "compliance"),
    "code_explanation": ("compliance", "human_review"),
    "unit_test_generation": ("syntax", "human_review"),
    "documentation_generation": ("compliance", "human_review"),
    "sql_generation": ("syntax", "human_review"),
    "software_architecture": ("compliance", "human_review"),
    "devops_config": ("syntax", "human_review"),
    "structured_output": ("structured", "compliance"),
    "instruction_following": ("compliance",),
}

# Tasks whose correctness cannot be established without a human judgement.
HUMAN_ONLY_METRICS = ("human_review",)

_ID_RE = re.compile(r"^m5-[a-z0-9]+-[0-9a-f]{12}$")

REQUIRED_FIELDS = (
    "id", "task_type", "prompt", "expected_behavior", "source_id",
    "license_id", "split", "provenance",
)

OPTIONAL_FIELDS = (
    "reference_answer", "test_specification", "language", "metadata",
    "checklist", "executable", "synthetic",
)


def stable_id(task_type: str, prompt: str) -> str:
    """Content-addressed id, stable across rebuilds."""
    digest = hashlib.sha256(
        f"{task_type}\x00{prompt}".encode("utf-8")
    ).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "", task_type.lower())[:16] or "task"
    return f"m5-{slug}-{digest}"


def normalise(text: str) -> str:
    """Normalised form for exact-duplicate detection.

    Lowercased, whitespace collapsed, unicode-normalised. This catches
    reformatting duplicates; it is **not** semantic deduplication and is not
    claimed to be.
    """
    import unicodedata

    folded = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(folded.split())


def text_hash(text: str) -> str:
    """SHA-256 of the *normalised* text, so trivial edits collapse."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def is_scorable(record) -> bool:
    """True when at least one automatic metric can be computed.

    "At least one" is deliberate. A code-generation task with an authored test
    but no format checklist is still scorable -- on functional correctness.
    Requiring *every* applicable metric to be satisfiable would silently drop
    exactly the tasks that have a trustworthy oracle.

    A record with neither a test nor any machine-checkable rule cannot be scored
    and must not enter an automated denominator.
    """
    metrics = [m for m in TASK_METRICS.get(record.get("task_type"), ())
               if m not in HUMAN_ONLY_METRICS]
    if not metrics:
        return False
    if "functional" in metrics and record.get("test_specification"):
        return True
    if ("structured" in metrics or "compliance" in metrics) \
            and record.get("checklist"):
        return True
    return False


def validate_record(record) -> list[str]:
    """Return problems. Empty means the record satisfies the contract."""
    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not an object"]

    problems = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing required field {field!r}")
    if problems:
        return problems

    if not isinstance(record["id"], str) or not _ID_RE.match(record["id"]):
        problems.append(f"id {record['id']!r} is not of the form m5-<task>-<12 hex>")

    if record.get("task_type") not in TASK_TYPES:
        problems.append(
            f"task_type {record.get('task_type')!r} is not one of {list(TASK_TYPES)}"
        )

    for field in ("prompt", "expected_behavior", "source_id", "license_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field} must be a non-empty string")

    if record.get("split") not in SPLITS:
        problems.append(f"split {record.get('split')!r} is not one of {list(SPLITS)}")

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("provenance must be an object")
    else:
        for field in ("origin", "license_evidence"):
            if not provenance.get(field):
                problems.append(f"provenance.{field} is required")

    if not isinstance(record.get("prompt"), str):
        problems.append("prompt must be a string")
    elif len(record["prompt"]) > 4000:
        problems.append(
            f"prompt is {len(record['prompt'])} chars; over the 4000-char limit"
        )

    language = record.get("language")
    if language is not None and not isinstance(language, str):
        problems.append("language must be a string when present")

    test_spec = record.get("test_specification")
    if test_spec is not None and not isinstance(test_spec, str):
        problems.append("test_specification must be a string when present")

    checklist = record.get("checklist")
    if checklist is not None:
        if not isinstance(checklist, list) or not all(
            isinstance(item, str) for item in checklist
        ):
            problems.append("checklist must be a list of strings")

    for field in OPTIONAL_FIELDS:
        if field in record and record[field] is None and field in REQUIRED_FIELDS:
            problems.append(f"{field} may not be null")

    return problems


def validate_dataset(path, verified_sources=None, require_provenance=True):
    """Validate a benchmark file and report counts, rejections and exclusions.

    ``verified_sources`` is a set of source ids authorised by the project's
    licence registry. When given, any example whose ``source_id`` is not in it
    is rejected -- regardless of what its own ``license_id`` says.
    """
    path = Path(path)
    records, malformed = [], []
    text = path.read_text(encoding="utf-8-sig")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            malformed.append({"line": number, "reason": str(exc)})

    accepted, rejected = [], []
    seen_ids, seen_prompts = set(), set()

    for record in records:
        rid = record.get("id", "<no id>")
        problems = validate_record(record) if require_provenance else []
        if problems:
            rejected.append({"id": rid, "reason": "schema", "detail": problems})
            continue
        if rid in seen_ids:
            rejected.append({"id": rid, "reason": "duplicate_id",
                             "detail": ["id already seen"]})
            continue
        seen_ids.add(rid)

        if verified_sources is not None and record.get("source_id") not in verified_sources:
            rejected.append({
                "id": rid, "reason": "unauthorized_source",
                "detail": [f"source_id {record.get('source_id')!r} is not in the "
                           f"verified-source registry; the example's own "
                           "license_id is not authorisation"],
            })
            continue

        prompt_key = normalise(record["prompt"])
        if prompt_key in seen_prompts:
            rejected.append({"id": rid, "reason": "duplicate_prompt",
                             "detail": ["normalised prompt already seen"]})
            continue
        seen_prompts.add(prompt_key)

        accepted.append(record)

    scorable = [r for r in accepted if is_scorable(r)]
    unscorable = [r for r in accepted if not is_scorable(r)]

    return {
        "path": str(path),
        "schema_version": SCHEMA,
        "scoring_version": SCORING_VERSION,
        "total_records": len(records),
        "malformed_lines": malformed,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejections": rejected,
        "scorable": len(scorable),
        "requires_human_review": len(unscorable),
        "human_review_ids": [r["id"] for r in unscorable],
        "by_task": {},
        "by_split": {},
    }


def manifest_hash(records) -> str:
    """Stable hash over the benchmark content, order-independent."""
    payload = json.dumps(
        sorted(
            (r.get("id"), normalise(r.get("prompt", ""))) for r in records
        ),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows