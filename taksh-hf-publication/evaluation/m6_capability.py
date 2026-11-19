"""M6 capability categories and evidence levels.

Two separate concerns, kept separate on purpose.

**Categories** answer "which coding tasks does the benchmark actually cover?".
A category with no eligible examples is reported ``UNAVAILABLE`` -- never a zero.

**Evidence levels** answer "how much has actually been proven?". They are
ordered and strictly cumulative in meaning: passing level N does not imply
level N+1, and this module never promotes automatically. Each level states the
evidence it requires.

The benchmark is 26 examples. Every level reached here is therefore reported
``preliminary``, because a single example moves a rate by several points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Benchmark task_type -> human-facing capability category.
CATEGORY_BY_TASK = {
    "code_generation": "code_generation",
    "code_completion": "code_completion",
    "code_explanation": "code_explanation",
    "sql_generation": "sql",
    "structured_output": "structured_output",
    "instruction_following": "instruction_compliance",
}

# Categories the milestone asks about that the frozen benchmark does not cover.
# Declared explicitly so a report can say "no eligible examples" rather than
# quietly omitting them.
DECLARED_CATEGORIES = [
    "code_generation", "code_completion", "debugging", "refactoring",
    "code_explanation", "unit_test_generation", "documentation", "sql",
    "structured_output", "instruction_compliance",
]

UNAVAILABLE = "UNAVAILABLE"


@dataclass
class CategoryResult:
    category: str
    eligible_examples: int = 0
    metrics: dict = field(default_factory=dict)
    status: str = UNAVAILABLE
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def categories_from_results(results) -> list[CategoryResult]:
    """Per-category coverage from persisted example records.

    Unlike :func:`categories_from_manifest`, this counts *examples*, not metric
    denominators. A checklist with two rules would otherwise make a single
    example look like two, which is the difference between "1 example" and
    "2 eligible examples" in a published table.
    """
    buckets: dict[str, dict] = {}
    for record in results:
        task = record.get("task_type")
        if not task:
            continue
        category = CATEGORY_BY_TASK.get(task, task)
        bucket = buckets.setdefault(
            category, {"examples": 0, "scorable": 0, "metrics": {}})
        bucket["examples"] += 1
        metrics = record.get("metrics") or {}
        if any((m.get("denominator") or 0) > 0 for m in metrics.values()):
            bucket["scorable"] += 1
        for name, metric in metrics.items():
            agg = bucket["metrics"].setdefault(
                name, {"numerator": 0, "denominator": 0})
            agg["numerator"] += metric.get("numerator") or 0
            agg["denominator"] += metric.get("denominator") or 0

    out = []
    for category in DECLARED_CATEGORIES:
        bucket = buckets.get(category)
        if bucket is None:
            out.append(CategoryResult(
                category=category, status=UNAVAILABLE,
                note="no eligible examples in the frozen M5 benchmark"))
            continue
        metrics = {}
        for name, agg in bucket["metrics"].items():
            denominator = agg["denominator"]
            metrics[name] = {
                "numerator": agg["numerator"], "denominator": denominator,
                "rate": (round(agg["numerator"] / denominator, 6)
                         if denominator else None)}
        entry = CategoryResult(
            category=category, eligible_examples=bucket["scorable"],
            metrics=metrics,
            status="SCORED" if bucket["scorable"] else UNAVAILABLE,
            note=("no scorable examples (all human review)"
                  if not bucket["scorable"] else
                  "scores are preliminary; denominator is small"
                  if bucket["scorable"] < 20 else ""))
        out.append(entry)
    return out


def categories_from_manifest(manifest) -> list[CategoryResult]:
    """Derive per-category coverage from a run manifest's ``by_task`` block.

    Use :func:`categories_from_results` when the persisted examples are
    available; this fallback can only see denominators and so may overcount a
    task whose metrics carry more than one rule per example.
    """
    by_task = manifest.get("by_task") or {}
    results: dict[str, CategoryResult] = {}

    for task, metrics in by_task.items():
        category = CATEGORY_BY_TASK.get(task, task)
        entry = results.setdefault(category, CategoryResult(category=category))
        # A task carries several metrics with different denominators (syntax 20,
        # compliance 6, human_review 0). Taking the first one reported a task as
        # UNAVAILABLE whenever ``human_review`` happened to be listed first.
        # The eligible count is the largest denominator a scorable metric
        # actually offered.
        denominators = [int(m.get("denominator") or 0)
                        for name, m in metrics.items()
                        if name != "human_review"]
        entry.eligible_examples += max(denominators, default=0)
        entry.metrics[task] = {
            name: {"numerator": m.get("numerator"),
                   "denominator": m.get("denominator"),
                   "rate": m.get("rate")}
            for name, m in metrics.items()}

    out = []
    for category in DECLARED_CATEGORIES:
        entry = results.get(category)
        if entry is None:
            out.append(CategoryResult(
                category=category, status=UNAVAILABLE,
                note="no eligible examples in the frozen M5 benchmark"))
            continue
        if entry.eligible_examples == 0:
            entry.status = UNAVAILABLE
            entry.note = "task present but no scorable examples (all human review)"
        else:
            entry.status = "SCORED"
            entry.note = ("scores are preliminary; denominator is small"
                          if entry.eligible_examples < 20 else "")
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Evidence levels
# --------------------------------------------------------------------------- #
@dataclass
class LevelDefinition:
    level: int
    name: str
    requires: str


LEVELS = [
    LevelDefinition(1, "RUNTIME_WORKS",
                    "the runtime loaded the model and produced a response "
                    "without error"),
    LevelDefinition(2, "GENERATION_WORKS",
                    "generation produced non-empty output on at least one "
                    "benchmark example"),
    LevelDefinition(3, "SYNTAX_CAPABILITY_DEMONSTRATED",
                    "at least one output parsed as valid Python and the "
                    "denominator is reported"),
    LevelDefinition(4, "FUNCTIONAL_CAPABILITY_DEMONSTRATED",
                    "at least one output passed a trusted functional oracle; "
                    "denominator > 0"),
    LevelDefinition(5, "INSTRUCTION_FOLLOWING_DEMONSTRATED",
                    "instruction-compliance numerator > 0 with a stated "
                    "denominator"),
    LevelDefinition(6, "USEFUL_CODING_CAPABILITY_SUPPORTED",
                    "functional numerator > 0 across MULTIPLE independent "
                    "examples on a benchmark large enough to be non-anecdotal"),
    LevelDefinition(7, "PRODUCT_READINESS_ESTABLISHED",
                    "human acceptance plus independent technical review"),
]

# The frozen benchmark is 26 examples. Until it grows, level 6 cannot be
# reached on this sample no matter how well a model scores.
MIN_EXAMPLES_FOR_USEFUL = 20

# Checklist rules that a stub answer satisfies without doing the task. A model
# that emits ``\n    """`` is "non-empty" and "has no code fences", so counting
# either as instruction following would turn a formatting accident into a
# capability claim. These rules only count when the output is non-degenerate.
VACUOUS_COMPLIANCE_RULES = frozenset({"non_empty", "no_fences"})

# Below this many stripped characters an answer is a stub, not an attempt.
MIN_SUBSTANTIVE_CHARS = 8


def compliance_evidence(results) -> dict:
    """Count compliance passes that reflect a real attempt.

    ``results`` is an iterable of example records carrying ``output`` and
    ``metrics``. A pass is *substantive* when the rule is not trivially
    satisfiable and the output is longer than a stub. This is what level 5 is
    checked against, so a vacuous 3/6 cannot be reported as instruction
    following.
    """
    examples = substantive = vacuous = 0
    rules_seen = set()
    for record in results:
        metric = (record.get("metrics") or {}).get("compliance") or {}
        if not metric.get("denominator"):
            continue
        examples += 1
        text = (record.get("output") or "").strip()
        real_attempt = len(text) >= MIN_SUBSTANTIVE_CHARS
        for check in metric.get("checks") or []:
            if not check.get("ok"):
                continue
            rule = check.get("rule")
            rules_seen.add(rule)
            if rule in VACUOUS_COMPLIANCE_RULES or not real_attempt:
                vacuous += 1
            else:
                substantive += 1
    return {
        "examples_with_checklist": examples,
        "substantive_passes": substantive,
        "vacuous_passes": vacuous,
        "rules_seen": sorted(r for r in rules_seen if r),
        "note": (
            "a pass is vacuous when the rule is non_empty/no_fences or the "
            "output is shorter than a stub; only substantive passes can "
            "support level 5"),
    }


def assess_levels(manifest, *, task_head="causal_lm", human_accepted=False,
                  technical_review=False, compliance_evidence=None) -> dict:
    """Determine the highest evidence level actually supported.

    Deliberately conservative: each level is checked against persisted run data,
    and level 6 additionally requires a benchmark large enough that a single
    lucky example cannot account for the result.
    """
    status = manifest.get("status")
    summary = manifest.get("summary") or {}
    evaluated = bool(manifest.get("complete")) and "summary" in manifest

    reached = []
    notes = []

    if not evaluated:
        return {
            "highest_level": 0,
            "highest_name": "NOT_EVALUATED",
            "reached": [],
            "levels": [asdict(level) for level in LEVELS],
            "preliminary": True,
            "note": (f"run status is {status!r}; no evidence level is claimed "
                     f"for a model that did not produce scored output"),
        }

    def metric(name):
        return summary.get(name) or {}

    synt = metric("syntax")
    func = metric("functional")
    comp = metric("compliance")

    if evaluated:
        reached.append(1)
    if (manifest.get("examples_evaluated") or 0) > 0:
        reached.append(2)
    if (synt.get("numerator") or 0) > 0:
        reached.append(3)
    if (func.get("numerator") or 0) > 0 and (func.get("denominator") or 0) > 0:
        reached.append(4)
    if (comp.get("numerator") or 0) > 0 and (comp.get("denominator") or 0) > 0:
        if compliance_evidence is None:
            notes.append(
                "compliance passes are uncorroborated: no per-rule evidence "
                "was supplied, so level 5 is not claimed")
        elif (compliance_evidence.get("substantive_passes") or 0) > 0:
            reached.append(5)
        else:
            notes.append(
                f"level 5 not claimed: all {comp.get('numerator')} compliance "
                f"passes are vacuous ({compliance_evidence.get('vacuous_passes')}"
                f" rule hits on stub or non_empty/no_fences answers), which is a "
                f"formatting artifact rather than instruction following")

    # Level 6 needs breadth, not a single hit.
    if 4 in reached and (func.get("numerator") or 0) >= 2 \
            and (manifest.get("examples_total") or 0) >= MIN_EXAMPLES_FOR_USEFUL:
        reached.append(6)
    elif 4 in reached:
        notes.append(
            "level 6 needs more than one functional pass on a benchmark of at "
            "least "
            f"{MIN_EXAMPLES_FOR_USEFUL} examples; the current sample cannot "
            "support a useful-coding claim")

    if human_accepted and technical_review and 6 in reached:
        reached.append(7)
    elif 6 in reached:
        notes.append(
            "level 7 requires human acceptance and independent technical "
            "review; neither has been given")

    highest = max(reached) if reached else 0
    if task_head in ("embedding", "masked_lm") and highest >= 3:
        notes.append(
            f"task head {task_head} cannot follow a coding instruction; syntax "
            f"or compliance hits are formatting artifacts, not capability")
        highest = min(highest, 2)

    return {
        "highest_level": highest,
        "highest_name": (next((l.name for l in LEVELS if l.level == highest),
                              "NOT_EVALUATED") if highest else "NOT_EVALUATED"),
        "reached": reached,
        "levels": [asdict(level) for level in LEVELS],
        "preliminary": True,
        "note": " ".join(notes) or (
            "highest level supported by the persisted run data"),
    }