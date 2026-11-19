"""TAKSH L0.7.6 — master roadmap registry validator.

Validates configs/roadmap/taksh-v3-roadmap.yaml for internal consistency:
IDs unique, statuses valid, dependencies resolvable, no cycles, releases map
to milestones, evidence references exist, prose cross-references align.

Never modifies any artifact or authorization state. Read-only by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

REGISTRY = REPO / "configs" / "roadmap" / "taksh-v3-roadmap.yaml"
ROADMAP_MD = REPO / "docs" / "TAKSH-MASTER-ROADMAP-v3.0.0.md"

VALID_STATUSES = {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "CONDITIONAL_PASS",
                  "QUALIFIED", "RELEASE_CANDIDATE", "GA", "DEFERRED"}

COMPLETEDISH = {"QUALIFIED", "CONDITIONAL_PASS", "BLOCKED", "IN_PROGRESS"}


def load_registry(path: Path = REGISTRY) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_registry(data: dict) -> list[str]:
    problems: list[str] = []

    if data.get("schema") != "taksh.v3_roadmap_registry/1":
        problems.append(f"schema must be taksh.v3_roadmap_registry/1, got {data.get('schema')!r}")
    for key in ("id", "version", "releases", "milestones"):
        if key not in data:
            problems.append(f"missing top-level field: {key}")
    if problems:
        return problems

    milestones = data["milestones"]
    ids = [m.get("id") for m in milestones]
    if len(set(ids)) != len(ids):
        seen = {}
        for i in ids:
            seen[i] = seen.get(i, 0) + 1
        dups = sorted(i for i, n in seen.items() if n > 1)
        problems.append(f"duplicate milestone ids: {dups}")
    if any(not i or not isinstance(i, str) for i in ids):
        problems.append("every milestone needs a non-empty id")

    id_set = set(ids)

    # statuses and required fields
    for m in milestones:
        mid = m.get("id")
        if m.get("status") not in VALID_STATUSES:
            problems.append(f"{mid}: invalid status {m.get('status')!r}")
        for key in ("title", "stage", "status", "dependencies", "prerequisites",
                    "deliverables", "exit_criteria", "evidence"):
            if key not in m:
                problems.append(f"{mid}: missing field {key}")
        if not m.get("deliverables"):
            problems.append(f"{mid}: deliverables must be non-empty")
        if not m.get("exit_criteria"):
            problems.append(f"{mid}: exit_criteria must be non-empty")
        if m.get("status") in ("QUALIFIED", "CONDITIONAL_PASS") and not m.get("evidence"):
            problems.append(f"{mid}: status {m['status']} requires evidence references")

    # dependencies + cycle detection
    for m in milestones:
        for dep in m.get("dependencies", []):
            if dep not in id_set:
                problems.append(f"{m['id']}: dependency {dep!r} does not exist")
    graph = {m["id"]: list(m.get("dependencies", [])) for m in milestones}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in id_set}

    def visit(node, stack):
        color[node] = GRAY
        for dep in graph.get(node, []):
            if dep not in color:
                continue  # already reported as a missing dependency; nothing to recurse into
            if color[dep] == GRAY:
                problems.append(f"dependency cycle: {' -> '.join(stack + [dep])}")
                return
            if color[dep] == WHITE:
                visit(dep, stack + [dep])
        color[node] = BLACK

    for i in id_set:
        if color[i] == WHITE:
            visit(i, [i])

    # dependency + cycle detection done above; stage integrity:
    for m in milestones:
        if m.get("stage") not in {"L0", "L1", "L2", "L3"}:
            problems.append(f"{m['id']}: stage must be one of L0..L3, got {m.get('stage')!r}")
    milestone_stages = {m["id"]: m["stage"] for m in milestones}

    # stages against releases
    stage_map = {r["id"]: r.get("stage") for r in data["releases"]}
    stage_releases = {r["id"]: set(r.get("milestones", [])) for r in data["releases"]}
    for stage_id, rel_milestones in stage_releases.items():
        expected = stage_map[stage_id]
        for mid in rel_milestones:
            if mid not in id_set:
                problems.append(f"release {stage_id} references missing milestone {mid!r}")
            if milestone_stages.get(mid) != expected:
                problems.append(f"release {stage_id} expects stage {expected!r} but milestone {mid!r} has {milestone_stages.get(mid)!r}")
    all_staged = set().union(*(stage_releases.values())) if stage_releases else set()
    missing_stage = id_set - all_staged
    if missing_stage:
        problems.append(f"milestones not mapped to any release: {sorted(missing_stage)}")

    # evidence existence for qualified/in-progress milestones
    for m in milestones:
        if m.get("status") in ("QUALIFIED", "CONDITIONAL_PASS", "IN_PROGRESS"):
            for ev in m.get("evidence", []):
                if not (REPO / ev).exists():
                    problems.append(f"{m['id']}: evidence path missing: {ev}")

    # markdown cross-reference
    if ROADMAP_MD.is_file():
        text = ROADMAP_MD.read_text(encoding="utf-8")
        for r in data["releases"]:
            if r["id"].lower() not in text.lower():
                problems.append(f"roadmap markdown missing release {r['id']}")
        for m in milestones[:20]:
            if m["id"] not in text:
                problems.append(f"roadmap markdown missing milestone {m['id']}")
    else:
        problems.append(f"roadmap markdown missing: {ROADMAP_MD}")

    return problems


def summarize(data: dict) -> dict:
    from collections import Counter

    return dict(Counter(m.get("status") for m in data.get("milestones", [])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l076_roadmap_validation.json")
    args = parser.parse_args(argv)

    data = load_registry()
    problems = validate_registry(data)
    summary = summarize(data)
    report = {
        "schema": "taksh.roadmap_validation/1",
        "generated_by": "evaluation/roadmap/roadmap_validator.py",
        "registry": "configs/roadmap/taksh-v3-roadmap.yaml",
        "valid": not problems,
        "problems": problems,
        "status_counts": summary,
        "milestone_count": len(data.get("milestones", [])),
        "release_count": len(data.get("releases", [])),
        "verdict": ("ROADMAP VALID — statuses and dependencies consistent; nothing authorized; "
                    "no owner decision, authorization or capability is implied")
        if not problems else "ROADMAP INCONSISTENT",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("ROADMAP INVALID:", p)
        return 2
    print(f"roadmap valid; milestones={report['milestone_count']} statuses={summary}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
