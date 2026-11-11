"""TAKSH L0.6.2 — DLCAE ↔ inference interface.

Clear interfaces for the future DLCAE ↔ Learning Engine ↔ inference loop,
strictly preserving the boundary rule: inference output never upgrades a
language's qualification status. Proposals are proposals; registry updates
require the owner.

Interfaces:
  1. identify_language()          — declared hint or registry probe (never capability)
  2. registry_lookup()            — status + qualification state from the committed registry
  3. comprehension_state_lookup() — DLCAE state for the language, if any
  4. retrieve_knowledge()         — KnowledgeSource protocol; NullKnowledgeSource default
  5. build_prompt()               — canonical prompt construction with diagnostics
  6. record_inference_feedback()  — evaluation feedback record (never scored here)
  7. propose_registry_update()    — PENDING_OWNER_REVIEW proposal; never writes the registry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Protocol

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_LANGUAGES = REPO / "configs" / "feasibility" / "l051_languages.yaml"
_EXTENSION_REGISTRY = REPO / "configs" / "foundation" / "l060r1_dynamic_language_registry.yaml"

_SUFFIX_HINTS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin", ".cs": "csharp",
    ".scala": "scala", ".pl": "perl", ".lua": "lua", ".r": "r", ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".hs": "haskell", ".clj": "clojure", ".ex": "elixir",
    ".exs": "elixir", ".erl": "erlang", ".dart": "dart", ".m": "objective_c",
    ".mm": "objective_c", ".jl": "julia", ".nim": "nim", ".zig": "zig",
    ".hs-boot": "haskell", ".ktm": "kotlin", ".kark": "karkain",
}


def load_registry() -> dict[str, dict]:
    import yaml

    data = yaml.safe_load(_LANGUAGES.read_text(encoding="utf-8"))
    registry = {}
    for lang in data.get("languages", []):
        registry[lang["id"]] = {**lang, "class": "language"}
    for tech in data.get("supporting_technologies", []):
        key = tech["name"].lower().replace(" ", "_")
        registry[key] = {**tech, "id": key, "class": "technology"}
    if _EXTENSION_REGISTRY.is_file():
        ext = yaml.safe_load(_EXTENSION_REGISTRY.read_text(encoding="utf-8"))
        for item in ext.get("entries", []):
            registry[item["id"]] = {**item,
                                    "id": item["id"],
                                    "status": item.get("qualification_status", "DISCOVERED"),
                                    "class": "language"}
    return registry


def identify_language(text: str | None = None, *, filename: str | None = None,
                      declared: str | None = None) -> dict:
    """Return {language_id, source, confidence, registered} — never capability."""
    registry = load_registry()
    if declared:
        lid = declared.strip().lower()
        return {"language_id": lid, "source": "declared",
                "registered": lid in registry, "confidence": 1.0}
    if filename:
        suffix = Path(filename).suffix.lower()
        lid = _SUFFIX_HINTS.get(suffix)
        if lid:
            return {"language_id": lid, "source": "filename_extension",
                    "registered": lid in registry, "confidence": 0.9}
    return {"language_id": None, "source": "unknown",
            "registered": False, "confidence": 0.0}


def registry_lookup(language_id: Optional[str]) -> dict:
    registry = load_registry()
    if not language_id:
        return {"language_id": None, "registered": False, "status": "UNREGISTERED"}
    entry = registry.get(language_id.lower())
    if entry is None:
        return {"language_id": language_id, "registered": False, "status": "UNREGISTERED"}
    return {
        "language_id": language_id,
        "registered": True,
        "class": entry.get("class"),
        "status": entry.get("status") or entry.get("qualification_status") or "REGISTERED",
        "canonical_name": entry.get("name", language_id),
        "benchmark_status": entry.get("benchmark_status"),
        "dataset_status": entry.get("dataset_status"),
        "execution_available_here": entry.get("execution_available_here"),
        "note": "registration never implies capability",
    }


def comprehension_state_lookup(language_id: Optional[str]) -> dict:
    """DLCAE comprehension state from the registry statuses; no writes."""
    info = registry_lookup(language_id)
    if not info["registered"]:
        return {**info, "comprehension_state": "NONE"}
    status = str(info.get("status", ""))
    if status in ("FUNCTIONALLY_QUALIFIED", "ENGINEERING_QUALIFIED"):
        state = "COMPREHENSION_COMPLETE"
    elif status == "COMPREHENSION_IN_PROGRESS":
        state = "COMPREHENSION_IN_PROGRESS"
    else:
        state = "NOT_STARTED"
    return {**info, "comprehension_state": state}


class KnowledgeItem(dict):
    pass


class KnowledgeSource(Protocol):
    def get(self, query: str, language_id: Optional[str]) -> list[KnowledgeItem]: ...


class NullKnowledgeSource:
    """Default: no knowledge base is wired up; returns nothing, says so."""

    status = "NO_KNOWLEDGE_BASE"

    def get(self, query: str, language_id: Optional[str]) -> list[KnowledgeItem]:
        return []


def retrieve_knowledge(query: str, language_id: Optional[str] = None,
                       source: Optional[KnowledgeSource] = None) -> dict:
    src = source if source is not None else NullKnowledgeSource()
    items = src.get(query, language_id)
    return {
        "language_id": language_id,
        "query": query,
        "source_status": getattr(src, "status", "CUSTOM"),
        "items": list(items),
        "note": "empty means no vetted knowledge exists; never fabricated",
    }


def build_prompt(task_id: Optional[str], language_id: Optional[str],
                 instruction: str, context_snippets: Optional[list[str]] = None) -> dict:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    parts = []
    if language_id:
        parts.append(f"Language: {language_id}")
    if context_snippets:
        parts.extend(s for s in context_snippets if s.strip())
    parts.append(instruction.strip())
    prompt = "\n\n".join(parts)
    return {
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "language_id": language_id,
        "task_id": task_id,
        "construction": "language-tagged instruction with optional vetted context snippets",
        "diagnostics": {"context_snippets": len(context_snippets or [])},
    }


def record_inference_feedback(record: dict) -> dict:
    """Normalize an inference trace into an evaluation feedback record.

    Never scores, never upgrades registry state; the frozen evaluation
    harness owns capability measurement.
    """
    return {
        "schema": "taksh.inference_feedback/1",
        "task_id": record.get("task_id"),
        "language_id": record.get("language_id"),
        "mode": record.get("mode"),
        "finish_reason": record.get("finish_reason"),
        "completion_tokens": record.get("completion_tokens"),
        "measured": True,
        "scored_by": "frozen evaluation harness only",
        "capability_verdict": None,
    }


def propose_registry_update(current: dict, proposed_status: str) -> dict:
    allowed = {"REGISTERED", "DISCOVERED", "COMPREHENSION_IN_PROGRESS",
               "FUNCTIONALLY_QUALIFIED", "ENGINEERING_QUALIFIED", "BLOCKED"}
    if proposed_status not in allowed:
        raise ValueError(f"proposed_status must be one of {sorted(allowed)}")
    return {
        "schema": "taksh.registry_update_proposal/1",
        "language_id": current.get("language_id"),
        "current_status": current.get("status"),
        "proposed_status": proposed_status,
        "state": "PENDING_OWNER_REVIEW",
        "written": False,
        "note": "proposal only; the registry is never mutated by inference code",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l062_dlcae_inference_interface.json")
    args = parser.parse_args(argv)

    probe = identify_language(filename="kernel.kark", declared=None)
    lookup = registry_lookup(probe["language_id"])
    comprehension = comprehension_state_lookup(probe["language_id"])
    knowledge = retrieve_knowledge("example: write a struct", probe["language_id"])
    feedback = record_inference_feedback({
        "task_id": "l062-probe", "language_id": probe["language_id"],
        "mode": "model_only", "finish_reason": "length", "completion_tokens": 4,
    })
    proposal = propose_registry_update(lookup, "COMPREHENSION_IN_PROGRESS")

    report = {
        "schema": "taksh.l062_dlcae_inference_interface/1",
        "generated_by": "evaluation/l062_dlcae_inference_interface.py",
        "interfaces": ["identify_language", "registry_lookup", "comprehension_state_lookup",
                       "retrieve_knowledge", "build_prompt", "record_inference_feedback",
                       "propose_registry_update"],
        "probe": {"identify": probe, "registry": lookup, "comprehension": comprehension,
                  "knowledge": knowledge, "feedback": feedback, "proposal": proposal},
        "invariants": [
            "inference output never changes registry status",
            "registration never implies capability",
            "proposals remain PENDING_OWNER_REVIEW until the owner writes the registry",
        ],
        "verdict": "DLCAE-INFERENCE INTERFACES QUALIFIED (interfaces only; no capability upgrades)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"registry probe: {probe['language_id']} registered={lookup['registered']} status={lookup.get('status')}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
