"""Decision register for the data, capability and infrastructure questions.

The identifiers D1-D6 keep the meanings the master roadmap gave them; they
are not redefined here. The v1.6.0 decisions use their own identifiers so the
roadmap's numbering stays intact.

Every entry states what is known, what is missing, who must decide, and what
happens if it stays unresolved. Nothing in this file approves anything.
"""

from __future__ import annotations

#: Roadmap decisions, with their original meanings preserved.
ROADMAP_DECISIONS = {
    "D1": {
        "topic": "source licensing and permitted use",
        "status": "unresolved",
        "evidence_available": "the v1.2.0 licence gate and the v1.6.0 source-manifest schema exist",
        "evidence_missing": "an owner's decision per source, with evidence references",
        "owner": "repository owner (legal decision)",
        "consequence_if_unresolved": "no production source is eligible; production training stays unauthorized",
    },
    "D2": {
        "topic": "provenance and evidence expectations",
        "status": "unresolved",
        "evidence_available": "provenance evidence fields in the source manifest",
        "evidence_missing": "an accepted chain-of-custody standard",
        "owner": "repository owner",
        "consequence_if_unresolved": "provenance cannot be established, so sources stay ineligible",
    },
    "D3": {
        "topic": "approved-source allowlist",
        "status": "empty",
        "evidence_available": "the registry and authorization gate",
        "evidence_missing": "the first approved source and its review record",
        "owner": "repository owner",
        "consequence_if_unresolved": "the corpus stays empty and the pipeline is unexercised on real material",
    },
    "D4": {
        "topic": "contamination policy and benchmark availability",
        "status": "partially addressed",
        "evidence_available": "a benchmark registry, contamination screening, and authored v1.4.0/v1.5.0 suites",
        "evidence_missing": "local copies of the published suites (HumanEval, MBPP)",
        "owner": "repository owner",
        "consequence_if_unresolved": "contamination coverage stays empty and the verdict stays 'unavailable'",
    },
    "D5": {
        "topic": "attribution, notice and redistribution obligations",
        "status": "unresolved",
        "evidence_available": "attribution and obligation fields in the source manifest",
        "evidence_missing": "per-source obligation decisions",
        "owner": "repository owner",
        "consequence_if_unresolved": "redistribution of any derived corpus stays blocked",
    },
    "D6": {
        "topic": "tokenizer vocabulary and model geometry",
        "status": "open",
        "evidence_available": "measured tokenizer cost (2.85-2.97 characters per token, "
        "32-43% identifier fragmentation) and the compatibility inspector",
        "evidence_missing": "a 32,768-id tokenizer artifact, an approved corpus, and a "
        "comparable coding measurement",
        "owner": "repository owner (architecture decision)",
        "consequence_if_unresolved": "the S3 rung stays unservable with the frozen tokenizer; "
        "the S2 rung remains the only usable geometry",
    },
}

#: v1.6.0 decisions, with their own identifiers.
V16_DECISIONS = {
    "DR-1": {
        "topic": "data licensing and rights",
        "status": "blocked on D1/D2/D5",
        "owner": "repository owner",
        "consequence_if_unresolved": "no production corpus",
    },
    "DR-2": {
        "topic": "source selection",
        "status": "no candidate approved",
        "owner": "repository owner",
        "consequence_if_unresolved": "the ingestion contract refuses everything",
    },
    "DR-3": {
        "topic": "corpus quality thresholds",
        "status": "defaults documented, not validated on real material",
        "owner": "repository owner after the first authorised corpus exists",
        "consequence_if_unresolved": "thresholds remain engineering defaults",
    },
    "DR-4": {
        "topic": "contamination policy",
        "status": "screening implemented; published suites unavailable",
        "owner": "repository owner",
        "consequence_if_unresolved": "contamination coverage stays empty",
    },
    "DR-5": {
        "topic": "training readiness",
        "status": "not authorized",
        "owner": "repository owner",
        "consequence_if_unresolved": "no real-data training run is possible",
    },
    "DR-6": {
        "topic": "tokenizer geometry",
        "status": "see D6",
        "owner": "repository owner",
        "consequence_if_unresolved": "S3 remains unusable",
    },
    "DR-7": {
        "topic": "model scaling",
        "status": "analysis only; no migration executed",
        "owner": "repository owner",
        "consequence_if_unresolved": "S2 stays the working rung",
    },
    "DR-8": {
        "topic": "compute availability",
        "status": "CPU-only local research",
        "owner": "repository owner",
        "consequence_if_unresolved": "training budgets stay small",
    },
    "DR-9": {
        "topic": "evaluation isolation",
        "status": "blocked: no isolation mechanism on this host",
        "owner": "environment owner",
        "consequence_if_unresolved": "functional correctness stays unmeasured",
    },
    "DR-10": {
        "topic": "agent integration readiness",
        "status": "transport verified in v1.4.0; re-verification pending after the "
        "v1.5.0 base-surface change",
        "owner": "repository owner (joint qualification)",
        "consequence_if_unresolved": "the IQ-1 evidence is stale for prompt rendering",
    },
}


def register() -> dict:
    """The full register, with the roadmap decisions first."""
    return {"roadmap": ROADMAP_DECISIONS, "v16": V16_DECISIONS}


def summary() -> list:
    """One line per decision, for a report or a console."""
    rows = []
    for group, decisions in register().items():
        for identifier, entry in decisions.items():
            rows.append(
                {
                    "id": identifier,
                    "group": group,
                    "topic": entry["topic"],
                    "status": entry["status"],
                    "owner": entry["owner"],
                }
            )
    return rows


def unresolved() -> list:
    """Every decision that is not settled."""
    return [row for row in summary() if row["status"] not in ("resolved", "decided")]
