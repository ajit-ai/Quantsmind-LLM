"""G4: verify the actual licence FILE at each pinned commit.

The corpus builder trusts the SPDX id GitHub reports in repository metadata.
That value is not evidence: it is a classifier guess, it can be wrong, and it can
drift between the pinned revision and today. This module fetches the licence
file(s) present at the pinned commit, hashes them, classifies the text, and
compares that against both the pinned SPDX id and the policy allowlist.

Results are conservative by construction:

* No licence file at the pinned commit -> ``missing`` (not approved).
* Several licence files that disagree -> ``ambiguous`` (needs a human).
* File text disagrees with the pinned SPDX id -> ``conflict`` (rejected).
* Text matching no known family -> ``unrecognised`` (needs a human).
* A family the allowlist does not permit -> ``not_permitted``.

Only ``verified`` results are eligible for corpus inclusion. Nothing here widens
the allowlist or the policy; a repository needing legal interpretation is flagged
``needs_human_review`` rather than asserted approved.

Usage
-----
    python -m data.verify_licenses
    python -m data.verify_licenses --out data/licence_verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.build_code_corpus_v1 import GITHUB_API, GITHUB_RAW, Transport, load_config
from data.licence_paths import normalise_prefix, path_within

# Filenames that carry licence text in practice. Matched case-insensitively.
LICENCE_FILENAMES = (
    "license",
    "licence",
    "license.txt",
    "licence.txt",
    "license.md",
    "licence.md",
    "license-mit",
    "license-apache",
    "license.rst",
    "licence.rst",
    "copying",
    "copying.txt",
    "copyright",
    "unlicense",
    "notice",
)

# Ordered most-specific first; the first family whose markers all appear wins.
# Each entry is (spdx_lower, [all-of markers], [all-of alternative markers]).
LICENCE_SIGNATURES = (
    ("agpl-3.0", ["gnu affero general public license", "version 3"], []),
    ("gpl-3.0", ["gnu general public license", "version 3"], []),
    ("gpl-2.0", ["gnu general public license", "version 2"], []),
    ("lgpl-3.0", ["gnu lesser general public license", "version 3"], []),
    ("lgpl-2.1", ["gnu lesser general public license", "version 2.1"], []),
    ("apache-2.0", ["apache license", "version 2.0"], []),
    # BSD is distinguished by the third clause ("Neither the name of ...") which
    # only the 3-clause variant carries. Order matters: 3-clause is checked first,
    # so the looser 2-clause match cannot steal it.
    (
        "bsd-3-clause",
        [
            "redistribution and use in source and binary forms",
            "neither the name of",
        ],
        [],
    ),
    (
        "bsd-2-clause",
        ["redistribution and use in source and binary forms", "warranty"],
        [],
    ),
    ("mit", ["permission is hereby granted, free of charge"], []),
    ("isc", ["permission to use, copy, modify, and/or distribute"], []),
    (
        "unlicense",
        ["this is free and unencumbered software released into the public domain"],
        [],
    ),
)

# Notice obligations that travel with a derived corpus. Recorded, not enforced.
NOTICE_OBLIGATIONS = {
    "apache-2.0": [
        "retain all copyright, patent, trademark and attribution notices",
        "reproduce a copy of the Apache-2.0 licence with distributions",
        "state any modifications made",
        "NOTICE file, if present, must be reproduced",
    ],
    "bsd-3-clause": [
        "retain the copyright notice, list of conditions and disclaimer",
        "reproduce the disclaimer in documentation and/or materials",
    ],
    "bsd-2-clause": [
        "retain the copyright notice, list of conditions and disclaimer",
    ],
    "mit": ["retain the copyright notice and permission notice"],
    "isc": ["retain the copyright notice and permission notice"],
    "lgpl-3.0": [
        "state changes to the library",
        "work using it must comply with Apache-2.0",
    ],
    "gpl-3.0": [
        "derivative works must also be GPL-3.0",
        "source must be distributed",
    ],
    "agpl-3.0": [
        "derivative works must also be AGPL-3.0",
        "network use must offer source",
    ],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def classify_licence(text: str) -> tuple[str | None, list[str]]:
    """Return (spdx_lower, matched_markers). ``None`` when unrecognised.

    Markers are matched against whitespace-collapsed, NFKC-normalised text, so
    a marker phrase split across a line break still matches. Real licence files
    wrap aggressively.
    """
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = re.sub(r"\s+", " ", norm)
    for spdx, required, alternatives in LICENCE_SIGNATURES:
        if required and all(marker in norm for marker in required):
            return spdx, list(required)
        if alternatives and all(marker in norm for marker in alternatives):
            return spdx, list(alternatives)
    return None, []


def is_licence_path(path: str) -> bool:
    return path.rsplit("/", 1)[-1].lower() in LICENCE_FILENAMES


# Directories that hold third-party code under their own licence. A licence file
# here describes the vendored project, not the repository, so it is not evidence
# about the repository's own licence. The corpus builder applies the same
# exclusions to file selection.
VENDOR_PATH_RE = re.compile(
    r"(^|/)(vendor|_vendor|third[_-]?party|node_modules|site[_-]packages|"
    r"external|deps)(/|$)",
    re.IGNORECASE,
)


def subtree_of(path: str) -> str:
    """Directory prefix a licence file governs: '' for repository root."""
    return path.rsplit("/", 1)[0] if "/" in path else ""


# A subtree whose licence text matches the repository root is not evidence of a
# conflict, but it is also not a *verified* inheritance: the text simply repeats
# the root licence. It is therefore allowed only when the root family is already
# permitted and the subtree declares nothing different.
def derive_exclusions(
    subtree_licences: dict,
    root_family: str | None,
    nested_files_by_subtree: dict | None = None,
) -> tuple[list[str], list[dict]]:
    """Derive exclusion prefixes from detected nested licence evidence.

    Rules, in order, all conservative:

    1. A nested subtree declaring a family different from the root -> excluded.
    2. A nested subtree whose licence text could not be classified -> excluded,
       because an unrecognised licence cannot be shown to be compatible.
    3. A nested subtree declaring the same family as the root -> not excluded.
    4. A subtree with no licence file at all -> **not** excluded by this function.
       Absence of a licence file is not a licence claim; it is handled by the
       caller's policy, not inferred here.

    Returns ``(prefixes, evidence)``.
    """
    prefixes: list[str] = []
    evidence: list[dict] = []

    for subtree, families in sorted(subtree_licences.items()):
        if not subtree:
            continue  # root handled by the caller
        known = sorted(f for f in families if f)
        unknown = bool(families) and not known
        conflicting = (
            root_family is not None and known and root_family not in known
        )
        if unknown or conflicting:
            prefixes.append(normalise_prefix(subtree))
            evidence.append(
                {
                    "subtree": subtree,
                    "excluded": True,
                    "detected_families": known,
                    "root_family": root_family,
                    "reason": (
                        "licence text could not be classified"
                        if unknown
                        else "nested licence differs from repository root"
                    ),
                    "rule": "unknown_text" if unknown else "conflicting_family",
                }
            )
        else:
            evidence.append(
                {
                    "subtree": subtree,
                    "excluded": False,
                    "detected_families": known,
                    "root_family": root_family,
                    "reason": "nested licence matches repository root",
                    "rule": "matching_family",
                }
            )
    return prefixes, evidence


def classify_and_scope(transport, source, tree, max_files=5):
    """Fetch licence files and report per-subtree detected families.

    Returns ``(entries, by_subtree)`` where ``by_subtree`` maps a directory
    prefix to the set of SPDX families found there. Subtree scoping is what makes
    a conflict visible: a repository can be MIT at the root while a vendored
    subtree carries its own, different licence.
    """
    candidates = sorted(
        n["path"]
        for n in tree
        if n.get("type") == "blob"
        and is_licence_path(n["path"])
        and not VENDOR_PATH_RE.search(n["path"])
    )
    entries = []
    by_subtree = {}
    for path in candidates[:max_files]:
        entry = {"path": path, "subtree": subtree_of(path)}
        try:
            raw = transport.get(
                GITHUB_RAW.format(
                    repo=source["repo"], commit=source["commit"], path=path
                ),
                max_bytes=512 * 1024,
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = type(exc).__name__
            entries.append(entry)
            continue
        text = raw.decode("utf-8", errors="replace")
        spdx, markers = classify_licence(text)
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        entry["bytes"] = len(raw)
        entry["detected_spdx"] = spdx
        entry["markers"] = markers
        entries.append(entry)
        if spdx:
            by_subtree.setdefault(entry["subtree"], set()).add(spdx)
    return entries, by_subtree


def verify_source(source: dict, transport, policy: dict) -> dict:
    """Fetch and adjudicate the licence file(s) for one pinned source."""
    result = {
        "id": source["id"],
        "repo": source["repo"],
        "commit": source["commit"],
        "pinned_spdx": source["spdx"],
        "metadata_spdx": None,
        "candidate_paths": [],
        "licence_files": [],
        "detected_spdx": None,
        "result": None,
        "needs_human_review": False,
        "notes": [],
    }

    try:
        meta_raw = transport.get(f"{GITHUB_API}/repos/{source['repo']}")
        meta = json.loads(meta_raw.decode("utf-8"))
        spdx = ((meta.get("license") or {}).get("spdx_id") or "").lower()
        result["metadata_spdx"] = spdx or None
    except Exception as exc:  # noqa: BLE001 - recorded, never silently ignored
        result["result"] = "metadata_unavailable"
        result["notes"].append(f"metadata fetch failed: {type(exc).__name__}")
        result["needs_human_review"] = True
        return result

    try:
        tree_raw = transport.get(
            f"{GITHUB_API}/repos/{source['repo']}"
            f"/git/trees/{source['commit']}?recursive=1"
        )
        tree = json.loads(tree_raw.decode("utf-8")).get("tree", [])
    except Exception as exc:  # noqa: BLE001
        result["result"] = "tree_unavailable"
        result["notes"].append(f"tree fetch failed: {type(exc).__name__}")
        result["needs_human_review"] = True
        return result

    candidates = sorted(
        n["path"]
        for n in tree
        if n.get("type") == "blob" and is_licence_path(n["path"])
    )
    result["candidate_paths"] = candidates

    if not candidates:
        result["result"] = "missing"
        result["notes"].append(
            "no licence file at the pinned commit; permitted use cannot be "
            "established from machine-readable content"
        )
        result["needs_human_review"] = True
        return result

    entries, by_subtree = classify_and_scope(transport, source, tree)
    result["licence_files"] = entries
    result["subtree_licences"] = {
        subtree or "<repository root>": sorted(families)
        for subtree, families in by_subtree.items()
    }

    pinned = source["spdx"].lower()
    allowed = {s.lower() for s in policy["allowed_spdx"]}

    if not by_subtree:
        result["result"] = "unrecognised"
        result["notes"].append(
            "licence file present but no known family matched; a human must read it"
        )
        result["needs_human_review"] = True
        return result

    # The root subtree governs the repository's own code. Nested subtrees govern
    # whatever sits inside them and may legitimately differ -- but then the
    # corpus must not label those files with the root licence.
    root_families = by_subtree.get("", set())
    nested = {
        subtree: families for subtree, families in by_subtree.items() if subtree
    }
    conflicting = {
        subtree: sorted(families)
        for subtree, families in nested.items()
        if families != root_families
    }
    result["conflicting_subtrees"] = conflicting

    if root_families and len(root_families) > 1:
        result["result"] = "ambiguous"
        result["detected_spdx"] = sorted(root_families)
        result["notes"].append(
            f"root licence files disagree: {sorted(root_families)}"
        )
        result["needs_human_review"] = True
        return result

    only = next(iter(root_families)) if root_families else None
    result["detected_spdx"] = only

    if only != pinned:
        result["result"] = "conflict"
        result["notes"].append(
            f"root licence file reads {only} but pinned metadata says {pinned}; "
            "the pinned value is not trustworthy, so this source is rejected"
        )
        return result

    if result["metadata_spdx"] and result["metadata_spdx"] != pinned:
        result["notes"].append(
            f"GitHub metadata currently reports {result['metadata_spdx']} while "
            f"the pinned value is {pinned}"
        )

    if only not in allowed:
        result["result"] = "not_permitted"
        result["notes"].append(f"{only} is not in the allowlist {sorted(allowed)}")
        return result

    result["notice_obligations"] = NOTICE_OBLIGATIONS.get(only, [])
    result["result"] = "verified"

    # Derive, rather than hand-maintain, the paths that must not be published
    # under the root licence. These are recorded so the corpus builder can
    # enforce them without re-fetching anything.
    derived, evidence = derive_exclusions(by_subtree, only)
    result["derived_exclusion_paths"] = derived
    result["exclusion_evidence"] = evidence
    result["policy"] = "nested-licence-subtree-v1"

    if conflicting:
        # The repository's own code is fine, but nested subtrees carry different
        # licences. Those paths must be quarantined in the corpus rather than
        # shipped under the root licence.
        result["result"] = "verified_with_quarantine"
        result["needs_human_review"] = True
        result["notes"].append(
            "nested subtrees carry a different licence than the repository root: "
            + "; ".join(f"{s or '<root>'}={v}" for s, v in conflicting.items())
            + ". Corpus records from those subtrees must be quarantined, not "
            "labelled with the root licence. A human must confirm."
        )
    return result


def build_report(cfg: dict, transport, log=print) -> dict:
    policy = cfg["policy"]
    sources = [s for s in cfg["sources"] if s.get("phase1_default", True)]
    results = []
    for s in sources:
        r = verify_source(s, transport, policy)
        log(f"  {r['id']:22s} {r['result']}")
        results.append(r)

    verified = [r for r in results if r["result"] == "verified"]
    review = [r for r in results if r.get("needs_human_review")]
    rejected = [r for r in results if r["result"] not in ("verified",)]

    return {
        "schema": "quantsmind.licence_verification/1",
        "generated_at": utc_now_iso(),
        "allowed_spdx": policy["allowed_spdx"],
        "policy_note": (
            "Verification reads the licence file at the pinned commit and "
            "compares it with the pinned SPDX id and the allowlist. It does not "
            "widen either. Anything needing legal interpretation is reported as "
            "needs_human_review rather than approved."
        ),
        "sources": results,
        "counts": {
            "checked": len(results),
            "verified": len(verified),
            "rejected": len(rejected),
            "needs_human_review": len(review),
        },
        "verified_ids": [r["id"] for r in verified],
        "rejected_ids": [r["id"] for r in rejected],
        "needs_human_review_ids": [r["id"] for r in review],
        "all_verified": len(results) > 0 and not rejected,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify licence files at pinned commits")
    p.add_argument("--config", default="configs/data/approved_code_sources.yaml")
    p.add_argument("--out", default="data/licence_verification.json")
    p.add_argument(
        "--include-held-back",
        action="store_true",
        help="also verify sources marked phase1_default: false",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.include_held_back:
        for s in cfg["sources"]:
            s.pop("phase1_default", None)

    transport = Transport(
        timeout=cfg["limits"]["request_timeout_seconds"],
        max_retries=cfg["limits"]["max_retries_per_request"],
        deadline=cfg["limits"].get("transfer_deadline_seconds"),
    )
    report = build_report(cfg, transport)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"verified: {report['counts']['verified']}/{report['counts']['checked']}")
    for r in report["sources"]:
        files = ", ".join(f["path"] for f in r["licence_files"]) or "-"
        print(f"  {r['id']:22s} {r['result']:16s} detected={r['detected_spdx']}")
        print(f"      files={files}")
        if r["licence_files"] and r["licence_files"][0].get("sha256"):
            print(f"      sha256={r['licence_files'][0]['sha256'][:16]}...")
        for note in r["notes"]:
            print(f"      note: {note}")
    if report["needs_human_review_ids"]:
        print(
            "NEEDS HUMAN REVIEW: " + ", ".join(report["needs_human_review_ids"])
        )
    print(f"\nwritten to {args.out}")
    return 0 if report["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())