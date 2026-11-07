"""M1 code-corpus builder: reproducible, licence-gated, bounded, resumable.

Replaces the search-based discovery in ``build_code_corpus.py``. That version asked
the GitHub search API for ``language:python stars:>100`` and ingested whatever came
back, so the corpus was not reproducible (ranking drifts), carried no provenance
guarantee, and could silently absorb a differently-licensed repository.

Design
------
* **Approved sources only.** Repositories come exclusively from
  ``configs/data/approved_code_sources.yaml``, each pinned to a full commit SHA.
  There is no discovery path.
* **Licence gate.** The live licence is re-read at run time and must match both the
  allowlist and the pinned value. A mismatch aborts that source and is recorded.
* **Bounded.** ``max_repos``, ``max_files``, ``max_total_bytes`` and per-file size
  limits. Hitting one sets ``truncated: true``; the run never reports success while
  truncated.
* **Resumable.** Accepted chunks are appended to the JSONL shard as they are
  validated, so an interrupted run resumes from the last complete record. The
  manifest is written once at the end, atomically, and carries ``complete``.
* **Honest failure.** A source that cannot be fetched is recorded with its error.
  An empty corpus exits non-zero.
* **Deterministic splits.** A chunk's split comes from the SHA-1 of its content, not
  its position, so assignment is stable across runs and collection order.
* **Leakage is verified, not assumed.** Exact-duplicate overlap across splits is
  structurally impossible because duplicates are removed first, but it is still
  checked; near-duplicate pairs straddling splits are counted and reported.

Usage
-----
    python -m data.build_code_corpus_v1 --out data/code_corpus_v1.jsonl
    python -m data.build_code_corpus_v1 --out data/code_corpus_v1.jsonl --resume
    python -m data.build_code_corpus_v1 --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.filters import ExactDedup, MinHashDedup, scrub_pii_report
from data.licence_paths import normalise_prefix, path_within

DEFAULT_CONFIG = "configs/data/approved_code_sources.yaml"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
GITHUB_API = "https://api.github.com"

# Paths that carry little training signal and would inflate the corpus with
# boilerplate, fixtures, tutorials and duplicated variants of the same module.
# ``_vendor``/``vendor`` and ``node_modules`` matter: vendored copies of other
# projects are third-party code under a different licence than the pinned repo,
# and recording this repo's licence for them would be wrong.
SKIP_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|doc|docs?_[^/]*|examples?|example|\.github|site|"
    r"news|benchmarks?|bench|scripts?|script|tutorials?|vendor|_vendor|"
    r"third[_-]?party|node_modules|site[_-]packages)(/|$)"
    r"|(^|/)(setup\.py|conftest\.py|noxfile\.py|tox\.ini)$",
    re.IGNORECASE,
)
GENERATED_RE = re.compile(r"(?i)(@generated|do not edit|automatically generated|auto-generated)")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


LICENCE_POLICY_VERSION = "licence-gate-v1"
# Results that may contribute records, and under what conditions.
LICENCE_ALLOWED_RESULTS = {"verified", "verified_with_quarantine"}


class LicenceGateError(RuntimeError):
    """A source cannot enter a corpus without acceptable licence evidence."""


def load_licence_verification(path: str, *, commit_by_source: dict) -> dict:
    """Load verification evidence and index it by source id.

    Fails loudly when the artifact is absent or unreadable. Missing evidence is
    never treated as permission.
    """
    p = Path(path)
    if not p.exists():
        raise LicenceGateError(
            f"licence verification artifact not found: {path}\n"
            "Corpus construction requires licence evidence. Run:\n"
            "  python -m data.verify_licenses\n"
            "A source with no verification is excluded, not assumed permitted."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise LicenceGateError(
            f"licence verification artifact is unreadable: {path}: {exc}"
        ) from exc

    by_source = {}
    for entry in data.get("sources", []):
        by_source[entry.get("id")] = entry

    # Evidence must correspond to the pinned revision actually being built.
    mismatched = [
        sid
        for sid, entry in by_source.items()
        if sid in commit_by_source and entry.get("commit") != commit_by_source[sid]
    ]
    if mismatched:
        raise LicenceGateError(
            "licence verification was produced for a different revision of: "
            + ", ".join(sorted(mismatched))
            + "\nRe-run `python -m data.verify_licenses` at the pinned commits."
        )
    return {"by_source": by_source, "generated_at": data.get("generated_at")}


def _subtree_is_excluded(subtree: str, prefixes) -> bool:
    """True when some exclusion prefix covers this subtree.

    An exclusion of an ancestor counts, so excluding ``src/`` also covers
    ``src/blib2to3``.
    """
    probe = normalise_prefix(subtree) + "."
    return any(path_within(probe, [p]) for p in (prefixes or []))


def licence_decision(source: dict, verification: dict | None, policy: dict) -> dict:
    """Decide whether one source may contribute records, and why.

    Returns a decision dict; ``allowed`` is False for anything that is not
    explicitly verified. Absence of evidence is a rejection, not a pass.
    """
    allowed_results = set(policy.get("allow_results", LICENCE_ALLOWED_RESULTS))
    decision = {
        "id": source["id"],
        "repo": source["repo"],
        "commit": source["commit"],
        "pinned_spdx": source["spdx"],
        "verification_result": None,
        "allowed": False,
        "reason": None,
        "exclusion_paths": list(source.get("quarantine_paths", [])),
        "exclusion_sources": ["config"],
    }

    if verification is None:
        decision["reason"] = "no licence verification evidence available"
        return decision

    decision["verification_result"] = verification.get("result")
    decision["verification_generated_at"] = verification.get("generated_at")

    result = verification.get("result")
    if result not in allowed_results:
        decision["reason"] = f"licence verification result is {result!r}"
        return decision

    # Derived exclusions come from the licence files actually found in the tree.
    derived = verification.get("derived_exclusion_paths") or []
    if derived:
        decision["exclusion_paths"] = sorted(
            set(decision["exclusion_paths"]) | {normalise_prefix(p) for p in derived}
        )
        decision["exclusion_sources"] = sorted(
            set(decision["exclusion_sources"]) | {"derived_from_licence_files"}
        )

    if result == "verified_with_quarantine":
        # Permitted only when every conflicting subtree is actually excluded,
        # by config decision or by evidence derived from the licence files.
        conflicting = verification.get("conflicting_subtrees") or {}
        uncovered = [
            sub
            for sub in conflicting
            if sub and not _subtree_is_excluded(sub, decision["exclusion_paths"])
        ]
        if uncovered:
            decision["reason"] = (
                "conflicting subtrees not excluded: " + ", ".join(sorted(uncovered))
            )
            return decision

    decision["allowed"] = True
    decision["reason"] = f"licence verification result is {result!r}"
    return decision


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("policy", "limits", "dedup", "splits", "sources"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing required top-level key '{key}'")
    return cfg


# --------------------------------------------------------------------------- #
# HTTP transport. Injectable so tests never touch the network.
# --------------------------------------------------------------------------- #
class ResponseTooLarge(Exception):
    """Raised when a response would exceed the caller's byte cap.

    Distinct from transport errors so callers can record ``too_large`` as a
    content rejection rather than a network failure.
    """


class Transport:
    def __init__(self, timeout=30, max_retries=3, pause=0.0, token=None, deadline=None):
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause = pause
        self.token = token or os.environ.get("GITHUB_TOKEN")
        # urlopen's timeout bounds each socket operation, not the whole transfer,
        # so a slow-drip response can stall a run forever. This is the hard cap
        # on total wall time for one request.
        self.deadline = deadline if deadline is not None else max(timeout * 2, 60)
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "too_large": 0}

    def _headers(self):
        h = {"Accept": "application/vnd.github+json", "User-Agent": "quantsmind-m1"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, url: str, max_bytes: int | None = None) -> bytes:
        import urllib.error
        import urllib.request

        last = None
        for attempt in range(1, self.max_retries + 1):
            self.stats["requests"] += 1
            try:
                req = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = self._read_capped(resp, max_bytes)
                if self.pause:
                    time.sleep(self.pause)
                return data
            except ResponseTooLarge:
                # Not transient: retrying cannot make a file smaller.
                self.stats["too_large"] += 1
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last = exc
                # 404 is a final answer, not a transient failure.
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    raise
                if attempt < self.max_retries:
                    self.stats["retries"] += 1
                    time.sleep(min(2 ** attempt, 8))
        self.stats["failures"] += 1
        raise last

    def _read_capped(self, resp, max_bytes: int | None) -> bytes:
        """Read a response under a byte cap and a total-transfer wall deadline.

        Content-Length is checked first so an oversized body is refused before it
        is read at all. GitHub's raw host normally sends no length, so the body is
        also read in chunks and abandoned the moment it exceeds the cap.
        """
        if max_bytes is not None:
            declared = resp.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > max_bytes:
                        raise ResponseTooLarge(
                            f"content-length {declared} exceeds cap {max_bytes}"
                        )
                except ValueError:
                    pass  # malformed header: fall back to the capped read
        deadline = time.monotonic() + self.deadline
        chunks = []
        total = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"transfer exceeded {self.deadline}s wall deadline"
                )
            chunk = resp.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ResponseTooLarge(f"body exceeds cap {max_bytes}")
            chunks.append(chunk)
        return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Content validation
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    """NFKC-normalise and strip trailing whitespace so dedup sees real equality."""
    text = unicodedata.normalize("NFKC", text)
    return "\n".join(line.rstrip() for line in text.split("\n"))


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() or c.isspace() for c in text) / len(text)


def reject_reason(text: str, path: str, policy: dict):
    """Return a rejection reason, or None if the chunk is acceptable."""
    raw_len = len(text.encode("utf-8", errors="ignore"))
    if raw_len < policy["min_file_bytes"]:
        return "too_small"
    if raw_len > policy["max_file_bytes"]:
        return "too_large"
    if not text.strip():
        return "empty"
    if SKIP_PATH_RE.search(path):
        return "path_excluded"
    if GENERATED_RE.search(text[:400]):
        return "generated_marker"
    if alpha_ratio(text) < policy["min_alpha_ratio"]:
        return "low_alpha_ratio"
    if policy.get("require_python_parses", True) and path.endswith(".py"):
        try:
            compile(text, path, "exec")
        except (SyntaxError, ValueError):
            return "syntax_error"
    return None


def content_split(text: str, splits: dict) -> str:
    """Deterministic split assignment derived from content, not position."""
    return _bucket_split(hashlib.sha1(text.encode("utf-8")).hexdigest(), splits)


def group_split(group_key: str, splits: dict) -> str:
    """Split assignment derived from a group identity (a repository, not a file).

    Every record from the same repository therefore lands in the same split, so
    near-identical files from one project cannot straddle train and test.
    """
    return _bucket_split(
        hashlib.sha1(("group:" + group_key).encode("utf-8")).hexdigest(), splits
    )


def assign_group_splits(groups, splits: dict) -> dict:
    """Assign whole groups to splits deterministically and as evenly as possible.

    Independent hashing does not work here. With only nine repositories and a
    60/20/20 target, independent assignment produced {train: 7, val: 0, test: 2} --
    a 1% holdout receives 0.09 expected groups, so an empty split is the normal
    outcome, not an anomaly.

    Instead the groups are ordered deterministically (by content hash of the
    group name, so input order cannot change the result) and then filled by
    largest remaining deficit against the configured fractions. This is the
    standard largest-remainder method: it hits the target fractions as closely as
    integer arithmetic allows, and it is reproducible.

    Returns ``{group_key: split}``.
    """
    groups = sorted(set(groups))
    if not groups:
        return {}
    order = sorted(groups, key=lambda g: hashlib.sha1(("order:" + g).encode()).hexdigest())
    names = ("train", "val", "test")
    fracs = [splits.get(n, 0.0) for n in names]
    total = sum(fracs) or 1.0
    targets = {n: f / total * len(groups) for n, f in zip(names, fracs)}
    assigned = {n: 0 for n in names}
    out = {}
    for g in order:
        # Largest deficit wins; ties broken by fixed train > val > test order so
        # the outcome never depends on dict or set iteration order.
        best = max(names, key=lambda n: (targets[n] - assigned[n], -names.index(n)))
        out[g] = best
        assigned[best] += 1
    return out


def _bucket_split(digest: str, splits: dict) -> str:
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    train = splits["train"]
    val = splits["val"]
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class CorpusBuilder:
    def __init__(
        self,
        cfg,
        transport,
        out_path,
        manifest_path,
        resume=False,
        dry_run=False,
        max_files_override=None,
        max_bytes_override=None,
        max_repos_override=None,
        log=print,
        licence_index=None,
    ):
        self.cfg = cfg
        self.transport = transport
        self.out_path = Path(out_path)
        self.manifest_path = Path(manifest_path)
        self.resume = resume
        self.dry_run = dry_run
        self.log = log

        limits = cfg["limits"]
        self.max_files = max_files_override or limits["max_files"]
        self.max_bytes = max_bytes_override or limits["max_total_bytes"]
        self.max_repos = max_repos_override or limits["max_repos"]

        self.exact = ExactDedup()
        self.fuzzy = MinHashDedup(
            num_perms=cfg["dedup"]["minhash_perms"],
            bands=cfg["dedup"]["minhash_bands"],
            threshold=cfg["dedup"]["minhash_threshold"],
        )
        self.counters = {"seen": 0, "kept": 0, "rejected": {}, "truncated_reasons": {}}
        self._integrity_failures = []
        self._group_split_map = None
        self._licence_index = licence_index
        self.bytes = 0
        self.truncated = False
        self.sources_result = []
        self._fh = None
        self._resumed_rows = 0

    def _reject(self, reason):
        self.counters["rejected"][reason] = self.counters["rejected"].get(reason, 0) + 1

    def _mark_truncated(self, reason):
        self.truncated = True
        self.counters["truncated_reasons"][reason] = (
            self.counters["truncated_reasons"].get(reason, 0) + 1
        )

    def _verify_license(self, source):
        """Re-read the live licence; return an error string to abort the source."""
        if not self.cfg["policy"].get("verify_license_at_runtime", True):
            return None
        raw = self.transport.get(f"{GITHUB_API}/repos/{source['repo']}")
        data = json.loads(raw.decode("utf-8"))
        live = ((data.get("license") or {}).get("spdx_id") or "").lower()
        allowed = {s.lower() for s in self.cfg["policy"]["allowed_spdx"]}
        pinned = source["spdx"].lower()
        if live != pinned:
            return f"licence drift: pinned {pinned}, live {live or 'none'}"
        if live not in allowed:
            return f"licence not in allowlist: {live or 'none'}"
        return None

    def _open_output(self):
        if self.dry_run:
            return
        mode = "a"
        if self.resume and self.out_path.exists():
            self._load_resume_state()
        else:
            # A fresh run restarts the shard; only --resume appends. Without
            # this a rerun would silently duplicate the previous corpus.
            mode = "w"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open(mode, encoding="utf-8")

    def _load_resume_state(self):
        """Rebuild dedup state and counters from complete records already on disk.

        A partially written final line (interrupted run) stops the scan; the shard
        is then truncated to the last complete record so appending stays valid JSONL.
        """
        raw = self.out_path.read_bytes()
        good_bytes = 0
        rows = 0
        text_bytes = 0
        for line in raw.split(b"\n"):
            if not line:
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.log("resume: discarding a truncated trailing record")
                break
            self.exact.add(row["text"])
            if self.cfg["dedup"].get("minhash"):
                self.fuzzy.add(row["text"])
            text_bytes += len(row["text"].encode("utf-8"))
            good_bytes += len(line) + 1
            rows += 1
        self._resumed_rows = rows
        self.counters["kept"] = rows
        self.bytes = text_bytes
        with self.out_path.open("r+b") as f:
            f.truncate(good_bytes)
        self.log(f"resume: {rows} chunks already collected, {text_bytes} text bytes")

    def _write_row(self, row):
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    # ------------------------------------------------------------- ingestion
    def licence_evidence(self, source: dict):
        """Verification record for a source, or None when none was supplied.

        Returning None is the correct, safe outcome: ``licence_decision`` treats
        absent evidence as a rejection rather than as permission.
        """
        if self._licence_index is None:
            return None
        return self._licence_index.get(source["id"])

    def _split_for(self, source, text: str) -> str:
        """Assign a split using the configured mode.

        ``content_hash``  -- per chunk (default; fine when records are
                              independent).
        ``source_group``  -- per repository. Every file of one repository lands in
                              one split, so near-identical files from the same
                              project cannot leak across train/val/test.
        """
        if self.cfg["splits"].get("mode", "content_hash") == "source_group":
            if self._group_split_map is None:
                self._group_split_map = assign_group_splits(
                    [s["repo"] for s in self.eligible_sources()], self.cfg["splits"]
                )
            return self._group_split_map.get(
                source["repo"], content_split(text, self.cfg["splits"])
            )
        return content_split(text, self.cfg["splits"])

    def _ingest_file(self, source, path, raw):
        self.counters["seen"] += 1
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self._reject("not_utf8")
            return False
        text = normalise(text)
        reason = reject_reason(text, path, self.cfg["policy"])
        if reason:
            self._reject(reason)
            return False
        if self.cfg["dedup"].get("exact") and not self.exact.add(text):
            self._reject("exact_duplicate")
            return False
        if self.cfg["dedup"].get("minhash") and not self.fuzzy.add(text):
            self._reject("near_duplicate")
            return False

        scrub = scrub_pii_report(text, mode="code")
        if scrub.preserved_reason:
            # A safe distinction could not be made, so the file is kept as-is and
            # counted rather than silently rewritten.
            self.counters["pii_preserved"] = self.counters.get("pii_preserved", 0) + 1
        else:
            for kind, n in scrub.findings:
                self.counters.setdefault("pii_replacements", {})
                self.counters["pii_replacements"][kind] = (
                    self.counters["pii_replacements"].get(kind, 0) + n
                )
        clean = scrub.clean
        digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()
        self._write_row(
            {
                "text": clean,
                "content_sha1": digest,
                "language": source["language"],
                "source_id": source["id"],
                "repo": source["repo"],
                "commit": source["commit"],
                "spdx": source["spdx"],
                "attribution": source.get("attribution", ""),
                "file_path": path,
                "retrieved_at": source["retrieved"],
                "split": self._split_for(source, clean),
            }
        )
        self.counters["kept"] += 1
        self.bytes += len(clean.encode("utf-8"))
        return True

    def _ingest_source(self, source):
        result = {
            "id": source["id"],
            "repo": source["repo"],
            "commit": source["commit"],
            "spdx": source["spdx"],
            "status": "ok",
            "state": "complete",
            "files_seen": 0,
            "files_kept": 0,
            "files_skipped_path": 0,
            "files_skipped_oversize": 0,
            "files_quarantined_licence": 0,
            "quarantine_paths": list(source.get("quarantine_paths", [])),
            "error": None,
        }

        # B1: the licence gate is checked before a single byte of this source is
        # transferred. A source without acceptable verification evidence
        # contributes nothing at all.
        decision = licence_decision(
            source, self.licence_evidence(source), self.cfg.get("licence_gate", {})
        )
        result["licence_gate"] = decision
        result["quarantine_paths"] = list(decision["exclusion_paths"])
        if not decision["allowed"]:
            result["status"] = "rejected_licence"
            result["state"] = "rejected_licence"
            result["error"] = decision["reason"]
            self.log(f"  BLOCKED {source['id']}: {decision['reason']}")
            return result

        try:
            lic_err = self._verify_license(source)
            if lic_err:
                result["status"] = "rejected_licence"
                result["state"] = "rejected_licence"
                result["error"] = lic_err
                self.log(f"  REJECT {source['id']}: {lic_err}")
                return result

            listing = self.transport.get(
                f"{GITHUB_API}/repos/{source['repo']}"
                f"/git/trees/{source['commit']}?recursive=1"
            )
            tree = json.loads(listing.decode("utf-8")).get("tree", [])
            exts = tuple(self.cfg["policy"]["extensions"])
            max_file = self.cfg["policy"]["max_file_bytes"]
            min_file = self.cfg["policy"]["min_file_bytes"]
            paths = []
            for n in tree:
                if n.get("type") != "blob" or not n["path"].endswith(exts):
                    continue
                # Path-based exclusions (tests, docs, vendored trees) are decided
                # from the listing so those files are never transferred at all.
                if SKIP_PATH_RE.search(n["path"]):
                    self._reject("path_excluded")
                    result["files_skipped_path"] += 1
                    continue
                # A subtree carrying a different licence than the repository
                # root, or explicitly quarantined by project decision (B2/B3).
                # Publishing it under the root licence would be a false licence
                # claim, so it is quarantined, not relabelled. Descendants of an
                # excluded subtree are excluded too.
                if path_within(n["path"], decision["exclusion_paths"]):
                    self._reject("quarantined_licence_conflict")
                    result["files_quarantined_licence"] += 1
                    continue
                # The tree listing carries each blob's size, so oversized files
                # are refused before any bytes are transferred. Without this a
                # multi-megabyte generated file can stall the whole run.
                size = n.get("size")
                if isinstance(size, int) and size > max_file:
                    self._reject("too_large")
                    result["files_skipped_oversize"] += 1
                    continue
                paths.append(n["path"])
            self.log(f"  {source['id']}: {len(paths)} candidate files")

            for path in paths:
                if self.counters["kept"] >= self.max_files:
                    self._mark_truncated("max_files")
                    break
                if self.bytes >= self.max_bytes:
                    self._mark_truncated("max_total_bytes")
                    break
                try:
                    raw = self.transport.get(
                        GITHUB_RAW.format(
                            repo=source["repo"], commit=source["commit"], path=path
                        ),
                        max_bytes=max_file,
                    )
                except ResponseTooLarge:
                    self._reject("too_large")
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad file is not fatal
                    self._reject("fetch_error")
                    result["error"] = f"{path}: {type(exc).__name__}"
                    continue
                result["files_seen"] += 1
                if self._ingest_file(source, path, raw):
                    result["files_kept"] += 1
            if self.truncated and result["status"] == "ok":
                result["state"] = "truncated"
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            result["status"] = "failed"
            result["state"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
                result["error"] += (
                    " (GitHub API rate limit for unauthenticated requests; "
                    "set GITHUB_TOKEN to raise it)"
                )
            self.log(f"  FAIL {source['id']}: {result['error']}")
        return result

    def eligible_sources(self):
        """Configured sources for this run, before any cap is applied."""
        return [s for s in self.cfg["sources"] if s.get("phase1_default", True)]

    def selected_sources(self):
        chosen = self.eligible_sources()
        return chosen[: self.max_repos] if self.max_repos else chosen

    def run(self):
        started = utc_now_iso()
        eligible = self.eligible_sources()
        sources = self.selected_sources()

        # A source cap that silently drops configured sources would make the
        # manifest claim coverage it does not have. Record it as truncation.
        if len(sources) < len(eligible):
            dropped = len(eligible) - len(sources)
            self._mark_truncated("max_repos")
            self.counters["sources_skipped_by_max_repos"] = dropped
            self.log(
                f"  max_repos={self.max_repos} drops {dropped} configured "
                "source(s); run is recorded as truncated"
            )

        if self.dry_run:
            return {
                "dry_run": True,
                "complete": len(sources) >= len(eligible),
                "would_process": [
                    {k: s[k] for k in ("id", "repo", "commit", "spdx")} for s in sources
                ],
                "eligible_source_count": len(eligible),
                "selected_source_count": len(sources),
                "would_truncate": len(sources) < len(eligible),
                "limits": {
                    "max_repos": self.max_repos,
                    "max_files": self.max_files,
                    "max_total_bytes": self.max_bytes,
                },
            }

        self._open_output()
        try:
            for source in sources:
                self.sources_result.append(self._ingest_source(source))
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

        manifest = self.build_manifest(started)
        self._write_manifest(manifest)
        return manifest

    # ---------------------------------------------------------------- output
    def config_fingerprint(self):
        """Stable digest of everything that changes what the corpus would contain.

        Timestamps and paths are excluded: re-running the same configuration
        against the same pinned revisions must reproduce this value.
        """
        material = {
            "corpus_version": self.cfg.get("corpus_version", "v1"),
            "policy": self.cfg["policy"],
            "limits": {
                "max_repos": self.max_repos,
                "max_files": self.max_files,
                "max_total_bytes": self.max_bytes,
            },
            "dedup": self.cfg["dedup"],
            "splits": self.cfg["splits"],
            "sources": [
                {k: s[k] for k in ("id", "repo", "commit", "spdx")}
                for s in self.selected_sources()
            ],
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def build_manifest(self, started):
        """Assemble the manifest by re-reading the shard, so it describes reality."""
        splits = {"train": 0, "val": 0, "test": 0}
        by_source = {}
        by_spdx = {}
        hash_to_split = {}
        exact_leaks = 0
        content_hashes = []
        repo_splits = {}
        group_counts = {"train": 0, "val": 0, "test": 0}

        if self.out_path.exists():
            with self.out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    splits[row["split"]] = splits.get(row["split"], 0) + 1
                    by_source[row["source_id"]] = by_source.get(row["source_id"], 0) + 1
                    by_spdx[row["spdx"]] = by_spdx.get(row["spdx"], 0) + 1
                    seen = repo_splits.setdefault(row["repo"], set())
                    seen.add(row["split"])
                    digest = row.get("content_sha1") or hashlib.sha1(
                        row["text"].encode("utf-8")
                    ).hexdigest()
                    content_hashes.append(f"sha1:{digest}")
                    prior = hash_to_split.get(digest)
                    if prior is None:
                        hash_to_split[digest] = row["split"]
                    elif prior != row["split"]:
                        exact_leaks += 1

        # A repository spanning two splits is only legitimate in content_hash
        # mode; in source_group mode it is a bug, so assert rather than report.
        overlapping_repos = sorted(r for r, s in repo_splits.items() if len(s) > 1)
        for repo, seen in repo_splits.items():
            group_counts[next(iter(seen))] = group_counts.get(next(iter(seen)), 0) + 1
        split_mode = self.cfg["splits"].get("mode", "content_hash")
        if split_mode == "source_group" and overlapping_repos:
            # Should be impossible: the split is a function of the repo. Recorded
            # as a hard integrity failure rather than silently accepted.
            self._integrity_failures.append(
                f"repository spans multiple splits in source_group mode: "
                f"{overlapping_repos}"
            )

        # Re-run the near-dup detector over the final corpus to report how many
        # near-duplicate pairs survive and how many straddle a split boundary.
        near_dup_pairs, near_dup_leaks = self._measure_near_duplicates()

        kept = sum(splits.values())
        bad_sources = [
            s for s in self.sources_result
            if s["status"] in ("failed", "rejected_licence")
        ]
        if bad_sources or kept == 0 or self._integrity_failures:
            status = "failed"
        elif self.truncated:
            status = "truncated"
        else:
            status = "complete"

        rejected = dict(self.counters["rejected"])
        deduplicated = rejected.get("exact_duplicate", 0) + rejected.get(
            "near_duplicate", 0
        )

        return {
            "schema": "quantsmind.code_corpus_manifest/1",
            "corpus": {
                "name": "code_corpus",
                "version": self.cfg.get("corpus_version", "v1"),
                "shard": self.out_path.name,
                "config_fingerprint": self.config_fingerprint(),
            },
            "started_at": started,
            "finished_at": utc_now_iso(),
            "retrieval": {
                "method": "github raw content at pinned commit SHA",
                "per_source": {
                    s["id"]: {"retrieved": s["retrieved"], "commit": s["commit"]}
                    for s in self.cfg["sources"]
                },
            },
            "config": {
                "policy": self.cfg["policy"],
                "limits": {
                    "max_repos": self.max_repos,
                    "max_files": self.max_files,
                    "max_total_bytes": self.max_bytes,
                },
                "dedup": self.cfg["dedup"],
                "splits": self.cfg["splits"],
            },
            "sources": self.sources_result,
            "sources_failed": len(bad_sources),
            "licensing": {
                "policy_version": LICENCE_POLICY_VERSION,
                "gate_enabled": any(
                    "licence_gate" in s for s in self.sources_result
                ),
                "allowed_spdx": self.cfg["policy"]["allowed_spdx"],
                "by_spdx": by_spdx,
                "rejected_at_verification": self.cfg.get("rejected", []),
                "attribution_recorded_per_chunk": True,
                "per_source": [
                    {
                        "id": s["id"],
                        "commit": s["commit"],
                        "verification_result": s.get("licence_gate", {}).get(
                            "verification_result"
                        ),
                        "allowed": s.get("licence_gate", {}).get("allowed"),
                        "reason": s.get("licence_gate", {}).get("reason"),
                        "exclusion_paths": s.get("licence_gate", {}).get(
                            "exclusion_paths", []
                        ),
                        "exclusion_sources": s.get("licence_gate", {}).get(
                            "exclusion_sources", []
                        ),
                        "files_excluded_by_licence": s.get(
                            "files_quarantined_licence", 0
                        ),
                    }
                    for s in self.sources_result
                ],
                "verification": (
                    "live licence re-read per run and matched against both the "
                    "pinned spdx value and the allowlist; a mismatch aborts that "
                    "source and the run is not reported complete"
                ),
            },
            "counts": {
                "seen": self.counters["seen"],
                "kept": kept,
                "resumed": self._resumed_rows,
                "deduplicated": deduplicated,
                "rejected": rejected,
                "rejected_total": sum(rejected.values()),
                "sources_skipped_by_max_repos": self.counters.get(
                    "sources_skipped_by_max_repos", 0
                ),
                "by_source": by_source,
            },
            "bytes": {"text_bytes": self.bytes},
            "splits": splits,
            "split_algorithm": {
                "method": split_mode,
                "group_by": "repo" if split_mode == "source_group" else "content",
                "seed": None,
                "formula": (
                    "bucket = int(sha1(key)[:8], 16) / 0xFFFFFFFF; train if "
                    "bucket < train_frac; val if < train_frac + val_frac; else test"
                ),
                "key": (
                    "group assignment by largest remainder over groups "
                    "ordered by sha1('order:' + repo)"
                    if split_mode == "source_group"
                    else "sha1(text)"
                ),
                "group_counts": group_counts,
                "groups_by_split": {
                    split: sorted(
                        r for r, s in repo_splits.items() if next(iter(s)) == split
                    )
                    for split in ("train", "val", "test")
                },
                "holdout_adequacy": (
                    "INSUFFICIENT: a source-group split can only allocate whole "
                    f"repositories, and there are {len(repo_splits)}. val holds "
                    f"{group_counts.get('val', 0)} repositories and test holds "
                    f"{group_counts.get('test', 0)}. Holdouts this small are "
                    "deterministic and reproducible but are NOT statistically "
                    "sufficient for model evaluation, and they are not "
                    "benchmark-independent (see the contamination audit)."
                    if split_mode == "source_group"
                    and min(
                        group_counts.get("val", 0), group_counts.get("test", 0)
                    ) < 3
                    else "at least 3 repositories per holdout"
                ),
                "note": (
                    "source_group mode keeps every file of one repository in a "
                    "single split. With few repositories the holdouts are small; "
                    "these splits are deterministic holdouts, not statistically "
                    "sufficient evaluation sets, and are not benchmark-"
                    "independent (see contamination_audit)."
                    if split_mode == "source_group"
                    else "A content-hash split needs no random seed. Source-level "
                    "grouping is not imposed."
                ),
            },
            "content_hashes": content_hashes,
            "leakage_check": {
                "method": "content SHA-1 across splits + MinHash near-dup re-scan",
                "exact_duplicate_chunks_in_multiple_splits": exact_leaks,
                "near_duplicate_pairs": near_dup_pairs,
                "near_duplicate_pairs_crossing_splits": near_dup_leaks,
                "repositories_spanning_multiple_splits": overlapping_repos,
                "repositories_in_more_than_one_split": len(overlapping_repos),
                "grouping": split_mode,
                "limitations": (
                    "MinHash over 8-word shingles at the configured threshold "
                    "catches near-identical text only. It is not semantic leakage "
                    "detection: paraphrases, renamed symbols and translated logic "
                    "are invisible to it."
                ),
            },
            "pii_scrub": {
                "mode": "token-aware (string and comment tokens only)",
                "replacements": self.counters.get("pii_replacements", {}),
                "preserved_untokenizable": self.counters.get("pii_preserved", 0),
                "limitations": (
                    "Patterns are India-specific (Aadhaar/PAN) plus a 10-digit "
                    "phone heuristic and email. Names, addresses, IPs, JWTs and "
                    "connection strings are not detected."
                ),
            },
            "network": dict(self.transport.stats),
            "integrity_failures": list(self._integrity_failures),
            "limits_reached": bool(self.counters["truncated_reasons"]),
            "truncated_reasons": dict(self.counters["truncated_reasons"]),
            "truncated": self.truncated,
            "status": status,
            "complete": status == "complete",
        }

    def _measure_near_duplicates(self):
        """Count surviving near-duplicate pairs and how many straddle a split.

        Re-scans the finished corpus with a fresh detector. The first pass counts
        pairs the detector would reject; a second pass tracks which split owns each
        LSH band key, so a band shared by chunks in different splits is counted as
        cross-split contamination rather than silently accepted.
        """
        if not self.out_path.exists():
            return 0, 0
        rows = []
        with self.out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        detector = MinHashDedup(
            num_perms=self.cfg["dedup"]["minhash_perms"],
            bands=self.cfg["dedup"]["minhash_bands"],
            threshold=self.cfg["dedup"]["minhash_threshold"],
        )
        pairs = 0
        band_owner = {}
        crossing = set()

        for row in rows:
            if not detector.add(row["text"]):
                pairs += 1
            sig = detector.signature(row["text"])
            if sig.size == 0:
                continue
            per_band = detector.rows
            for band in range(detector.num_perms // per_band):
                key = (band, sig[band * per_band : (band + 1) * per_band].tobytes())
                owner = band_owner.get(key)
                if owner is None:
                    band_owner[key] = row["split"]
                elif owner != row["split"]:
                    crossing.add(key)

        return pairs, len(crossing)

    def _write_manifest(self, manifest):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the M1 external code corpus from approved, pinned sources"
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default="data/code_corpus_v1.jsonl")
    p.add_argument("--manifest", default="data/code_corpus_v1.manifest.json")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--dry-run", action="store_true", help="print the plan, touch no network")
    p.add_argument("--max-repos", type=int, default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--include-held-back", action="store_true",
                   help="also process sources marked phase1_default: false")
    p.add_argument(
        "--licence-verification",
        default="data/licence_verification.json",
        help="licence verification artifact required by the build gate",
    )
    p.add_argument(
        "--skip-licence-gate",
        action="store_true",
        help="DISABLES the mandatory licence gate. Every source will be recorded "
        "as unverified and blocked. Provided only so a failed gate is visible "
        "rather than mysterious.",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.include_held_back:
        for s in cfg["sources"]:
            s.pop("phase1_default", None)

    licence_index = None
    if not args.skip_licence_gate:
        selected = [
            s for s in cfg["sources"] if s.get("phase1_default", True)
        ]
        try:
            licence_index = load_licence_verification(
                args.licence_verification,
                commit_by_source={s["id"]: s["commit"] for s in selected},
            )
        except LicenceGateError as exc:
            raise SystemExit(str(exc))
        licence_index = licence_index["by_source"]
        if not licence_index:
            raise SystemExit(
                f"{args.licence_verification} contains no source results; "
                "nothing can be verified."
            )
    else:
        print(
            "WARNING: --skip-licence-gate is set. The mandatory licence gate is "
            "OFF; sources will be blocked as unverified. Do not publish the "
            "result of this run."
        )

    transport = Transport(
        timeout=cfg["limits"]["request_timeout_seconds"],
        max_retries=cfg["limits"]["max_retries_per_request"],
        pause=cfg["limits"].get("pause_between_requests_seconds", 0.0),
        deadline=cfg["limits"].get("transfer_deadline_seconds"),
    )

    if not args.resume and not args.dry_run:
        # Previous artifacts are evidence, not litter. A fresh run moves them
        # aside rather than deleting them, so a run that fails (rate limit, network,
        # licence rejection) cannot destroy the corpus that is already on disk.
        out = Path(args.out)
        if out.exists():
            stale_out = out.with_suffix(out.suffix + ".stale")
            stale_out.unlink(missing_ok=True)
            out.replace(stale_out)
        old = Path(args.manifest)
        if old.exists():
            stale = old.with_suffix(old.suffix + ".stale")
            stale.unlink(missing_ok=True)
            old.replace(stale)

    builder = CorpusBuilder(
        cfg,
        transport,
        out_path=args.out,
        manifest_path=args.manifest,
        resume=args.resume,
        dry_run=args.dry_run,
        max_files_override=args.max_files,
        max_bytes_override=args.max_bytes,
        max_repos_override=args.max_repos,
        licence_index=licence_index,
    )
    manifest = builder.run()

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    print(f"\nkept={manifest['counts']['kept']} splits={manifest['splits']}")
    print(f"status={manifest['status']} truncated={manifest['truncated']}")
    print(f"licence spread={manifest['licensing']['by_spdx']}")
    leak = manifest["leakage_check"]
    print(
        "leakage: exact_cross_split="
        f"{leak['exact_duplicate_chunks_in_multiple_splits']} "
        f"near_dup_pairs={leak['near_duplicate_pairs']} "
        f"near_dup_crossing={leak['near_duplicate_pairs_crossing_splits']}"
    )
    if manifest["status"] != "complete":
        print(
            f"RUN STATUS: {manifest['status'].upper()} -- see manifest "
            "'status'/'sources'/'truncated_reasons' for detail"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())