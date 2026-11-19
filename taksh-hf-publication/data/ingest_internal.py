import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from data.build_code_corpus import (
    LANGUAGE_EXTENSIONS,
    chunk_code,
    detect_language,
    has_valid_structure,
    is_minified,
)
from data.filters import ExactDedup, MinHashDedup, quality_check, scrub_pii

SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "[REDACTED_SECRET]"),
    (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
    (r"ghp_[0-9a-zA-Z]{36}", "[REDACTED_GITHUB_TOKEN]"),
]


def scrub_secrets(text: str) -> str:
    import re
    for pattern, replacement in SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def scan_for_secrets(text: str) -> list[str]:
    import re
    findings = []
    for pattern, _ in SECRET_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            findings.extend(matches[:3])
    return findings


def get_git_info(repo_path: Path) -> dict:
    info = {"commit": None, "branch": None, "remote": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["remote"] = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=repo_path, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        pass
    return info


def ingest_repo(repo_path: Path, repo_name: str | None = None) -> list[dict]:
    if repo_name is None:
        repo_name = repo_path.name

    git_info = get_git_info(repo_path)
    items = []

    for file_path in sorted(repo_path.rglob("*")):
        if not file_path.is_file():
            continue
        if ".git" in file_path.parts:
            continue
        lang = detect_language(file_path.name)
        if lang is None:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if is_minified(text):
            continue
        if not has_valid_structure(text, lang):
            continue
        ok, _ = quality_check(text, 100, 100_000, 0.3)
        if not ok:
            continue

        relative_path = str(file_path.relative_to(repo_path))
        items.append({
            "text": text,
            "language": lang,
            "source": f"internal:{repo_name}",
            "file_path": relative_path,
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
        })

    return items


def main():
    p = argparse.ArgumentParser(description="Ingest internal codebase for corporate fine-tuning")
    p.add_argument("--repo", required=True, help="Path to internal git repository")
    p.add_argument("--repo-name", default=None, help="Name identifier for the repo")
    p.add_argument("--out", default="data/internal_corpus_v0.jsonl")
    p.add_argument("--stats", default="data/internal_corpus_v0.stats.json")
    p.add_argument("--scan-secrets", action="store_true", help="Scan and report potential secrets")
    p.add_argument("--chunk-lines", type=int, default=200)
    args = p.parse_args()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        raise SystemExit(f"Repository not found: {repo_path}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting: {repo_path}")
    items = ingest_repo(repo_path, args.repo_name)
    print(f"Found {len(items)} code files")

    dedup = ExactDedup()
    fuzzy = MinHashDedup()

    stats = {
        "seen": 0,
        "kept": 0,
        "rejected_quality": 0,
        "rejected_dup": 0,
        "rejected_fuzzy": 0,
        "chunks_emitted": 0,
        "by_language": {},
        "secrets_found": 0,
        "files_with_secrets": [],
    }

    with out.open("w", encoding="utf-8") as f:
        for item in items:
            stats["seen"] += 1

            if args.scan_secrets:
                secrets = scan_for_secrets(item["text"])
                if secrets:
                    stats["secrets_found"] += len(secrets)
                    stats["files_with_secrets"].append(item["file_path"])

            clean_text = scrub_secrets(item["text"])
            clean_text = scrub_pii(clean_text)

            for piece in chunk_code(clean_text, args.chunk_lines):
                if not dedup.add(piece):
                    stats["rejected_dup"] += 1
                    continue
                if not fuzzy.add(piece):
                    stats["rejected_fuzzy"] += 1
                    continue
                row = {
                    "text": piece,
                    "language": item["language"],
                    "source": item["source"],
                    "file_path": item["file_path"],
                    "git_commit": item["git_commit"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats["chunks_emitted"] += 1
                stats["by_language"][item["language"]] = stats["by_language"].get(item["language"], 0) + 1

    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(f"seen={stats['seen']} kept={stats['kept']} chunks={stats['chunks_emitted']}")
    if args.scan_secrets and stats["secrets_found"] > 0:
        print(f"WARNING: {stats['secrets_found']} potential secrets found in {len(stats['files_with_secrets'])} files")
        for fp in stats["files_with_secrets"][:10]:
            print(f"  - {fp}")


if __name__ == "__main__":
    main()
