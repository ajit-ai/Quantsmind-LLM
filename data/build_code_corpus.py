import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from data.filters import ExactDedup, MinHashDedup, quality_check, scrub_pii

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
    "isc", "unlicense", "cc0-1.0", "wtfpl",
}

MIN_CODE_CHARS = 100
MAX_CODE_CHARS = 100_000
MIN_ALPHA_RATIO = 0.3


def detect_language(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return LANGUAGE_EXTENSIONS.get(ext)


def is_minified(text: str) -> bool:
    lines = text.split("\n")
    if len(lines) < 3:
        return False
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    return avg_line_len > 200


def has_valid_structure(text: str, language: str) -> bool:
    if language == "python":
        try:
            compile(text, "<string>", "exec")
            return True
        except SyntaxError:
            return False
    return True


def fetch_github_repos(query: str, max_repos: int = 50, token: str | None = None) -> list[dict]:
    import urllib.parse

    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page={max_repos}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "Quantsmind-LLM-code-corpus")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("items", [])
    except Exception as e:
        print(f"GitHub API error: {e}")
        return []


def fetch_repo_files(repo_full_name: str, token: str | None = None) -> list[dict]:
    url = f"https://api.github.com/repos/{repo_full_name}/git/trees/HEAD?recursive=1"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "Quantsmind-LLM-code-corpus")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return [
                item for item in data.get("tree", [])
                if item.get("type") == "blob"
                and Path(item.get("path", "")).suffix.lower() in LANGUAGE_EXTENSIONS
            ]
    except Exception as e:
        print(f"GitHub tree error for {repo_full_name}: {e}")
        return []


def fetch_file_content(repo_full_name: str, path: str, token: str | None = None) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo_full_name}/HEAD/{path}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Quantsmind-LLM-code-corpus")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def fetch_license(repo_full_name: str, token: str | None = None) -> str | None:
    url = f"https://api.github.com/repos/{repo_full_name}/license"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "Quantsmind-LLM-code-corpus")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("license", {}).get("spdx_id", "").lower()
    except Exception:
        return None


def chunk_code(text: str, chunk_lines: int = 200, min_lines: int = 10) -> list[str]:
    lines = text.split("\n")
    if len(lines) <= chunk_lines:
        return [text] if len(lines) >= min_lines else []
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        piece = "\n".join(lines[start:end]).strip()
        if len(piece.split("\n")) >= min_lines:
            chunks.append(piece)
        start = end
    return chunks


def process_local_files(raw_dir: Path) -> list[tuple[str, str, str]]:
    results = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        lang = detect_language(path.name)
        if lang is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        results.append((text, lang, f"local:{path.parent.name}"))
    return results


def main():
    p = argparse.ArgumentParser(description="Build code corpus from public GitHub repos and local files")
    p.add_argument("--raw-dir", default=None, help="Local directory with code files to include")
    p.add_argument("--out", default="data/code_corpus_v0.jsonl")
    p.add_argument("--stats", default="data/code_corpus_v0.stats.json")
    p.add_argument("--max-repos", type=int, default=50)
    p.add_argument("--max-files-per-repo", type=int, default=100)
    p.add_argument("--chunk-lines", type=int, default=200)
    p.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"))
    p.add_argument("--languages", nargs="+", default=["python", "javascript", "typescript"])
    p.add_argument("--no-github", action="store_true", help="Skip GitHub fetch, only process local files")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dedup = ExactDedup()
    fuzzy = MinHashDedup()

    stats = {
        "seen": 0,
        "kept": 0,
        "rejected_quality": 0,
        "rejected_dup": 0,
        "rejected_fuzzy": 0,
        "rejected_minified": 0,
        "rejected_syntax": 0,
        "rejected_license": 0,
        "chunks_emitted": 0,
        "by_language": {},
        "by_source": {},
    }

    all_items: list[tuple[str, str, str]] = []

    if args.raw_dir:
        raw_path = Path(args.raw_dir)
        if raw_path.exists():
            all_items.extend(process_local_files(raw_path))
            print(f"Local files: {len(all_items)} files found")

    if not args.no_github:
        for lang in args.languages:
            query = f"language:{lang} stars:>100"
            repos = fetch_github_repos(query, args.max_repos, args.github_token)
            print(f"GitHub search '{query}': {len(repos)} repos found")

            for repo in repos:
                repo_name = repo.get("full_name", "")
                license_id = fetch_license(repo_name, args.github_token)
                if license_id and license_id not in PERMISSIVE_LICENSES:
                    stats["rejected_license"] += 1
                    continue

                files = fetch_repo_files(repo_name, args.github_token)
                files = files[: args.max_files_per_repo]
                print(f"  {repo_name}: {len(files)} code files")

                for f in files:
                    path = f.get("path", "")
                    if detect_language(path) != lang:
                        continue
                    content = fetch_file_content(repo_name, path, args.github_token)
                    if content:
                        all_items.append((content, lang, f"github:{repo_name}"))

    print(f"Total items to process: {len(all_items)}")

    with out.open("w", encoding="utf-8") as f:
        for text, lang, source in all_items:
            stats["seen"] += 1
            if is_minified(text):
                stats["rejected_minified"] += 1
                continue
            if not has_valid_structure(text, lang):
                stats["rejected_syntax"] += 1
                continue
            ok, _ = quality_check(text, MIN_CODE_CHARS, MAX_CODE_CHARS, MIN_ALPHA_RATIO)
            if not ok:
                stats["rejected_quality"] += 1
                continue

            for piece in chunk_code(text, args.chunk_lines):
                if not dedup.add(piece):
                    stats["rejected_dup"] += 1
                    continue
                if not fuzzy.add(piece):
                    stats["rejected_fuzzy"] += 1
                    continue
                clean = scrub_pii(piece)
                f.write(
                    json.dumps(
                        {"text": clean, "language": lang, "source": source},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["kept"] += 1
                stats["chunks_emitted"] += 1
                stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(
        f"seen={stats['seen']} kept={stats['kept']} "
        f"dup={stats['rejected_dup']} quality_reject={stats['rejected_quality']} "
        f"minified={stats['rejected_minified']} syntax={stats['rejected_syntax']}"
    )


if __name__ == "__main__":
    main()
