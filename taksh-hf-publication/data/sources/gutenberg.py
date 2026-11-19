import argparse
import json
import urllib.request
from pathlib import Path

CURATED = {
    3300: "The Wealth of Nations — Adam Smith",
    833: "The Theory of the Leisure Class — Thorstein Veblen",
    33076: "The Principles of Political Economy — John Stuart Mill (vol 1 excerpt set)",
}

URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
]


def fetch(gutenberg_id: int) -> str | None:
    for pattern in URL_PATTERNS:
        url = pattern.format(id=gutenberg_id)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
    return None


def main():
    p = argparse.ArgumentParser(description="Download public-domain economics texts")
    p.add_argument("--out-dir", default="data/raw/gutenberg")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for gid, title in CURATED.items():
        dest = out_dir / f"{gid}.txt"
        if dest.exists():
            manifest[str(gid)] = {"title": title, "status": "cached"}
            continue
        body = fetch(gid)
        if body is None:
            manifest[str(gid)] = {"title": title, "status": "failed"}
            print(f"skip {gid} ({title}): download failed")
            continue
        dest.write_text(body, encoding="utf-8")
        manifest[str(gid)] = {"title": title, "status": "ok", "chars": len(body)}
        print(f"ok {gid}: {title} ({len(body)} chars)")

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out_dir / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
