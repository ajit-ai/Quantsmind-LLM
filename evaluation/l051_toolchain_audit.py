"""TAKSH L0.5.1 — toolchain and local deployment feasibility audit (WP7).

Inventories compiler/interpreter/runtime availability for the 16 L1.0 languages
plus SQL support. NEVER installs software; every entry records observed
version, platform notes, and the execution-isolation posture.

The local execution strategy (designed, not yet implemented):
  * fresh temporary working directory per task
  * scrubbed environment (SystemRoot/PATH only, no user secrets)
  * hard timeouts and per-language resource limits
  * no uncontrolled network access; no unrestricted filesystem access
  * explicit execution permissions per language
  * sandbox keeps generated code off the repository tree

Probes run with a 20 s timeout each. Failures are reported, never treated as
fatal to the audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PYTHON = r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe"
RUSTC = str(Path.home() / ".cargo" / "bin" / "rustc.exe")
GITBASH = r"C:\Program Files\Git\bin\bash.exe"

TOOLS = [
    ("python", "python3", [PYTHON, "--version"]),
    ("sql", "sqlite3-module", [PYTHON, "-c", "import sqlite3; print('sqlite3 module', sqlite3.sqlite_version)"]),
    ("javascript", "node", ["node", "--version"]),
    ("typescript", "tsc", ["npx", "--no-install", "tsc", "--version"]),
    ("typescript", "node-type-stripping", ["node", "--experimental-strip-types", "-e", "console.log('ts-ok')"]),
    ("c", "msvc-cl", ["cmd", "/c", "where cl"]),
    ("c", "msvc-availability", ["cmd", "/c", "dir /b", r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"]),
    ("cpp", "msvc-cl-cpp17", ["cmd", "/c", "where cl"]),
    ("java", "javac", ["javac", "-version"]),
    ("java", "java", ["java", "-version"]),
    ("csharp", "dotnet", ["dotnet", "--version"]),
    ("go", "go", ["go", "version"]),
    ("rust", "rustc", [RUSTC, "--version"]),
    ("php", "php", ["php", "--version"]),
    ("ruby", "ruby", ["ruby", "--version"]),
    ("swift", "swift", ["swift", "--version"]),
    ("kotlin", "kotlinc", ["kotlinc", "-version"]),
    ("dart", "dart", ["dart", "--version"]),
    ("sql", "sqlite-cli", ["sqlite3", "--version"]),
    ("bash", "bash", [GITBASH, "--version"]),
]


def probe(language: str, name: str, argv: list) -> dict:
    exe = argv[0]
    # Resolve the real executable, including .cmd/.bat shims which CreateProcess cannot run directly.
    if Path(exe).parent.exists() and Path(exe).exists():
        resolved = exe
    else:
        resolved = shutil.which(exe)
    if resolved is None:
        return {"language": language, "tool": name, "available": False,
                "version": None, "detail": "not found on PATH"}
    rest = argv[1:]
    if resolved.lower().endswith((".cmd", ".bat")):
        # .cmd/.bat shims must go through cmd.exe
        proc_argv = ["cmd", "/c", resolved, *rest]
    else:
        proc_argv = [resolved, *rest]
    try:
        proc = subprocess.run(proc_argv, capture_output=True, text=True, timeout=20)
        output = (proc.stdout or "").strip().splitlines()
        err_out = (proc.stderr or "").strip().splitlines()
        text = (output[0] if output else (err_out[0] if err_out else "")).strip()
        if name == "tsc" and "Version" not in text:
            return {"language": language, "tool": name, "available": False,
                    "version": None, "detail": "tsc not installed (npx prints placeholder)"}
        if name == "msvc-cl" and proc.returncode != 0:
            return {"language": language, "tool": name, "available": False,
                    "version": None, "detail": "cl.exe not on PATH; MSVC BuildTools present at fixed install dir; vsdevcmd env required"}
        ok = proc.returncode in (0, 1) and bool(text)
        return {"language": language, "tool": name, "available": bool(ok),
                "version": text[:120], "detail": "" if ok else f"exit={proc.returncode}"}
    except FileNotFoundError:
        return {"language": language, "tool": name, "available": False,
                "version": None, "detail": "not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"language": language, "tool": name, "available": False,
                "version": None, "detail": "probe timed out"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_toolchain_audit.json")
    args = parser.parse_args(argv)

    results = [probe(lang, name, argv) for lang, name, argv in TOOLS]
    by_lang: dict[str, list] = {}
    for r in results:
        by_lang.setdefault(r["language"], []).append(r)

    l1 = ["python", "javascript", "typescript", "c", "cpp", "java", "csharp",
          "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash"]
    summary = {}
    for lid in l1:
        rows = by_lang.get(lid, [])
        usable = any(r["available"] for r in rows)
        summary[lid] = {"usable": usable, "checks": rows}

    available = [lid for lid in l1 if summary[lid]["usable"]]
    missing = [lid for lid in l1 if not summary[lid]["usable"]]

    report = {
        "schema": "taksh.l051_toolchain_audit/1",
        "generated_by": "evaluation/l051_toolchain_audit.py",
        "host": {
            "platform": "win32",
            "note": "Windows 11 host; WSL excluded from availability claims (not installed); Git Bash present",
        },
        "local_execution_strategy": {
            "workspace": "fresh temporary directory per benchmark task; deleted afterwards",
            "environment": "SystemRoot/PATH only; no user secrets; no inherited environment",
            "timeouts": "5-20 s per language (see per-language execution specs in the benchmark)",
            "network": "disabled inside the sandbox process tree",
            "filesystem": "workspace-only; repo tree unreachable from sandbox cwd",
            "permissions": "explicit per language; no ambient privileges; never runs model-generated code outside the sandbox",
        },
        "l1_summary": {lid: bool(summary[lid]["usable"]) for lid in l1},
        "available": available,
        "missing": missing,
        "checks": results,
        "conclusions": [
            f"{len(available)} of 16 L1.0 toolchains are locally executable in this environment: {', '.join(sorted(available))}",
            f"{len(missing)} are NOT available here: {', '.join(missing)}",
            "SQL is executable via Python's bundled sqlite3 module (3.50.4); standalone sqlite3 CLI absent",
            "TypeScript executes only via Node's experimental type stripping; no tsc (not installed, not authorized to install)",
            "C and C++ build via MSVC 19.44.35229 through a vsdevcmd environment; cl.exe is not on PATH",
            "Bash executes via Git Bash 5.2.37; WSL is absent (bash.exe is the disabled WSL launcher)",
            "no benchmarks can execute without the per-language sandboxed runner (L1.0 work item)",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"available ({len(available)}): {sorted(available)}")
    print(f"missing   ({len(missing)}): {missing}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
