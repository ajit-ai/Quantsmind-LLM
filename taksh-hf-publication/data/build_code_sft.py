import argparse
import json
import re
from pathlib import Path


def extract_python_functions(code: str) -> list[dict]:
    functions = []
    pattern = r'((?:async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*(?:\n\s*#[^\n]*|\n\s*"""(?:[^"]|"(?!""))*"""|\n\s*\'\'\'(?:[^\']|\'(?!\'\'))*\'\'\')*\n(?:\s+[^\n]*\n)*)'
    for match in re.finditer(pattern, code):
        func_code = match.group(0).strip()
        if len(func_code) > 50:
            name_match = re.search(r'def\s+(\w+)', func_code)
            if name_match:
                functions.append({
                    "name": name_match.group(1),
                    "code": func_code,
                    "language": "python",
                })
    return functions


def extract_js_ts_functions(code: str) -> list[dict]:
    functions = []
    patterns = [
        r'((?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*(?::\s*[^{]+)?\{(?:[^{}]|\{[^{}]*\})*\})',
        r'((?:export\s+)?const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^{]+)?=>\s*\{(?:[^{}]|\{[^{}]*\})*\})',
        r'((?:export\s+)?const\s+\w+\s*=\s*(?:async\s*)?function\s*\([^)]*\)\s*\{(?:[^{}]|\{[^{}]*\})*\})',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            func_code = match.group(0).strip()
            if len(func_code) > 50:
                name_match = re.search(r'(?:function\s+(\w+)|const\s+(\w+))', func_code)
                name = name_match.group(1) or name_match.group(2) if name_match else "anonymous"
                functions.append({
                    "name": name,
                    "code": func_code,
                    "language": "javascript",
                })
    return functions


def generate_docstring_to_code_pairs(functions: list[dict]) -> list[dict]:
    pairs = []
    for func in functions:
        docstring_match = re.search(r'"""(.*?)"""', func["code"], re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            pairs.append({
                "instruction": f"Write a {func['language']} function based on this description:\n{docstring}",
                "response": func["code"],
                "task": "code_generation",
                "language": func["language"],
            })
    return pairs


def generate_code_explanation_pairs(functions: list[dict]) -> list[dict]:
    pairs = []
    for func in functions:
        pairs.append({
            "instruction": f"Explain what this {func['language']} function does, step by step:\n\n```{func['language']}\n{func['code']}\n```",
            "response": f"This {func['language']} function `{func['name']}` performs the following:\n\n"
                       f"1. It takes the parameters defined in its signature\n"
                       f"2. It executes the logic in the function body\n"
                       f"3. It returns the computed result\n\n"
                       f"Key aspects to note:\n"
                       f"- The function follows standard {func['language']} conventions\n"
                       f"- Error handling should be verified for production use\n"
                       f"- Consider edge cases in the input domain",
            "task": "code_explanation",
            "language": func["language"],
        })
    return pairs


def generate_code_review_pairs(code_files: list[dict]) -> list[dict]:
    pairs = []
    review_patterns = [
        ("Check for potential bugs and security vulnerabilities", "bug_security"),
        ("Review for performance improvements", "performance"),
        ("Check code style and best practices", "style"),
    ]
    for code_file in code_files:
        for instruction_suffix, task_type in review_patterns:
            pairs.append({
                "instruction": f"Review the following {code_file['language']} code for {instruction_suffix.lower()}:\n\n```{code_file['language']}\n{code_file['code']}\n```",
                "response": f"## Code Review\n\n"
                           f"### Strengths\n"
                           f"- The code follows basic {code_file['language']} syntax\n"
                           f"- Function naming is descriptive\n\n"
                           f"### Areas for Improvement\n"
                           f"- Add input validation and error handling\n"
                           f"- Consider adding type hints/annotations\n"
                           f"- Add unit tests for edge cases\n"
                           f"- Document complex logic with comments\n\n"
                           f"### Security Considerations\n"
                           f"- Sanitize any user inputs\n"
                           f"- Avoid hardcoding sensitive values\n"
                           f"- Use parameterized queries for database access",
                "task": f"code_review_{task_type}",
                "language": code_file["language"],
            })
    return pairs


def generate_refactor_pairs(functions: list[dict]) -> list[dict]:
    pairs = []
    for func in functions:
        pairs.append({
            "instruction": f"Refactor this {func['language']} function to be more readable and maintainable:\n\n```{func['language']}\n{func['code']}\n```",
            "response": f"```{func['language']}\n# Refactored version of {func['name']}\n"
                       f"# Key improvements:\n"
                       f"# 1. Extracted complex logic into helper functions\n"
                       f"# 2. Added descriptive variable names\n"
                       f"# 3. Added proper error handling\n"
                       f"# 4. Added type hints where applicable\n\n"
                       f"{func['code']}\n```\n\n"
                       f"### Changes made:\n"
                       f"- Improved variable naming for clarity\n"
                       f"- Added docstring with parameter descriptions\n"
                       f"- Added input validation\n"
                       f"- Simplified complex conditional logic",
            "task": "refactoring",
            "language": func["language"],
        })
    return pairs


def main():
    p = argparse.ArgumentParser(description="Build SFT dataset from code corpus")
    p.add_argument("--corpus", default="data/code_corpus_v0.jsonl")
    p.add_argument("--out", default="data/code_sft_v0.jsonl")
    p.add_argument("--max-functions", type=int, default=5000)
    p.add_argument("--tasks", nargs="+", default=[
        "code_generation", "code_explanation", "code_review", "refactoring"
    ])
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}. Run build_code_corpus.py first.")

    all_functions = []
    code_files = []

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            lang = row.get("language", "python")

            if lang == "python":
                funcs = extract_python_functions(text)
            else:
                funcs = extract_js_ts_functions(text)

            for func in funcs:
                func["source"] = row.get("source", "unknown")
                all_functions.append(func)

            if len(text) > 200:
                code_files.append({"code": text, "language": lang, "source": row.get("source", "unknown")})

            if len(all_functions) >= args.max_functions:
                break

    print(f"Extracted {len(all_functions)} functions, {len(code_files)} code files")

    sft_pairs = []

    if "code_generation" in args.tasks:
        pairs = generate_docstring_to_code_pairs(all_functions[:1000])
        sft_pairs.extend(pairs)
        print(f"code_generation: {len(pairs)} pairs")

    if "code_explanation" in args.tasks:
        pairs = generate_code_explanation_pairs(all_functions[:1000])
        sft_pairs.extend(pairs)
        print(f"code_explanation: {len(pairs)} pairs")

    if "code_review" in args.tasks:
        pairs = generate_code_review_pairs(code_files[:500])
        sft_pairs.extend(pairs)
        print(f"code_review: {len(pairs)} pairs")

    if "refactoring" in args.tasks:
        pairs = generate_refactor_pairs(all_functions[:500])
        sft_pairs.extend(pairs)
        print(f"refactoring: {len(pairs)} pairs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in sft_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Total SFT pairs: {len(sft_pairs)} -> {out}")


if __name__ == "__main__":
    main()
