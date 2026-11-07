import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.server_security import (
    add_standard_args,
    api_key_from_env,
    install_auth,
    load_trusted_state,
    resolve_device,
    serve,
)

app = FastAPI(title="Taksh Coder — Code Review API")

_model = None
_tokenizer = None
_device = None

# There is deliberately no POST /load. The previous implementation accepted a
# `ckpt` query parameter and called torch.load(weights_only=False) on it with no
# authentication while binding 0.0.0.0 -- unauthenticated remote code execution.
# The checkpoint is now chosen once, on the command line, by the operator.

REVIEW_SYSTEM_PROMPT = (
    "You are Taksh Coder, an expert code reviewer specializing in Python and "
    "JavaScript/TypeScript. Review code for:\n"
    "1. Bugs and logic errors\n"
    "2. Security vulnerabilities (OWASP Top 10)\n"
    "3. Performance issues\n"
    "4. Code style and best practices\n"
    "5. Missing error handling\n\n"
    "Format your review as JSON with 'issues' array containing objects with "
    "'severity' (critical/major/minor/info), 'line' (line number or null), "
    "'message' (description), and 'suggestion' (fix recommendation)."
)

ISSUE_PATTERNS = {
    "bug": [
        (r"except\s*:", "Bare except clause catches all exceptions including KeyboardInterrupt"),
        (r"==\s*True|!=\s*False", "Unnecessary comparison to boolean literal"),
        (r"print\(", "Debug print statement left in code"),
    ],
    "security": [
        (r"eval\(", "Use of eval() is dangerous and can lead to code injection"),
        (r"exec\(", "Use of exec() is dangerous and can lead to code injection"),
        (r"shell\s*=\s*True", "shell=True in subprocess is a security risk"),
        (r"verify\s*=\s*False", "Disabling SSL verification is a security risk"),
        (r"md5", "MD5 is cryptographically broken, use SHA-256 or better"),
    ],
    "performance": [
        (r"for\s+.*\n\s*.*\.append\(", "Consider using list comprehension instead of append in loop"),
        (r"\.keys\(\)\s+in\s+", "Use 'key in dict' instead of 'key in dict.keys()'"),
    ],
}


class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    context: str = ""


class ReviewIssue(BaseModel):
    severity: str
    line: int | None
    message: str
    suggestion: str


class ReviewResponse(BaseModel):
    issues: list[ReviewIssue]
    summary: str
    model: str


def static_analysis(code: str, language: str) -> list[ReviewIssue]:
    issues = []
    lines = code.split("\n")
    for category, patterns in ISSUE_PATTERNS.items():
        for pattern, message in patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    severity = "critical" if category == "security" else "major" if category == "bug" else "minor"
                    issues.append(ReviewIssue(
                        severity=severity,
                        line=i,
                        message=message,
                        suggestion=f"Review and fix: {message}",
                    ))
    return issues


def model_review(model, tokenizer, code: str, language: str, device: str) -> str:
    prompt = (
        f"<|system|>{REVIEW_SYSTEM_PROMPT}\n"
        f"<|user|>\nReview this {language} code:\n```{language}\n{code}\n```\n<|assistant|>\n"
    )
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(512):
            idx_cond = x if x.size(1) <= model.cfg["max_seq_len"] else x[:, -model.cfg["max_seq_len"]:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.3
            v, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.token_to_id("<eos>"):
                break
    generated = tokenizer.decode(x[0].tolist())
    return generated[len(prompt):] if generated.startswith(prompt) else generated


@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    static_issues = static_analysis(request.code, request.language)
    model_output = model_review(_model, _tokenizer, request.code, request.language, _device)

    try:
        model_json = json.loads(model_output)
        model_issues = [ReviewIssue(**i) for i in model_json.get("issues", [])]
    except (json.JSONDecodeError, TypeError):
        model_issues = []

    all_issues = static_issues + model_issues
    critical = sum(1 for i in all_issues if i.severity == "critical")
    major = sum(1 for i in all_issues if i.severity == "major")
    minor = sum(1 for i in all_issues if i.severity == "minor")

    if critical > 0:
        summary = f"Found {critical} critical, {major} major, {minor} minor issues. Immediate attention required."
    elif major > 0:
        summary = f"Found {major} major, {minor} minor issues. Review recommended."
    elif minor > 0:
        summary = f"Found {minor} minor issues. Code is generally good."
    else:
        summary = "No issues found. Code looks clean."

    return ReviewResponse(issues=all_issues, summary=summary, model="quantsmind-coder")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


def main():
    global _model, _tokenizer, _device

    p = argparse.ArgumentParser(description="Taksh Coder — Code Review API")
    add_standard_args(p, default_port=8001)
    args = p.parse_args()

    _device = resolve_device()
    _model, _model_cfg, _tokenizer = load_trusted_state(args.ckpt, args.tokenizer, _device)
    print(f"loaded {_model_cfg.get('name')} context={_model_cfg.get('max_seq_len')} device={_device}")

    auth_on = install_auth(app, api_key_from_env())
    serve(app, p, args, "Code Review API", auth_on)


if __name__ == "__main__":
    main()