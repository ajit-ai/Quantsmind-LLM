"""Verify an installed Taksh wheel against a real M7 checkpoint.

Run this with the Python executable of the target virtual environment. It
asserts that Taksh and serving were imported from the installed package rather
than from the repository, then exercises the OpenAI-compatible API.

No model output is treated as capability evidence; the chat call is a serving
smoke test only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--skip-serving", action="store_true")
    return parser.parse_args(argv)


def _remove_repo_from_path(repo: Path) -> None:
    repo = repo.resolve()

    def keep(entry: str) -> bool:
        if not entry:
            return True
        try:
            candidate = Path(entry).resolve()
        except OSError:
            return True
        return candidate != repo

    sys.path[:] = [entry for entry in sys.path if keep(entry)]


def _assert_installed(module, repo: Path) -> str:
    path = getattr(module, "__file__", None)
    if not path:
        raise AssertionError(f"{module.__name__} has no file location")
    resolved = Path(path).resolve()
    if resolved.is_relative_to(repo):
        raise AssertionError(f"{module.__name__} was imported from the repository: {resolved}")
    return str(resolved)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.resolve()
    _remove_repo_from_path(repo)

    import taksh
    import serving

    package_path = _assert_installed(taksh, repo)
    serving_path = _assert_installed(serving, repo)
    if taksh.__version__ != args.expect_version:
        raise AssertionError(
            f"installed version {taksh.__version__!r} != expected {args.expect_version!r}"
        )

    summary: dict[str, object] = {
        "version": taksh.__version__,
        "release_status": taksh.__release_status__,
        "package_path": package_path,
        "serving_path": serving_path,
        "serving_checked": not args.skip_serving,
    }
    if args.skip_serving:
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.json_out:
            args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    checkpoint = (args.checkpoint or repo / "runs" / "m7_sft" / "checkpoint_best.pt").resolve()
    tokenizer = (args.tokenizer or repo / "tokenizer" / "qm_code_bpe_v0.json").resolve()
    configs = (args.configs or repo / "configs").resolve()
    for label, path in (("checkpoint", checkpoint), ("tokenizer", tokenizer), ("configs", configs)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    from fastapi.testclient import TestClient
    from serving import openai_server

    with tempfile.TemporaryDirectory(prefix="taksh-serving-check-") as tmp:
        work = Path(tmp)
        shutil.copytree(configs, work / "configs")
        original_cwd = Path.cwd()
        try:
            os.chdir(work)

            openai_server.load_state(str(checkpoint), str(tokenizer))
            openai_server.state.api_key = None
            client = TestClient(openai_server.app)

            health = client.get("/health")
            if health.status_code != 200:
                raise AssertionError(f"/health returned {health.status_code}: {health.text}")
            health_body = health.json()
            if health_body.get("version") != args.expect_version:
                raise AssertionError(f"/health version {health_body.get('version')!r} mismatch")
            if health_body.get("model_loaded") is not True:
                raise AssertionError("/health did not report a loaded model")

            models = client.get("/v1/models")
            if models.status_code != 200:
                raise AssertionError(f"/v1/models returned {models.status_code}: {models.text}")
            model_rows = models.json().get("data") or []
            if not model_rows or not model_rows[0].get("id"):
                raise AssertionError("/v1/models did not report the loaded model")

            completion = client.post(
                "/v1/chat/completions",
                json={
                    "model": "quantsmind-coder",
                    "messages": [{"role": "user", "content": "Write a Python function that adds two integers."}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": False,
                },
            )
            if completion.status_code != 200:
                raise AssertionError(
                    f"/v1/chat/completions returned {completion.status_code}: {completion.text}"
                )
            completion_body = completion.json()
            completion_tokens = completion_body.get("usage", {}).get("completion_tokens", 0)
            if completion_tokens <= 0:
                raise AssertionError("chat completion reported no generated tokens")

            summary.update(
                {
                    "health_status": health.status_code,
                    "health": health_body,
                    "models_status": models.status_code,
                    "model_id": model_rows[0]["id"],
                    "completion_status": completion.status_code,
                    "completion_tokens": completion_tokens,
                    "completion_text": completion_body["choices"][0]["message"]["content"],
                    "capability_note": "HTTP success only; not evidence of correct or useful output.",
                }
            )
        finally:
            os.chdir(original_cwd)

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
