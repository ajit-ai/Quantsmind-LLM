"""TAKSH publication and distribution readiness (read-only).

This module answers one question: **which distribution channels could Taksh use,
and what is actually blocking each one right now?**

It never publishes, uploads, tags, opens a network connection, or writes outside
the requested report path. It reads committed source and evidence files only.

Run:

    python -m evaluation.publication_readiness
    python -m evaluation.publication_readiness --out reports/taksh_publication_readiness.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")

# Every check below is deliberately conservative. A missing artifact is reported
# as missing; nothing is inferred from a filename alone.
CHECKPOINT_CANDIDATES = {
    "runs/taksh-foundation-111m/checkpoint_last.pt": "canonical_candidate_last",
    "runs/taksh-foundation-111m/checkpoint_best.pt": "canonical_candidate_best",
    "runs/v180_sft/checkpoint_last.pt": "legacy_incompatible_candidate",
    "runs/m7_sft/checkpoint_best.pt": "historical_m7_candidate",
}

TOKENIZER = "tokenizer/qm_code_bpe_v0.json"
FOUNDATION_CONFIG = "configs/model/foundation_111m.yaml"
FROZEN_BENCHMARK_CANDIDATES = (
    "evaluation/l02_python_bench_v1.json",
    "data/benchmarks/l02_python_bench_v1.json",
)
MODEL_CARD_CANDIDATES = (
    "docs/model_cards/taksh-foundation-111m.md",
    "MODEL_CARD.md",
)
CAPABILITY_EVAL_CANDIDATES = (
    "reports/taksh_capability_evaluation.json",
    "reports/l130_capability_evaluation.json",
)
SAFETENSORS_CANDIDATES = (
    "model.safetensors",
    "export/taksh-foundation-111m/model.safetensors",
)
CORPUS_SNAPSHOT = "artifacts/l072_l04_freeze/snapshot.json"


def _path(rel: str) -> Path:
    return REPO / rel


def _sha256(rel_or_path: str | Path) -> str | None:
    path = _path(rel_or_path) if isinstance(rel_or_path, str) else rel_or_path
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(rel: str) -> dict[str, Any]:
    path = _path(rel)
    return {
        "path": rel,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path),
    }


def _first_existing(candidates: tuple[str, ...]) -> str | None:
    for rel in candidates:
        if _path(rel).is_file():
            return rel
    return None


def _version() -> str:
    text = _path("taksh/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _release_status() -> str:
    text = _path("taksh/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__release_status__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _routes(rel: str) -> list[str]:
    """Extract FastAPI route decorators without importing the server or torch."""
    path = _path(rel)
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    routes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
                continue
            if func.value.id != "app" or func.attr not in {"get", "post", "put", "delete"}:
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                routes.append(f"{func.attr.upper()} {decorator.args[0].value}")
    return sorted(set(routes))


def _tokenizer_state() -> dict[str, Any]:
    path = _path(TOKENIZER)
    state: dict[str, Any] = {
        "path": TOKENIZER,
        "exists": path.is_file(),
        "sha256": _sha256(path),
        "vocab_size": None,
        "status": "BLOCKED",
    }
    if not path.is_file():
        state["detail"] = "canonical tokenizer file is absent"
        return state
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        vocab = doc.get("model", {}).get("vocab", {})
        state["vocab_size"] = len(vocab)
    except (json.JSONDecodeError, OSError) as exc:
        state["detail"] = f"tokenizer could not be read: {type(exc).__name__}"
        return state
    state["status"] = "PASS" if state["vocab_size"] == 5822 else "PARTIAL"
    state["detail"] = (
        "canonical tokenizer, vocab matches the foundation config"
        if state["vocab_size"] == 5822
        else "vocab differs from the 5,822-token foundation binding"
    )
    return state


def _foundation_state() -> dict[str, Any]:
    path = _path(FOUNDATION_CONFIG)
    state: dict[str, Any] = {
        "path": FOUNDATION_CONFIG,
        "exists": path.is_file(),
        "sha256": _sha256(path),
        "id": None,
        "parameters": None,
        "status": "BLOCKED",
    }
    if not path.is_file():
        state["detail"] = "canonical foundation config is absent"
        return state
    try:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        state["id"] = doc.get("id")
    except Exception as exc:  # noqa: BLE001 - config parsing is evidence
        state["detail"] = f"config could not be parsed: {type(exc).__name__}"
        return state

    # Prefer the generated engineering report when it is present and its own
    # config hash matches the current file. Otherwise report no parameter count
    # rather than inventing one.
    report_path = _path("reports/l110_native_model_foundation.json")
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reported = report.get("foundation_config", {})
            state["parameters"] = reported.get("parameters")
            report_hash = reported.get("sha256")
            if report_hash is None:
                report_hash = (
                    report.get("tokenizer", {})
                    .get("fingerprints", {})
                    .get("foundation_config_sha256")
                )
            if report_hash and report_hash != state["sha256"]:
                state["evidence_drift"] = (
                    "reports/l110_native_model_foundation.json records a different "
                    "foundation-config hash; regenerate evidence before publication"
                )
        except (json.JSONDecodeError, OSError):
            pass
    state["status"] = "PASS" if state["id"] == "taksh-foundation-111m" else "PARTIAL"
    state["detail"] = (
        "canonical foundation geometry is committed"
        if state["status"] == "PASS"
        else "foundation config does not match the canonical identifier"
    )
    return state


def _checkpoints() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel, role in CHECKPOINT_CANDIDATES.items():
        path = _path(rel)
        seen.add(rel)
        found.append(
            {
                "path": rel,
                "role": role,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": "NOT_HASHED_WEIGHT_FILE",
            }
        )
    runs = _path("runs")
    if runs.is_dir():
        for path in sorted(runs.rglob("checkpoint_*.pt")):
            rel = path.relative_to(REPO).as_posix()
            if rel in seen:
                continue
            found.append(
                {
                    "path": rel,
                    "role": "discovered_checkpoint",
                    "exists": True,
                    "bytes": path.stat().st_size,
                    "sha256": "NOT_HASHED_WEIGHT_FILE",
                }
            )
    return found


def _frozen_benchmark_state() -> dict[str, Any]:
    for rel in FROZEN_BENCHMARK_CANDIDATES:
        path = _path(rel)
        if path.is_file():
            return {
                "path": rel,
                "exists": True,
                "sha256": _sha256(path),
                "status": "NOT EVALUATED",
                "historical_result": "0/30 preserved; no model has been scored against it",
                "detail": "artifact is present; model capability has not been measured",
            }
    return {
        "path": FROZEN_BENCHMARK_CANDIDATES[0],
        "exists": False,
        "sha256": None,
        "status": "BLOCKED",
        "historical_result": "0/30 preserved; not re-scored",
        "detail": "frozen benchmark artifact is absent on this ref",
    }


def _m7_history() -> dict[str, Any]:
    """Read the historical v1.0.0 M7 outcome without re-running anything."""
    path = _path("reports/M7_OUTCOME.json")
    history: dict[str, Any] = {
        "source": "reports/M7_OUTCOME.json",
        "exists": path.is_file(),
        "outcome": None,
        "locked_syntax": None,
        "functional": None,
        "evidence_level": None,
    }
    if not path.is_file():
        return history
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return history
    locked = doc.get("locked_generation", {})
    attempted = int(locked.get("attempted", 0))
    syntax_passed = round(float(locked.get("syntax_rate", 0.0)) * attempted)
    history.update(
        {
            "outcome": doc.get("outcome"),
            "locked_syntax": f"{syntax_passed}/{attempted} syntax-valid",
            "functional": f"{locked.get('functional_passed', 0)}/"
            f"{locked.get('functional_attempted', 0)} functional",
            "evidence_level": doc.get("evidence_level"),
        }
    )
    return history


def _channel(
    channel_id: str,
    name: str,
    audience: str,
    status: str,
    summary: str,
    *,
    evidence: dict[str, Any] | None = None,
    missing: list[str] | None = None,
    owner_gate: str = "OWNER_AUTHORIZATION",
    first_action: str = "",
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"invalid status for {channel_id}: {status}")
    return {
        "id": channel_id,
        "name": name,
        "audience": audience,
        "status": status,
        "summary": summary,
        "evidence": evidence or {},
        "missing": missing or [],
        "owner_gate": owner_gate,
        "first_action": first_action,
    }


def evaluate() -> dict[str, Any]:
    """Return a deterministic, read-only readiness assessment."""
    version = _version()
    routes = _routes("serving/openai_server.py")
    route_paths = {route.split(" ", 1)[1] for route in routes if " " in route}
    server_surface_complete = {
        "/v1/models",
        "/v1/chat/completions",
        "/health",
    } <= route_paths
    checkpoints = _checkpoints()
    canonical_present = any(
        item["exists"] and item["role"].startswith("canonical_candidate")
        for item in checkpoints
    )
    foundation = _foundation_state()
    tokenizer = _tokenizer_state()
    benchmark = _frozen_benchmark_state()
    m7 = _m7_history()
    corpus = _file_evidence(CORPUS_SNAPSHOT)

    model_card = _first_existing(MODEL_CARD_CANDIDATES)
    capability_eval = _first_existing(CAPABILITY_EVAL_CANDIDATES)
    safetensors_export = _first_existing(SAFETENSORS_CANDIDATES)
    model_available = canonical_present
    # A checkpoint alone is not a Hugging Face model release. Require the model
    # card, capability evidence and a portable safetensors export as well.
    hf_model_ready = all(
        (canonical_present, model_card, capability_eval, safetensors_export)
    )
    model_status = "PASS" if hf_model_ready else "BLOCKED"

    package_evidence = {
        "pyproject": _file_evidence("pyproject.toml"),
        "license": _file_evidence("LICENSE"),
        "notice": _file_evidence("NOTICE"),
        "readme": _file_evidence("README.md"),
        "changelog": _file_evidence("CHANGELOG.md"),
        "release_procedure": _file_evidence("docs/RELEASING.md"),
        "version": version,
        "release_status": _release_status(),
    }

    channels = [
        _channel(
            "github_source_release",
            "GitHub source release",
            "Developers and researchers who build from source",
            "PARTIAL",
            "The v1.0.0 source package was published; a new tag or release from "
            "this branch is not authorized and has not been prepared.",
            evidence={
                "package": package_evidence,
                "historical_release": "v1.0.0 GitHub source release published 2026-10-21",
                "historical_pypi": "blocked on registry credentials",
            },
            missing=[
                "owner authorization for a new tag or GitHub release",
                "verified release commit and fresh release evidence for this branch",
            ],
            owner_gate="OWNER_RELEASE_DECISION",
            first_action="Run scripts/prepare_release.py after the owner authorizes a release.",
        ),
        _channel(
            "pypi",
            "PyPI source package",
            "Python developers installing with pip",
            "PARTIAL",
            "Package metadata, Apache-2.0 licensing and release tooling exist. "
            "No upload is authorized and credentials have not been used.",
            evidence={
                "package": package_evidence,
                "build_tooling": "scripts/prepare_release.py; twine check; isolated wheel install",
                "contains_weights": False,
                "contains_data": False,
            },
            missing=[
                "registry credentials (historical upload was blocked on them)",
                "owner upload authorization",
                "fresh wheel/sdist build and isolated-install evidence for this branch",
            ],
            owner_gate="OWNER_PYPI_DECISION",
            first_action="Build with scripts/prepare_release.py; upload only after explicit owner approval.",
        ),
        _channel(
            "huggingface_model",
            "Hugging Face model repository",
            "Model users (transformers, inference endpoints, hosted providers)",
            model_status,
            (
                "There is no canonical taksh-foundation-111m checkpoint to upload. "
                "The legacy v180_sft checkpoint is geometry-incompatible and must "
                "not be substituted."
                if not canonical_present
                else "A canonical checkpoint exists, but the model card, capability "
                "evidence and portable safetensors export are still required."
            ),
            evidence={
                "foundation": foundation,
                "tokenizer": tokenizer,
                "checkpoints": checkpoints,
                "expected_parameters": 110_658_816,
                "legacy_v180_sft": "GEOMETRY_INCOMPATIBLE (d_model 384 vs 768); never substituted",
                "checkpoint_format": "PyTorch state_dict for the custom QMModel",
                "transformers_compatible": False,
                "inference_endpoint_ready": False,
                "model_card": model_card,
                "capability_evaluation": capability_eval,
                "safetensors_export": safetensors_export,
                "ready": hf_model_ready,
            },
            missing=[
                "canonical taksh-foundation-111m checkpoint",
                "model card with architecture, tokenizer, training data and limits",
                "capability evidence against a frozen benchmark",
                "transformers-compatible export or a custom inference container",
                "weight-license decision and owner upload authorization",
            ],
            owner_gate="OWNER_MODEL_RELEASE_DECISION",
            first_action="Do not upload. First produce and evaluate a canonical checkpoint under owner authorization.",
        ),
        _channel(
            "huggingface_dataset",
            "Hugging Face dataset repository",
            "Researchers who want to reproduce or audit the corpus",
            "BLOCKED",
            "The corpus record is research-only and the snapshot is not present "
            "on this ref. Licence, provenance and PII review are not complete, so "
            "no dataset card or upload can be honest.",
            evidence={
                "frozen_research_corpus": corpus,
                "frozen_corpus_scope": "Python-only research corpus; not a product training corpus",
                "contamination_audit": _file_evidence("data/contamination_audit.json"),
                "licence_gate": _file_evidence("data/verify_licenses.py"),
            },
            missing=[
                "licence and provenance clearance for every source",
                "PII/secret review completion",
                "published-benchmark decontamination evidence",
                "dataset card and owner upload authorization",
            ],
            owner_gate="OWNER_DATA_RELEASE_DECISION",
            first_action="Keep private. Publish only after a documented licence and privacy review.",
        ),
        _channel(
            "openai_compatible_local",
            "Local OpenAI-compatible server",
            "Cline, OpenCode and any OpenAI-compatible client",
            "PARTIAL" if server_surface_complete else "BLOCKED",
            (
                "The OpenAI-compatible surface exists, but --ckpt is required at "
                "startup and there is no canonical checkpoint to load. It cannot "
                "serve a usable Taksh model yet."
                if not canonical_present
                else "The OpenAI-compatible surface exists and a canonical "
                "checkpoint is present. Tool calling and streaming remain "
                "unsupported, so agentic clients are still blocked."
            ),
            evidence={
                "entry_point": "taksh-openai = serving.openai_server:main",
                "routes": routes,
                "surface_complete": server_surface_complete,
                "model_loaded": False,
                "unavailable_behavior": (
                    "GET /v1/models returns an empty list with unavailable_reason; "
                    "POST /v1/chat/completions returns HTTP 503"
                ),
                "checkpoint_argument": "--ckpt (required)",
                "default_bind": "127.0.0.1:8002",
                "chat_completions": "non-streaming only",
                "streaming": "rejected with HTTP 400 (not implemented)",
                "tool_calling": "rejected with HTTP 400 (not implemented)",
            },
            missing=[
                "canonical checkpoint loaded with --ckpt",
                "tool-calling and streaming support for agentic clients",
                "capability evaluation of the served model",
                "security review before exposing beyond localhost",
            ],
            owner_gate="OWNER_RUNTIME_AND_MODEL_AUTHORIZATION",
            first_action="Start the server only when a canonical checkpoint exists; keep it bound to localhost.",
        ),
        _channel(
            "cline",
            "Cline (VS Code coding agent)",
            "Developers using Cline with a custom OpenAI-compatible provider",
            "BLOCKED",
            "Cline can point at an OpenAI-compatible base URL, but Taksh rejects "
            "tool calls. Cline's agentic coding loop therefore cannot run against "
            "Taksh today, even if a checkpoint is loaded.",
            evidence={
                "integration": "OpenAI-compatible base URL",
                "base_url": "http://127.0.0.1:8002/v1",
                "model_identifier": "read from GET /v1/models after the server loads a checkpoint",
                "tool_calling": "not implemented; HTTP 400",
                "requires": ["serving.openai_server", "canonical checkpoint", "tool calling"],
            },
            missing=[
                "canonical checkpoint",
                "tool-calling support or a documented chat-only mode",
                "end-to-end request test from Cline",
                "documented context window and tool-calling limitations",
            ],
            owner_gate="OWNER_INTEGRATION_DECISION",
            first_action="Document the base URL and model id; test only against a canonical checkpoint.",
        ),
        _channel(
            "opencode",
            "OpenCode",
            "Developers using OpenCode with a custom provider",
            "BLOCKED",
            "OpenCode supports custom OpenAI-compatible providers, but Taksh "
            "rejects tool calls. The agentic coding loop cannot run against Taksh "
            "today, even if a checkpoint is loaded.",
            evidence={
                "integration": "OpenCode V2 custom OpenAI-compatible provider in opencode.json",
                "provider_package": "@opencode/ai/providers/openai-compatible",
                "base_url": "http://127.0.0.1:8002/v1",
                "model_identifier": "taksh-foundation-111m (verify with GET /v1/models)",
                "tool_calling": "not implemented; HTTP 400",
                "requires": ["serving.openai_server", "canonical checkpoint", "tool calling"],
            },
            missing=[
                "canonical checkpoint",
                "tool-calling support or a documented chat-only mode",
                "end-to-end request test from OpenCode",
                "documented compatibility and context limits",
            ],
            owner_gate="OWNER_INTEGRATION_DECISION",
            first_action="Document the provider block; test only against a canonical checkpoint.",
        ),
        _channel(
            "openrouter",
            "OpenRouter",
            "Users of a hosted, paid model marketplace",
            "NOT STARTED",
            "OpenRouter lists models served by hosting providers; it does not run "
            "a local checkpoint. Taksh has no hosted endpoint, no pricing/terms "
            "decision and no capability evidence, so no listing can be submitted.",
            evidence={
                "requires_hosted_inference": True,
                "requires_model_or_provider_weights": True,
                "requires_pricing_and_terms": True,
                "requires_capability_evidence": True,
                "submission_status": "none",
            },
            missing=[
                "hosted inference endpoint or provider partnership",
                "commercial terms and pricing decision",
                "capability and safety evidence suitable for a public listing",
                "owner authorization to publish or submit",
            ],
            owner_gate="OWNER_COMMERCIAL_AND_RELEASE_DECISION",
            first_action="Do not submit. Revisit only after a hosted, evaluated model exists.",
        ),
    ]

    report = {
        "schema": "taksh.publication_readiness/1",
        "generated_by": "evaluation/publication_readiness.py",
        "generated_on": "2026-11-12",
        "product": {
            "name": "Taksh",
            "version": version,
            "license": "Apache-2.0",
            "package": "taksh",
        },
        "program": {
            "branch": "feature/taksh-five-version-ga-program",
            "implementation_version": "1.1.0",
            "package_version": version,
            "versions": ["1.1.0", "1.2.0", "1.3.0", "1.4.0", "2.0.0"],
        },
        "policy": {
            "publication_authorized": False,
            "publication_executed": False,
            "network_access": "none",
            "training_authorized": False,
            "compute_authorized": False,
            "model_weights_published": False,
            "datasets_published": False,
            "allowed_statuses": list(ALLOWED_STATUSES),
        },
        "capability": {
            "model_availability": "MODEL_UNAVAILABLE" if not model_available else "MODEL_AVAILABLE",
            "actual_coding_capability": "NOT EVALUATED",
            "frozen_python_benchmark": benchmark,
            "historical_m7": m7,
            "repository_tooling": {
                "status": "PARTIAL",
                "detail": "engineering-qualified indexing, validation and orchestration; no quality claim",
            },
            "karkain": {
                "status": "BLOCKED",
                "state": "DISCOVERED",
                "detail": "KCC 0xC0000135; TAKSH-KARK-BENCH 0 tasks",
            },
            "production_ga": {"status": "NOT STARTED"},
        },
        "channels": channels,
        "readiness_summary": {
            channel["id"]: channel["status"] for channel in channels
        },
        "verdict": {
            "code": "NO_MODEL_OR_DATASET_CHANNEL_READY",
            "source_package": "PARTIAL",
            "model": "BLOCKED",
            "dataset": "BLOCKED",
            "hosted_api": "NOT STARTED",
            "statement": "Taksh is not publishable as a model or dataset, and it "
            "is not ready for agentic coding clients. Only the source package and "
            "a chat-only local OpenAI-compatible surface are prepared, and neither "
            "may be published without explicit owner authorization.",
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="reports/taksh_publication_readiness.json",
        help="report path (default: reports/taksh_publication_readiness.json)",
    )
    args = parser.parse_args(argv)

    report = evaluate()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"product: {report['product']['name']} {report['product']['version']}")
    print(f"model:   {report['capability']['model_availability']}")
    print(f"coding:  {report['capability']['actual_coding_capability']}")
    print("channels:")
    for channel in report["channels"]:
        print(f"  {channel['status']:<12} {channel['name']}")
    print(f"verdict: {report['verdict']['code']}")
    print(f"report:  {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
