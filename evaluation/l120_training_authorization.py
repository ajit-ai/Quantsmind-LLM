"""Build the L1.2.0 training-authorization packet and readiness report.

The packet is a request for authorization, not an authorization. This module
reads committed metadata only: model config, tokenizer, licence evidence,
corpus manifest, token metadata and checkpoint path. It never trains, acquires
data, installs packages, contacts a provider, creates a checkpoint or
publishes anything.

Run:

    python -m evaluation.l120_training_authorization
    python -m evaluation.l120_training_authorization --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PACKET = "configs/roadmap/l120_training_authorization.yaml"
DEFAULT_OUT = "reports/l120_training_authorization.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")
DECISION_STATUSES = ("GRANTED", "NOT GRANTED", "DECLINED")
REQUIRED_GATES = ("model", "tokenizer", "corpus_licences", "corpus_scale", "corpus_payload", "compute", "checkpoint")
REQUIRED_DECISIONS = ("D-TRAIN-1", "D-COMPUTE-2", "D-CORPUS-3", "D-CHECKPOINT-4")
AUTHORIZATION_POLICY_KEYS = (
    "training_authorized",
    "compute_authorized",
    "dataset_acquisition_authorized",
    "checkpoint_creation_authorized",
    "capability_evaluation_authorized",
    "publication_authorized",
    "training_ci_enabled",
)


class TrainingAuthorizationError(RuntimeError):
    """The packet is malformed or claims an unauthorized capability."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise TrainingAuthorizationError(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise TrainingAuthorizationError(
            f"unreadable JSON in {path}: {type(exc).__name__}"
        ) from exc


def load_packet(path: str | Path = PACKET) -> dict[str, Any]:
    packet_path = _path(path)
    if not packet_path.is_file():
        raise TrainingAuthorizationError(f"packet not found: {packet_path}")
    document = yaml.safe_load(packet_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise TrainingAuthorizationError(f"packet is not a mapping: {packet_path}")
    return document


def _validate(packet: dict[str, Any]) -> None:
    errors: list[str] = []
    if packet.get("schema") != "taksh.l120_training_authorization/1":
        errors.append(f"unexpected schema: {packet.get('schema')!r}")
    if packet.get("program_label") != "1.2.0":
        errors.append(f"program_label must be '1.2.0', got {packet.get('program_label')!r}")
    if packet.get("status") not in ALLOWED_STATUSES:
        errors.append(f"invalid packet status: {packet.get('status')!r}")

    policy = packet.get("policy") or {}
    for key in AUTHORIZATION_POLICY_KEYS:
        if not isinstance(policy.get(key), bool):
            errors.append(f"policy.{key} must be a boolean")

    for section in ("model", "corpus", "compute", "checkpoint"):
        if not isinstance(packet.get(section), dict):
            errors.append(f"missing mapping: {section}")

    decisions = packet.get("owner_decisions") or []
    if not decisions:
        errors.append("owner_decisions must not be empty")
    granted: set[str] = set()
    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"owner_decisions[{index}] is not a mapping")
            continue
        decision_id = str(decision.get("id", ""))
        if not decision_id:
            errors.append(f"owner_decisions[{index}] is missing id")
        if decision_id in seen:
            errors.append(f"duplicate owner decision: {decision_id}")
        seen.add(decision_id)
        if decision.get("status") not in DECISION_STATUSES:
            errors.append(
                f"owner decision {decision_id or index} has invalid status "
                f"{decision.get('status')!r}"
            )
        if decision.get("status") == "GRANTED":
            granted.add(decision_id)

    if not packet.get("acceptance_criteria"):
        errors.append("acceptance_criteria must not be empty")

    # Fail closed: an authorization flag may not be true while its decision is
    # absent or ungranted. No code change may grant an owner decision.
    decision_for_flag = {
        "training_authorized": "D-TRAIN-1",
        "compute_authorized": "D-COMPUTE-2",
        "dataset_acquisition_authorized": "D-CORPUS-3",
        "checkpoint_creation_authorized": "D-CHECKPOINT-4",
        "capability_evaluation_authorized": "D-EVAL-5",
    }
    for flag, decision_id in decision_for_flag.items():
        if policy.get(flag) is True and decision_id not in granted:
            errors.append(
                f"policy.{flag} is true but {decision_id} is not GRANTED"
            )

    if errors:
        raise TrainingAuthorizationError("; ".join(errors))


def _gate(status: str, **details: Any) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise TrainingAuthorizationError(f"invalid gate status: {status}")
    return {"status": status, **details}


def _inspect_model(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = packet["model"]
    config_path = _path(model["config"])
    tokenizer_path = _path(model["tokenizer"])
    evidence_path = _path(model["foundation_evidence"])
    expected_parameters = int(model["expected_parameters"])
    expected_vocab = int(model["expected_vocab_size"])

    config_doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tokenizer_doc = _read_json(tokenizer_path)
    evidence = _read_json(evidence_path)

    recorded = evidence.get("tokenizer", {}).get("fingerprints", {})
    recorded_config_lf = recorded.get("foundation_config_sha256")
    recorded_tokenizer = recorded.get("tokenizer_sha256")
    recorded_parameters = evidence.get("foundation_config", {}).get("parameters")
    recorded_matches = evidence.get("foundation_config", {}).get("matches_expected")

    config_lf = _sha256_lf(config_path)
    tokenizer_sha = _sha256(tokenizer_path)
    vocab_size = len(tokenizer_doc.get("model", {}).get("vocab", {}))

    model_ok = (
        config_doc.get("id") == model["id"]
        and config_doc.get("model", {}).get("vocab_size") == expected_vocab
        and recorded_parameters == expected_parameters
        and recorded_matches is True
        and recorded_config_lf == config_lf
    )
    tokenizer_ok = (
        vocab_size == expected_vocab
        and recorded_tokenizer == tokenizer_sha
        and config_doc.get("tokenizer", {}).get("path") == model["tokenizer"]
    )

    model_gate = _gate(
        "PASS" if model_ok else "BLOCKED",
        id=config_doc.get("id"),
        config=_relative(config_path),
        sha256_raw=_sha256(config_path),
        sha256_lf=config_lf,
        recorded_sha256_lf=recorded_config_lf,
        parameters=recorded_parameters,
        expected_parameters=expected_parameters,
        parameters_match=recorded_parameters == expected_parameters,
        evidence=_relative(evidence_path),
    )
    tokenizer_gate = _gate(
        "PASS" if tokenizer_ok else "BLOCKED",
        path=_relative(tokenizer_path),
        sha256_raw=tokenizer_sha,
        vocab_size=vocab_size,
        expected_vocab_size=expected_vocab,
        recorded_sha256=recorded_tokenizer,
    )
    return model_gate, tokenizer_gate


def _inspect_licences(packet: dict[str, Any]) -> dict[str, Any]:
    corpus = packet["corpus"]
    evidence_path = _path(corpus["licence_evidence"])
    evidence = _read_json(evidence_path)
    sources = evidence.get("sources") or []

    approved = {"verified", "verified_with_quarantine"}
    checked = len(sources)
    verified = [source for source in sources if source.get("result") == "verified"]
    conditional = [
        source
        for source in sources
        if source.get("result") == "verified_with_quarantine"
    ]
    unresolved = [
        str(source.get("id"))
        for source in sources
        if source.get("result") not in approved or source.get("needs_human_review") is True
    ]
    required_unresolved = int(corpus["required_unresolved_licences"])
    ok = checked > 0 and len(unresolved) <= required_unresolved

    return _gate(
        "PASS" if ok else "BLOCKED",
        evidence=_relative(evidence_path),
        checked=checked,
        verified=len(verified),
        verified_with_quarantine=len(conditional),
        unresolved=len(unresolved),
        unresolved_ids=unresolved,
        required_unresolved=required_unresolved,
        policy=(
            "verified_with_quarantine remains conditional on the recorded "
            "exclusions and is not a human licence ruling."
        ),
    )


def _inspect_corpus(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = packet["corpus"]
    manifest_path = _path(corpus["manifest"])
    metadata_path = _path(corpus["token_metadata"])
    manifest = _read_json(manifest_path)
    metadata = _read_json(metadata_path)

    observed_tokens = int(metadata.get("num_tokens", 0))
    observed_documents = int(metadata.get("num_documents", 0))
    observed_repositories = len(manifest.get("sources") or [])
    required_tokens = int(corpus["required_tokens"])
    required_repositories = int(corpus["required_repositories"])

    scale_ok = (
        observed_tokens >= required_tokens
        and observed_repositories >= required_repositories
    )
    scale_gate = _gate(
        "PASS" if scale_ok else "BLOCKED",
        manifest=_relative(manifest_path),
        token_metadata=_relative(metadata_path),
        observed_tokens=observed_tokens,
        required_tokens=required_tokens,
        token_shortfall=max(0, required_tokens - observed_tokens),
        observed_repositories=observed_repositories,
        required_repositories=required_repositories,
        repository_shortfall=max(0, required_repositories - observed_repositories),
        observed_documents=observed_documents,
        corpus_sha256=metadata.get("corpus_sha256"),
        tokenizer_sha256=metadata.get("tokenizer_sha256"),
    )

    payload_path = _path(corpus["payload"])
    tokens_path = _path(corpus["tokens"])
    snapshot_path = _path(corpus["frozen_snapshot_required"])
    data_card_path = _path(corpus["data_card"])
    present = {
        "manifest": manifest_path.is_file(),
        "token_metadata": metadata_path.is_file(),
        "payload": payload_path.is_file(),
        "tokens": tokens_path.is_file(),
        "frozen_snapshot": snapshot_path.is_file(),
        "data_card": data_card_path.is_file(),
    }
    payload_ok = all(present.values())
    payload_gate = _gate(
        "PASS" if payload_ok else "BLOCKED",
        **{f"{key}_present": value for key, value in present.items()},
        payload=_relative(payload_path),
        tokens=_relative(tokens_path),
        frozen_snapshot=_relative(snapshot_path),
        note=(
            "Payload and token files are gitignored. Their absence on this ref "
            "means no governed training materialization exists here."
        ),
    )
    return scale_gate, payload_gate


def _inspect_compute(packet: dict[str, Any]) -> dict[str, Any]:
    compute = packet["compute"]
    authorized = compute.get("authorization") == "GRANTED"
    budget = float(compute.get("max_budget_usd", 0))
    gpu_hours = float(compute.get("gpu_hours_authorized", 0))
    ok = authorized and budget > 0 and gpu_hours > 0
    return _gate(
        "PASS" if ok else "BLOCKED",
        authorization=compute.get("authorization"),
        max_budget_usd=budget,
        gpu_hours_authorized=gpu_hours,
        local_host_profile=compute.get("local_host_profile"),
        prohibited=list(compute.get("prohibited") or []),
    )


def _inspect_checkpoint(packet: dict[str, Any]) -> dict[str, Any]:
    checkpoint = packet["checkpoint"]
    target = _path(checkpoint["target"])
    decisions = {
        str(decision.get("id")): decision.get("status")
        for decision in packet.get("owner_decisions") or []
    }
    creation_authorized = (
        packet["policy"].get("checkpoint_creation_authorized") is True
        and decisions.get("D-CHECKPOINT-4") == "GRANTED"
    )
    exists = target.is_dir() or target.is_file()
    ok = exists and creation_authorized
    return _gate(
        "PASS" if ok else "BLOCKED",
        id=checkpoint["id"],
        target=_relative(target),
        exists=exists,
        creation_authorized=creation_authorized,
        existing_artifact=checkpoint.get("existing_artifact"),
        forbidden_artifacts=list(packet["model"].get("forbidden_artifacts") or []),
    )


def inspect(path: str | Path = PACKET) -> dict[str, Any]:
    """Return a deterministic authorization report; never writes state."""
    packet = load_packet(path)
    _validate(packet)

    model_gate, tokenizer_gate = _inspect_model(packet)
    corpus_scale_gate, corpus_payload_gate = _inspect_corpus(packet)
    gates = {
        "model": model_gate,
        "tokenizer": tokenizer_gate,
        "corpus_licences": _inspect_licences(packet),
        "corpus_scale": corpus_scale_gate,
        "corpus_payload": corpus_payload_gate,
        "compute": _inspect_compute(packet),
        "checkpoint": _inspect_checkpoint(packet),
    }

    decisions = [
        {
            "id": decision["id"],
            "title": decision["title"],
            "status": decision["status"],
            "evidence_required": decision["evidence_required"],
        }
        for decision in packet["owner_decisions"]
    ]
    granted = {
        decision["id"]
        for decision in decisions
        if decision["status"] == "GRANTED"
    }

    policy = packet["policy"]
    authorization_flags_clear = all(
        policy.get(key) is False
        for key in (
            "training_authorized",
            "compute_authorized",
            "dataset_acquisition_authorized",
            "checkpoint_creation_authorized",
            "capability_evaluation_authorized",
            "publication_authorized",
            "training_ci_enabled",
        )
    )
    gates_pass = all(gates[name]["status"] == "PASS" for name in REQUIRED_GATES)
    decisions_pass = all(decision_id in granted for decision_id in REQUIRED_DECISIONS)
    ready_to_train = (
        gates_pass
        and decisions_pass
        and policy.get("training_authorized") is True
        and policy.get("compute_authorized") is True
        and policy.get("dataset_acquisition_authorized") is True
        and policy.get("checkpoint_creation_authorized") is True
    )

    overall = "PASS" if ready_to_train else "BLOCKED"
    if overall not in ALLOWED_STATUSES:
        raise TrainingAuthorizationError(f"invalid overall status: {overall}")

    blockers = [
        f"{name}: {gates[name]['status']}"
        for name in REQUIRED_GATES
        if gates[name]["status"] != "PASS"
    ]
    blockers.extend(
        f"{decision['id']} {decision['title']}: {decision['status']}"
        for decision in decisions
        if decision["status"] != "GRANTED"
    )

    return {
        "schema": "taksh.l120_training_authorization_report/1",
        "generated_by": "evaluation/l120_training_authorization.py",
        "program_label": packet["program_label"],
        "packet": _relative(_path(path)),
        "status": overall,
        "verdict": "TRAINING_AUTHORIZED" if ready_to_train else "TRAINING_NOT_AUTHORIZED",
        "ready_to_train": ready_to_train,
        "training_executed": False,
        "checkpoint_created": False,
        "authorization": {
            "preparation_only": policy.get("preparation_only"),
            "training_authorized": policy.get("training_authorized"),
            "compute_authorized": policy.get("compute_authorized"),
            "dataset_acquisition_authorized": policy.get("dataset_acquisition_authorized"),
            "checkpoint_creation_authorized": policy.get("checkpoint_creation_authorized"),
            "capability_evaluation_authorized": policy.get("capability_evaluation_authorized"),
            "publication_authorized": policy.get("publication_authorized"),
            "training_ci_enabled": policy.get("training_ci_enabled"),
            "authorization_flags_clear": authorization_flags_clear,
        },
        "gates": gates,
        "blockers": blockers,
        "acceptance_criteria": packet["acceptance_criteria"],
        "owner_decisions": decisions,
        "policy": {
            "no_training": True,
            "no_checkpoint": True,
            "no_dataset_acquisition": True,
            "no_network_access": True,
            "no_publication": True,
            "no_capability_claim": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        default=PACKET,
        help=f"authorization packet (default: {PACKET})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = inspect(args.packet)
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"program label: {report['program_label']}")
        print(f"status: {report['status']}")
        print(f"ready to train: {report['ready_to_train']}")
        print(f"verdict: {report['verdict']}")
        print("gates:")
        for name, gate in report["gates"].items():
            print(f"  {gate['status']:<10} {name}")
        print(f"blockers: {len(report['blockers'])}")
        print(f"report: {out_path}")
    return 0 if report["ready_to_train"] else 1


if __name__ == "__main__":
    sys.exit(main())
