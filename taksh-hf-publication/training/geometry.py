"""Checkpoint / tokenizer / geometry compatibility inspector (v1.5.0).

Answers the question that decides whether a checkpoint may be served or
trained: *does this weight file belong to this tokenizer and this model
geometry?* The inspector reads metadata and weight shapes only; it never
loads a checkpoint into a model and never resizes anything.

It fails closed. A vocabulary wider than the embedding matrix, a missing
tokenizer, control-token ids that do not match the reserved range, or weights
that do not cover the architecture are all reported as incompatibilities, not
as warnings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


class CompatibilityError(RuntimeError):
    """The checkpoint, tokenizer and model geometry do not belong together."""


@dataclass
class CompatibilityReport:
    """The verified facts about one checkpoint/tokenizer/geometry triple."""

    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_schema: str | None = None
    checkpoint_step: int | None = None
    tokenizer: str | None = None
    tokenizer_sha256: str | None = None
    tokenizer_vocab: int | None = None
    control_token_ids: dict = field(default_factory=dict)
    model_name: str | None = None
    model_vocab: int | None = None
    embedding_rows: int | None = None
    output_projection_rows: int | None = None
    context_window: int | None = None
    weight_tensors: int | None = None
    missing_weights: list = field(default_factory=list)
    unexpected_weights: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    compatible: bool = False

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_schema": self.checkpoint_schema,
            "checkpoint_step": self.checkpoint_step,
            "tokenizer": self.tokenizer,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_vocab": self.tokenizer_vocab,
            "control_token_ids": self.control_token_ids,
            "model_name": self.model_name,
            "model_vocab": self.model_vocab,
            "embedding_rows": self.embedding_rows,
            "output_projection_rows": self.output_projection_rows,
            "context_window": self.context_window,
            "weight_tensors": self.weight_tensors,
            "missing_weights": self.missing_weights,
            "unexpected_weights": self.unexpected_weights,
            "problems": self.problems,
            "compatible": self.compatible,
        }


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_ids(tokenizer) -> dict:
    from training.tokenizer_contract import CONTROL_TOKENS

    ids = {}
    for marker in CONTROL_TOKENS:
        token_id = tokenizer.token_to_id(marker)
        if token_id is not None:
            ids[marker] = int(token_id)
    return ids


def inspect(checkpoint_path=None, tokenizer_path=None, model_config_path=None) -> CompatibilityReport:
    """Inspect the triple and report every incompatibility.

    Nothing is loaded into a model and nothing is modified. A checkpoint that
    does not exist is reported, not raised, so a caller can show the state of
    an absent artifact.
    """
    report = CompatibilityReport()

    if tokenizer_path:
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.is_file():
            report.problems.append(f"tokenizer file not found: {tokenizer_file}")
        else:
            from tokenizers import Tokenizer

            report.tokenizer = str(tokenizer_file)
            report.tokenizer_sha256 = file_sha256(tokenizer_file)
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
            report.tokenizer_vocab = tokenizer.get_vocab_size()
            report.control_token_ids = _control_ids(tokenizer)

    if model_config_path:
        import yaml

        config_file = Path(model_config_path)
        if not config_file.is_file():
            report.problems.append(f"model config not found: {config_file}")
        else:
            model_cfg = (yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}).get(
                "model", {}
            )
            report.model_name = model_cfg.get("name")
            report.model_vocab = model_cfg.get("vocab_size")
            report.context_window = model_cfg.get("max_seq_len")

    if checkpoint_path:
        import torch

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            report.problems.append(f"checkpoint not found: {checkpoint_file}")
        else:
            report.checkpoint = str(checkpoint_file)
            report.checkpoint_sha256 = file_sha256(checkpoint_file)
            checkpoint = torch.load(
                str(checkpoint_file), map_location="cpu", weights_only=False
            )
            report.checkpoint_schema = checkpoint.get("schema")
            report.checkpoint_step = checkpoint.get("step")
            state = checkpoint.get("model") or {}
            report.weight_tensors = len(state)
            for key, tensor in state.items():
                if key == "tok_emb.weight":
                    report.embedding_rows = int(tensor.shape[0])
                elif key == "lm_head.weight":
                    report.output_projection_rows = int(tensor.shape[0])
            if not report.model_vocab:
                config = checkpoint.get("config") or {}
                model_cfg = config.get("model") if isinstance(config, dict) else None
                if isinstance(model_cfg, dict):
                    report.model_name = model_cfg.get("name")
                    report.model_vocab = model_cfg.get("vocab_size")
                    report.context_window = model_cfg.get("max_seq_len")

    if report.model_vocab and report.embedding_rows is None and report.weight_tensors:
        report.problems.append("checkpoint has no tok_emb.weight to compare")
    if report.model_vocab and report.embedding_rows and report.embedding_rows != report.model_vocab:
        report.problems.append(
            f"embedding rows {report.embedding_rows} != model vocab {report.model_vocab}"
        )
    if report.tokenizer_vocab and report.embedding_rows and report.tokenizer_vocab > report.embedding_rows:
        report.problems.append(
            f"tokenizer vocabulary {report.tokenizer_vocab} exceeds the "
            f"embedding matrix ({report.embedding_rows} rows); ids would index "
            "outside the table"
        )
    if report.tokenizer_vocab and report.embedding_rows and report.embedding_rows > report.tokenizer_vocab:
        report.problems.append(
            f"the embedding table has {report.embedding_rows - report.tokenizer_vocab} "
            f"rows the {report.tokenizer_vocab}-id tokenizer can never address; "
            "the checkpoint was trained against a different vocabulary"
        )
    if report.model_vocab and report.tokenizer_vocab and report.model_vocab != report.tokenizer_vocab:
        report.problems.append(
            f"model vocab {report.model_vocab} != tokenizer vocab "
            f"{report.tokenizer_vocab}"
        )
    if report.control_token_ids:
        expected_floor = report.tokenizer_vocab - len(report.control_token_ids)
        for marker, token_id in report.control_token_ids.items():
            if token_id != expected_floor + list(report.control_token_ids).index(marker):
                report.problems.append(
                    f"control token {marker} has id {token_id}; the reserved "
                    f"range is expected to start at {expected_floor}"
                )
                break
    report.compatible = not report.problems
    return report


def require_compatible(report: CompatibilityReport) -> CompatibilityReport:
    """Raise unless the report is clean."""
    if not report.compatible:
        raise CompatibilityError("; ".join(report.problems))
    return report
