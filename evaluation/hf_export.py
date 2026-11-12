"""TAKSH Hugging Face export preparation (no upload).

Converts a **pre-existing** canonical checkpoint into a portable export
directory:

    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    README.md
    SHA256SUMS
    taksh_export_manifest.json

The tool is deliberately dry-run by default. It never trains, creates a
checkpoint, downloads data, contacts a network service, or uploads anything.
It refuses a missing or geometry-incompatible checkpoint instead of producing a
misleading artifact.

Run:

    python -m evaluation.hf_export --checkpoint <path> --out dist/hf/taksh-foundation-111m
    python -m evaluation.hf_export --checkpoint <path> --out <dir> --execute
    python -m evaluation.hf_export --verify <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")

DEFAULT_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"
DEFAULT_MODEL_CONFIG = "configs/model/foundation_111m.yaml"
DEFAULT_OUTPUT = "dist/hf/taksh-foundation-111m"

EXPORT_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
    "SHA256SUMS",
    "taksh_export_manifest.json",
)

FILE_KINDS = {
    "model.safetensors": "weights",
    "config.json": "config",
    "tokenizer.json": "tokenizer",
    "tokenizer_config.json": "tokenizer_config",
    "special_tokens_map.json": "tokenizer_config",
    "README.md": "model_card",
    "SHA256SUMS": "checksums",
    "taksh_export_manifest.json": "manifest",
}

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class ExportError(RuntimeError):
    """The export cannot be prepared or written safely."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_model_config(path: str | Path = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    """Load the ``model:`` section and validate the fields an export needs."""
    import yaml

    config_path = _path(path)
    if not config_path.is_file():
        raise ExportError(f"model config not found: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_cfg = document.get("model", document)
    required = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "ffn_dim",
        "max_seq_len",
    )
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ExportError(
            "model config is missing required fields: " + ", ".join(missing)
        )
    return model_cfg


def load_foundation_document(path: str | Path = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    import yaml

    config_path = _path(path)
    if not config_path.is_file():
        raise ExportError(f"model config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def count_parameters(model_cfg: dict[str, Any]) -> int | None:
    """Count parameters with the real model on the meta device (no weights)."""
    try:
        import torch
        from training.model import QMModel

        with torch.device("meta"):
            model = QMModel(model_cfg)
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:  # noqa: BLE001 - a count is evidence, not a hard failure
        return None


def load_tokenizer_info(path: str | Path = DEFAULT_TOKENIZER) -> dict[str, Any]:
    tokenizer_path = _path(path)
    if not tokenizer_path.is_file():
        raise ExportError(f"tokenizer not found: {tokenizer_path}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = {
        name: tokenizer.token_to_id(name)
        for name in SPECIAL_TOKENS
    }
    return {
        "path": _relative(tokenizer_path),
        "sha256": _sha256_file(tokenizer_path),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "special_token_ids": {name: int(value) for name, value in ids.items() if value is not None},
    }


def to_hf_config(
    model_cfg: dict[str, Any],
    *,
    foundation_document: dict[str, Any] | None = None,
    tokenizer_info: dict[str, Any] | None = None,
    parameter_count: int | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Map the native Taksh geometry to a JSON config.

    The result is a portable description, not a ``transformers`` auto-loading
    claim. ``trust_remote_code`` is not a substitute for the missing Taksh
    modeling files.
    """
    document = foundation_document or {}
    tokenizer_info = tokenizer_info or {}
    special_ids = tokenizer_info.get("special_token_ids", {})
    config: dict[str, Any] = {
        "architectures": ["QMModel"],
        "model_type": "taksh_qm",
        "hidden_size": int(model_cfg["d_model"]),
        "intermediate_size": int(model_cfg["ffn_dim"]),
        "num_hidden_layers": int(model_cfg["n_layers"]),
        "num_attention_heads": int(model_cfg["n_heads"]),
        "num_key_value_heads": int(model_cfg["n_kv_heads"]),
        "max_position_embeddings": int(model_cfg["max_seq_len"]),
        "vocab_size": int(model_cfg["vocab_size"]),
        "rope_theta": float(model_cfg.get("rope_theta", 10000.0)),
        "rms_norm_eps": float(model_cfg.get("norm_eps", 1e-5)),
        "tie_word_embeddings": bool(model_cfg.get("tie_embeddings", True)),
        "torch_dtype": "float32",
        "taksh": {
            "schema": "taksh.hf_config/1",
            "product_id": document.get("id"),
            "model_name": model_cfg.get("name"),
            "parameter_count": parameter_count,
            "checkpoint_sha256": checkpoint_sha256,
            "custom_architecture": True,
            "transformers_auto_load": False,
            "load_with": "evaluation.hf_export.load_export",
        },
    }
    for name in SPECIAL_TOKENS:
        token_id = special_ids.get(name)
        if token_id is not None:
            config[f"{name.strip('<>')}_token_id"] = int(token_id)
    return config


def native_model_config_from_hf(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Map an export ``config.json`` back to the native ``QMModel`` fields.

    This is the inverse of :func:`to_hf_config`. It rejects a config that does not
    contain enough geometry to construct the model instead of guessing.
    """
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
        "vocab_size",
    )
    missing = [key for key in required if key not in hf_config]
    if missing:
        raise ExportError(
            "export config is missing required fields: " + ", ".join(missing)
        )

    taksh = hf_config.get("taksh") or {}
    n_heads = int(hf_config["num_attention_heads"])
    return {
        "name": taksh.get("model_name") or hf_config.get("model_type", "taksh"),
        "vocab_size": int(hf_config["vocab_size"]),
        "d_model": int(hf_config["hidden_size"]),
        "n_layers": int(hf_config["num_hidden_layers"]),
        "n_heads": n_heads,
        "n_kv_heads": int(hf_config.get("num_key_value_heads", n_heads)),
        "ffn_dim": int(hf_config["intermediate_size"]),
        "max_seq_len": int(hf_config["max_position_embeddings"]),
        "rope_theta": float(hf_config.get("rope_theta", 10000.0)),
        "norm_eps": float(hf_config.get("rms_norm_eps", 1e-5)),
        "tie_embeddings": bool(hf_config.get("tie_word_embeddings", True)),
    }


def load_export(output_dir: str | Path, *, device: str = "cpu"):
    """Load an export directory into the native ``QMModel``.

    The export is a portable artifact, not a ``transformers`` checkpoint. This
    loader reads ``config.json`` and ``model.safetensors`` and applies the same
    strict parameter-name/shape checks as checkpoint loading.
    """
    from safetensors.torch import load_file
    from training.checkpoint import load_model_weights
    from training.model import QMModel

    output_path = _path(output_dir)
    config_path = output_path / "config.json"
    weights_path = output_path / "model.safetensors"
    if not config_path.is_file():
        raise ExportError(f"export config not found: {config_path}")
    if not weights_path.is_file():
        raise ExportError(f"export weights not found: {weights_path}")

    try:
        hf_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ExportError(f"export config is unreadable: {type(exc).__name__}") from exc
    model_cfg = native_model_config_from_hf(hf_config)

    tensors = load_file(str(weights_path), device="cpu")
    if hf_config.get("tie_word_embeddings") and "lm_head.weight" not in tensors:
        # safetensors cannot store two names for one storage, so the export keeps
        # one copy of the tied embedding. Restore the alias expected by QMModel.
        embedding = tensors.get("tok_emb.weight")
        if embedding is None:
            raise ExportError(
                "tied export has neither lm_head.weight nor tok_emb.weight"
            )
        tensors["lm_head.weight"] = embedding

    model = QMModel(model_cfg)
    load_model_weights(model, tensors, source=str(weights_path))
    return model.to(device).eval(), model_cfg


def build_model_card(
    *,
    model_cfg: dict[str, Any],
    hf_config: dict[str, Any],
    checkpoint_path: str,
    checkpoint_sha256: str | None,
    tokenizer_info: dict[str, Any],
    compatibility: dict[str, Any],
    evaluation_path: str | None = None,
    evaluation_sha256: str | None = None,
    weight_license: str | None = None,
) -> str:
    """Build an honest model card. It never invents a capability score."""
    product_id = hf_config.get("taksh", {}).get("product_id") or "taksh-foundation-111m"
    parameters = hf_config.get("taksh", {}).get("parameter_count")
    parameter_text = f"{parameters:,}" if isinstance(parameters, int) else "not computed"
    license_lines = (
        [f"license: {weight_license}"]
        if weight_license
        else ["license: other", "license_name: taksh-weights-license-pending"]
    )
    front_matter = "\n".join(
        [
            "---",
            *license_lines,
            "library_name: taksh",
            "tags:",
            "  - taksh",
            "  - code",
            "  - custom-architecture",
            "---",
        ]
    )

    evaluation_section = [
        "## Evaluation",
        "",
        "**Actual coding capability: NOT EVALUATED.**",
        "",
        "No capability benchmark is bundled with this export. Historical negative",
        "results are preserved for context, not presented as this model's score:",
        "",
        "| Test | Historical result | Status |",
        "|---|---|---|",
        "| M7 locked syntax | 0/40 syntax-valid | historical; not re-run |",
        "| M7 functional probes | 0/8 passed | historical; not re-run |",
        "| Frozen Python benchmark | 0/30 | preserved; not re-scored |",
        "",
    ]
    if evaluation_path and evaluation_sha256:
        evaluation_section += [
            "An evaluation report was supplied with this export:",
            "",
            f"- Path: `{evaluation_path}`",
            f"- SHA-256: `{evaluation_sha256}`",
            "",
            "Read the report for its protocol, raw outputs and limitations. This",
            "card does not summarize or reinterpret it.",
            "",
        ]
    else:
        evaluation_section += [
            "No evaluation report was supplied. Do not infer capability from the",
            "existence of weights, a successful export, or an HTTP 200 from a",
            "serving endpoint.",
            "",
        ]

    return "\n".join(
        [
            front_matter,
            "",
            f"# {product_id}",
            "",
            "> **Status:** portable export prepared from a pre-existing checkpoint.",
            "> **Actual coding capability: NOT EVALUATED.**",
            "",
            "## Model summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Product id | `{product_id}` |",
            f"| Native model name | `{model_cfg.get('name', 'unknown')}` |",
            f"| Parameters | {parameter_text} |",
            f"| Architecture | Taksh `QMModel` (custom PyTorch) |",
            f"| Hidden size | {hf_config.get('hidden_size')} |",
            f"| Layers | {hf_config.get('num_hidden_layers')} |",
            f"| Attention | {hf_config.get('num_attention_heads')} query / "
            f"{hf_config.get('num_key_value_heads')} KV heads |",
            f"| FFN size | {hf_config.get('intermediate_size')} (SwiGLU) |",
            f"| Context | {hf_config.get('max_position_embeddings')} tokens |",
            f"| Vocabulary | {hf_config.get('vocab_size')} |",
            f"| Tied embeddings | {hf_config.get('tie_word_embeddings')} |",
            f"| Tokenizer SHA-256 | `{tokenizer_info.get('sha256')}` |",
            f"| Checkpoint SHA-256 | `{checkpoint_sha256}` |",
            "",
            "## Architecture",
            "",
            "RMSNorm, RoPE, grouped-query attention and SwiGLU, implemented in",
            "`training/model.py`. This is **not** a `transformers` architecture.",
            "`transformers.from_pretrained` will not load it without a custom",
            "modeling implementation or a separate conversion.",
            "",
            "## Load",
            "",
            "Use the Taksh export loader, not `AutoModelForCausalLM`:",
            "",
            "```python",
            "from evaluation.hf_export import load_export",
            "model, cfg = load_export('<export-directory>', device='cpu')",
            "```",
            "",
            "The loader reads `config.json` and `model.safetensors` and applies the",
            "same strict parameter checks as the native checkpoint loader.",
            "",
            "## Compatibility evidence",
            "",
            "```json",
            json.dumps(compatibility, indent=2)[:4000],
            "```",
            "",
            *evaluation_section,
            "## Intended use",
            "",
            "- Local research and engineering inspection.",
            "- Architecture, tokenizer and export validation.",
            "- Reproducibility work with explicit owner authorization.",
            "",
            "## Limitations",
            "",
            "- No claim of code-generation quality; capability is **NOT EVALUATED**.",
            "- No tool-calling support in the current serving surface.",
            "- No streaming support in the current serving surface.",
            "- No production, security, legal or safety qualification.",
            "- Chat/instruction behaviour is not established by this export.",
            "",
            "## License",
            "",
            "Source code is Apache-2.0. The weight license is a separate owner",
            "decision; this card does not assume the code license applies to",
            "weights.",
            "",
            "## Provenance",
            "",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Checkpoint SHA-256: `{checkpoint_sha256}`",
            f"- Tokenizer: `{tokenizer_info.get('path')}`",
            f"- Tokenizer SHA-256: `{tokenizer_info.get('sha256')}`",
            "",
            "This export was prepared locally. Nothing was uploaded.",
            "",
        ]
    )


@dataclass
class ExportPlan:
    status: str
    checkpoint: str
    output_dir: str
    dry_run: bool
    model_config: dict[str, Any]
    hf_config: dict[str, Any]
    tokenizer_info: dict[str, Any]
    compatibility: dict[str, Any]
    files: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_card: str = ""
    config_json: str = ""
    evaluation_path: str | None = None
    evaluation_sha256: str | None = None
    weight_license: str | None = None
    generated_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taksh.hf_export_plan/1",
            "status": self.status,
            "generated_at": self.generated_at,
            "checkpoint": self.checkpoint,
            "output_dir": self.output_dir,
            "dry_run": self.dry_run,
            "model_config": self.model_config,
            "hf_config": self.hf_config,
            "tokenizer": self.tokenizer_info,
            "compatibility": self.compatibility,
            "files": self.files,
            "missing": self.missing,
            "warnings": self.warnings,
            "evaluation": {
                "path": self.evaluation_path,
                "sha256": self.evaluation_sha256,
                "status": "NOT EVALUATED" if not self.evaluation_path else "REPORT_SUPPLIED",
            },
            "weight_license": self.weight_license,
            "publication": {
                "uploaded": False,
                "network_access": "none",
                "authorized": False,
            },
            "error": self.error,
        }


def plan_export(
    checkpoint: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    tokenizer: str | Path = DEFAULT_TOKENIZER,
    model_config: str | Path = DEFAULT_MODEL_CONFIG,
    evaluation: str | Path | None = None,
    weight_license: str | None = None,
) -> ExportPlan:
    """Build a validated export plan. This never writes weight files."""
    generated_at = _utc_now()
    checkpoint_path = _path(checkpoint)
    output_path = _path(output_dir)
    tokenizer_path = _path(tokenizer)
    model_config_path = _path(model_config)

    missing: list[str] = []
    warnings: list[str] = []
    compatibility: dict[str, Any] = {}
    model_cfg: dict[str, Any] = {}
    tokenizer_info: dict[str, Any] = {}
    hf_config: dict[str, Any] = {}
    error: str | None = None

    try:
        model_cfg = load_model_config(model_config_path)
        foundation_document = load_foundation_document(model_config_path)
    except ExportError as exc:
        error = str(exc)
        missing.append("valid model config")
        foundation_document = {}

    try:
        tokenizer_info = load_tokenizer_info(tokenizer_path)
    except Exception as exc:  # noqa: BLE001 - report an unusable tokenizer
        error = error or str(exc)
        missing.append("tokenizer")

    if not checkpoint_path.is_file():
        missing.append("checkpoint")
        error = error or f"checkpoint not found: {checkpoint_path}"
    else:
        from training import geometry

        report = geometry.inspect(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path if tokenizer_path.is_file() else None,
            model_config_path=model_config_path if model_config_path.is_file() else None,
        )
        compatibility = report.to_dict()
        if not report.compatible:
            missing.extend(report.problems or ["checkpoint/tokenizer/geometry mismatch"])
            error = error or "checkpoint is not compatible with the canonical geometry"

    parameter_count = count_parameters(model_cfg) if model_cfg else None
    checkpoint_sha = compatibility.get("checkpoint_sha256")
    if model_cfg:
        hf_config = to_hf_config(
            model_cfg,
            foundation_document=foundation_document,
            tokenizer_info=tokenizer_info,
            parameter_count=parameter_count,
            checkpoint_sha256=checkpoint_sha,
        )
        model_card = build_model_card(
            model_cfg=model_cfg,
            hf_config=hf_config,
            checkpoint_path=_relative(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            tokenizer_info=tokenizer_info,
            compatibility=compatibility,
            evaluation_path=_relative(_path(evaluation)) if evaluation else None,
            evaluation_sha256=(
                _sha256_file(_path(evaluation))
                if evaluation and _path(evaluation).is_file()
                else None
            ),
            weight_license=weight_license,
        )
        config_json = json.dumps(hf_config, indent=2) + "\n"
    else:
        model_card = ""
        config_json = ""

    if evaluation and not _path(evaluation).is_file():
        warnings.append(f"evaluation report not found: {_path(evaluation)}")
    if evaluation and _path(evaluation).is_file():
        evaluation_path = _relative(_path(evaluation))
        evaluation_sha256 = _sha256_file(_path(evaluation))
    else:
        evaluation_path = None
        evaluation_sha256 = None

    status = "PASS" if not missing and model_cfg and tokenizer_info else "BLOCKED"
    if status not in ALLOWED_STATUSES:
        raise ExportError(f"invalid export status: {status}")

    files = [
        {
            "path": name,
            "kind": FILE_KINDS.get(name, "unknown"),
            "written": False,
            "sha256": None,
        }
        for name in EXPORT_FILES
    ]

    return ExportPlan(
        status=status,
        checkpoint=_relative(checkpoint_path),
        output_dir=_relative(output_path),
        dry_run=True,
        model_config=model_cfg,
        hf_config=hf_config,
        tokenizer_info=tokenizer_info,
        compatibility=compatibility,
        files=files,
        missing=missing,
        warnings=warnings,
        model_card=model_card,
        config_json=config_json,
        evaluation_path=evaluation_path,
        evaluation_sha256=evaluation_sha256,
        weight_license=weight_license,
        generated_at=generated_at,
        error=error,
    )


def _dedupe_tied_weights(
    state: dict[str, Any], hf_config: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    import torch

    tensors = {
        key: value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not hf_config.get("tie_word_embeddings"):
        return tensors
    embed = tensors.get("tok_emb.weight")
    head = tensors.get("lm_head.weight")
    if embed is None or head is None:
        return tensors
    if embed.data_ptr() == head.data_ptr() or torch.equal(embed, head):
        tensors.pop("lm_head.weight")
        warnings.append(
            "dropped lm_head.weight: it is tied to tok_emb.weight; preserving one copy"
        )
    else:
        warnings.append(
            "checkpoint has an untied lm_head.weight while the config declares tied "
            "embeddings; both tensors are preserved"
        )
    return tensors


def _prepare_output_dir(output_path: Path) -> None:
    """Create the export directory, refusing to clobber unrelated files."""
    if output_path.exists():
        if not output_path.is_dir():
            raise ExportError(f"export path is not a directory: {output_path}")
        if any(output_path.iterdir()):
            if not (output_path / "taksh_export_manifest.json").is_file():
                raise ExportError(
                    "export directory is not empty and has no Taksh manifest; "
                    f"refusing to overwrite: {output_path}"
                )
    else:
        output_path.mkdir(parents=True)


def execute_export(
    plan: ExportPlan,
    *,
    state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an already-validated export plan. Refuses a blocked plan."""
    if plan.status != "PASS":
        raise ExportError(
            "refusing to write a blocked export plan"
            + (": " + "; ".join(plan.missing) if plan.missing else "")
        )
    if not plan.config_json or not plan.model_card:
        raise ExportError(
            "refusing to write an export plan without config.json and README.md content"
        )
    if state_dict is None:
        from serving.local_inference import torch_load_checkpoint

        checkpoint = torch_load_checkpoint(str(_path(plan.checkpoint)))
        state_dict = checkpoint.get("model") or {}
        if not state_dict:
            raise ExportError("checkpoint has no 'model' state_dict")

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExportError(
            "safetensors is required to write an export; install safetensors"
        ) from exc

    output_path = _path(plan.output_dir)
    _prepare_output_dir(output_path)

    tensors = _dedupe_tied_weights(state_dict, plan.hf_config, plan.warnings)
    if not tensors:
        raise ExportError("no tensor weights to export")

    import torch

    tensors = {
        key: value.detach().to("cpu").contiguous()
        for key, value in tensors.items()
    }
    save_file(tensors, str(output_path / "model.safetensors"))

    (output_path / "config.json").write_text(plan.config_json, encoding="utf-8")
    (output_path / "README.md").write_text(plan.model_card, encoding="utf-8")
    shutil.copyfile(_path(plan.tokenizer_info["path"]), output_path / "tokenizer.json")

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": int(plan.model_config.get("max_seq_len", 2048)),
        "clean_up_tokenization_spaces": False,
    }
    (output_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8"
    )
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    (output_path / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema": "taksh.hf_export/1",
        "generated_at": plan.generated_at,
        "product": {
            "id": plan.hf_config.get("taksh", {}).get("product_id"),
            "name": plan.model_config.get("name"),
            "parameter_count": plan.hf_config.get("taksh", {}).get("parameter_count"),
        },
        "checkpoint": {
            "path": plan.checkpoint,
            "sha256": plan.compatibility.get("checkpoint_sha256"),
            "schema": plan.compatibility.get("checkpoint_schema"),
            "step": plan.compatibility.get("checkpoint_step"),
        },
        "tokenizer": plan.tokenizer_info,
        "compatibility": plan.compatibility,
        "evaluation": {
            "path": plan.evaluation_path,
            "sha256": plan.evaluation_sha256,
            "status": "NOT EVALUATED" if not plan.evaluation_path else "REPORT_SUPPLIED",
        },
        "weight_license": plan.weight_license,
        "files": [],
        "warnings": plan.warnings,
        "publication": {
            "uploaded": False,
            "network_access": "none",
            "authorized": False,
        },
    }

    for name in EXPORT_FILES:
        if name in {"SHA256SUMS", "taksh_export_manifest.json"}:
            continue
        file_path = output_path / name
        if not file_path.is_file():
            raise ExportError(f"expected export file was not written: {name}")
        digest = _sha256_file(file_path)
        manifest["files"].append(
            {"path": name, "bytes": file_path.stat().st_size, "sha256": digest}
        )

    (output_path / "taksh_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest_digest = _sha256_file(output_path / "taksh_export_manifest.json")
    manifest["files"].append(
        {
            "path": "taksh_export_manifest.json",
            "bytes": (output_path / "taksh_export_manifest.json").stat().st_size,
            "sha256": manifest_digest,
        }
    )

    sums = "\n".join(
        f"{entry['sha256']}  {entry['path']}" for entry in manifest["files"]
    ) + "\n"
    (output_path / "SHA256SUMS").write_text(sums, encoding="utf-8")

    plan.dry_run = False
    for entry in plan.files:
        file_path = output_path / entry["path"]
        entry["written"] = file_path.is_file()
        entry["sha256"] = _sha256_file(file_path) if file_path.is_file() else None
    return manifest


def verify_export(output_dir: str | Path) -> dict[str, Any]:
    """Verify file presence and hashes without loading weights."""
    output_path = _path(output_dir)
    manifest_path = output_path / "taksh_export_manifest.json"
    problems: list[str] = []
    if not manifest_path.is_file():
        return {
            "schema": "taksh.hf_export_verification/1",
            "status": "BLOCKED",
            "directory": _relative(output_path),
            "problems": ["taksh_export_manifest.json is missing"],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "schema": "taksh.hf_export_verification/1",
            "status": "BLOCKED",
            "directory": _relative(output_path),
            "problems": [f"manifest is unreadable: {type(exc).__name__}"],
        }

    if manifest.get("schema") != "taksh.hf_export/1":
        problems.append(
            f"unexpected manifest schema: {manifest.get('schema')!r}"
        )

    for entry in manifest.get("files", []):
        file_path = output_path / entry["path"]
        if not file_path.is_file():
            problems.append(f"missing file: {entry['path']}")
            continue
        actual = _sha256_file(file_path)
        if entry.get("sha256") and actual != entry["sha256"]:
            problems.append(f"hash mismatch: {entry['path']}")
    for name in EXPORT_FILES:
        if not (output_path / name).is_file():
            problems.append(f"missing required file: {name}")
    return {
        "schema": "taksh.hf_export_verification/1",
        "status": "PASS" if not problems else "BLOCKED",
        "directory": _relative(output_path),
        "problems": problems,
        "files_checked": len(manifest.get("files", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="pre-existing checkpoint to export")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--evaluation", default=None)
    parser.add_argument("--weight-license", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the export; default is a dry run",
    )
    parser.add_argument(
        "--verify",
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help="verify an existing export directory instead of planning",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.verify is not None:
        result = verify_export(args.verify)
        print(json.dumps(result, indent=2) if args.json else result["status"])
        return 0 if result["status"] == "PASS" else 1

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --verify is used")

    plan = plan_export(
        args.checkpoint,
        output_dir=args.out,
        tokenizer=args.tokenizer,
        model_config=args.model_config,
        evaluation=args.evaluation,
        weight_license=args.weight_license,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(f"status: {plan.status}")
        print(f"checkpoint: {plan.checkpoint}")
        print(f"output: {plan.output_dir}")
        print(f"parameters: {plan.hf_config.get('taksh', {}).get('parameter_count')}")
        print(f"files: {', '.join(EXPORT_FILES)}")
        for problem in plan.missing:
            print(f"  BLOCKED: {problem}")
        for warning in plan.warnings:
            print(f"  WARNING: {warning}")

    if plan.status != "PASS":
        return 1
    if not args.execute:
        print("dry run: pass --execute to write the export")
        return 0

    manifest = execute_export(plan)
    print(f"wrote export to {plan.output_dir}")
    print(f"files: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
