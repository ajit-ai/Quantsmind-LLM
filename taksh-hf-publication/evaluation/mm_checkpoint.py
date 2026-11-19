"""MM-aware checkpoint loading shared by the evaluation harness.

Handles both text-only ``QMModel`` checkpoints and multimodal
``QMMultimodal`` checkpoints.  The checkpoint schema produced by
``training/train_mm.py`` is read-only here: ``model`` (state dict) and
``cfg`` (text config) are required; multimodal checkpoints carry the vision
config either as a YAML path under ``config["run"]["vision_config"]`` or,
for hermetic tests, as an inline dict.

This module never modifies a checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch

from training.model import QMModel
from training.multimodal import QMMultimodal
from training.pretrain import load_yaml


class LoadedCheckpoint:
    """Minimal view over a loaded checkpoint.

    Attributes
    ----------
    path : Path
        Absolute path of the checkpoint file on disk.
    model : QMModel | QMMultimodal
        The (eval-mode) model.  For multimodal checkpoints this is a
        ``QMMultimodal``; for text-only checkpoints it is a ``QMModel``.
    is_mm : bool
        True when the checkpoint contains vision/projector weights.
    text_cfg : dict
        The text (language-model) configuration dict.
    config : dict | None
        The full train-run ``config`` dict, if present.
    metadata : dict | None
        The additive ``metadata`` dict, if present.
    """

    def __init__(self, path: Path, model, text_cfg: dict, is_mm: bool, config, metadata):
        self.path = Path(path)
        self.model = model
        self.is_mm = is_mm
        self.text_cfg = dict(text_cfg)
        self.config = config
        self.metadata = metadata

    @property
    def text_model(self):
        """The text (language) model to use for text-only evaluation."""
        if self.is_mm:
            return self.model.text
        return self.model


def _resolve_vision_cfg(raw, checkpoint_path: Path) -> dict:
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        if not p.is_absolute():
            here = checkpoint_path.parent / p
            if not here.is_file():
                here = Path(".").resolve() / p
            p = here
        if not p.is_file():
            raise FileNotFoundError(
                f"vision config referenced by checkpoint does not exist: {p}"
            )
        data = load_yaml(p)
        if "vision" not in data:
            raise ValueError(f"vision config file missing 'vision' key: {p}")
        return dict(data["vision"])
    if isinstance(raw, dict):
        required = {"image_size", "patch_size", "d_model"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"inline vision config missing keys: {sorted(missing)}")
        return dict(raw)
    raise ValueError(
        f"vision_config must be a YAML path or inline dict, got {type(raw).__name__}"
    )


def _resolve_text_cfg(raw, checkpoint_path: Path) -> dict:
    """Resolve a text (model) config from a YAML path or an inline dict.

    Real checkpoints store it in two ways: train_mm checkpoints carry an
    inline ``cfg`` dict, while text-only M0 checkpoints reference a YAML
    ``model`` file under ``config["run"]["model_config"]``.  YAML paths are
    resolved relative to the checkpoint, then relative to the repo root.
    """
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        if not p.is_absolute():
            here = checkpoint_path.parent / p
            if not here.is_file():
                here = Path(".").resolve() / p
            p = here
        if not p.is_file():
            raise FileNotFoundError(f"model config referenced by checkpoint does not exist: {p}")
        data = load_yaml(p)
        if "model" not in data:
            raise ValueError(f"model config file missing 'model' key: {p}")
        cfg = dict(data["model"])
        if "vocab_size" not in cfg:
            raise ValueError(f"model config has no vocab_size: {p}")
        return cfg
    if isinstance(raw, dict) and "vocab_size" in raw:
        return dict(raw)
    raise ValueError(
        f"text config must be an inline dict with 'vocab_size' or a YAML path, "
        f"got {type(raw).__name__}"
    )


def load_checkpoint(path, device: str = "cpu", vision_cfg=None) -> LoadedCheckpoint:
    """Load a text-only or multimodal checkpoint for evaluation.

    Parameters
    ----------
    path : str | Path
        Path to a ``checkpoint_last.pt``-style checkpoint.
    device : str
        Target device (default ``"cpu"``).
    vision_cfg : dict | str | None
        Override for the vision config.  When omitted, multimodal
        checkpoints resolve it from ``config["run"]["vision_config"]``.

    Returns
    -------
    LoadedCheckpoint

    Raises
    ------
    FileNotFoundError
        If the checkpoint (or a referenced vision config file) is missing.
    ValueError
        If the checkpoint lacks a usable ``cfg``/``model`` state dict, or a
        multimodal checkpoint has no resolvable vision config.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"checkpoint must be a dict with a 'model' state dict (got {type(ckpt).__name__})"
        )
    state = ckpt["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint 'model' state dict is empty or malformed")

    text_cfg = ckpt.get("cfg")
    if not (isinstance(text_cfg, dict) and "vocab_size" in text_cfg):
        run = (ckpt.get("config") or {}).get("run") or {}
        if run.get("model_config"):
            text_cfg = _resolve_text_cfg(run["model_config"], path)
    if not (isinstance(text_cfg, dict) and "vocab_size" in text_cfg):
        raise ValueError(
            "checkpoint missing a usable text 'cfg' dict (vocab_size required); "
            "no config['run']['model_config'] fallback either"
        )

    is_mm = any(
        k.startswith("vision.") or k.startswith("projector.") for k in state
    )

    if is_mm:
        raw = vision_cfg
        if raw is None:
            run = (ckpt.get("config") or {}).get("run") or {}
            raw = run.get("vision_config")
            if raw is None:
                raise ValueError(
                    "multimodal checkpoint has no config['run']['vision_config']; "
                    "pass vision_cfg=... explicitly"
                )
        resolved = _resolve_vision_cfg(raw, path)
        model = QMMultimodal(text_cfg, resolved)
    else:
        model = QMModel(text_cfg)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        path=path,
        model=model,
        text_cfg=text_cfg,
        is_mm=is_mm,
        config=ckpt.get("config"),
        metadata=ckpt.get("metadata"),
    )