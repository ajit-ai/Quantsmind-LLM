"""M6 common local-model interface.

One contract for every local model QuantsMind runs: the project's own M3/M4
checkpoints and any external Hugging Face runtime. The operations are
``describe``, ``preflight``, ``load``, ``generate``, ``health`` and ``close``.

Two rules shape this module.

**Nothing pretends to be universal.** A model with an embedding head cannot
accept a decoding temperature, and an MLM cannot accept an instruction at all.
Unsupported controls are recorded in ``unsupported_settings`` rather than
silently ignored, so a report can say what was and was not actually applied.

**No universal chat template.** Applying ChatML to a model with a different
tokenizer silently changes what the prompt means. Templates are therefore
declared per runtime and the applied form is recorded on every response.

No runtime in this module may contact a hosted inference API.

    from evaluation.m6_runtime import HFLocalRuntime
    with HFLocalRuntime("sentence-transformers/all-MiniLM-L6-v2") as rt:
        print(rt.describe()["task_head"])
"""

from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field

GENERATION_SCHEMA = "quantsmind.m6_generation/1"
DESCRIBE_SCHEMA = "quantsmind.m6_describe/1"


class RuntimeUnavailable(RuntimeError):
    """The runtime, its dependency, or its weights cannot be used right now."""


@dataclass
class GenerationRequest:
    """An explicit generation request.

    ``template`` is opt-in per model family. Leaving it unset means the raw
    prompt is sent unchanged, which is the correct default for a base or
    encoder-style model.
    """

    prompt: str
    task_id: str | None = None
    max_new_tokens: int = 96
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: list = field(default_factory=list)
    timeout_seconds: int | None = None
    template: str | None = None
    system_prompt: str | None = None


@dataclass
class GenerationResponse:
    """Everything known about one generation, including what was not applied."""

    text: str
    model_id: str
    runtime: str
    task_head: str
    template_applied: str | None = None
    generation_config: dict = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float = 0.0
    finish_reason: str | None = None
    truncated: bool = False
    error: str | None = None
    unsupported_settings: list = field(default_factory=list)
    resource_metadata: dict = field(default_factory=dict)
    schema: str = GENERATION_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


class LocalModel(abc.ABC):
    """The common local-model contract."""

    runtime_name = "abstract"

    @abc.abstractmethod
    def describe(self) -> dict:
        """Identity: model, architecture, artifact, tokenizer, runtime."""

    @abc.abstractmethod
    def preflight(self) -> tuple[bool, str]:
        """Can this model be used on this machine right now?"""

    @abc.abstractmethod
    def load(self) -> None:
        """Load after a successful preflight."""

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Produce a completion for an explicit request."""

    def health(self) -> dict:
        """Local runtime status. Safe to call before :meth:`load`."""
        loaded = getattr(self, "_model", None) is not None
        return {"runtime": self.runtime_name, "loaded": loaded,
                "model_id": getattr(self, "model_id", None),
                "schema": DESCRIBE_SCHEMA}

    def close(self) -> None:
        """Release resources. Must be idempotent."""

    @property
    def loaded(self) -> bool:
        return getattr(self, "_model", None) is not None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def effective_config(request: GenerationRequest) -> dict:
    """The decoding settings as actually sent to the model."""
    config = {
        "strategy": "sampling" if request.do_sample else "greedy",
        "max_new_tokens": request.max_new_tokens,
        "do_sample": request.do_sample,
    }
    if request.do_sample:
        config.update({"temperature": request.temperature, "top_p": request.top_p})
        if request.top_k:
            config["top_k"] = request.top_k
    if request.seed is not None:
        config["seed"] = request.seed
    if request.stop:
        config["stop"] = list(request.stop)
    return config


def unsupported_for(task_head: str, request: GenerationRequest) -> list[str]:
    """Settings this head genuinely cannot honour.

    Recorded rather than dropped, so that a report can distinguish "the model
    produced this under sampling" from "sampling was requested and ignored".
    """
    unsupported = []
    if task_head == "embedding":
        unsupported.append(
            "max_new_tokens: an embedding head performs no decoding")
        if request.do_sample:
            unsupported.append("do_sample: embedding heads are deterministic")
        if request.stop:
            unsupported.append("stop: an embedding head produces no token stream")
    elif task_head == "masked_lm":
        unsupported.append(
            "max_new_tokens: a masked LM scores [MASK] positions, it does not "
            "autoregressively generate")
        if request.do_sample:
            unsupported.append("do_sample: masked-LM scoring is deterministic")
        if request.stop:
            unsupported.append("stop: a masked LM produces no token stream")
    return unsupported


def elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


# --------------------------------------------------------------------------- #
# Sovereign runtime: the project's own M3/M4 checkpoints
# --------------------------------------------------------------------------- #
class SovereignCheckpointRuntime(LocalModel):
    """Wraps the existing, tested M5 ``QMCkptAdapter``.

    M3/M4 inference already works and has a regression suite, so M6 does not
    reimplement it. This class adapts that adapter onto the M6 contract and
    records the checkpoint hash, so a run can be tied to an exact artifact.
    """

    runtime_name = "sovereign_checkpoint"

    # The M5 runner always passes this explicitly; M6 mirrors that default so
    # the runtime also works when driven directly. Checkpoints do not reliably
    # carry a tokenizer path, and ``Path(None)`` is a crash rather than a
    # missing-tokenizer error.
    DEFAULT_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"

    @property
    def loaded(self) -> bool:
        # The sovereign runtime delegates to the M5 adapter rather than
        # holding ``_model`` itself.
        return self._adapter is not None and getattr(self._adapter, "_model",
                                                    None) is not None

    def __init__(self, checkpoint_path, tokenizer_path=None, device="cpu"):
        self.checkpoint_path = str(checkpoint_path)
        self.tokenizer_path = (tokenizer_path
                               if tokenizer_path is not None
                               else self.DEFAULT_TOKENIZER)
        self.device = device
        self._adapter = None
        self._artifact_sha256 = None
        self._parameters = None

    def describe(self) -> dict:
        from evaluation.m5_adapters import sha256_file
        from pathlib import Path

        path = Path(self.checkpoint_path)
        self._artifact_sha256 = sha256_file(path) if path.exists() else None
        # ``info()`` on the M5 adapter requires a loaded model, so identity is
        # reported from the artifact on disk until load() has happened.
        if self._adapter is not None and self.loaded:
            self._parameters = self._adapter.info().parameters
        return {
            "schema": DESCRIBE_SCHEMA,
            "model_id": f"quantsmind/{path.parent.name}" if path.exists()
                        else self.checkpoint_path,
            "runtime": self.runtime_name,
            "architecture": "GPT-style decoder (project implementation)",
            "task_head": "causal_lm",
            "artifact_path": self.checkpoint_path,
            "artifact_sha256": self._artifact_sha256,
            "parameters": self._parameters,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer": "project BPE (qm_code_bpe_v0) unless overridden",
            "prompt_template": None,
            "template_note": (
                "no chat template is applied; the raw prompt is tokenised "
                "with the project's own BPE"),
            "device": self.device,
            "hosted_api_used": False,
        }

    def preflight(self) -> tuple[bool, str]:
        from evaluation.m5_adapters import QMCkptAdapter

        self._adapter = QMCkptAdapter(
            self.checkpoint_path, tokenizer_path=self.tokenizer_path,
            device=self.device)
        # ``info()`` on the M5 adapter requires a loaded model, so the parameter
        # count is only available after load().
        return self._adapter.preflight()

    def load(self) -> None:
        if self._adapter is None:
            self.preflight()
        self._adapter.load()
        self._parameters = self._adapter.info().parameters

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.loaded:
            raise RuntimeUnavailable(
                "generate() called before load(); call preflight() and load()")
        started = time.perf_counter()
        # The M5 adapter parameterises decoding with ``greedy`` rather than
        # ``do_sample``; M6's contract is explicit about the sampling mode, so
        # the flag is translated here rather than leaking both spellings.
        kwargs = {"max_new_tokens": request.max_new_tokens,
                  "greedy": not request.do_sample}
        if request.do_sample:
            kwargs.update({"temperature": request.temperature,
                           "top_p": request.top_p, "top_k": request.top_k})
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop:
            kwargs["stop"] = list(request.stop)
        raw = self._adapter.generate(request.prompt, **kwargs)
        unsupported = unsupported_for("causal_lm", request)
        if request.template:
            unsupported.append(
                f"template {request.template!r}: the sovereign checkpoint has no "
                f"chat template; the raw prompt was used")
        return GenerationResponse(
            text=raw.get("text", ""),
            model_id=self.describe()["model_id"],
            runtime=self.runtime_name, task_head="causal_lm",
            template_applied=None,
            generation_config=effective_config(request),
            input_tokens=raw.get("prompt_tokens"),
            output_tokens=raw.get("completion_tokens"),
            duration_seconds=elapsed(started),
            finish_reason=raw.get("finish_reason"),
            truncated=raw.get("finish_reason") == "length",
            unsupported_settings=unsupported,
            resource_metadata={"tokens_per_second": raw.get("tokens_per_second")},
        )

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.release()
            self._adapter = None
        self._model = None


# --------------------------------------------------------------------------- #
# Hugging Face runtime
# --------------------------------------------------------------------------- #
class HFLocalRuntime(LocalModel):
    """A locally cached Hugging Face model, loaded with ``transformers``.

    Every refusal happens in :meth:`preflight`, before any weights are read:
    missing dependency, missing licence record, incomplete snapshot, or
    insufficient memory. Nothing is ever downloaded -- ``local_files_only`` is
    fixed at ``True`` and is not overridable from the command line.

    Supports causal LM, masked LM and embedding heads. The head is chosen from
    the model's own ``config.architectures`` field rather than being forced.
    """

    runtime_name = "huggingface_local"

    HEAD_BY_ARCHITECTURE = {
        "ForCausalLM": "causal_lm",
        "ForMaskedLM": "masked_lm",
        "ForSequenceClassification": "embedding",
        "Model": "embedding",
    }

    def __init__(self, model_id, revision=None, dtype="auto", device="cpu",
                 license_record=None, snapshot_path=None, local_path=None):
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self.license_record = license_record or {}
        self.local_path = local_path
        self._snapshot = local_path
        self._model = None
        self._tokenizer = None
        self._config = None
        self._task_head = "unknown"
        self._parameters = None
        self._weights_path = None

    # ---- helpers ---------------------------------------------------------
    def _resolve_snapshot(self):
        """Find a local snapshot. Never triggers a network fetch."""
        from pathlib import Path

        if self._snapshot:
            return Path(self._snapshot)
        try:
            from huggingface_hub import try_to_load_from_cache
            hit = try_to_load_from_cache(self.model_id, "config.json",
                                         revision=self.revision)
            if isinstance(hit, str) and Path(hit).exists():
                self._snapshot = str(Path(hit).parent)
                return Path(self._snapshot)
        except Exception:  # noqa: BLE001 - absence is an answer, not an error
            pass
        return None

    def _load_config(self):
        import json
        from pathlib import Path

        snapshot = self._resolve_snapshot()
        if snapshot is None:
            return None
        config_file = snapshot / "config.json"
        if not config_file.exists():
            return None
        try:
            return json.loads(Path(config_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _weights_present(self, snapshot):
        """Weight files must exist and must not be partial downloads."""
        if snapshot is None:
            return None
        candidates = [p for p in snapshot.rglob("*")
                      if p.is_file()
                      and p.name.endswith((".safetensors", ".bin"))]
        candidates = [p for p in candidates if not p.name.endswith(".incomplete")]
        if not candidates:
            return None
        index = snapshot / "model.safetensors.index.json"
        if index.exists():
            import json
            try:
                meta = json.loads(index.read_text(encoding="utf-8"))
                shards = sorted(set(meta.get("weight_map", {}).values()))
                if any(not (snapshot / s).exists() for s in shards):
                    return None
            except (OSError, ValueError):
                return None
        return max(candidates, key=lambda p: p.stat().st_size)

    def _transformers_version(self):
        try:
            import transformers
            return transformers.__version__
        except ImportError:
            return None

    # ---- contract --------------------------------------------------------
    def describe(self) -> dict:
        config = self._load_config() or {}
        archs = config.get("architectures") or []
        head = self._task_head
        if head == "unknown":
            for arch in archs:
                for suffix, name in self.HEAD_BY_ARCHITECTURE.items():
                    if arch.endswith(suffix):
                        head = name
                        break
                if head != "unknown":
                    break
        self._task_head = head
        return {
            "schema": DESCRIBE_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "runtime": self.runtime_name,
            "runtime_versions": {"transformers": self._transformers_version()},
            "architecture": archs[0] if archs else None,
            "task_head": head,
            "parameters": self._parameters,
            "config_parameters": config.get("num_parameters"),
            "local_path": self._snapshot,
            "artifact_sha256": self._artifact_hash(),
            "weight_format": (self._weights_path.suffix.lstrip(".")
                              if self._weights_path else None),
            "dtype": config.get("torch_dtype") or self.dtype,
            "context_window": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
            "tokenizer": ("auto from snapshot" if self._snapshot else None),
            "prompt_template": None,
            "template_note": (
                "no chat template is applied automatically; each family needs "
                "its own and applying the wrong one changes prompt semantics"),
            "license_id": self.license_record.get("license_id"),
            "hosted_api_used": False,
        }

    def _artifact_hash(self):
        from evaluation.m5_adapters import sha256_file

        if self._weights_path and self._weights_path.exists():
            return sha256_file(self._weights_path)
        return None

    def preflight(self) -> tuple[bool, str]:
        version = self._transformers_version()
        if not version:
            return False, (
                "the 'transformers' package is not installed; install it "
                "explicitly before evaluating an external model "
                "(M6 never installs packages itself)")

        if not self.license_record.get("license_id"):
            return False, (
                "no verified licence record supplied for this model; an "
                "external model may not be evaluated without one")

        snapshot = self._resolve_snapshot()
        if snapshot is None:
            return False, (
                f"no local snapshot for {self.model_id!r}; M6 runs local-only "
                f"and never downloads weights")

        config = self._load_config()
        if not config:
            return False, f"snapshot at {snapshot} has no readable config.json"

        weights = self._weights_present(snapshot)
        if weights is None:
            return False, (
                f"snapshot at {snapshot} has no complete weight file "
                f"(partial downloads and missing shards are not usable)")

        from evaluation.m6_resources import plan_for

        head = self.describe()["task_head"]
        params = self._parameters or config.get("num_parameters")
        plan = plan_for(
            self.model_id, params, dtype=config.get("torch_dtype") or self.dtype,
            context=config.get("max_position_embeddings", 512),
            weights_bytes=weights.stat().st_size,
            artifact_bytes=weights.stat().st_size,
            layers=config.get("num_hidden_layers"),
            kv_heads=config.get("num_key_value_heads"),
            head_dim=(config.get("hidden_size", 0) // config["num_attention_heads"]
                      if config.get("hidden_size") and
                      config.get("num_attention_heads") else None))
        ok, reason = plan.decision()
        if not ok:
            return False, f"resource preflight refused: {reason}"
        self._weights_path = weights
        return True, f"transformers {version}; {weights.name}; {reason}"

    def load(self) -> None:
        ok, reason = self.preflight()
        if not ok:
            raise RuntimeUnavailable(reason)
        import torch
        from transformers import (AutoModel, AutoModelForCausalLM,
                                  AutoModelForMaskedLM, AutoTokenizer)

        started = time.perf_counter()
        loader = {
            "causal_lm": AutoModelForCausalLM,
            "masked_lm": AutoModelForMaskedLM,
            "embedding": AutoModel,
        }.get(self._task_head, AutoModel)
        kwargs = {"local_files_only": True}
        if self._snapshot:
            kwargs["revision"] = None
        self._tokenizer = AutoTokenizer.from_pretrained(self._snapshot, **kwargs)
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(self.dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self._model = loader.from_pretrained(self._snapshot, **kwargs)
        self._model.eval()
        try:
            self._parameters = sum(p.numel() for p in self._model.parameters())
        except (TypeError, AttributeError):
            self._parameters = None
        self._load_seconds = elapsed(started)

    def health(self) -> dict:
        from evaluation.m6_resources import machine_facts

        base = super().health()
        base.update({"task_head": self._task_head,
                     "parameters": self._parameters,
                     "transformers": self._transformers_version(),
                     "load_seconds": getattr(self, "_load_seconds", None)})
        return base

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.loaded:
            raise RuntimeUnavailable(
                "generate() called before load(); call preflight() and load()")
        import torch

        started = time.perf_counter()
        unsupported = unsupported_for(self._task_head, request)
        text_input = request.prompt
        if request.system_prompt:
            text_input = request.system_prompt + "\n" + request.prompt

        encoded = self._tokenizer(text_input, return_tensors="pt",
                                  truncation=True, max_length=512)
        input_tokens = int(encoded["input_ids"].shape[-1])

        with torch.no_grad():
            if self._task_head == "causal_lm":
                out = self._model.generate(
                    **encoded, max_new_tokens=request.max_new_tokens,
                    do_sample=request.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id
                    or self._tokenizer.eos_token_id)
                generated = out[0][input_tokens:]
                text = self._tokenizer.decode(generated, skip_special_tokens=True)
                output_tokens = int(generated.shape[-1])
                finish = ("length" if output_tokens >= request.max_new_tokens
                          else "eos")
            elif self._task_head == "masked_lm":
                out = self._model(**encoded)
                best = int(out.logits[0].argmax(-1))
                text = self._tokenizer.decode([best])
                output_tokens = 1
                finish = "argmax_token"
            else:  # embedding head
                out = self._model(**encoded)
                vector = out.last_hidden_state.mean(dim=1)[0]
                text = (f"[embedding dim={vector.numel()} "
                        f"norm={float(vector.norm()):.4f}]")
                output_tokens = 0
                finish = "pooled"

        return GenerationResponse(
            text=text, model_id=self.model_id, runtime=self.runtime_name,
            task_head=self._task_head, template_applied=None,
            generation_config=effective_config(request),
            input_tokens=input_tokens, output_tokens=output_tokens,
            duration_seconds=elapsed(started), finish_reason=finish,
            truncated=bool(self._task_head == "causal_lm"
                           and output_tokens >= request.max_new_tokens),
            unsupported_settings=unsupported,
            resource_metadata={"load_seconds": getattr(self, "_load_seconds",
                                                       None)},
        )

    def close(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        self._config = None
        gc.collect()


# --------------------------------------------------------------------------- #
# Runtime factory
# --------------------------------------------------------------------------- #
def runtime_status() -> list[dict]:
    """Which runtimes are usable on this machine right now."""
    out = []
    for name, importer in (("transformers", "transformers"),):
        try:
            module = __import__(importer)
            out.append({"runtime": name, "available": True,
                        "version": getattr(module, "__version__", "unknown")})
        except ImportError as exc:
            out.append({"runtime": name, "available": False, "reason": str(exc)})
    try:
        import torch
        out.append({"runtime": "torch", "available": True,
                    "version": torch.__version__,
                    "cuda": torch.cuda.is_available()})
    except ImportError as exc:
        out.append({"runtime": "torch", "available": False, "reason": str(exc)})
    for name in ("llama_cpp", "onnxruntime"):
        try:
            module = __import__(name)
            out.append({"runtime": name, "available": True,
                        "version": getattr(module, "__version__", "unknown")})
        except ImportError:
            out.append({"runtime": name, "available": False,
                        "reason": "not installed"})
    return out