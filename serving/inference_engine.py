"""TAKSH L0.6.2 — qualified local inference engine.

A single, coherent local inference architecture (logical components, not
microservices):

  ModelLoader           explicit load states; safe weights_only loading;
                        config/tokenizer/state-dict compatibility validation;
                        architecture fingerprint verification; memory preflight;
                        clean unload.
  TokenizerAdapter      load, fingerprint, vocab-parity check, diagnostics.
  ContextManager        tokenizer-aware accounting; rejects overflow; explicit,
                        documented truncation policy (never silent).
  SamplingController    temperature / top-k / top-p / presence & frequency
                        penalties / seeded dedicated RNG per request.
  GenerationController  parameter validation, bounded generation, EOS handling,
                        stop sequences, token accounting.
  Streaming             token-by-token chunk generator.
  CancellationController / Deadline  cooperative cancellation and timeouts via
                        threading.Event + monotonic deadline.
  InferenceTrace        structured per-request record for the evaluation
                        harness; never scores capability.

Everything on this path is local CPU inference. No network access, no GPU
unless explicitly authorized, no provider fallback, no checkpoint mutation, no
execution of generated code.

Binding rules honoured here:
  * a checkpoint is marked READY only after actual load + validation succeed;
  * the full 111M artifact is loaded only when host memory clears preflight
    (see evaluation/l062_inference_qualification.py) — never silently;
  * deterministic mode uses a dedicated torch.Generator per request, so
    concurrent requests do not share sampling state;
  * generated text is never executed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.local_inference import (  # noqa: E402
    _apply_penalties,
    _filter_logits,
    resolve_model_config,
    torch_load_checkpoint,
)

FP32_BYTES_PER_PARAM = 4
DEFAULT_CONTEXT_LIMIT = 2048
HARD_MAX_NEW_TOKENS = 256
MAX_STOP_SEQUENCES = 8
MAX_STOP_SEQUENCE_CHARS = 128
MAX_PROMPT_CHARS = 200_000


class LoadState(str, Enum):
    NOT_LOADED = "NOT_LOADED"
    PREFLIGHT = "PREFLIGHT"
    LOADING = "LOADING"
    READY = "READY"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    UNLOADED = "UNLOADED"


class InferenceError(RuntimeError):
    """Base for structured inference failures (never a bare traceback)."""


class ConfigurationError(InferenceError):
    """The inference request itself is invalid."""


class ContextOverflowError(ConfigurationError):
    """Prompt exceeds the configured context (or prompt+max_new_tokens would)."""


class EngineNotReady(InferenceError):
    """Generation requested while not in the READY state."""


class CheckpointIncompatibleError(InferenceError):
    """Checkpoint/tokenizer failed to load compatibly (or corrupt)."""


class ResourceBlockedError(InferenceError):
    """Host resources insufficient for the requested load (fail-closed)."""


class CancelledError(InferenceError):
    """Request was cancelled cooperatively."""


class GenerationFailedError(InferenceError):
    """Generation failed mid-stream (OOM, numerical issue, backend error)."""


class GenerationTimeoutError(InferenceError):
    """Request exceeded its deadline cooperatively."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_fingerprint(model_cfg: dict) -> str:
    blob = json.dumps(model_cfg, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def foundation_config_path() -> Path:
    return REPO / "configs" / "foundation" / "l060_foundation.yaml"


def canonical_model_cfg() -> dict:
    import yaml

    return yaml.safe_load(foundation_config_path().read_text(encoding="utf-8"))["model"]


def recommended_headroom(estimate_bytes: int) -> int:
    """Deterministic safety margin: 25% of the estimate, at least 32 MiB.

    Not a performance guarantee — just the guard margin applied by preflight
    and by the load-time check. For the 111M foundation model this is
    ~111 MB; for tiny synthetic fixtures it is 32 MiB.
    """
    return max(32 * 1024 * 1024, int(estimate_bytes) // 4)


def estimate_model_bytes(model_cfg: dict) -> int:
    """Upper-bound fp32 estimate derived from the config, not measured.

    The state-dict is instantiated only during an actual load; this estimate
    exists for preflight and is labelled as an estimate everywhere.
    """
    v = int(model_cfg["vocab_size"]); d = int(model_cfg["d_model"])
    L = int(model_cfg["n_layers"]); h = int(model_cfg["ffn_dim"]); kv = int(model_cfg["n_kv_heads"])
    heads = int(model_cfg["n_heads"]); hd = d // heads
    # tied embeddings: count once
    params = v * d
    # per layer: attention q (d*d), k/v (d * kv*hd each), o (d*d),
    # two RMSNorms (2d), FFN 3 * d * h
    attn = d * d + 2 * d * (kv * hd) + d * d
    ffn = 3 * d * h
    params += L * (attn + ffn + 2 * d)
    # final norm
    params += d
    return params * FP32_BYTES_PER_PARAM


def memory_snapshot() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        return {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "percent_used": float(vm.percent),
            "process_rss_bytes": int(proc.memory_info().rss),
            "tool": "psutil",
            "measured": True,
        }
    except Exception:
        return {"total_bytes": None, "available_bytes": None,
                "percent_used": None, "process_rss_bytes": None,
                "tool": "unavailable", "measured": False}


# --------------------------------------------------------------------------- #
# Tokenizer adapter
# --------------------------------------------------------------------------- #
class TokenizerAdapter:
    def __init__(self, tokenizer):
        self._tok = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "TokenizerAdapter":
        path = Path(path)
        if not path.is_file():
            raise CheckpointIncompatibleError(f"tokenizer not found: {path}")
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(path))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointIncompatibleError(
                f"tokenizer {path} failed to load: {type(exc).__name__}: {exc}"
            ) from exc
        adapter = cls(tok)
        adapter.path = path
        adapter.sha256 = file_sha256(path)
        return adapter

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise ConfigurationError("prompt must be a string")
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def eos_id(self) -> Optional[int]:
        return self._tok.token_to_id("<eos>")

    def compatibility_report(self, model_cfg: dict) -> dict:
        vocab_ok = self.vocab_size() == int(model_cfg["vocab_size"])
        return {
            "tokenizer_path": str(getattr(self, "path", "<memory>")),
            "tokenizer_sha256": getattr(self, "sha256", None),
            "vocab_size": self.vocab_size(),
            "model_vocab_size": int(model_cfg["vocab_size"]),
            "vocab_match": vocab_ok,
            "note": "fingerprint is of the tokenizer file bytes",
        }


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #
@dataclass
class ContextPlan:
    prompt_tokens: int
    max_new_tokens: int
    context_limit: int
    remaining_prompt_budget: int
    remaining_total_budget: int
    utilization: float
    truncation_policy: str
    truncated: bool
    prompt_ids: list[int]


class ContextManager:
    def __init__(self, context_limit: int = DEFAULT_CONTEXT_LIMIT):
        if not isinstance(context_limit, int) or context_limit <= 0:
            raise ConfigurationError(f"context_limit must be a positive int, got {context_limit!r}")
        self.context_limit = context_limit

    def plan(self, prompt_ids: list[int], max_new_tokens: int,
             truncation_policy: str = "reject") -> ContextPlan:
        if truncation_policy not in ("reject", "left"):
            raise ConfigurationError(
                f"truncation_policy must be 'reject' or 'left', got {truncation_policy!r}"
            )
        limit = self.context_limit
        truncated = False
        ids = list(prompt_ids)

        if max_new_tokens > limit - 1:
            raise ConfigurationError(
                f"max_new_tokens {max_new_tokens} leaves no room for any prompt within "
                f"context limit {limit}"
            )

        if len(ids) > limit - max_new_tokens:
            if truncation_policy == "reject":
                raise ContextOverflowError(
                    f"prompt has {len(ids)} tokens; context limit is {limit} and "
                    f"max_new_tokens is {max_new_tokens}. "
                    "Use truncation_policy='left' to opt into left truncation."
                )
            ids = ids[-(limit - max_new_tokens):]
            truncated = True

        prompt_tokens = len(ids)
        if prompt_tokens + max_new_tokens > limit:
            raise ContextOverflowError(
                f"prompt_tokens({prompt_tokens}) + max_new_tokens({max_new_tokens}) "
                f"exceeds context limit {limit}"
            )

        remaining = limit - prompt_tokens
        return ContextPlan(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            context_limit=limit,
            remaining_prompt_budget=remaining,
            remaining_total_budget=limit - prompt_tokens - max_new_tokens,
            utilization=(prompt_tokens + max_new_tokens) / limit,
            truncation_policy=truncation_policy,
            truncated=truncated,
            prompt_ids=ids,
        )


# --------------------------------------------------------------------------- #
# Sampling controller
# --------------------------------------------------------------------------- #
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0                 # <= 0 disables top-k filtering
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None     # None => non-deterministic; int => dedicated generator
    max_new_tokens: int = 64

    def validate(self) -> "SamplingParams":
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise ConfigurationError("max_new_tokens must be an integer")
        if not 1 <= self.max_new_tokens <= HARD_MAX_NEW_TOKENS:
            raise ConfigurationError(
                f"max_new_tokens must be in [1, {HARD_MAX_NEW_TOKENS}], got {self.max_new_tokens}"
            )
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ConfigurationError("temperature must be a number")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ConfigurationError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise ConfigurationError("top_k must be an integer (<=0 disables)")
        if self.top_k < 0:
            raise ConfigurationError(f"top_k must be >= 0 (0 disables), got {self.top_k}")
        if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
            raise ConfigurationError("top_p must be a number")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ConfigurationError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        for name, val in (("presence_penalty", self.presence_penalty),
                          ("frequency_penalty", self.frequency_penalty)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ConfigurationError(f"{name} must be a number")
            if not -2.0 <= float(val) <= 2.0:
                raise ConfigurationError(f"{name} must be in [-2.0, 2.0], got {val}")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ConfigurationError("seed must be an integer or null")
        return self

    @property
    def deterministic(self) -> bool:
        return float(self.temperature) == 0.0

    def recorded(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "presence_penalty": float(self.presence_penalty),
            "frequency_penalty": float(self.frequency_penalty),
            "seed": self.seed,
            "max_new_tokens": int(self.max_new_tokens),
            "deterministic": self.deterministic,
        }


class SamplingController:
    """Samples with a dedicated, per-request torch.Generator.

    This isolates random state between concurrent requests; a shared global
    RNG is never touched when a seed is provided.
    """

    @staticmethod
    def make_generator(seed: Optional[int]) -> torch.Generator:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        else:
            g.seed()  # nondeterministic, still isolated from other requests
        return g

    @staticmethod
    def sample(logits: torch.Tensor, params: SamplingParams,
               generator: torch.Generator) -> int:
        if params.deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = _filter_logits(logits, float(params.temperature), int(params.top_k), float(params.top_p))
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


# --------------------------------------------------------------------------- #
# Cancellation controller
# --------------------------------------------------------------------------- #
class CancellationController:
    def __init__(self):
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str) -> threading.Event:
        with self._lock:
            event = threading.Event()
            self._events[request_id] = event
            return event

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._events.get(request_id)
            if event is None:
                return False
            event.set()
            return True

    def cleanup(self, request_id: str) -> None:
        with self._lock:
            self._events.pop(request_id, None)

    def inflight(self) -> list[str]:
        with self._lock:
            return sorted(self._events)


# --------------------------------------------------------------------------- #
# Request / response / trace
# --------------------------------------------------------------------------- #
@dataclass
class InferenceRequest:
    prompt: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    stop: Optional[list[str]] = None
    language_id: Optional[str] = None
    language_hint: Optional[str] = None
    task_id: Optional[str] = None
    mode: str = "general"          # general | model_only | compiler_assisted
    truncation_policy: str = "reject"
    timeout_seconds: Optional[float] = None
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:16]}")


@dataclass
class InferenceTrace:
    request_id: str
    model_config_fingerprint: str
    tokenizer_sha256: Optional[str]
    device: str
    language_id: Optional[str]
    language_qualification_status: Optional[str]
    task_id: Optional[str]
    mode: str
    sampling: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    truncated: bool = False
    cancelled: bool = False
    timed_out: bool = False

    def to_eval_record(self) -> dict:
        """Evaluation-harness record. Contains no capability verdict."""
        return {
            "schema": "taksh.inference_trace/1",
            "request_id": self.request_id,
            "task_id": self.task_id,
            "language_id": self.language_id,
            "language_qualification_status": self.language_qualification_status,
            "mode": self.mode,
            "model_config_fingerprint": self.model_config_fingerprint,
            "tokenizer_sha256": self.tokenizer_sha256,
            "device": self.device,
            "sampling": self.sampling,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "capability_claim": None,
            "note": "trace only; scoring happens in the frozen evaluation harness, never here",
        }


# --------------------------------------------------------------------------- #
# Model loader
# --------------------------------------------------------------------------- #
class ModelLoader:
    def __init__(self, checkpoint: str | Path, tokenizer_path: str | Path,
                 device: str = "cpu"):
        if device != "cpu":
            import os

            from evaluation.resource_guard import authorization_granted

            if not authorization_granted():
                raise ResourceBlockedError(
                    f"device {device!r} requires explicit owner authorization "
                    "(TAKSH_EXECUTE_AUTHORIZED=I_AUTHORIZE_EXPENSIVE_EXECUTION); cpu default"
                )
        self.checkpoint = Path(checkpoint)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device
        self.state = LoadState.NOT_LOADED
        self.last_error: Optional[str] = None
        self.model = None
        self.model_cfg: Optional[dict] = None
        self.tokenizer: Optional[TokenizerAdapter] = None
        self.load_report: dict = {}
        self._cancel = CancellationController()

    # -- discovery ------------------------------------------------------------
    @staticmethod
    def discover_checkpoints(root: str | Path) -> list[dict]:
        root = Path(root)
        found = []
        if root.is_dir():
            for p in sorted(root.rglob("*.pt")):
                try:
                    found.append({"path": str(p), "size_bytes": p.stat().st_size,
                                  "sha256": file_sha256(p)})
                except OSError as exc:
                    found.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
        return found

    # -- preflight -------------------------------------------------------------
    def preflight(self, headroom_bytes: int | None = None) -> dict:
        self.state = LoadState.PREFLIGHT
        problems: list[str] = []
        if not self.checkpoint.is_file():
            problems.append(f"checkpoint not found: {self.checkpoint}")
        if not self.tokenizer_path.is_file():
            problems.append(f"tokenizer not found: {self.tokenizer_path}")

        mem = memory_snapshot()
        # Upper-bound estimate from an *unresolved* config is not possible yet,
        # so preflight uses the measured checkpoint file size as the lower bound
        # and reports it. The resolved-config fp32 upper bound is computed in
        # load() before weights are instantiated.
        file_bytes = self.checkpoint.stat().st_size if self.checkpoint.is_file() else None
        headroom = headroom_bytes if headroom_bytes is not None else recommended_headroom(file_bytes or 0)
        required = (file_bytes or 0) + headroom
        fits = (mem["available_bytes"] is not None and file_bytes is not None
                and mem["available_bytes"] >= required)

        report = {
            "device": self.device,
            "checkpoint": str(self.checkpoint),
            "checkpoint_file_bytes": file_bytes,
            "headroom_bytes": headroom_bytes,
            "estimated_required_bytes": required,
            "available_bytes": mem["available_bytes"],
            "fits_available_memory": bool(fits),
            "memory": mem,
            "problems": problems,
            "note": ("estimate = measured checkpoint file size + headroom; "
                     "fp32 parameter upper bound is validated during load; "
                     "this is a guard, not a performance measurement"),
        }
        if problems or not fits:
            if not fits:
                problems.append(
                    f"available memory {mem['available_bytes']} B < estimated required "
                    f"{required} B (checkpoint {file_bytes} B + headroom {headroom_bytes} B)"
                )
            report["problems"] = problems
            self.state = LoadState.BLOCKED
            report["decision"] = "BLOCKED"
            return report
        report["decision"] = "PROCEED"
        return report

    # -- fingerprints -----------------------------------------------------------
    def architecture_fingerprint_of_checkpoint(self) -> str:
        ckpt = torch_load_checkpoint(str(self.checkpoint))
        model_cfg = resolve_model_config(ckpt)
        return architecture_fingerprint(model_cfg)

    # -- load -------------------------------------------------------------------
    def load(self, *, allow_full_model: bool = False) -> dict:
        """Load and validate the artifact. READY is set only on full success.

        ``allow_full_model=False`` refuses to instantiate the artifact at all —
        used by lightweight tests. Real loading requires the caller to opt in
        after preflight clears.
        """
        if not allow_full_model:
            self.state = LoadState.BLOCKED
            self.last_error = "load() requires allow_full_model=True after preflight"
            return {"state": self.state.value, "loaded": False, "reason": self.last_error}

        self.state = LoadState.LOADING
        started = time.perf_counter()
        try:
            mem_before = memory_snapshot()
            ckpt = torch_load_checkpoint(str(self.checkpoint))
            model_cfg = resolve_model_config(ckpt)
            fingerprint = architecture_fingerprint(model_cfg)

            # resolved-config fp32 upper bound; refuse before instantiating if impossible
            params_upper = estimate_model_bytes(model_cfg)
            mem_now = memory_snapshot()
            headroom = recommended_headroom(params_upper)
            if mem_now["available_bytes"] is None or mem_now["available_bytes"] < params_upper + headroom:
                avail = mem_now["available_bytes"]
                self.state = LoadState.BLOCKED
                self.last_error = (
                    f"resolved fp32 upper bound {params_upper} B + headroom {headroom} B "
                    f"exceeds available memory {avail} B"
                )
                return {"state": self.state.value, "loaded": False, "reason": self.last_error,
                        "estimated_upper_bound_bytes": params_upper}

            from training.checkpoint import load_model_weights
            from training.model import QMModel

            model = QMModel(model_cfg)
            report = load_model_weights(model, ckpt["model"], source=str(self.checkpoint))
            model = model.to(self.device).eval()

            tokenizer = TokenizerAdapter.load(self.tokenizer_path)
            tok_report = tokenizer.compatibility_report(model_cfg)
            if not tok_report["vocab_match"]:
                raise CheckpointIncompatibleError(
                    f"tokenizer vocab {tok_report['vocab_size']} != model vocab {tok_report['model_vocab_size']}"
                )

            ckpt_sha = file_sha256(self.checkpoint)
            self.model, self.model_cfg, self.tokenizer = model, model_cfg, tokenizer
            self.state = LoadState.READY
            mem_after = memory_snapshot()
            self.load_report = {
                "state": self.state.value,
                "loaded": True,
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": ckpt_sha,
                "checkpoint_size_bytes": self.checkpoint.stat().st_size,
                "model_config_fingerprint": fingerprint,
                "tokenizer_sha256": tokenizer.sha256,
                "parameters": int(model.num_params()),
                "device": self.device,
                "state_dict": report.summary() if hasattr(report, "summary") else {
                    "loaded": len(report.loaded), "missing": report.missing,
                    "unexpected": report.unexpected, "mismatched": report.mismatched},
                "memory": {"before": mem_before, "after": mem_after},
                "load_seconds": round(time.perf_counter() - started, 3),
            }
            return self.load_report
        except ResourceBlockedError:
            raise
        except CheckpointIncompatibleError as exc:
            self.state = LoadState.FAILED
            self.last_error = str(exc)
            return {"state": self.state.value, "loaded": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 — classify, never bare
            self.state = LoadState.FAILED
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {"state": self.state.value, "loaded": False, "reason": self.last_error}

    def unload(self) -> dict:
        self.model = None
        self.tokenizer = None
        self.model_cfg = None
        gc.collect()
        self.state = LoadState.UNLOADED
        return {"state": self.state.value, "loaded": False}

    @property
    def cancel_controller(self) -> CancellationController:
        return self._cancel

    # -- inference ---------------------------------------------------------------
    def _require_ready(self) -> None:
        if self.state is not LoadState.READY:
            raise EngineNotReady(f"engine state is {self.state.value}; load() must succeed first")

    def generate(self, request: InferenceRequest) -> "InferenceResponse":
        return list(self.stream(request))[-1].response  # type: ignore[union-attr]

    def stream(self, request: InferenceRequest) -> Iterator["InferenceChunk"]:
        self._require_ready()
        params = request.sampling.validate()
        assert self.tokenizer is not None and self.model is not None and self.model_cfg is not None

        stop_list = list(request.stop or [])
        if len(stop_list) > MAX_STOP_SEQUENCES:
            raise ConfigurationError(f"at most {MAX_STOP_SEQUENCES} stop sequences")
        for s in stop_list:
            if not isinstance(s, str) or not s or len(s) > MAX_STOP_SEQUENCE_CHARS:
                raise ConfigurationError("stop sequences must be non-empty strings <= 128 chars")
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ConfigurationError("prompt must be a non-empty string")
        if len(request.prompt) > MAX_PROMPT_CHARS:
            raise ConfigurationError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
        if request.timeout_seconds is not None and request.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive or null")
        if request.mode not in ("general", "model_only", "compiler_assisted"):
            raise ConfigurationError(f"mode must be general|model_only|compiler_assisted, got {request.mode!r}")

        context = ContextManager(int(self.model_cfg["max_seq_len"]))
        prompt_ids = self.tokenizer.encode(request.prompt)
        plan = context.plan(prompt_ids, params.max_new_tokens, request.truncation_policy)

        event = self._cancel.register(request.request_id)
        generator = SamplingController.make_generator(params.seed)
        deadline = time.monotonic() + request.timeout_seconds if request.timeout_seconds else None
        trace = InferenceTrace(
            request_id=request.request_id,
            model_config_fingerprint=architecture_fingerprint(self.model_cfg),
            tokenizer_sha256=getattr(self.tokenizer, "sha256", None),
            device=self.device,
            language_id=request.language_id,
            language_qualification_status=None,
            task_id=request.task_id,
            mode=request.mode,
            sampling=params.recorded(),
            prompt_tokens=plan.prompt_tokens,
            truncated=plan.truncated,
        )

        generated: list[int] = []
        finish_reason = "length"
        eos_id = self.tokenizer.eos_id()
        decoded_so_far = ""
        try:
            for step in range(params.max_new_tokens):
                if event.is_set():
                    finish_reason = "cancelled"
                    trace.cancelled = True
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    finish_reason = "timeout"
                    trace.timed_out = True
                    break
                try:
                    token = self._decode_step(plan.prompt_ids + generated, eos_id, generated,
                                              params, generator)
                except Exception as exc:  # noqa: BLE001 — classify, never bare
                    raise GenerationFailedError(
                        f"generation failed at step {step}: {type(exc).__name__}: {exc}"
                    ) from exc
                if token is None:               # EOS reached
                    finish_reason = "stop"
                    break
                generated.append(token)
                full_text = self.tokenizer.decode(generated)
                hit = next((s for s in stop_list if s and s in full_text), None)
                if hit is not None:
                    finish_reason = "stop"
                    generated = self.tokenizer.encode(full_text[: full_text.index(hit)])
                    break
                delta = full_text[len(decoded_so_far):]
                decoded_so_far = full_text
                yield InferenceChunk(
                    request_id=request.request_id, text=delta,
                    token_id=token, index=len(generated) - 1,
                )

            trace.completion_tokens = len(generated)
            trace.finish_reason = finish_reason
            response = InferenceResponse(
                request_id=request.request_id,
                text=self.tokenizer.decode(generated),
                token_ids=list(generated),
                finish_reason=finish_reason,
                prompt_tokens=plan.prompt_tokens,
                completion_tokens=len(generated),
                truncated=plan.truncated,
                context_limit=plan.context_limit,
                utilization=plan.utilization,
                sampling=params.recorded(),
                device=self.device,
                trace=trace,
            )
            yield InferenceChunk(request_id=request.request_id, final=True, response=response)
        finally:
            self._cancel.cleanup(request.request_id)

    def _decode_step(self, ids: list[int], eos_id: Optional[int], generated: list[int],
                     params: SamplingParams, generator: torch.Generator) -> Optional[int]:
        limit = int(self.model_cfg["max_seq_len"])  # type: ignore[index]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        if x.size(1) > limit:
            x = x[:, -limit:]
        logits, _ = self.model(x)  # type: ignore[operator]
        next_logits = _apply_penalties(logits[:, -1, :], generated,
                                       params.presence_penalty, params.frequency_penalty)
        tok = SamplingController.sample(next_logits, params, generator)
        if eos_id is not None and tok == eos_id:
            return None
        return tok


@dataclass
class InferenceResponse:
    request_id: str
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool
    context_limit: int
    utilization: float
    sampling: dict
    device: str
    trace: InferenceTrace


@dataclass
class InferenceChunk:
    request_id: str
    text: str = ""
    token_id: Optional[int] = None
    index: int = 0
    final: bool = False
    response: Optional[InferenceResponse] = None


def build_engine(checkpoint: str | Path, tokenizer_path: str | Path,
                 device: str = "cpu") -> ModelLoader:
    return ModelLoader(checkpoint, tokenizer_path, device)
