"""Model availability gate and session (ENG3).

Rules (mandatory):

- explicit checkpoint + tokenizer configuration, always — never an implicit default;
- the canonical ``taksh-foundation-111m`` identity is enforced: the resolved
  checkpoint configuration must match the canonical foundation geometry and its
  committed configuration fingerprint — ``v180_sft`` (d_model 384 / 28.8M) is
  rejected, never silently substituted;
- a missing or incompatible checkpoint yields MODEL_UNAVAILABLE or
  GENERATION_BLOCKED, never a fallback;
- inference uses the existing ``serving.inference_engine`` (no second engine);
- model output is tagged ``MODEL_GENERATED`` and is never tool-observed,
  tool-verified or human-confirmed;
- test doubles (``StaticModelDouble``) are labelled ``test_double`` and never
  enter the canonical gate.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from serving.inference_engine import (  # noqa: E402 — production engine is the only path
    InferenceRequest, ModelLoader, SamplingParams, architecture_fingerprint,
    file_sha256, estimate_model_bytes, canonical_model_cfg,
)
from taksh.dlcae.types import fingerprint, utcnow_iso

FOUNDATION_CFG_SHA256 = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
CANONICAL_MODEL_ID = "taksh-foundation-111m"
CANONICAL_GEOMETRY = {"vocab_size": 5822, "d_model": 768, "n_layers": 12,
                      "n_heads": 12, "n_kv_heads": 6, "ffn_dim": 3072,
                      "max_seq_len": 2048}


class ModelCapabilityState:
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_READY = "MODEL_READY"
    GENERATION_COMPLETED = "GENERATION_COMPLETED"
    GENERATION_FAILED = "GENERATION_FAILED"
    GENERATION_BLOCKED = "GENERATION_BLOCKED"

    ALL = frozenset({MODEL_UNAVAILABLE, MODEL_READY, GENERATION_COMPLETED,
                     GENERATION_FAILED, GENERATION_BLOCKED})


def canonical_fingerprint() -> str:
    cfg = canonical_model_cfg()
    return architecture_fingerprint(cfg)


def geometry_of(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg.get(k) for k in CANONICAL_GEOMETRY}


def canonical_geometry_match(cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    resolved = geometry_of(cfg)
    mismatches = {k: (resolved.get(k), v) for k, v in CANONICAL_GEOMETRY.items()
                  if resolved.get(k) != v}
    return (not mismatches), mismatches


class ModelSession:
    """An explicitly configured, gated model session.

    ``checkpoint=None`` means "no artifact configured" — the session reports
    MODEL_UNAVAILABLE and every generation request returns GENERATION_BLOCKED.
    """

    def __init__(self, checkpoint: str | Path | None, tokenizer: str | Path | None, *,
                 device: str = "cpu", allow_full_model: bool = False,
                 model_kind: str = "canonical") -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.tokenizer = Path(tokenizer) if tokenizer else None
        self.device = device
        self.allow_full_model = allow_full_model
        self.model_kind = model_kind  # canonical | test_double
        self.state = ModelCapabilityState.MODEL_UNAVAILABLE
        self.reason: str | None = None
        self._loader: ModelLoader | None = None
        self.last_trace: dict[str, Any] | None = None
        self.last_trace_fingerprint: str | None = None
        self.generation_config: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def assess(self) -> dict[str, Any]:
        """Fail-closed availability assessment. Nothing is invented."""
        problems: list[str] = []
        if self.model_kind == "test_double":
            return self._test_double_assessment()
        if self.checkpoint is None or not str(self.checkpoint).strip():
            problems.append("no checkpoint configured (explicit artifact required; none provided)")
        elif not self.checkpoint.is_file():
            problems.append(f"checkpoint not found: {self.checkpoint}")
        if self.tokenizer is None or not str(self.tokenizer).strip():
            problems.append("no tokenizer configured (explicit artifact required)")
        elif not self.tokenizer.is_file():
            problems.append(f"tokenizer not found: {self.tokenizer}")
        if problems:
            self.state = ModelCapabilityState.MODEL_UNAVAILABLE
            self.reason = "; ".join(problems)
            return self._report(problems)

        try:
            from serving.local_inference import resolve_model_config, torch_load_checkpoint

            ckpt = torch_load_checkpoint(str(self.checkpoint))
            cfg = resolve_model_config(ckpt)
        except Exception as exc:  # noqa: BLE001
            self.state = ModelCapabilityState.MODEL_UNAVAILABLE
            self.reason = f"checkpoint unreadable: {type(exc).__name__}: {exc}"
            return self._report([self.reason])

        resolved_fingerprint = architecture_fingerprint(cfg)
        canonical_fp = canonical_fingerprint()
        geometry_ok, geometry_mismatch = canonical_geometry_match(cfg)
        canonical_match = geometry_ok and resolved_fingerprint == canonical_fp
        ckpt_sha = file_sha256(self.checkpoint)
        tok_sha = file_sha256(self.tokenizer)

        if not canonical_match:
            self.state = ModelCapabilityState.MODEL_UNAVAILABLE
            self.reason = (
                "configured checkpoint does not match taksh-foundation-111m "
                f"(geometry_mismatch={geometry_mismatch}; "
                f"fingerprint_match={resolved_fingerprint == canonical_fp}). "
                "GEOMETRY_INCOMPATIBLE artifacts (e.g. v180_sft, d_model 384) are "
                "never substituted")
            return self._report([self.reason], fingerprints={
                "checkpoint_sha256": ckpt_sha, "tokenizer_sha256": tok_sha,
                "resolved_fingerprint": resolved_fingerprint,
                "canonical_fingerprint": canonical_fp,
                "geometry_mismatch": geometry_mismatch})

        # host capacity check before declaring READY
        loader = ModelLoader(self.checkpoint, self.tokenizer, self.device)
        preflight = loader.preflight()
        if preflight.get("decision") != "PROCEED":
            self.state = ModelCapabilityState.MODEL_UNAVAILABLE
            self.reason = ("host preflight blocked load: "
                           + "; ".join(preflight.get("problems", ["unknown"])))
            return self._report([self.reason], fingerprints={
                "checkpoint_sha256": ckpt_sha, "tokenizer_sha256": tok_sha,
                "resolved_fingerprint": resolved_fingerprint,
                "canonical_fingerprint": canonical_fp,
                "preflight": preflight})

        if self.allow_full_model:
            report = loader.load(allow_full_model=True)
            if report.get("state") == "READY":
                self.state = ModelCapabilityState.MODEL_READY
                self.reason = None
                self._loader = loader
                self.generation_config = {"loader": "serving.inference_engine.ModelLoader",
                                          "device": self.device}
                return self._report([], fingerprints={
                    "checkpoint_sha256": ckpt_sha, "tokenizer_sha256": tok_sha,
                    "resolved_fingerprint": resolved_fingerprint,
                    "canonical_fingerprint": canonical_fp,
                    "parameters": report.get("parameters")})
            self.state = ModelCapabilityState.MODEL_UNAVAILABLE
            self.reason = f"load blocked: {report.get('reason')}"
            return self._report([self.reason], fingerprints={
                "checkpoint_sha256": ckpt_sha, "tokenizer_sha256": tok_sha,
                "resolved_fingerprint": resolved_fingerprint})
        self._loader = loader
        self.generation_config = {"loader": "serving.inference_engine.ModelLoader",
                                  "device": self.device, "load": "NOT_INSTANTIATED"}
        return self._report([], fingerprints={
            "checkpoint_sha256": ckpt_sha, "tokenizer_sha256": tok_sha,
            "resolved_fingerprint": resolved_fingerprint,
            "canonical_fingerprint": canonical_fp,
            "note": "assessment-only mode: artifact verified but not instantiated "
                    "(callers must opt in explicitly)"})

    def _test_double_assessment(self) -> dict[str, Any]:
        self.state = ModelCapabilityState.MODEL_READY
        return {"state": "MODEL_READY",
                "model_kind": "test_double",
                "canonical": False,
                "reason": "deterministic test double — never the canonical model, "
                          "never capability evidence",
                "fingerprints": {"checkpoint_sha256": None, "tokenizer_sha256": None,
                                 "resolved_fingerprint": None,
                                 "canonical_fingerprint": canonical_fingerprint()},
                "checked": utcnow_iso(),
                "problems": []}

    def _report(self, problems: list[str], *, fingerprints: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"state": self.state,
                "model_kind": self.model_kind,
                "canonical": self.model_kind == "canonical",
                "model_id": CANONICAL_MODEL_ID if self.model_kind == "canonical" else "test-double",
                "reason": self.reason,
                "problems": problems,
                "fingerprints": fingerprints or {},
                "checked": utcnow_iso()}

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, *, seed: int | None = None,
                 max_new_tokens: int = 64, temperature: float = 0.0,
                 task_id: str | None = None, mode: str = "model_only") -> dict[str, Any]:
        """Generate once per the configured gate. Blocked unless MODEL_READY."""
        if self.state is ModelCapabilityState.MODEL_UNAVAILABLE and self.model_kind != "test_double":
            if self._loader is None and self.checkpoint is not None:
                assessment = self.assess()
                if assessment["state"] != ModelCapabilityState.MODEL_READY:
                    self.state = ModelCapabilityState.GENERATION_BLOCKED
                    return self._generation_blocked(assessment["reason"])
            else:
                self.state = ModelCapabilityState.GENERATION_BLOCKED
                return self._generation_blocked(self.reason or "model not available")
        started = time.perf_counter()
        try:
            request = InferenceRequest(
                prompt=prompt,
                sampling=SamplingParams(temperature=temperature, seed=seed,
                                        max_new_tokens=max_new_tokens),
                task_id=task_id, language_id="python", mode=mode)
            response = self._loader.generate(request)
        except Exception as exc:  # noqa: BLE001
            self.state = ModelCapabilityState.GENERATION_FAILED
            return {"state": self.state,
                    "error": f"{type(exc).__name__}: {exc}",
                    "mode": mode, "model_kind": self.model_kind,
                    "capability_claim": None,
                    "generated_at": utcnow_iso()}
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        self.state = ModelCapabilityState.GENERATION_COMPLETED
        trace = response.trace.to_eval_record() if hasattr(response, "trace") else {}
        self.generation_config = {"temperature": temperature, "seed": seed,
                                  "max_new_tokens": max_new_tokens, "mode": mode}
        self.last_trace = trace
        self.last_trace_fingerprint = fingerprint(trace)
        return {"state": self.state,
                "text": response.text,
                "trace_fingerprint": self.last_trace_fingerprint,
                "model_kind": self.model_kind,
                "model_id": ("taksh-foundation-111m" if self.model_kind == "canonical"
                             else "static-test-double"),
                "model_identity": {
                    "canonical_id": CANONICAL_MODEL_ID,
                    "model_config_fingerprint": trace.get("model_config_fingerprint"),
                    "tokenizer_sha256": trace.get("tokenizer_sha256"),
                    "checkpoint_sha256": (self._loader.load_report.get("checkpoint_sha256")
                                          if self._loader else None),
                    "device": response.device,
                },
                "generation": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "finish_reason": response.finish_reason,
                    "latency_ms": elapsed_ms,
                    "sampling": self.generation_config,
                },
                "trace": trace,
                "evidence_category": "MODEL_GENERATED",
                "capability_claim": None,
                "generated_at": utcnow_iso()}

    def _generation_blocked(self, reason: str) -> dict[str, Any]:
        self.last_trace = None
        self.last_trace_fingerprint = fingerprint({"blocked": reason, "at": utcnow_iso()})
        return {"state": self.state, "reason": reason,
                "model_kind": self.model_kind, "capability_claim": None,
                "evidence_category": "MODEL_GENERATED",
                "generated_at": utcnow_iso()}


class StaticModelDouble:
    """Deterministic test double implementing the model-session surface.

    Always ``model_kind='test_double'``; never presented as TAKSH output; never
    canonical; output only exercises the pipeline machinery.
    """

    model_kind = "test_double"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls = 0
        self.state = "MODEL_READY"
        self.last_trace = None
        self.last_trace_fingerprint = None
        self.generation_config = {"model_kind": "test_double"}

    def assess(self) -> dict[str, Any]:
        return {"state": "MODEL_READY", "model_kind": "test_double",
                "canonical": False, "problems": [],
                "reason": "deterministic test double — never the canonical model",
                "checked": utcnow_iso()}

    def generate(self, prompt: str, *, seed: int | None = None,
                 max_new_tokens: int = 64, temperature: float = 0.0,
                 task_id: str | None = None, mode: str = "model_only") -> dict[str, Any]:
        self.calls += 1
        if self.calls - 1 < len(self.responses):
            text = self.responses[self.calls - 1]
        else:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            text = f"# static-test-double response for {digest}\n"
        self.last_trace_fingerprint = fingerprint({"double": text[:64], "call": self.calls})
        return {"state": ModelCapabilityState.GENERATION_COMPLETED,
                "text": text,
                "trace_fingerprint": self.last_trace_fingerprint, "model_kind": "test_double",
                "model_id": "static-test-double",
                "model_identity": {"canonical_id": None,
                                   "model_config_fingerprint": None,
                                   "tokenizer_sha256": None,
                                   "checkpoint_sha256": None,
                                   "device": "none (test double)"},
                "generation": {"prompt_tokens": len(prompt.split()),
                               "completion_tokens": len(text.split()),
                               "finish_reason": "stop",
                               "latency_ms": 0.0,
                               "sampling": {"seed": seed, "temperature": temperature,
                                            "max_new_tokens": max_new_tokens,
                                            "mode": mode}},
                "trace": {"kind": "test_double", "call": self.calls},
                "evidence_category": "MODEL_GENERATED",
                "capability_claim": None,
                "generated_at": utcnow_iso()}
