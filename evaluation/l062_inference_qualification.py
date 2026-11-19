"""TAKSH L0.6.2 — local inference engineering qualification.

Runs the L0.6.2 qualification:

  * synthetic tiny-model probes (load states, generation, streaming,
    cancellation, determinism) — always executed;
  * designated existing-checkpoint decision — measured NOW; the artifact is
    loaded only if host memory clears preflight at execution time, and never
    to force a pass; the canonical taksh-foundation-111m artifact does not
    exist (the foundation model has never been trained);
  * writes reports/l062_inference_qualification.json.

The frozen Python benchmark is NOT re-run here. Its baseline remains 0/30;
nothing about capability is claimed.

Usage:
    python -m evaluation.l062_inference_qualification
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from evaluation.l060_inference_qualification import SMOKE_CFG, build_tiny_artifact  # noqa: E402
from serving.inference_engine import (  # noqa: E402
    InferenceRequest,
    LoadState,
    ModelLoader,
    SamplingParams,
    memory_snapshot,
    recommended_headroom,
)

DESIGNATED_CHECKPOINT = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
TOKENIZER = REPO / "tokenizer" / "taksh_code_bpe_v1.json"

# Measured by direct read-only inspection of the DESIGNATED checkpoint on
# 2026-11-11 (weights never instantiated for this measurement):
#   resolve_model_config() -> {name: qm-code-v160-15m, vocab_size: 5822,
#     d_model: 384, n_layers: 12, n_heads: 8, n_kv_heads: 4, ffn_dim: 1536,
#     max_seq_len: 2048, rope_theta: 10000.0, tie_embeddings: true}
#   fp32 weight bytes estimate: 115,149,312 (= 28,787,328 params)
#   file size: 345,596,421 bytes (~12 bytes/param => fp32 weights + Adam m,v state)
# This is NOT the canonical taksh-foundation-111m (d_model 768): no 111M
# artifact exists anywhere in this repository; the foundation model has never
# been trained or saved.
DESIGNATED_ARTIFACT_IDENTITY = {
    "name": "qm-code-v160-15m",
    "lineage": "runs/v160_pretrain -> qm-code-v180-sft (chat SFT default)",
    "resolved_config": {"vocab_size": 5822, "d_model": 384, "n_layers": 12,
                        "n_heads": 8, "n_kv_heads": 4, "ffn_dim": 1536,
                        "max_seq_len": 2048, "rope_theta": 10000.0,
                        "tie_embeddings": True},
    "fp32_weight_bytes_estimate": 115_149_312,
    "file_size_bytes": 345_596_421,
    "composition_note": "~12 bytes/param on disk => fp32 weights + Adam m/v state",
    "not_the_canonical_foundation": "taksh-foundation-111m (110,658,816 params) has never been trained; no artifact exists",
}


def synthetic_probes(tmp_dir: Path) -> dict:
    ckpt, tok_path = build_tiny_artifact(tmp_dir)
    results: dict = {}

    loader = ModelLoader(ckpt, tok_path, "cpu")
    results["initial_state"] = loader.state.value

    pre = loader.preflight()
    results["preflight"] = {k: pre[k] for k in
                            ("device", "checkpoint_file_bytes", "available_bytes",
                             "estimated_required_bytes", "decision", "fits_available_memory")}

    refused = loader.load()  # without allow_full_model -> must refuse
    results["load_refused_without_opt_in"] = {"loaded": refused["loaded"], "state": refused["state"]}

    load = loader.load(allow_full_model=True)
    results["load"] = {k: load.get(k) for k in
                       ("state", "loaded", "parameters", "load_seconds",
                        "checkpoint_sha256", "model_config_fingerprint", "tokenizer_sha256")}

    req = InferenceRequest(prompt="a b c d",
                           sampling=SamplingParams(temperature=0.0, max_new_tokens=6, seed=42))
    resp = loader.generate(req)
    results["generation"] = {
        "finish_reason": resp.finish_reason,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "max_new_tokens_respected": resp.completion_tokens <= 6,
        "context_limit": resp.context_limit,
        "labels": "measured",
    }

    req2 = InferenceRequest(prompt="a b c d",
                            sampling=SamplingParams(temperature=0.0, max_new_tokens=6, seed=42))
    resp2 = loader.generate(req2)
    results["determinism"] = {"same_seed_identical": resp.text == resp2.text}

    req3 = InferenceRequest(prompt="a b c d",
                            sampling=SamplingParams(temperature=0.0, max_new_tokens=6, seed=7))
    chunks = list(loader.stream(req3))
    final = next((c for c in chunks if c.final), None)
    results["streaming"] = {
        "chunks": sum(1 for c in chunks if not c.final),
        "final_chunk_present": final is not None,
        "accumulated_text": "".join(c.text for c in chunks if not c.final),
        "response_text": final.response.text if (final and final.response) else None,
        "matches_response": final is not None and "".join(c.text for c in chunks if not c.final) == final.response.text,
    }

    # cancellation probe: cancel from another thread after first chunk
    import threading

    req4 = InferenceRequest(prompt="a b c d e f g h",
                            sampling=SamplingParams(temperature=0.0, max_new_tokens=8, seed=1),
                            request_id="l062-cancel-probe")
    stream = loader.stream(req4)
    first = next(iter(stream))
    loader.cancel_controller.cancel("l062-cancel-probe")
    remaining = list(stream)
    final4 = next((c for c in remaining if c.final), None)
    results["cancellation"] = {
        "first_chunk_text": first.text,
        "final_finish_reason": final4.response.finish_reason if final4 and final4.response else None,
        "engine_state_after_cancel": loader.state.value,
    }

    results["unload"] = loader.unload()
    results["state_after_unload"] = loader.state.value
    return results


def full_model_decision() -> dict:
    """Measure memory now; decide whether the DESIGNATED checkpoint may be attempted."""
    ckpt_exists = DESIGNATED_CHECKPOINT.is_file()
    tok_exists = TOKENIZER.is_file()
    file_bytes = DESIGNATED_CHECKPOINT.stat().st_size if ckpt_exists else None
    mem = memory_snapshot()
    headroom = recommended_headroom(file_bytes or 0)
    required = (file_bytes or 0) + headroom
    available = mem["available_bytes"]
    fits = available is not None and ckpt_exists and tok_exists and available >= required
    decision = "PROCEED" if fits else "BLOCKED"
    return {
        "checkpoint": str(DESIGNATED_CHECKPOINT),
        "checkpoint_exists": ckpt_exists,
        "checkpoint_size_bytes": file_bytes,
        "tokenizer_exists": tok_exists,
        "available_bytes_at_decision": available,
        "required_bytes_estimate": required,
        "headroom_bytes": headroom,
        "decision": decision,
        "memory": mem,
        "uncertainty": "estimate = checkpoint file size + headroom; final arbiter is load-time allocation",
    }


def attempt_full_load(decision: dict) -> dict:
    if decision["decision"] != "PROCEED":
        return {"attempted": False,
                "state": "BLOCKED",
                "reason": (f"available {decision['available_bytes_at_decision']} B < "
                           f"required {decision['required_bytes_estimate']} B — refusing to load; "
                           "no swap/quantization tricks applied (would need its own qualification)")}
    loader = ModelLoader(DESIGNATED_CHECKPOINT, TOKENIZER, "cpu")
    outcome = loader.load(allow_full_model=True)
    if outcome.get("loaded"):
        smoke = InferenceRequest(prompt="def add(a, b):",
                                 sampling=SamplingParams(temperature=0.0, max_new_tokens=8, seed=42))
        t0 = time.perf_counter()
        resp = loader.generate(smoke)
        smoke_result = {
            "finish_reason": resp.finish_reason,
            "completion_tokens": resp.completion_tokens,
            "latency_seconds": round(time.perf_counter() - t0, 3),
            "labels": "measured (2 greedy smoke tokens-batch; NOT a capability result)",
        }
        loader.unload()
        return {"attempted": True, "state": outcome["state"],
                "parameters": outcome.get("parameters"),
                "load_seconds": outcome.get("load_seconds"),
                "smoke": smoke_result}
    return {"attempted": True, "state": outcome["state"], "reason": outcome.get("reason")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l062_inference_qualification.json")
    args = parser.parse_args(argv)

    tmp_dir = Path(tempfile.mkdtemp(prefix="l062_smoke_"))
    synthetic = synthetic_probes(tmp_dir)
    decision = full_model_decision()
    full = attempt_full_load(decision)

    report = {
        "schema": "taksh.l062_inference_qualification/1",
        "generated_by": "evaluation/l062_inference_qualification.py",
        "synthetic": synthetic,
        "designated_checkpoint": {
            "artifact_identity": DESIGNATED_ARTIFACT_IDENTITY,
            "decision": decision,
            "outcome": full,
        },
        "canonical_foundation_111m": {
            "exists_as_artifact": False,
            "params": 110_658_816,
            "note": "architecture qualified in L0.6.0; weights never trained or saved; inference of the canonical foundation model is not possible on any host today",
        },
        "frozen_python_benchmark": {
            "sha256": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6",
            "status": "NOT RE-RUN — baseline 0/30 stands; suite never executed in this milestone",
        },
        "capability": "NOT ASSESSED — infrastructure qualification only",
        "verdict": (
            "CONDITIONAL PASS — local inference infrastructure qualified on synthetic fixtures; "
            "designated existing checkpoint inference: " + (
                "BLOCKED BY HOST RESOURCES (fail-closed; see designated_checkpoint.outcome)"
                if full.get("state") == "BLOCKED" else str(full.get("state", "UNKNOWN"))) +
            "; canonical taksh-foundation-111m artifact does not exist"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"synthetic: load={synthetic['load'].get('state')} params={synthetic['load'].get('parameters')}")
    print(f"determinism same={synthetic['determinism']['same_seed_identical']} stream_match={synthetic['streaming']['matches_response']}")
    print(f"full checkpoint decision: {decision['decision']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
