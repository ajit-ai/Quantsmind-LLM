"""Focused tests for the TAKSH L0.6.2 local-inference qualification.

Synthetic tiny models + temporary artifacts only. The full 111M artifact is
never instantiated here. The frozen capability benchmark is never executed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import serving.inference_engine as eng  # noqa: E402
from evaluation import l062_dlcae_inference_interface as dlcae  # noqa: E402
from evaluation import l062_karkain_inference as kark  # noqa: E402
from evaluation.l060_inference_qualification import SMOKE_CFG, build_tiny_artifact  # noqa: E402
from serving.local_api import create_app  # noqa: E402
from serving.inference_engine import (  # noqa: E402
    ConfigurationError,
    ContextManager,
    ContextOverflowError,
    InferenceRequest,
    LoadState,
    ModelLoader,
    SamplingParams,
    SamplingController,
)

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
CANONICAL_ARCH_FINGERPRINT = "92da3497cec631b49f7a4c2be7a60cfb74e95c2e8e248a102546f8a4a99d1c14"


@pytest.fixture()
def tiny(tmp_path):
    ckpt, tok = build_tiny_artifact(tmp_path)
    return ckpt, tok


@pytest.fixture()
def ready_loader(tmp_path):
    ckpt, tok = build_tiny_artifact(tmp_path)
    loader = ModelLoader(ckpt, tok, "cpu")
    loader.load(allow_full_model=True)
    return loader


def req(prompt="a b c d", **kwargs):
    sampling_keys = {"temperature", "top_k", "top_p", "presence_penalty",
                     "frequency_penalty", "seed", "max_new_tokens"}
    sampling = {k: kwargs.pop(k) for k in list(kwargs) if k in sampling_keys}
    sampling.setdefault("max_new_tokens", 4)
    sampling.setdefault("seed", 42)
    sampling.setdefault("temperature", 0.0)
    return InferenceRequest(prompt=prompt, sampling=SamplingParams(**sampling), **kwargs)


# --------------------------------------------------------------------------- #
# 1. Model loading + 2. Configuration compatibility
# --------------------------------------------------------------------------- #
def test_initial_state_is_not_loaded(tiny):
    ckpt, tok = tiny
    assert ModelLoader(ckpt, tok, "cpu").state is LoadState.NOT_LOADED


def test_load_requires_explicit_opt_in(tiny):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    out = loader.load()
    assert out["loaded"] is False
    assert loader.state is LoadState.BLOCKED


def test_load_success_sets_ready_and_reports(tiny):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    out = loader.load(allow_full_model=True)
    assert out["loaded"] is True
    assert loader.state is LoadState.READY
    assert out["parameters"] == 10816
    assert len(out["checkpoint_sha256"]) == 64


def test_load_missing_checkpoint_fails_closed(tiny):
    _, tok = tiny
    loader = ModelLoader(Path("definitely_missing_ckpt.pt"), tok, "cpu")
    out = loader.load(allow_full_model=True)
    assert out["loaded"] is False
    assert loader.state is LoadState.FAILED
    assert out["reason"]


def test_unload_releases_model(ready_loader):
    ready_loader.unload()
    assert ready_loader.state is LoadState.UNLOADED
    with pytest.raises(Exception):
        ready_loader.generate(req())


def test_discover_checkpoints(tmp_path, tiny):
    ckpt, _ = tiny
    found = ModelLoader.discover_checkpoints(tmp_path)
    assert any(Path(f["path"]).samefile(ckpt) and len(f["sha256"]) == 64 for f in found)


def test_architecture_fingerprint_canonical_stable():
    cfg = eng.canonical_model_cfg()
    assert eng.architecture_fingerprint(cfg) == CANONICAL_ARCH_FINGERPRINT


def test_estimated_foundation_bytes_matches_known_params():
    assert eng.estimate_model_bytes(eng.canonical_model_cfg()) == 110_658_816 * 4


def test_architecture_fingerprint_differs_tiny():
    assert eng.architecture_fingerprint(SMOKE_CFG) != CANONICAL_ARCH_FINGERPRINT


# --------------------------------------------------------------------------- #
# 3. Tokenizer compatibility
# --------------------------------------------------------------------------- #
def test_tokenizer_fingerprint_matches_file(tiny):
    from serving.inference_engine import TokenizerAdapter

    _, tok = tiny
    adapter = TokenizerAdapter.load(tok)
    assert adapter.sha256 == hashlib.sha256(Path(tok).read_bytes()).hexdigest()
    report = adapter.compatibility_report(SMOKE_CFG)
    assert report["vocab_match"] is True


def test_tokenizer_vocab_mismatch_rejected(tmp_path, tiny):
    from serving.inference_engine import TokenizerAdapter

    ckpt, _ = tiny
    bad_cfg = dict(SMOKE_CFG, vocab_size=SMOKE_CFG["vocab_size"] + 1)
    adapter = TokenizerAdapter.load(tiny[1])
    assert adapter.compatibility_report(bad_cfg)["vocab_match"] is False


# --------------------------------------------------------------------------- #
# 4. Context enforcement
# --------------------------------------------------------------------------- #
def test_context_rejects_overlong_prompt():
    cm = ContextManager(8)
    with pytest.raises(ContextOverflowError):
        cm.plan([1] * 10, 1, "reject")


def test_context_rejects_prompt_plus_generation():
    cm = ContextManager(8)
    with pytest.raises(ContextOverflowError):
        cm.plan([1] * 6, 3, "reject")


def test_context_left_truncation_explicit():
    cm = ContextManager(8)
    plan = cm.plan(list(range(10)), 1, "left")
    assert plan.truncated is True
    assert plan.prompt_tokens == 7
    assert plan.prompt_ids == list(range(3, 10))
    assert plan.prompt_tokens + plan.max_new_tokens <= cm.context_limit


def test_context_utilization_reported():
    cm = ContextManager(16)
    plan = cm.plan([1] * 4, 4, "reject")
    assert plan.utilization == pytest.approx(0.5)
    assert plan.remaining_total_budget == 8


# --------------------------------------------------------------------------- #
# 5. Generation parameter validation + 6. deterministic generation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1}, {"temperature": 2.1}, {"temperature": True},
    {"top_k": -1}, {"top_k": 1.5},
    {"top_p": 0.0}, {"top_p": 1.01},
    {"presence_penalty": 2.1}, {"frequency_penalty": -2.1},
    {"seed": 1.5}, {"max_new_tokens": 0}, {"max_new_tokens": 257},
])
def test_invalid_sampling_params_rejected(kwargs):
    merged = {"max_new_tokens": 4, **kwargs}
    with pytest.raises(ConfigurationError):
        SamplingParams(**merged).validate()


def test_empty_prompt_rejected(ready_loader):
    with pytest.raises(ConfigurationError):
        ready_loader.generate(req(prompt="   "))


def test_determinism_same_seed_identical(tiny):
    ckpt, tok = tiny
    loaders = []
    for _ in range(2):
        loader = ModelLoader(ckpt, tok, "cpu")
        loader.load(allow_full_model=True)
        loaders.append(loader)
    outs = [l.generate(req()).token_ids for l in loaders]
    assert outs[0] == outs[1]


def test_determinism_greedy_two_runs(ready_loader):
    first = ready_loader.generate(req())
    second = ready_loader.generate(req())
    assert first.token_ids == second.token_ids
    assert first.sampling["deterministic"] is True


def test_sampling_seed_isolation_concurrent(ready_loader):
    results: dict[int, list[list[int]]] = {1: [], 2: []}

    def worker(seed: int):
        for _ in range(2):
            results[seed].append(
                ready_loader.generate(req(temperature=1.0, top_k=0, max_new_tokens=12,
                                          seed=seed)).token_ids)

    threads = [threading.Thread(target=worker, args=(s,)) for s in (1, 2, 1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[1][0] == results[1][1]       # same seed, same sequence under contention
    assert results[2][0] == results[2][1]


# --------------------------------------------------------------------------- #
# 7. Sampling controller behaviour
# --------------------------------------------------------------------------- #
def test_greedy_sampling_picks_argmax():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0
    g = SamplingController.make_generator(123)
    assert SamplingController.sample(logits, SamplingParams(temperature=0.0), g) == 3


def test_top_k_one_is_argmax_under_random_sampling():
    logits = torch.randn(1, 50)
    g = SamplingController.make_generator(1)
    assert SamplingController.sample(logits, SamplingParams(temperature=1.0, top_k=1), g) == int(
        torch.argmax(logits).item())


def test_seeded_multinomial_reproducible():
    logits = torch.randn(1, 30)
    g1 = SamplingController.make_generator(99)
    g2 = SamplingController.make_generator(99)
    p = SamplingParams(temperature=0.9, top_k=0, top_p=1.0)
    assert SamplingController.sample(logits, p, g1) == SamplingController.sample(logits, p, g2)


# --------------------------------------------------------------------------- #
# 8. Streaming + 9. cancellation + 10. timeout + 11. concurrency isolation
# --------------------------------------------------------------------------- #
def test_streaming_chunks_cover_response_text(ready_loader):
    req_obj = req(max_new_tokens=4)
    chunks = list(ready_loader.stream(req_obj))
    token_chunks = [c for c in chunks if not c.final]
    final = next(c for c in chunks if c.final)
    assert "".join(c.text for c in token_chunks) == final.response.text
    assert [c.index for c in token_chunks] == list(range(len(token_chunks)))
    assert final.response.completion_tokens == len(token_chunks)


def test_streaming_cancel_mid_generation(ready_loader):
    req_obj = req(max_new_tokens=8, request_id="cancel-me")
    stream = ready_loader.stream(req_obj)
    next(iter(stream))                      # pull one chunk
    assert ready_loader.cancel_controller.cancel("cancel-me") is True
    remainder = list(stream)
    final = next(c for c in remainder if c.final)
    assert final.response.finish_reason == "cancelled"
    assert ready_loader.state is LoadState.READY   # cancellation never corrupts the engine


def test_cancel_unknown_request_id(ready_loader):
    assert ready_loader.cancel_controller.cancel("nope") is False


def test_timeout_produces_timeout_finish(ready_loader):
    req_obj = req(max_new_tokens=8, timeout_seconds=1e-6, request_id="timeout-me")
    final = next(c for c in ready_loader.stream(req_obj) if c.final)
    assert final.response.finish_reason == "timeout"
    assert final.response.completion_tokens == 0
    assert ready_loader.state is LoadState.READY


def test_invalid_timeout_rejected(ready_loader):
    req_obj = req(max_new_tokens=4, timeout_seconds=0.0)
    with pytest.raises(ConfigurationError):
        list(ready_loader.stream(req_obj))


def test_stop_sequence_terminates_early(ready_loader):
    first_pass = ready_loader.generate(req(max_new_tokens=6))
    word = next((w for w in first_pass.text.split() if w), None)
    if word is None:
        pytest.skip("tiny greedy completion produced no decodable word")
    second_pass = ready_loader.generate(req(max_new_tokens=6, stop=[word]))
    assert second_pass.finish_reason == "stop"
    assert word not in second_pass.text


# --------------------------------------------------------------------------- #
# 12. Resource preflight (fail-closed)
# --------------------------------------------------------------------------- #
def test_memory_snapshot_measured_or_honest():
    snap = eng.memory_snapshot()
    assert "measured" in snap
    if snap["measured"]:
        assert snap["total_bytes"] and snap["total_bytes"] > 0


def test_headroom_floor_and_scaling():
    assert eng.recommended_headroom(0) == 32 * 1024 * 1024
    assert eng.recommended_headroom(442_635_264) == 442_635_264 // 4


def test_preflight_blocks_when_memory_insufficient(tiny, monkeypatch):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    monkeypatch.setattr(eng, "memory_snapshot",
                        lambda: {"total_bytes": 10, "available_bytes": 1,
                                 "percent_used": 100.0, "process_rss_bytes": 1,
                                 "tool": "test-fixture", "measured": True})
    out = loader.preflight()
    assert out["decision"] == "BLOCKED"
    assert out["fits_available_memory"] is False


def test_preflight_proceeds_for_tiny_artifact(tiny, monkeypatch):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    monkeypatch.setattr(eng, "memory_snapshot",
                        lambda: {"total_bytes": 16 * 2**30, "available_bytes": 12 * 2**30,
                                 "percent_used": 25.0, "process_rss_bytes": 1 * 2**30,
                                 "tool": "test-fixture", "measured": True})
    out = loader.preflight()
    assert out["decision"] == "PROCEED"


def test_preflight_blocks_when_checkpoint_missing(tiny, monkeypatch):
    _, tok = tiny
    loader = ModelLoader(Path("missing_ckpt.pt"), tok, "cpu")
    monkeypatch.setattr(eng, "memory_snapshot",
                        lambda: {"total_bytes": 16 * 2**30, "available_bytes": 12 * 2**30,
                                 "percent_used": 25.0, "process_rss_bytes": 1 * 2**30,
                                 "tool": "test-fixture", "measured": True})
    out = loader.preflight()
    assert out["decision"] == "BLOCKED"
    assert any("not found" in p for p in out["problems"])


# --------------------------------------------------------------------------- #
# 13. API request/response + 14. structured errors
# --------------------------------------------------------------------------- #
def _client(tiny, ready=True):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    if ready:
        loader.load(allow_full_model=True)
    from fastapi.testclient import TestClient

    return TestClient(create_app(loader)), loader


def test_api_health(tiny):
    client, _ = _client(tiny)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "local"
    assert body["external_providers"] == "disabled"


def test_api_models_before_and_after_load(tiny):
    client, _ = _client(tiny, ready=False)
    body = client.get("/v1/models").json()
    assert body["data"] == []
    assert "unavailable_reason" in body
    client, _ = _client(tiny, ready=True)
    body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == "taksh-foundation-111m"
    assert body["data"][0]["quantsmind"]["capability_claim"] is None
    assert len(body["data"][0]["quantsmind"]["checkpoint_fingerprint"]) == 64


def test_api_resource_blocked_when_not_loaded(tiny):
    client, _ = _client(tiny, ready=False)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}]})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "resource_blocked"


def test_api_chat_completion_subset(tiny):
    client, _ = _client(tiny)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "a b c"}],
        "max_tokens": 3, "temperature": 0, "seed": 42})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["provider"] == "local"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert body["quantsmind"]["capability_claim"] is None


def test_api_streaming_sse(tiny):
    client, _ = _client(tiny)
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "a b c"}],
            "max_tokens": 2, "temperature": 0, "stream": True}) as resp:
        payload = "".join(resp.iter_text())
    assert "chat.completion.chunk" in payload
    assert "data: [DONE]" in payload


def test_api_rejects_unsupported_param_tools(tiny):
    client, _ = _client(tiny)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "a b c"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "configuration_error"


def test_api_validation_error_envelope(tiny):
    client, _ = _client(tiny)
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_api_native_generate_and_cancel(tiny):
    client, _ = _client(tiny)
    resp = client.post("/v1/generate", json={
        "prompt": "a b c", "max_new_tokens": 2, "temperature": 0.0, "seed": 42,
        "language_id": "python", "task_id": "t-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "local"
    assert body["trace"]["language_id"] == "python"
    assert body["labels"]["token_counts"] == "measured"
    cancel = client.post("/v1/cancel", json={"request_id": "does-not-exist"}).json()
    assert cancel["cancelled"] is False


# --------------------------------------------------------------------------- #
# 15. Multilingual metadata + 16. DLCAE integration
# --------------------------------------------------------------------------- #
def test_dlcae_identify_declared_and_filename():
    assert dlcae.identify_language(declared="python")["language_id"] == "python"
    assert dlcae.identify_language(filename="kernel.kark")["language_id"] == "karkain"
    assert dlcae.identify_language()["language_id"] is None


def test_dlcae_registry_lookup_base_language():
    info = dlcae.registry_lookup("python")
    assert info["registered"] is True
    assert info["status"] == "REGISTERED"
    with_kark = dlcae.registry_lookup("karkain")
    assert with_kark["registered"] is True
    assert with_kark["status"] == "DISCOVERED"
    assert dlcae.registry_lookup("not_a_language_x")["status"] == "UNREGISTERED"


def test_dlcae_inference_never_upgrades_registry():
    before = dlcae.registry_lookup("karkain")["status"]
    dlcae.record_inference_feedback({"task_id": "t", "language_id": "karkain",
                                     "mode": "model_only", "finish_reason": "length",
                                     "completion_tokens": 4})
    proposal = dlcae.propose_registry_update(dlcae.registry_lookup("karkain"),
                                             "COMPREHENSION_IN_PROGRESS")
    assert dlcae.registry_lookup("karkain")["status"] == before
    assert proposal["state"] == "PENDING_OWNER_REVIEW"
    assert proposal["written"] is False


def test_dlcae_knowledge_defaults_empty_and_honest():
    out = dlcae.retrieve_knowledge("hello", "python")
    assert out["items"] == []
    assert out["source_status"] == "NO_KNOWLEDGE_BASE"


def test_dlcae_build_prompt_requires_instruction():
    with pytest.raises(ValueError):
        dlcae.build_prompt("t", "python", "   ")
    out = dlcae.build_prompt("t", "python", "write a function")
    assert out["language_id"] == "python"
    assert "write a function" in out["prompt"]


# --------------------------------------------------------------------------- #
# 17. Karkain integration contract
# --------------------------------------------------------------------------- #
def test_karkain_identify_variants():
    assert kark.identify_karkain_prompt(filename="app.kark")["is_karkain"] is True
    assert kark.identify_karkain_prompt(language_hint="Karkain")["is_karkain"] is True
    assert kark.identify_karkain_prompt("hello world")["is_karkain"] is False


def test_karkain_metadata_pinned():
    meta = kark.karkain_inference_metadata()
    assert meta["spec_version"] == "1.1.0"
    assert meta["kir_version"] == "KIR v1 (SSA)"
    assert meta["qualification_status"] == "DISCOVERED"
    assert meta["training_eligibility"] == []


def test_karkain_validate_passes():
    assert kark.validate() == []


def test_karkain_compiler_assisted_carries_no_fake_evidence():
    ctx = kark.build_karkain_context("do x", mode="compiler_assisted")
    assert ctx["evidence"] == "none"
    assert "0xC0000135" in ctx["note"] or "exit 3" in ctx["note"]
    with pytest.raises(ValueError):
        kark.build_karkain_context("do x", mode="compiler_assisted_plus_plus")


def test_karkain_task_classification():
    assert kark.classify_karkain_task("fix the bug")["task_class"] == "bugfix"
    assert kark.classify_karkain_task("dump the KIR")["task_class"] == "kir_inspection"
    assert kark.classify_karkain_task("write something")["task_class"] == "feature_impl"


# --------------------------------------------------------------------------- #
# 18. Observability + request replay
# --------------------------------------------------------------------------- #
def test_trace_eval_record_shape(ready_loader):
    out = ready_loader.generate(req(language_id="python", task_id="task-1", mode="model_only"))
    record = out.trace.to_eval_record()
    for key in ("schema", "request_id", "task_id", "language_id", "mode",
                "model_config_fingerprint", "sampling", "prompt_tokens",
                "completion_tokens", "finish_reason", "capability_claim"):
        assert key in record
    assert record["capability_claim"] is None
    assert record["model_config_fingerprint"] == eng.architecture_fingerprint(ready_loader.model_cfg)


def test_request_replay_identical(ready_loader):
    req_obj = req(max_new_tokens=5)
    first = ready_loader.generate(req_obj)
    second = ready_loader.generate(req_obj)
    assert first.token_ids == second.token_ids
    assert first.sampling == second.sampling


def test_checkpoint_tokenizer_fingerprints_match_load_report(tiny):
    ckpt, tok = tiny
    loader = ModelLoader(ckpt, tok, "cpu")
    loader.load(allow_full_model=True)
    assert loader.load_report["checkpoint_sha256"] == hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert loader.load_report["tokenizer_sha256"] == hashlib.sha256(Path(tok).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    path = REPO / "evaluation" / "l02_python_bench_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_l01_checkpoint_preserved():
    path = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_foundation_config_preserved():
    path = REPO / "configs" / "foundation" / "l060_foundation.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FOUNDATION_SHA


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA
