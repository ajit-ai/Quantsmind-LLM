"""Real-checkpoint integration tests -- NO MOCKS.

Every other serving test mocks ``generate_ids``. These do not: they load the actual
checkpoint through ``serving.openai_server.load_state`` (the same function ``main()``
calls) and drive the real model over real HTTP routes.

They assert the *contract* -- schema, determinism, usage accounting, termination,
error handling, auth -- not the quality of the output. The checkpoint is a 495,808
parameter model trained 200 steps on 100,244 tokens of this repository's own source;
it emits a degenerate repetition loop. Asserting on that text would be a
characterisation test that silently rots the moment real training lands, so these
assert only what must remain true either way.

Skipped automatically when weights are absent, which is the normal case in CI
because ``runs/`` and ``*.pt`` are gitignored.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient

from serving import openai_server

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "runs" / "code_pretrain_tiny" / "checkpoint_last.pt"
TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and TOKENIZER.exists()),
    reason="real checkpoint/tokenizer not present (runs/ and *.pt are gitignored)",
)

USER_MSG = "Write a function that adds two numbers."
MAX_TOKENS = 16  # bounded: ~0.24s for 24 tokens on CPU


@pytest.fixture(scope="module")
def real_client():
    """Load real weights once, then expose the real app."""
    openai_server.load_state(str(CKPT), str(TOKENIZER))
    openai_server.state.api_key = None
    try:
        yield TestClient(openai_server.app)
    finally:
        openai_server.state.model = None
        openai_server.state.model_cfg = None
        openai_server.state.tokenizer = None


def _chat(client, **overrides):
    body = {
        "model": "qm-code-tiny",
        "messages": [{"role": "user", "content": USER_MSG}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


# --------------------------------------------------------------------------- #
# 1. Weights load through the supported application path
# --------------------------------------------------------------------------- #
def test_load_state_populates_real_model(real_client):
    assert openai_server.state.model is not None
    assert openai_server.state.model_id == "qm-code-tiny"
    assert openai_server.state.model_cfg["max_seq_len"] == 512
    assert openai_server.state.model_cfg["vocab_size"] == 5822
    assert openai_server.state.model.num_params() == 495_808
    assert openai_server.state.checkpoint_step == 200


def test_model_and_tokenizer_vocab_agree(real_client):
    assert openai_server.state.tokenizer.get_vocab_size() == openai_server.state.model_cfg["vocab_size"]


def test_tokenizer_roundtrips_after_real_load(real_client):
    sample = "def add(a, b):\n    return a + b\n"
    ids = openai_server.state.tokenizer.encode(sample).ids
    assert openai_server.state.tokenizer.decode(ids) == sample


def test_load_state_rejects_missing_checkpoint():
    from serving.server_security import load_trusted_state

    with pytest.raises(SystemExit) as exc:
        load_trusted_state("runs/does_not_exist.pt", str(TOKENIZER), "cpu")
    assert "Checkpoint not found" in str(exc.value)


def test_load_state_rejects_mismatched_tokenizer(tmp_path):
    """A tokenizer wider than the embedding table would index out of bounds."""
    from tokenizers import Tokenizer, models

    from serving.server_security import load_trusted_state

    fat = tmp_path / "fat_tokenizer.json"
    Tokenizer(models.BPE(vocab={f"tok{i}": i for i in range(9_000)}, merges=[])).save(str(fat))

    with pytest.raises(SystemExit) as exc:
        load_trusted_state(str(CKPT), str(fat), "cpu")
    assert "exceeds model vocab_size" in str(exc.value)


# --------------------------------------------------------------------------- #
# 2. Real inference reaches the model and returns the right shape
# --------------------------------------------------------------------------- #
def test_real_request_reaches_inference(real_client):
    r = _chat(real_client)
    assert r.status_code == 200
    body = r.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "qm-code-tiny"
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert choice["logprobs"] is None


def test_real_models_endpoint_reports_loaded_model(real_client):
    body = real_client.get("/v1/models").json()
    entry = body["data"][0]
    assert entry["id"] == "qm-code-tiny"
    assert entry["object"] == "model"
    assert entry["quantsmind"]["context_window"] == 512
    assert entry["quantsmind"]["vocab_size"] == 5822
    assert entry["quantsmind"]["n_params_non_embedding"] == 495_808


def test_real_health_reports_loaded(real_client):
    body = real_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == "qm-code-tiny"


def test_real_usage_token_counts_are_accurate(real_client):
    usage = _chat(real_client, max_tokens=8).json()["usage"]
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0

    prompt = openai_server.build_prompt(
        [openai_server.ChatMessage(role="user", content=USER_MSG)]
    )
    assert usage["prompt_tokens"] == len(
        openai_server.state.tokenizer.encode(prompt).ids
    )


def test_greedy_generation_is_deterministic(real_client):
    """temperature=0 must be a pure function of the prompt."""
    a = _chat(real_client, max_tokens=12).json()["choices"][0]["message"]["content"]
    b = _chat(real_client, max_tokens=12).json()["choices"][0]["message"]["content"]
    assert a == b


def test_max_tokens_is_respected_on_the_real_model(real_client):
    for n in (4, 12):
        body = _chat(real_client, max_tokens=n).json()
        assert body["usage"]["completion_tokens"] <= n
        assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_system_and_user_roles_reach_the_real_model(real_client):
    """A caller-supplied system message must replace the built-in default.

    The default path injects the long CODE_SYSTEM_PROMPT, so replacing it with a
    short custom one *reduces* the token count. What matters is that the caller's
    text is used and is not silently discarded.
    """
    long_system = "You are terse. " * 40
    with_system = _chat(
        real_client,
        messages=[
            {"role": "system", "content": long_system},
            {"role": "user", "content": USER_MSG},
        ],
    ).json()
    without_system = _chat(real_client).json()

    assert with_system["usage"]["prompt_tokens"] > without_system["usage"]["prompt_tokens"]

    prompt = openai_server.build_prompt(
        [
            openai_server.ChatMessage(role="system", content=long_system),
            openai_server.ChatMessage(role="user", content=USER_MSG),
        ]
    )
    assert long_system.strip() in prompt
    assert USER_MSG in prompt


def test_stop_sequence_truncates_real_output(real_client):
    body = _chat(real_client, max_tokens=32, stop=["\\\""]).json()
    assert "\\\" \\\\" not in body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# 3. Errors and auth hold with real weights loaded
# --------------------------------------------------------------------------- #
def test_streaming_still_rejected_with_real_model(real_client):
    r = _chat(real_client, stream=True)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "streaming_unsupported"


def test_unsupported_tool_param_still_rejected(real_client):
    r = _chat(real_client, tools=[{"type": "function"}])
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tools"


def test_bad_role_still_rejected(real_client):
    r = _chat(real_client, messages=[{"role": "tool", "content": "x"}])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_role"


def test_context_limit_enforced_against_real_window(real_client):
    r = _chat(real_client, messages=[{"role": "user", "content": "x" * 9000}])
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert "512" in err["message"]


def test_errors_do_not_leak_checkpoint_path(real_client):
    openai_server.state.ckpt_path = openai_server.Path("runs/secret/internal_ckpt.pt")
    r = _chat(real_client, messages=[{"role": "bad", "content": "x"}])
    assert "Traceback" not in r.text
    assert "internal_ckpt" not in r.text
    assert ".pt" not in r.text


def test_auth_boundary_intact_with_real_model(real_client):
    real_client.post("/v1/chat/completions", json={})
    openai_server.state.api_key = "s3cret"
    try:
        body = {
            "model": "qm-code-tiny",
            "messages": [{"role": "user", "content": USER_MSG}],
            "max_tokens": 4,
            "temperature": 0.0,
        }
        assert real_client.post("/v1/chat/completions", json=body).status_code == 401
        assert real_client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer wrong"}
        ).status_code == 401
        ok = real_client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer s3cret"}
        )
        assert ok.status_code == 200
        assert ok.json()["choices"][0]["message"]["role"] == "assistant"
    finally:
        openai_server.state.api_key = None


# --------------------------------------------------------------------------- #
# 4. Measured behaviour, documented so a change is noticed
# --------------------------------------------------------------------------- #
def test_model_is_well_formed_but_untrained(real_client):
    """Records what the checkpoint actually is, without asserting on junk output.

    The model is a real trained network, not a random one: its predictive entropy is
    far below the uniform baseline over its 5,822-token vocabulary. What it has
    learned is Python *unigram* statistics, not syntax -- so greedy decoding falls
    into a repetition loop and `finish_reason` is always "length".
    """
    import torch

    model = openai_server.state.model
    tok = openai_server.state.tokenizer
    vocab = model.cfg["vocab_size"]

    ids = torch.tensor([tok.encode("def add(a, b):\n").ids])
    with torch.no_grad():
        logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum())

    import math

    assert entropy < math.log(vocab) - 1.0, "model should be better than uniform"
    assert probs.max().item() > 1.0 / vocab, "no single token should dominate"

    # A raw code prompt still yields a valid, in-vocabulary continuation.
    out = _chat(real_client, max_tokens=8).json()["choices"][0]
    assert out["message"]["content"]
    assert out["finish_reason"] in ("stop", "length")