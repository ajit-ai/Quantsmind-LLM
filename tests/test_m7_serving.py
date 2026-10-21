"""Real-checkpoint serving compatibility for the M7 fine-tuned model."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient

from serving import openai_server

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "runs" / "m7_sft" / "checkpoint_best.pt"
TOKENIZER = REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and TOKENIZER.exists()),
    reason="M7 checkpoint/tokenizer not present (runs/ and *.pt are gitignored)",
)


@pytest.fixture(scope="module")
def m7_client():
    openai_server.load_state(str(CKPT), str(TOKENIZER))
    openai_server.state.api_key = None
    try:
        yield TestClient(openai_server.app)
    finally:
        openai_server.state.model = None
        openai_server.state.model_cfg = None
        openai_server.state.tokenizer = None


def test_m7_checkpoint_loads_through_serving_path(m7_client):
    assert openai_server.state.model is not None
    assert openai_server.state.model_id == "qm-code-m3-9m"
    assert openai_server.state.checkpoint_step == 120
    assert openai_server.state.model.num_params() == 9_359_104
    assert openai_server.state.model_cfg["vocab_size"] == 5_822
    assert openai_server.state.tokenizer.get_vocab_size() == 5_822


def test_m7_model_is_discoverable(m7_client):
    body = m7_client.get("/v1/models").json()
    entry = body["data"][0]

    assert entry["id"] == "qm-code-m3-9m"
    assert entry["quantsmind"]["checkpoint_step"] == 120
    assert entry["quantsmind"]["context_window"] == 512
    assert entry["quantsmind"]["vocab_size"] == 5_822
    assert entry["quantsmind"]["n_params_non_embedding"] == 9_359_104


def test_m7_chat_completion_runs_without_quality_claim(m7_client):
    response = m7_client.post(
        "/v1/chat/completions",
        json={
            "model": "qm-code-m3-9m",
            "messages": [{"role": "user", "content": "Write a function."}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "qm-code-m3-9m"
    assert body["usage"]["completion_tokens"] <= 4
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
