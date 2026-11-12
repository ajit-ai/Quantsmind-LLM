"""v1.1.0 acceptance gates. Offline, CPU-sized, bounded."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from training.chat_template import (
    CONTRACT_V1,
    SYSTEM_PROMPT,
    render_chat,
    render_completion,
    render_for,
    render_prompt,
)
from training.checkpoint_schema import (
    ContractMismatch,
    SCHEMA_V1,
    SCHEMA_V2,
    build_provenance,
    verify_against,
)
from training.determinism import set_determinism
from training.model import QMModel
from training.tokenizer_contract import (
    CONTRACT_V2,
    CONTROL_TOKENS,
    TokenizerContractError,
    build_contract_tokenizer,
    contract_ids,
    detect_contract,
    require_contract,
    tokenizer_fingerprint,
)

V0 = "tokenizer/qm_code_bpe_v0.json"
TINY = {
    "name": "v110-tiny", "vocab_size": 5828, "d_model": 32, "n_layers": 1,
    "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 64,
    "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True,
}


@pytest.fixture(scope="module")
def tok_v0():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(V0)


@pytest.fixture(scope="module")
def tok_v2():
    return build_contract_tokenizer(V0)


def _model():
    return QMModel(TINY)


def _prov(tok, contract, seed=7):
    return build_provenance(
        model_cfg=TINY,
        parameter_count=sum(p.numel() for p in _model().parameters()),
        contract_id=contract,
        config={"model": TINY},
        seed=seed,
        tokenizer=tok,
        tokenizer_path=V0,
    )

def test_gate1_train_and_serve_render_identical_prompt(tok_v2):
    q = "write a function that adds two numbers"
    training = render_for(tok_v2, SYSTEM_PROMPT, q)
    serving = render_chat(
        tok_v2, [{"role": "user", "content": q}], default_system=SYSTEM_PROMPT
    )
    assert training == serving
    assert tok_v2.encode(training).ids == tok_v2.encode(serving).ids
    assert training.startswith("<|system|>")
    assert training.endswith("<|assistant|>")


def test_gate1_legacy_renderer_is_unchanged(tok_v0):
    assert render_for(tok_v0, SYSTEM_PROMPT, "hi") == render_prompt(
        SYSTEM_PROMPT, "hi"
    )
    assert render_chat(
        tok_v0,
        [{"role": "user", "content": "hi"}],
        default_system=SYSTEM_PROMPT,
    ) == render_prompt(SYSTEM_PROMPT, "hi")


def test_gate1_completion_has_prompt_as_literal_prefix(tok_v2):
    prompt = render_chat(
        tok_v2, [{"role": "user", "content": "q"}], default_system=SYSTEM_PROMPT
    )
    full = render_completion(SYSTEM_PROMPT, "q", "a", contract_id=CONTRACT_V2)
    assert full.startswith(prompt)
    assert full.endswith("<|end_of_turn|>")


def test_gate1_sft_uses_the_same_contract(tok_v2, tmp_path):
    from training.sft import SFTPairDataset

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"instruction": "UNIQUEQ", "response": "UNIQUER"}) + "\n",
        encoding="utf-8",
    )
    ds = SFTPairDataset(str(path), tok_v2, max_length=256)
    assert ds.contract == CONTRACT_V2
    input_ids, labels = ds[0]
    text = tok_v2.decode(input_ids.tolist(), skip_special_tokens=False)
    assert "UNIQUEQ" in text and "UNIQUER" in text
    eot = contract_ids(tok_v2)["<|end_of_turn|>"]
    assert labels[labels != -100].tolist().count(eot) == 1


def test_gate2_control_tokens_are_single_ids(tok_v2):
    ids = contract_ids(tok_v2)
    assert set(ids) == set(CONTROL_TOKENS)
    for marker, token_id in ids.items():
        assert tok_v2.id_to_token(token_id) == marker
        assert tok_v2.encode(marker, add_special_tokens=False).ids == [token_id]
    assert len(set(ids.values())) == len(CONTROL_TOKENS)
    assert min(ids.values()) >= 5822


def test_gate2_derived_tokenizer_is_deterministic(tok_v2):
    again = build_contract_tokenizer(V0)
    assert tokenizer_fingerprint(again) == tokenizer_fingerprint(tok_v2)
    from tokenizers import Tokenizer

    reloaded = Tokenizer.from_str(tok_v2.to_str())
    assert tokenizer_fingerprint(reloaded) == tokenizer_fingerprint(tok_v2)
    assert contract_ids(reloaded) == contract_ids(tok_v2)


def test_gate2_legacy_tokenizer_refuses_v2(tok_v0):
    assert detect_contract(tok_v0) == CONTRACT_V1
    assert contract_ids(tok_v0) is None
    with pytest.raises(TokenizerContractError):
        require_contract(tok_v0, CONTRACT_V2)

def test_gate3_v2_server_emits_tokens_not_delimiters(tok_v2, monkeypatch):
    import serving.openai_server as srv

    monkeypatch.setattr(srv.state, "tokenizer", tok_v2, raising=False)
    prompt = srv.build_prompt(
        [{"role": "system", "content": "be brief"},
         {"role": "user", "content": "hi"}]
    )
    enc = tok_v2.encode(prompt)
    assert "<|" not in enc.tokens
    ids = contract_ids(tok_v2)
    assert enc.ids.count(ids["<|system|>"]) == 1
    assert enc.ids.count(ids["<|user|>"]) == 1
    assert enc.ids.count(ids["<|assistant|>"]) == 1


def test_gate3_legacy_server_refuses_multi_turn(tok_v0, monkeypatch):
    import serving.openai_server as srv

    monkeypatch.setattr(srv.state, "tokenizer", tok_v0, raising=False)
    single = srv.build_prompt([{"role": "user", "content": "q"}])
    assert single == render_prompt(SYSTEM_PROMPT, "q")
    assert "<|" not in single
    with pytest.raises(ValueError):
        srv.build_prompt(
            [{"role": "user", "content": "a"},
             {"role": "assistant", "content": "b"},
             {"role": "user", "content": "c"}]
        )

def test_gate4_causal_mask_hides_future_tokens():
    set_determinism(1)
    model = _model().eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        a = model(x)[0]
        y = x.clone()
        y[:, -4:] = 7
        b = model(y)[0]
    assert torch.allclose(a[:, :-4], b[:, :-4], atol=1e-5)


def test_gate4_labels_are_shifted_and_prompt_masked(tok_v2, tmp_path):
    from training.sft import SFTPairDataset

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"instruction": "MASKME", "response": "SUPERVISE"}) + "\n",
        encoding="utf-8",
    )
    ds = SFTPairDataset(str(path), tok_v2, max_length=256)
    x, y = ds[0]
    keep = y != -100
    assert torch.equal(y[:-1][keep[:-1]], x[1:][keep[:-1]])
    masked = tok_v2.decode(x[~keep].tolist())
    assert "MASKME" in masked and "SUPERVISE" not in masked
    assert tok_v2.decode(x[keep].tolist()).strip() == "SUPERVISE"


def test_gate4_padding_is_excluded_and_init_is_seeded():
    from training.sft import collate_sft

    a = (torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 5]))
    b = (torch.tensor([1, 2]), torch.tensor([2, 3]))
    x, y = collate_sft([a, b], pad_id=0)
    assert x.shape == y.shape == (2, 4)
    assert (y[1, 2:] == -100).all()
    _, loss = _model()(x, targets=y)
    assert torch.isfinite(loss)

    set_determinism(5)
    m1 = _model()
    set_determinism(5)
    m2 = _model()
    assert all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))

def test_gate5_v2_roundtrip_and_refusals(tmp_path, tok_v0, tok_v2):
    from training.pretrain import load_checkpoint_full, save_checkpoint

    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    prov = _prov(tok_v2, CONTRACT_V2)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, m, opt, 3, {"model": TINY}, provenance=prov)

    m2 = _model()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step, ck = load_checkpoint_full(path, m2, opt2)
    assert step == 3 and ck["schema"] == SCHEMA_V2
    assert ck["provenance"]["contract_id"] == CONTRACT_V2
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p, q)

    with pytest.raises(ContractMismatch):
        verify_against(prov, tokenizer=tok_v0)
    with pytest.raises(ContractMismatch):
        verify_against(prov, tokenizer=tok_v2, contract_id=CONTRACT_V1)
    with pytest.raises(ContractMismatch):
        verify_against(prov, model_cfg={**TINY, "d_model": 64})


def test_gate5_legacy_v1_checkpoint_loads_explicitly(tmp_path):
    from training.pretrain import load_checkpoint_full, save_checkpoint

    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, m, opt, 1, {"model": TINY})

    m2 = _model()
    step, ck = load_checkpoint_full(path, m2, torch.optim.AdamW(m2.parameters()))
    assert step == 1 and ck["schema"] == SCHEMA_V1

def test_gate6_tiny_sample_learning_is_controlled():
    import math

    set_determinism(11)
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(11)
    x = torch.randint(10, 500, (4, 24), generator=g)
    y = torch.cat([x[:, 1:], x[:, :1]], dim=1)

    losses = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    baseline = math.log(TINY["vocab_size"])
    final = sum(losses[-5:]) / 5
    assert losses[0] > baseline * 0.8
    assert final < baseline - 1.0
    assert final < losses[0] - 1.0
    assert all(math.isfinite(v) for v in losses)


def _train(model, x, y, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
    return opt


def test_gate7_save_reload_and_resume(tmp_path):
    from training.pretrain import load_checkpoint_full, save_checkpoint

    g = torch.Generator().manual_seed(13)
    x = torch.randint(0, 200, (2, 12), generator=g)
    y = torch.randint(0, 200, (2, 12), generator=g)

    set_determinism(13)
    a = _model()
    _train(a, x, y, 10)

    set_determinism(13)
    b = _model()
    opt_b = _train(b, x, y, 5)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, b, opt_b, 5, {"model": TINY})

    c = _model()
    opt_c = torch.optim.AdamW(c.parameters(), lr=1e-3)
    step, _ = load_checkpoint_full(path, c, opt_c)
    assert step == 5
    for p, q in zip(b.parameters(), c.parameters()):
        assert torch.equal(p, q)

    for _ in range(5):
        opt_c.zero_grad(set_to_none=True)
        _, loss = c(x, targets=y)
        loss.backward()
        opt_c.step()

    for p, q in zip(a.parameters(), c.parameters()):
        assert torch.allclose(p, q, atol=0.0, rtol=0.0)

