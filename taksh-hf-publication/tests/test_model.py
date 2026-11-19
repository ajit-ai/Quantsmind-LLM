import pytest

torch = pytest.importorskip("torch")

from training.model import QMModel
from training.pretrain import lr_at

CFG = {
    "name": "test-tiny",
    "vocab_size": 128,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 88,
    "max_seq_len": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "tie_embeddings": True,
}


def build_model():
    torch.manual_seed(0)
    return QMModel(CFG)


def test_forward_shapes():
    m = build_model()
    x = torch.randint(0, CFG["vocab_size"], (2, 16))
    logits, _ = m(x)
    assert logits.shape == (2, 16, CFG["vocab_size"])


def test_loss_finite_and_backward():
    m = build_model()
    x = torch.randint(0, CFG["vocab_size"], (2, 16))
    y = torch.randint(0, CFG["vocab_size"], (2, 16))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss).all()
    loss.backward()


def test_ties_embeddings():
    m = build_model()
    assert m.lm_head.weight is m.tok_emb.weight


def test_gqa_shapes():
    attn = m_blocks(build_model())[0].attn
    assert attn.wk.weight.shape[0] == CFG["n_kv_heads"] * (CFG["d_model"] // CFG["n_heads"])


def m_blocks(m):
    return list(m.blocks)


def test_loss_decreases_over_steps():
    m = build_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, CFG["vocab_size"], (4, 16))
    y = torch.randint(0, CFG["vocab_size"], (4, 16))
    first = None
    for i in range(15):
        _, loss = m(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_lr_schedule():
    peak, warmup, total, minr = 1e-3, 10, 100, 0.1
    assert lr_at(0, total, peak, warmup, minr) < peak
    assert lr_at(warmup - 1, total, peak, warmup, minr) == pytest.approx(peak)
    assert lr_at(total - 1, total, peak, warmup, minr) >= peak * minr * 0.99
