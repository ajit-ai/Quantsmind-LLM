import pytest

torch = pytest.importorskip("torch")

from training.model import Attention, QMModel, build_rope
from training.multimodal import QMMultimodal, load_text_state


def tiny_cfg(vocab: int) -> dict:
    return {
        "vocab_size": vocab,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 88,
        "max_seq_len": 128,
        "rope_theta": 10000.0,
        "norm_eps": 1.0e-5,
        "tie_embeddings": True,
    }


TINY_VISION_CFG = {
    "image_size": 32,
    "patch_size": 8,
    "in_channels": 3,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "ffn_dim": 88,
    "norm_eps": 1.0e-5,
}


def make_m0(vocab: int = 64):
    torch.manual_seed(1)
    return QMModel(tiny_cfg(vocab))


def make_mm(vocab: int = 68):
    torch.manual_seed(2)
    text_cfg = tiny_cfg(vocab)
    return QMMultimodal(text_cfg, dict(TINY_VISION_CFG))


def save_m0_ckpt(m0, path):
    torch.save(
        {
            "model": m0.state_dict(),
            "optimizer": None,
            "step": 3,
            "config": {"run": {"model_config": "configs/model/local_15m_mm.yaml"}},
        },
        path,
    )


def test_m0_checkpoint_loading(tmp_path):
    m0 = make_m0()
    ckpt = tmp_path / "m0.pt"
    save_m0_ckpt(m0, ckpt)
    mm = make_mm()
    ignored = load_text_state(mm, str(ckpt))
    assert ignored == []


def test_vocab_extension_preserves_rows(tmp_path):
    m0 = make_m0()
    ckpt = tmp_path / "m0.pt"
    save_m0_ckpt(m0, ckpt)
    mm = make_mm()
    load_text_state(mm, str(ckpt))
    torch.testing.assert_close(mm.text.tok_emb.weight[:64], m0.tok_emb.weight, atol=0, rtol=0)
    # extended rows kept their own initialization (not copied from M0)
    assert not torch.equal(mm.text.tok_emb.weight[64:], m0.tok_emb.weight[:4])


def test_original_model_logits_preserved(tmp_path):
    m0 = make_m0()
    ckpt = tmp_path / "m0.pt"
    save_m0_ckpt(m0, ckpt)
    mm = make_mm()
    load_text_state(mm, str(ckpt))
    torch.manual_seed(7)
    idx = torch.randint(0, 64, (2, 20))
    ref_logits, _ = m0(idx)
    mm_logits, _ = mm.text(idx)
    # mm.text has vocab 68; original 64 columns must be bit-identical.
    assert torch.equal(ref_logits, mm_logits[:, :, :64])


def test_default_attention_preserved():
    torch.manual_seed(3)
    attn = Attention(32, 4, 2, 10000.0)
    x = torch.randn(2, 10, 32)
    cos, sin = build_rope(10, 8, 10000.0, torch.device("cpu"), torch.float32)
    out_default = attn(x, cos, sin)
    causal_mask = torch.tril(torch.ones(10, 10, dtype=torch.bool))
    out_masked = attn(x, cos, sin, attn_mask=causal_mask.bool().unsqueeze(0).unsqueeze(0))
    assert torch.equal(out_default, out_masked)


def test_text_only_ckpt_roundtrip(tmp_path):
    m = make_m0()
    ckpt = tmp_path / "m0.pt"
    torch.save({"model": m.state_dict(), "step": 5}, ckpt)
    m2 = make_m0()
    m2.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    torch.manual_seed(9)
    idx = torch.randint(0, 64, (2, 20))
    y = torch.randint(0, 64, (2, 20))
    _, l1 = m(idx, targets=y)
    _, l2 = m2(idx, targets=y)
    assert torch.equal(l1, l2)


def test_checkpoint_both_config_conventions(tmp_path):
    mm = make_mm()
    text_cfg = tiny_cfg(68)
    train_cfg = {"run": {"model_config": "configs/model/local_15m_mm.yaml", "vision_config": "configs/vision/vision_m1.yaml"}}
    ckpt_path = tmp_path / "mm.pt"
    torch.save({"model": mm.state_dict(), "step": 1, "config": train_cfg, "cfg": text_cfg}, ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["config"]["run"]["model_config"].endswith("local_15m_mm.yaml")
    assert ckpt["cfg"]["vocab_size"] == 68
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_dim", "rope_theta", "norm_eps"):
        assert key in ckpt["cfg"]