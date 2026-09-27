from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import yaml

from training.multimodal import (
    CAPTION_START,
    IMG_END,
    IMG_START,
    TEXT_EOS,
    Projector,
    QMMultimodal,
    build_mm_mask,
)

TEXT_CFG = {
    "vocab_size": 68,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 88,
    "max_seq_len": 256,
    "rope_theta": 10000.0,
    "norm_eps": 1.0e-5,
    "tie_embeddings": True,
}

VISION_CFG = {
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


def build_mm():
    torch.manual_seed(0)
    return QMMultimodal(TEXT_CFG, VISION_CFG)


def test_projector_shape():
    p = Projector(32, 64, 32)
    out = p(torch.randn(2, 17, 32))
    assert out.shape == (2, 17, 32)


def test_multimodal_forward():
    m = build_mm()
    imgs = torch.randn(2, 3, 32, 32)
    caps = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    lengths = torch.tensor([3, 4])
    logits, loss = m(imgs, caps, lengths)
    assert logits.shape == (2, m.first_text_pos + caps.shape[1] + 1, TEXT_CFG["vocab_size"])
    assert torch.isfinite(loss)


def test_mask_shape():
    mask = build_mm_mask([(0, 20)], 40)
    assert mask.dtype == torch.bool
    assert mask.shape == (40, 40)


def test_mask_semantics():
    img_end = 20
    seq = 30
    mask = build_mm_mask([(0, img_end)], seq)
    # image -> image: fully visible
    assert mask[0, 5].item() and mask[5, 0].item()
    # image -> text: blocked
    assert not mask[3, img_end].item()
    assert not mask[0, 25].item()
    # text -> image: visible
    assert mask[25, 5].item()
    # text -> text: causal
    assert mask[25, 25].item()
    assert not mask[25, 26].item()


def test_special_token_layout():
    m = build_mm()
    imgs = torch.randn(2, 3, 32, 32)
    caps = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    lengths = torch.tensor([2, 4])
    embeds, targets, mask = m.build_sequence(imgs, caps, lengths)
    total = embeds.shape[1]
    assert total == m.first_text_pos + caps.shape[1] + 1
    assert mask.shape == (1, 1, total, total)
    # special tokens come from the text embedding table
    assert torch.equal(embeds[0, 0, :], m.text.tok_emb.weight[m.img_start])
    assert torch.equal(embeds[0, m.img_end_pos, :], m.text.tok_emb.weight[m.img_end])
    assert torch.equal(embeds[0, m.cap_start_pos, :], m.text.tok_emb.weight[m.caption_start])
    assert torch.equal(embeds[0, total - 1, :], m.text.tok_emb.weight[m.text_eos])
    # caption start position predicts first caption token
    assert (targets[:, m.cap_start_pos] == caps[:, 0]).all()


def test_conditioning_changes_output():
    m = build_mm()
    caps = torch.tensor([[6, 7, 8, 9]])
    lengths = torch.tensor([3])
    l1, _ = m(torch.randn(1, 3, 32, 32), caps, lengths)
    l2, _ = m(torch.randn(1, 3, 32, 32), caps, lengths)
    assert (l1 - l2).abs().max().item() > 1e-6


def test_mm_gradients_reach_vision_and_projector():
    m = build_mm()
    imgs = torch.randn(2, 3, 32, 32)
    caps = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    lengths = torch.tensor([3, 4])
    _, loss = m(imgs, caps, lengths)
    loss.backward()
    vit_grads = [p.grad for p in m.vision.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    proj_grads = [p.grad for p in m.projector.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert vit_grads, "VisionEncoder received no gradients"
    assert proj_grads, "Projector received no gradients"
    for g in vit_grads + proj_grads:
        assert torch.isfinite(g).all()


def test_configs_align():
    root = Path(__file__).resolve().parents[1]
    text = yaml.safe_load((root / "configs/model/local_15m_mm.yaml").read_text(encoding="utf-8"))["model"]
    vis = yaml.safe_load((root / "configs/vision/vision_m1.yaml").read_text(encoding="utf-8"))["vision"]
    assert text["vocab_size"] == 49156
    assert vis["d_model"] == text["d_model"]
    assert text["n_heads"] % text["n_kv_heads"] == 0