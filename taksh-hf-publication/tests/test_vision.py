import pytest

torch = pytest.importorskip("torch")

from training.vision import VisionEncoder

CFG = {
    "image_size": 112,
    "patch_size": 16,
    "in_channels": 3,
    "d_model": 128,
    "n_layers": 6,
    "n_heads": 4,
    "n_kv_heads": 4,
    "ffn_dim": 512,
    "norm_eps": 1.0e-5,
}


def test_patch_count():
    v = VisionEncoder(CFG)
    assert v.num_patches == 49  # (112 // 16) ** 2
    assert v.num_tokens == 50  # + CLS


def test_output_shape():
    v = VisionEncoder(CFG)
    out = v(torch.zeros(2, 3, 112, 112))
    assert out.shape == (2, 50, 128)


def test_configurable_resolution():
    v = VisionEncoder({**CFG, "image_size": 224})
    out = v(torch.zeros(2, 3, 224, 224))
    assert v.num_patches == 196
    assert out.shape == (2, 197, 128)


def test_forward_finite():
    v = VisionEncoder(CFG)
    out = v(torch.randn(2, 3, 112, 112))
    assert torch.isfinite(out).all()


def test_gradients_finite():
    v = VisionEncoder(CFG)
    out = v(torch.randn(2, 3, 112, 112))
    out.pow(2).mean().backward()
    grads = [p.grad for p in v.parameters() if p.grad is not None]
    assert grads, "no parameters received gradients"
    assert all(torch.isfinite(g).all() for g in grads)