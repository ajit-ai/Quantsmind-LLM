import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_theta: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.attn_norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.attn = Attention(d, cfg["n_heads"], cfg["n_kv_heads"], cfg.get("rope_theta", 10000.0))
        self.ffn_norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.ffn = SwiGLU(d, cfg["ffn_dim"])

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class QMModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        d = cfg["d_model"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.lm_head = nn.Linear(d, cfg["vocab_size"], bias=False)
        if cfg.get("tie_embeddings", False):
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        d = self.cfg["d_model"]
        head_dim = d // self.cfg["n_heads"]
        theta = self.cfg.get("rope_theta", 10000.0)
        dtype = self.tok_emb.weight.dtype
        cos, sin = build_rope(t, head_dim, theta, idx.device, dtype)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.get("tie_embeddings", False):
            n -= self.tok_emb.weight.numel()
            n -= self.lm_head.weight.numel()
        return n
