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
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


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

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
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

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# Numeric model-config fields. YAML 1.1 does not parse bare `1e-5` as a float,
# so a hand-written `norm_eps: 1e-5` arrives here as the string "1e-5" and fails
# deep inside RMSNorm with `Tensor + str`. Coercing here turns that into an
# explicit, actionable error at construction time.
_NUMERIC_CFG_FIELDS = (
    "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_dim",
    "max_seq_len", "rope_theta", "norm_eps",
)
_INT_CFG_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                   "ffn_dim", "max_seq_len")


def _coerce_numeric_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    for field in _NUMERIC_CFG_FIELDS:
        if field not in cfg:
            continue
        value = cfg[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model config field {field!r} is {value!r}, which is not a "
                    "number. YAML parses a bare '1e-5' as a string, not a float; "
                    "write it as 1.0e-05 or 0.00001."
                ) from exc
            if field in _INT_CFG_FIELDS:
                value = int(value)
        cfg[field] = value
    return cfg


class QMModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cfg = _coerce_numeric_cfg(cfg)
        self.cfg = dict(cfg)
        d = cfg["d_model"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.lm_head = nn.Linear(d, cfg["vocab_size"], bias=False)
        if cfg.get("tie_embeddings", False):
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        return self.decode(self.embed(idx), targets=targets)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(idx)

    def decode(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        b, t, _ = x.shape
        d = self.cfg["d_model"]
        head_dim = d // self.cfg["n_heads"]
        theta = self.cfg.get("rope_theta", 10000.0)
        dtype = self.tok_emb.weight.dtype
        cos, sin = build_rope(t, head_dim, theta, x.device, dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
        return logits, loss

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        targets: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        return self.decode(embeds, targets=targets, attn_mask=attn_mask)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.get("tie_embeddings", False):
            n -= self.tok_emb.weight.numel()
            n -= self.lm_head.weight.numel()
        return n
