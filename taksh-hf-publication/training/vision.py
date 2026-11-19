import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import RMSNorm


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        assert h == self.image_size and w == self.image_size, "unexpected input image size"
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(b, -1, c * p * p)
        return self.proj(x)


class VitAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


class VitBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.norm1 = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.attn = VitAttention(d, cfg["n_heads"], cfg["n_kv_heads"])
        self.norm2 = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.up = nn.Linear(d, cfg["ffn_dim"], bias=False)
        self.gate = nn.Linear(d, cfg["ffn_dim"], bias=False)
        self.down = nn.Linear(cfg["ffn_dim"], d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.down(F.silu(self.gate(self.norm2(x))) * self.up(self.norm2(x)))
        return x + h


class VisionEncoder(nn.Module):
    """Patch-based vision encoder (ViT) — bidirectional, learned pos embeddings, no RoPE."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        image_size = cfg["image_size"]
        patch_size = cfg["patch_size"]
        d = cfg["d_model"]
        self.patch_embed = PatchEmbed(image_size, patch_size, cfg.get("in_channels", 3), d)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = self.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, d))
        self.blocks = nn.ModuleList(VitBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())