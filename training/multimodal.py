import torch
import torch.nn as nn

from training.model import QMModel
from training.vision import VisionEncoder

# Reserved ids in the multimodal vocab (49156 = 49152 BPE + 4 specials).
IMG_START = 49152
IMG_END = 49153
CAPTION_START = 49154
TEXT_EOS = 49155
# Future image-token vocab begins at 51200 (never allocated in M1.1).
IMAGE_TOKENS_START = 51200

# Reserved-but-unimplemented special ids.
NEXT_SPECIALS = (49156, 49157, 49158, 49159)


def build_mm_mask(image_spans: list[tuple[int, int]], seq_len: int) -> torch.Tensor:
    """Boolean [seq_len, seq_len] attention mask for interleaved image/text sequences.

    image_spans: list of (start, end) EXCLUSIVE ranges occupied by image content.
    Image rows attend to image rows only (never text); text rows attend to all image
    tokens plus causal text tokens. Compatible with F.scaled_dot_product_attention
    (True = allowed).
    """
    if image_spans:
        text_start = max(end for _, end in image_spans)
    else:
        text_start = 0
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(text_start, seq_len):
        mask[i, i + 1 :] = False  # text -> text causal
    for i in range(text_start):
        mask[i, text_start:] = False  # image -> text blocked
    return mask


class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMMultimodal(nn.Module):
    """VisionEncoder -> Projector -> shared QM decoder.

    Layout: [IMG_START] [image tokens (V+1 including CLS)] [IMG_END] [CAPTION_START]
    [caption tokens] [TEXT_EOS]. Caption loss is computed on the text region only,
    using -100 (ignore_index) elsewhere.
    """

    def __init__(self, text_cfg: dict, vision_cfg: dict):
        super().__init__()
        assert vision_cfg["d_model"] == text_cfg["d_model"]
        self.text_cfg = dict(text_cfg)
        self.vision = VisionEncoder(vision_cfg)
        self.projector = Projector(vision_cfg["d_model"], 256, text_cfg["d_model"])
        self.text = QMModel(text_cfg)
        # Special ids are the last 4 slots of the text vocab (real config: 49152..49155).
        self.first_special = text_cfg["vocab_size"] - 4
        self.img_start = self.first_special
        self.img_end = self.first_special + 1
        self.caption_start = self.first_special + 2
        self.text_eos = self.first_special + 3

    @property
    def image_tokens(self) -> int:
        return self.vision.num_patches + 1

    @property
    def img_end_pos(self) -> int:
        return 1 + self.image_tokens

    @property
    def cap_start_pos(self) -> int:
        return self.img_end_pos + 1

    @property
    def first_text_pos(self) -> int:
        return self.cap_start_pos + 1

    @property
    def image_region(self) -> tuple[int, int]:
        return 0, self.first_text_pos  # IMG_START..CAPTION_START inclusive prefix

    def _special_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text.embed(ids)

    def build_sequence(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embeds, targets, mask) for a full MM training sequence."""
        b = images.shape[0]
        device = images.device
        cap_t = caption_ids.shape[1]

        vis = self.projector(self.vision(images))  # (B, V+1, d)
        ids = torch.full((b, 1), self.img_start, dtype=torch.long, device=device)
        start = self._special_embed(ids)
        ids = torch.full((b, 1), self.img_end, dtype=torch.long, device=device)
        end = self._special_embed(ids)
        ids = torch.full((b, 1), self.caption_start, dtype=torch.long, device=device)
        cs = self._special_embed(ids)
        cap_emb = self.text.embed(caption_ids)
        ids = torch.full((b, 1), self.text_eos, dtype=torch.long, device=device)
        eos = self._special_embed(ids)

        embeds = torch.cat([start, vis, end, cs, cap_emb, eos], dim=1)
        total = embeds.shape[1]

        targets = torch.full((b, total), -100, dtype=torch.long, device=device)
        cap_eos = torch.full((b, cap_t + 1), -100, dtype=torch.long, device=device)
        for i in range(b):
            ln = int(caption_lengths[i].item())
            cap_eos[i, :ln] = caption_ids[i, :ln]
            cap_eos[i, ln] = self.text_eos
        targets[:, self.cap_start_pos : self.cap_start_pos + cap_t + 1] = cap_eos

        mask = build_mm_mask([self.image_region], total).to(device)
        return embeds, targets, mask.bool().unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_lengths: torch.Tensor | None = None,
    ):
        if caption_lengths is None:
            caption_lengths = caption_ids.new_full((caption_ids.shape[0],), caption_ids.shape[1])
        embeds, targets, mask = self.build_sequence(images, caption_ids, caption_lengths)
        logits, loss = self.text.forward_embeds(embeds, targets=targets, attn_mask=mask)
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        return self.text.num_params(non_embedding) + sum(
            p.numel() for p in self.vision.parameters()
        ) + sum(p.numel() for p in self.projector.parameters())


def load_text_state(model, ckpt_path: str, strict: bool = False) -> list[str]:
    """Load an M0 text-only checkpoint into a QMMultimodal's text backbone.

    Copy compatible tensors exactly (key + shape match) and extend the tied
    tok_emb/lm_head from 49152 -> 49156 by keeping the new rows' initialization.
    Returns the list of new/ignored parameter names.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) and "model" in raw else raw
    new_sd = model.text.state_dict()
    ignored = []
    for key, value in sd.items():
        if key not in new_sd:
            ignored.append(key)
            continue
        if new_sd[key].shape == value.shape:
            new_sd[key] = value.detach().clone()
            continue
        # shape mismatch: allow head-dimension growth (vocab extension)
        if (
            value.ndim == 2
            and new_sd[key].shape[0] > value.shape[0]
            and value.shape[1:] == new_sd[key].shape[1:]
        ):
            new_sd[key][: value.shape[0]] = value.detach()
        else:
            ignored.append(key)
    missing = [k for k in new_sd if k not in sd]
    strict_failed = ignored + missing
    if strict and strict_failed:
        raise RuntimeError(f"strict load failed: {strict_failed}")
    model.text.load_state_dict(new_sd)
    return strict_failed


@torch.no_grad()
def probe_conditioning(model, img_a: torch.Tensor, img_b: torch.Tensor, caption_ids: torch.Tensor):
    """Compare next-token distributions at CAPTION_START given two different images.

    Returns (max_abs_logit_diff, argmax_token_a, argmax_token_b).
    """
    model.eval()
    la, _ = model(img_a, caption_ids)
    lb, _ = model(img_b, caption_ids)
    pos = model.cap_start_pos
    da = la[:, pos, :]
    db = lb[:, pos, :]
    diff = (da - db).abs().max().item()
    return diff, int(da.argmax(dim=-1)), int(db.argmax(dim=-1))