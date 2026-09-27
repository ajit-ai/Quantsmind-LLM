"""Deterministic autoregressive generation for the multimodal model.

This is *measurement* generation used by the evaluation harness: greedy
argmax decoding with the same image-mask placement used during training, so
that caption/VQA outputs are reproducible.  It is not a training component.
"""
from __future__ import annotations

from pathlib import Path

import torch

from training.image_data import preprocess_image
from training.multimodal import build_mm_mask


def image_tensor_file(path: str | Path, size: int = 112, device: str = "cpu") -> torch.Tensor:
    """Load an image file and preprocess it to ``(1, 3, size, size)`` float32.

    Uses ``training.image_data.preprocess_image`` (RGB, LANCZOS cover-resize,
    center crop, [0, 1]) so evaluation preprocessing matches the training
    contract exactly.
    """
    from PIL import Image

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with Image.open(path) as img:
        arr = preprocess_image(img, size=size)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).float()


def _prefix_embeds(model, image: torch.Tensor, text_prefix_ids):
    """Build the image + optional text-prefix embedding row for one sample."""
    if image.shape[0] != 1:
        raise ValueError("generation currently supports a single sample (batch size 1)")
    device = image.device

    def row(ids):
        t = torch.tensor([[ids]], dtype=torch.long, device=device)
        return model.text.embed(t)

    with torch.no_grad():
        vis = model.projector(model.vision(image))
        parts = [
            row(model.img_start),
            vis,
            row(model.img_end),
            row(model.caption_start),
        ]
        if text_prefix_ids:
            t = torch.tensor([list(text_prefix_ids)], dtype=torch.long, device=device)
            parts.append(model.text.embed(t))
    return torch.cat(parts, dim=1)


def generate_continuation(
    model,
    image: torch.Tensor,
    text_prefix_ids=None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[int]:
    """Greedily generate token ids continuing from an image (+ optional text).

    Prediction follows the training layout: every position predicts the next
    token, so position ``cap_start_pos + len(text_prefix_ids)`` predicts the
    first generated token.  Generation stops when ``text_eos`` is sampled or
    ``max_new_tokens`` is reached.

    Parameters
    ----------
    model : QMMultimodal
        A multimodal model in eval mode.
    image : torch.Tensor
        ``(1, 3, H, W)`` float32 image on the same device as the model.
    text_prefix_ids : list[int] | None
        Token ids placed in the text region after CAPTION_START (e.g. a VQA
        question).  Empty/None generates a plain caption.
    max_new_tokens : int
        Hard cap on generated tokens.
    temperature : float
        Sampling temperature.  ``0.0`` (default) means deterministic argmax.

    Returns
    -------
    list[int]
        Generated token ids (excluding the special-image prefix and any EOS).
    """
    if not model.__class__.__name__ == "QMMultimodal":
        raise TypeError("generate_continuation requires a QMMultimodal model")
    text_prefix_ids = list(text_prefix_ids or [])
    embeds = _prefix_embeds(model, image, text_prefix_ids)
    q = len(text_prefix_ids)
    out = []
    for _ in range(max_new_tokens):
        L = embeds.shape[1]
        mask = build_mm_mask([model.image_region], L).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).to(image.device)
        with torch.no_grad():
            logits, _ = model.text.decode(embeds, targets=None, attn_mask=mask)
        pos = model.cap_start_pos + q + len(out)
        dist = logits[0, pos]
        if temperature <= 0.0 or temperature is None:
            tok = int(dist.argmax().item())
        else:
            probs = torch.softmax(dist.float() / temperature, dim=-1)
            tok = int(torch.multinomial(probs, 1).item())
        if tok == model.text_eos:
            break
        out.append(tok)
        t = torch.tensor([[tok]], dtype=torch.long, device=image.device)
        with torch.no_grad():
            embeds = torch.cat([embeds, model.text.embed(t)], dim=1)
    return out


def generate_caption(model, image: torch.Tensor, max_new_tokens: int = 32) -> list[int]:
    """Generate a zero-shot caption for a single image."""
    return generate_continuation(model, image, text_prefix_ids=None, max_new_tokens=max_new_tokens)


def generate_answer(
    model, image: torch.Tensor, question_ids, max_new_tokens: int = 16
) -> list[int]:
    """Generate an answer for a single image+VQA-question."""
    return generate_continuation(model, image, text_prefix_ids=question_ids, max_new_tokens=max_new_tokens)


def decode_ids(tokenizer, ids, vocab_limit: int = 49152) -> str:
    """Decode token ids to text, ignoring ids beyond the tokenizer's vocab.

    Generated ids may fall in the special-token range (>= ``vocab_limit``) which
    the BPE tokenizer cannot map; those are dropped rather than raising.
    """
    safe = [i for i in ids if i < vocab_limit]
    if not safe:
        return ""
    return tokenizer.decode(safe)