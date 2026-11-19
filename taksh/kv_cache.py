"""Key-Value cache for Taksh transformer inference.

Caches previous key/value tensors per decoder layer so each generated token
reuses prior computations instead of recomputing the full prompt every step.

This is v1.5.0 Week 2-3 work. It does not modify checkpoints, tokenizer, or
training data. It only changes inference-time computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class KVCacheEntry:
    """Cached key/value tensors for one decoder layer."""

    key: torch.Tensor      # shape: [batch, seq_len, n_kv_heads, head_dim]
    value: torch.Tensor    # shape: [batch, seq_len, n_kv_heads, head_dim]


class KVCache:
    """Layer-wise KV cache for autoregressive generation.

    Attributes:
        n_layers: Number of decoder layers.
        cache: Mapping from layer index to KVCacheEntry (None until first use).
        seq_len: Current cached sequence length (prompt + generated tokens so far).
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.cache: Dict[int, Optional[KVCacheEntry]] = {i: None for i in range(n_layers)}
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value to the cache and return the full sequence.

        Args:
            layer_idx: Decoder layer index.
            key: New key tensor, shape [batch, n_kv_heads, new_seq_len, head_dim].
            value: New value tensor, shape [batch, n_kv_heads, new_seq_len, head_dim].

        Returns:
            (full_key, full_value) — the complete cached sequence for this layer.
            Both have shape [batch, n_kv_heads, full_seq_len, head_dim].
        """
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = KVCacheEntry(key=key, value=value)
        else:
            prev = self.cache[layer_idx]
            # seq_len is at dim=2: [batch, n_kv_heads, seq, head_dim]
            self.cache[layer_idx] = KVCacheEntry(
                key=torch.cat([prev.key, key], dim=2),
                value=torch.cat([prev.value, value], dim=2),
            )
        self.seq_len = self.cache[layer_idx].key.size(2)
        return self.cache[layer_idx].key, self.cache[layer_idx].value

    def get(self, layer_idx: int) -> Optional[KVCacheEntry]:
        """Return the cache entry for a layer, or None if not yet populated."""
        return self.cache.get(layer_idx)

    def clear(self) -> None:
        """Reset the cache (call between unrelated prompts)."""
        self.cache = {i: None for i in range(self.n_layers)}
        self.seq_len = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"KVCache(n_layers={self.n_layers}, seq_len={self.seq_len})"
