"""Report whether and how much two Taksh checkpoints differ, tensor by tensor.

Usage:
    python -m scripts.compare_checkpoint_params BASE CAND [--full]

Prints the number of tensors that differ and the max absolute difference for a
sample of tensors (all tensors with --full). Loading uses the hardened loader
(weights_only=True plus the narrow NumPy allowlist retry).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--full", action="store_true")
    p.add_argument("--sample", type=int, default=8)
    args = p.parse_args(argv)

    import torch
    from serving.local_inference import torch_load_checkpoint

    base = torch_load_checkpoint(args.base)["model"]
    cand = torch_load_checkpoint(args.candidate)["model"]
    shared = set(base) & set(cand)
    only_base = set(base) - set(cand)
    only_cand = set(cand) - set(base)
    print(f"tensors: base={len(base)} candidate={len(cand)} shared={len(shared)}")
    if only_base:
        print("only in base:", sorted(only_base)[:5])
    if only_cand:
        print("only in candidate:", sorted(only_cand)[:5])

    diffs = []
    for name in sorted(shared):
        b, c = base[name], cand[name]
        if b.shape != c.shape:
            diffs.append((name, float("inf")))
            continue
        d = (c.float() - b.float()).abs().max().item() if b.numel() else 0.0
        diffs.append((name, d))
    changed = [d for d in diffs if d[1] > 0.0]
    identical = len(diffs) - len(changed)
    print(f"identical tensors: {identical} | changed tensors: {len(changed)}")
    for name, d in sorted(changed, key=lambda item: item[1])[: args.sample]:
        print(f"  {d:.3e}  {name}")
    if args.full:
        for name, d in changed:
            print(f"  {d:.3e}  {name}")
    identical_all = len(changed) == 0
    print("parameters changed:", "NO" if identical_all else "YES")
    return 1 if identical_all else 0


if __name__ == "__main__":
    sys.exit(main())
