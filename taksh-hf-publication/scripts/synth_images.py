"""Generate deterministic synthetic image/caption pairs for the M1.1 fit-check.

Hermetic: no network, no real data, numpy-only drawing. Every index maps to a
fixed (color, shape, position, size) tuple so results are reproducible across runs.
"""

import argparse
from pathlib import Path

import numpy as np

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR_RGB = {
    "red": (220, 45, 45),
    "blue": (45, 80, 220),
    "green": (45, 165, 65),
    "yellow": (235, 200, 45),
    "purple": (145, 55, 200),
    "orange": (235, 135, 40),
}
SHAPES = ["square", "circle", "triangle", "diamond", "rectangle"]
POSITIONS = ["left", "center", "right", "top", "bottom", "offset"]
POSITION_XY = {
    "left": (0.25, 0.5),
    "center": (0.5, 0.5),
    "right": (0.75, 0.5),
    "top": (0.5, 0.26),
    "bottom": (0.5, 0.74),
    "offset": (0.32, 0.68),
}
SIZES = [("small", 13), ("big", 22)]


def make_shape_mask(h: int, w: int, cx: float, cy: float, r: float, shape: str) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    if shape == "square":
        return (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    if shape == "circle":
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if shape == "diamond":
        return np.abs(xx - cx) + np.abs(yy - cy) <= r
    if shape == "rectangle":
        return (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r * 0.55)
    if shape == "triangle":
        v0 = np.array([cx, cy - 1.2 * r])
        v1 = np.array([cx - 1.0 * r, cy + 0.7 * r])
        v2 = np.array([cx + 1.0 * r, cy + 0.7 * r])
        pts = np.stack([xx, yy], axis=-1)

        def s(a, b, q):
            return (q[..., 1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (q[..., 0] - a[0])

        d0 = s(v0, v1, pts)
        d1 = s(v1, v2, pts)
        d2 = s(v2, v0, pts)
        return (d0 >= 0) & (d1 >= 0) & (d2 >= 0)
    raise ValueError(f"unknown shape: {shape}")


def make_image(h: int, w: int, color: str, shape: str, cx: float, cy: float, r: float) -> np.ndarray:
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    sel = make_shape_mask(h, w, cx, cy, r, shape)
    canvas[sel] = np.array(COLOR_RGB[color], dtype=np.uint8)
    return canvas


def attribute_for_index(i: int) -> tuple[str, str, str, str, float]:
    color = COLORS[i % len(COLORS)]
    shape = SHAPES[(i // len(COLORS)) % len(SHAPES)]
    position = POSITIONS[(i // (len(COLORS) * len(SHAPES))) % len(POSITIONS)]
    size, r = SIZES[(i // (len(COLORS) * len(SHAPES) * len(POSITIONS))) % len(SIZES)]
    return color, shape, position, size, float(r)


def generate_synthetic_pairs(n: int = 256, image_size: int = 112):
    maxn = len(COLORS) * len(SHAPES) * len(POSITIONS) * len(SIZES)
    assert n <= maxn, f"requested {n} pairs but only {maxn} combinations exist"
    images = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    captions = []
    meta = []
    for i in range(n):
        color, shape, position, size, r = attribute_for_index(i)
        cx, cy = POSITION_XY[position]
        images[i] = make_image(image_size, image_size, color, shape, cx * image_size, cy * image_size, r)
        captions.append(f"a {size} {color} {shape} is on the {position}")
        meta.append(f"{color}|{shape}|{position}|{size}")
    return images, captions, meta


def main():
    p = argparse.ArgumentParser(description="Generate deterministic synthetic image/caption pairs")
    p.add_argument("--out", default="runs/mm_synth/data.npz")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    images, captions, meta = generate_synthetic_pairs(args.n, args.image_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        images=images,
        captions=np.array(captions),
        meta=np.array(meta),
        n=args.n,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"wrote {out}: {images.shape[0]} pairs (deterministic, seed={args.seed})")


if __name__ == "__main__":
    main()