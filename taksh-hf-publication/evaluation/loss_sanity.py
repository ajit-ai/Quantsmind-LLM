"""Loss-sanity review helper.

Implements the ``loss_sanity`` rule from ``evaluation/promotion.yaml``
(manual review): scans a ``train_mm``-style log for unrecovered loss spikes in
the final 20% of training.  The helper produces a review packet for a human;
it never makes the promotion decision itself.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

_LINE_RE = re.compile(r"[^\d]*step\s+(\d+)\s*/\s*(\d+)\s+loss=([0-9.eE+\-]+)")

CRITERION = "loss sanity (manual review): no unrecovered loss spikes in the final 20% of training"


def _parse_log(text: str) -> list[tuple[int, float]]:
    steps = []
    for m in _LINE_RE.finditer(text):
        try:
            steps.append((int(m.group(1)), float(m.group(3))))
        except ValueError:
            continue
    if not steps:
        raise ValueError("no train_mm loss lines found (expected 'step N/M loss=...')")
    return steps


def review_loss_log(text: str, trailing_fraction: float = 0.2,
                    spike_factor: float = 1.5,
                    recovery_factor: float = 1.25) -> dict:
    """Review a train_mm log for unrecovered loss spikes.

    Parameters
    ----------
    text : str
        Raw training log contents.
    trailing_fraction : float
        Fraction of the training run treated as the "final window" (0.2).
    spike_factor : float
        A step is a spike when loss > ``spike_factor * baseline``.
    recovery_factor : float
        A spike is recovered when a later step is ``<= recovery_factor * baseline``.

    Returns
    -------
    dict
        A review packet: ``status`` in ``{"clean", "review_required"}``,
        flags, baseline stats, spike records, and the criterion string.
    """
    steps = _parse_log(text)
    if not steps:
        raise ValueError("no train_mm loss lines found")

    last_step = steps[-1][0]
    cutoff = last_step * (1.0 - trailing_fraction)
    window = [s for s in steps if s[0] >= cutoff]
    prior = [s for s in steps if s[0] < cutoff]

    flags = []

    if len(steps) < 5:
        flags.append("insufficient_training_steps_to_judge_final_20%")

    if not prior:
        flags.append("no_steps_before_final_window_for_baseline")
        baseline = None
    else:
        baseline = statistics.median(loss for _, loss in prior)

    spikes = []
    if baseline is not None:
        for step, loss in window:
            if loss > spike_factor * baseline:
                recovered = any(l2 <= recovery_factor * baseline for s2, l2 in steps if s2 > step)
                spikes.append({
                    "step": step, "loss": round(float(loss), 6),
                    "baseline_median": round(baseline, 6),
                    "recovered": bool(recovered),
                })
    unrecovered = [s for s in spikes if not s["recovered"]]
    if unrecovered:
        flags.append("unrecovered_loss_spikes_in_final_20%")

    status = "review_required" if (unrecovered or any(
        f.startswith("insufficient") or f.startswith("no_") for f in flags)) else "clean"

    return {
        "status": status,
        "criterion": CRITERION,
        "review_kind": "manual_review",
        "total_steps_logged": len(steps),
        "last_step": last_step,
        "final_window_start_step": cutoff,
        "baseline_median": round(baseline, 6) if baseline is not None else None,
        "final_window_steps": len(window),
        "spikes": spikes,
        "unrecovered_spikes": unrecovered,
        "flags": flags,
    }


def read_log_text(path: str | Path) -> str:
    """Read a training log, sniffing UTF-8 vs UTF-16 (Windows redirects)."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig", errors="replace")
    if b"\x00" in raw[:4096]:
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.loss_sanity",
        description="Review a train_mm log for unrecovered loss spikes (manual review helper).",
    )
    parser.add_argument("--log", required=True, help="training log file (train_mm format)")
    parser.add_argument("--out", help="write the review packet as JSON here")
    args = parser.parse_args(argv)

    text = read_log_text(args.log)
    try:
        packet = review_loss_log(text)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(packet, indent=2), encoding="utf-8")
    print(json.dumps(packet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())