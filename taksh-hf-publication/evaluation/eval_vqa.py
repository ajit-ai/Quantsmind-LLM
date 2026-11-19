"""Minimal VQA smoke evaluation.

For each record ``{image, question, expected_answer}`` the model generates an
answer conditioned on the image and the question, and the result is compared
with the expected answer using an exact normalized match.  The pass/fail
outcomes are *reported only* (exit code is always 0); this is measurement
infrastructure and does not by itself promote or demote a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from evaluation.generate import decode_ids, generate_answer, image_tensor_file
from evaluation.mm_checkpoint import load_checkpoint


def normalize_answer(text: str) -> str:
    """Lowercase, strip surrounding whitespace, collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def evaluate_records(model, tokenizer, records, max_new_tokens=16, image_size=112,
                     device="cpu"):
    """Evaluate a list of VQA records.

    Each record must contain ``image`` (path), ``question``, ``expected_answer``.
    ``image`` paths are resolved relative to the records file directory when
    they are relative.  Returns per-record results, a summary, and any
    per-record errors (bad/missing files, wrong types).
    """
    results = []
    errors = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or not {"image", "question", "expected_answer"} <= set(rec):
            errors.append({"index": i, "error": "record missing image/question/expected_answer"})
            continue
        img_path = rec["image"]
        if not Path(img_path).is_absolute():
            img_path = str(Path(".") / img_path)
        if not Path(img_path).is_file():
            errors.append({"index": i, "error": f"image file not found: {img_path}"})
            continue
        try:
            ids = tokenizer.encode(rec["question"]).ids
            image = image_tensor_file(img_path, size=image_size, device=device)
            gen_ids = generate_answer(model, image, ids, max_new_tokens=max_new_tokens)
            generated = decode_ids(tokenizer, gen_ids)
            expected = normalize_answer(rec["expected_answer"])
            got = normalize_answer(generated)
            results.append({
                "index": i,
                "question": rec["question"],
                "expected_answer": rec["expected_answer"],
                "generated_answer": generated,
                "pass": bool(got == expected),
            })
        except Exception as exc:  # noqa: BLE001 - report per-record and continue
            errors.append({"index": i, "error": str(exc)})

    summary = {
        "records": len(records),
        "evaluated": len(results),
        "errors": len(errors),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": sum(1 for r in results if not r["pass"]),
        "measurement_only": True,
    }
    return {"summary": summary, "results": results, "errors": errors}


def _iter_records(path: Path):
    """Yield records from a JSON array file, a single JSON object, or JSONL."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    if isinstance(obj, list):
        yield from obj
    else:
        yield obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.eval_vqa",
        description="Minimal VQA smoke evaluation (measurement only).",
    )
    parser.add_argument("--ckpt", required=True, help="MM checkpoint (.pt)")
    parser.add_argument("--tokenizer", default="tokenizer/qm_bpe_v0.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--records", required=True,
                        help="JSON or JSONL list of {image, question, expected_answer}")
    parser.add_argument("--out", help="write JSON report here")
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    loaded = load_checkpoint(args.ckpt, device=args.device)
    if not loaded.is_mm:
        raise SystemExit(f"error: {args.ckpt} is not a multimodal checkpoint")

    records = list(_iter_records(Path(args.records)))
    report = evaluate_records(
        loaded.model, tokenizer, records,
        max_new_tokens=args.max_new_tokens, image_size=args.image_size, device=args.device,
    )
    report["checkpoint"] = str(loaded.path)
    report["note"] = "measurement only; no promotion decision"

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    print(f"records={s['records']} evaluated={s['evaluated']} errors={s['errors']} "
          f"passed={s['passed']} failed={s['failed']} (measurement only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())