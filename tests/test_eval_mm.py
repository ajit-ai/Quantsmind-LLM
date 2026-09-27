"""Hermetic tests for the M1 evaluation harness.

These tests use only tiny in-memory models and tmp_path fixtures: no real data,
no network, no GPU.  They verify the measurement plumbing (checkpoint loading,
generation, reporting) — they make no claim that any checkpoint is good.
"""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.model import QMModel
from training.multimodal import QMMultimodal

from evaluation import generate
from evaluation.eval_caption import evaluate_image, evaluate_manifest
from evaluation.eval_text import _deltas, evaluate_text, score_items
from evaluation.eval_vqa import evaluate_records, normalize_answer
from evaluation.loss_sanity import review_loss_log
from evaluation.mm_checkpoint import load_checkpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOKENIZER_PATH = ROOT / "tokenizer" / "qm_bpe_v0.json"

# Vocab 49156 (real layout) so tokenizer-produced ids stay in embedding range
# and the 4 trailing special ids (49152..49155) match the training contract.
TEXT_CFG = {
    "vocab_size": 49156,
    "d_model": 32,
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "ffn_dim": 88,
    "max_seq_len": 512,
    "rope_theta": 10000.0,
    "norm_eps": 1.0e-5,
    "tie_embeddings": True,
}

VISION_CFG = {
    "image_size": 32,
    "patch_size": 8,
    "in_channels": 3,
    "d_model": 32,
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "ffn_dim": 88,
    "norm_eps": 1.0e-5,
}


def build_text():
    torch.manual_seed(0)
    return QMModel(TEXT_CFG)


def build_mm():
    torch.manual_seed(0)
    return QMMultimodal(TEXT_CFG, VISION_CFG)


def save_ckpt(tmp_path: Path, model, text_cfg, config=None, metadata=None, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": text_cfg,
            "config": config,
            "metadata": metadata,
            "step": 0,
        },
        path,
    )
    return path


def make_png(path: Path, size=32) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8))).save(path)
    return path


def make_tokens(path: Path, n=7000, vocab=49156) -> Path:
    rng = np.random.default_rng(0)
    path.write_bytes(rng.integers(0, vocab, n, dtype=np.uint16).tobytes())
    return path


def make_seed(path: Path) -> Path:
    items = [
        {"id": "a1", "category": "concept", "question": "what is one", "reference": "a", "requires_tool": False},
        {"id": "a2", "category": "arithmetic", "question": "what is two", "reference": "b", "requires_tool": False},
    ]
    path.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    return path


# ---------------------------------------------------------------- checkpoint loader


def test_load_text_only_checkpoint(tmp_path):
    path = save_ckpt(tmp_path, build_text(), TEXT_CFG)
    ckpt = load_checkpoint(path)
    assert not ckpt.is_mm
    assert ckpt.text_cfg["vocab_size"] == 49156
    assert ckpt.text_model is ckpt.model
    assert ckpt.config is None


def test_load_mm_checkpoint_inline_vision(tmp_path):
    config = {"run": {"model_config": "tiny", "vision_config": VISION_CFG}}
    path = save_ckpt(tmp_path, build_mm(), TEXT_CFG, config=config)
    ckpt = load_checkpoint(path)
    assert ckpt.is_mm
    assert ckpt.text_model is ckpt.model.text
    assert ckpt.model.image_tokens == (32 // 8) ** 2 + 1


def test_load_mm_requires_vision_cfg(tmp_path):
    path = save_ckpt(tmp_path, build_mm(), TEXT_CFG, config={"run": {}})
    with pytest.raises(ValueError, match="vision_config"):
        load_checkpoint(path)
    ckpt = load_checkpoint(path, vision_cfg=VISION_CFG)
    assert ckpt.is_mm


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_missing_cfg(tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"model": build_text().state_dict()}, path)
    with pytest.raises(ValueError, match="cfg"):
        load_checkpoint(path)


def test_load_bad_state(tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"model": "garbage", "cfg": TEXT_CFG}, path)
    with pytest.raises(ValueError, match="state dict"):
        load_checkpoint(path)


# ---------------------------------------------------------------- generation


def test_image_tensor_file_shape(tmp_path):
    p = make_png(tmp_path / "a.png")
    t = generate.image_tensor_file(p, size=32)
    assert t.shape == (1, 3, 32, 32)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_generate_caption_deterministic_and_in_vocab(tmp_path):
    model = build_mm()
    img = torch.rand(1, 3, 32, 32)
    ids1 = generate.generate_caption(model, img, max_new_tokens=8)
    ids2 = generate.generate_caption(model, img, max_new_tokens=8)
    assert ids1 == ids2
    assert all(0 <= i < TEXT_CFG["vocab_size"] for i in ids1)
    assert len(ids1) <= 8


def test_generate_answer_deterministic():
    model = build_mm()
    img = torch.rand(1, 3, 32, 32)
    q = [5, 6, 7]
    a1 = generate.generate_answer(model, img, q, max_new_tokens=8)
    a2 = generate.generate_answer(model, img, q, max_new_tokens=8)
    assert a1 == a2


def test_generate_requires_multimodal():
    model = build_text()
    img = torch.rand(1, 3, 32, 32)
    with pytest.raises(TypeError, match="QMMultimodal"):
        generate.generate_continuation(model, img)


def test_decode_ids_filters_specials():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    ids = tok.encode("alpha beta").ids
    assert generate.decode_ids(tok, ids) == "alpha beta"
    assert generate.decode_ids(tok, [49152, 49153, 49155]) == ""
    assert generate.decode_ids(tok, list(ids) + [49155]) == "alpha beta"


# ---------------------------------------------------------------- caption eval


def test_evaluate_image_record(tmp_path):
    model = build_mm()
    img = torch.rand(1, 3, 32, 32)
    rep = evaluate_image(model, None, img, max_new_tokens=8)
    assert set(rep) >= {"caption", "tokens", "generated_len"}
    assert rep["generated_len"] == len(rep["tokens"]) <= 8
    assert rep["caption"] == "" or isinstance(rep["caption"], str)


def test_evaluate_manifest_reports_errors(tmp_path):
    model = build_mm()
    img: Path = make_png(tmp_path / "imgs" / "a.png")
    manifest = {
        "schema_version": 1,
        "preprocessing_version": "mm1_v1",
        "dataset": {"name": "tiny"},
        "samples": [
            {
                "sample_id": "s1",
                "split": "train",
                "image_path": str(img.relative_to(tmp_path)),
                "caption": "a photo",
                "source": "tiny-fixture",
            },
            {
                "sample_id": "s2",
                "split": "train",
                "image_path": "imgs/missing.png",
                "caption": "will not load",
                "source": "tiny-fixture",
            },
        ],
    }
    mp = tmp_path / "MANIFEST.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    out = evaluate_manifest(model, None, mp, split="train", image_size=32, max_new_tokens=4)
    assert out["summary"]["requested"] == 2
    assert out["summary"]["evaluated"] == 1
    assert out["summary"]["errors"] == 1
    assert len(out["errors"]) == 1


# ---------------------------------------------------------------- VQA eval


def test_normalize_answer():
    assert normalize_answer("  A Cat  sits " ) == "a cat sits"
    assert normalize_answer(None) == ""


def test_vqa_evaluate_records_structure(tmp_path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = build_mm()
    img = make_png(tmp_path / "q.png")
    records = [
        {"image": str(img), "question": "what is shown", "expected_answer": "a cat"},
        {"image": "does-not-exist.png", "question": "x", "expected_answer": "y"},
        {"question": "missing keys"},
    ]
    out = evaluate_records(model, tok, records, image_size=32, max_new_tokens=4)
    assert out["summary"]["records"] == 3
    assert out["summary"]["evaluated"] == 1
    assert out["summary"]["errors"] == 2
    assert out["results"][0]["pass"] in (True, False)
    assert out["results"][0]["generated_answer"] is not None


def test_vqa_records_file_json_and_jsonl(tmp_path):
    img = make_png(tmp_path / "q.png")
    json_file = tmp_path / "r.json"
    json_file.write_text(
        json.dumps([{"image": str(img), "question": "q", "expected_answer": "a"}]),
        encoding="utf-8",
    )
    jsonl_file = tmp_path / "r.jsonl"
    jsonl_file.write_text(
        json.dumps({"image": str(img), "question": "q", "expected_answer": "a"}) + "\n",
        encoding="utf-8",
    )
    bom_file = tmp_path / "bom.json"
    bom_file.write_bytes(b'\xef\xbb\xbf' + json.dumps([{"image": str(img), "question": "q", "expected_answer": "a"}]).encode("utf-8"))
    from evaluation.eval_vqa import _iter_records

    assert len(list(_iter_records(json_file))) == 1
    assert len(list(_iter_records(jsonl_file))) == 1
    assert len(list(_iter_records(bom_file))) == 1


# ---------------------------------------------------------------- text eval


def test_text_eval_heldout(tmp_path):
    ckpt = save_ckpt(tmp_path, build_text(), TEXT_CFG)
    tokens = make_tokens(tmp_path / "toks.bin")
    rep = evaluate_text(load_checkpoint(ckpt).text_model, lambda s: [1, 2], None, tokens,
                        seq_len=64, max_windows=8)
    assert isinstance(rep["heldout_perplexity"], float)
    assert rep["heldout_perplexity"] > 0.0
    assert rep["seed_summary"] is None


def test_text_eval_seed(tmp_path):
    ckpt = save_ckpt(tmp_path, build_text(), TEXT_CFG)
    seed = make_seed(tmp_path / "seed.jsonl")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    rep = evaluate_text(load_checkpoint(ckpt).text_model, lambda s: tok.encode(s).ids,
                        str(seed), None)
    assert rep["seed_summary"]["n_items"] == 2
    assert set(rep["seed_summary"]["by_category"]) == {"arithmetic", "concept"}
    assert isinstance(rep["seed_summary"]["overall_mean_nll"], float)


def test_text_eval_deltas_between_models(tmp_path):
    ckpt_a = save_ckpt(tmp_path, build_text(), TEXT_CFG, name="a.pt")
    ckpt_b = save_ckpt(tmp_path, build_text(), TEXT_CFG, name="b.pt")
    tokens = make_tokens(tmp_path / "toks.bin")
    seed = make_seed(tmp_path / "seed.jsonl")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    enc = lambda s: tok.encode(s).ids  # noqa: E731
    a = evaluate_text(load_checkpoint(ckpt_a).text_model, enc, str(seed), str(tokens), seq_len=64, max_windows=8)
    b = evaluate_text(load_checkpoint(ckpt_b).text_model, enc, str(seed), str(tokens), seq_len=64, max_windows=8)
    d = _deltas(a, b)
    assert "heldout_perplexity" in d
    assert isinstance(d["heldout_perplexity"]["delta"], float)
    assert "seed_overall_nll_delta" in d


def test_score_items_matches_evaluate_pilot_summarize(tmp_path):
    seed = make_seed(tmp_path / "seed.jsonl")
    from evaluation.load_eval import load_eval_set
    from evaluation.evaluate_pilot import summarize
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = load_checkpoint(save_ckpt(tmp_path, build_text(), TEXT_CFG)).text_model
    rows = score_items(model, list(load_eval_set(seed)), lambda s: tok.encode(s).ids)
    summary = summarize(rows)
    assert summary == {"overall_mean_nll": summary["overall_mean_nll"], "by_category": summary["by_category"], "n_items": 2}


# ---------------------------------------------------------------- loss sanity


def _log(pairs):
    return "\n".join(
        f"iter {i} step {s}/{max(s for s, _ in pairs)} loss={l:.4f} lr=0.0 "
        for i, (s, l) in enumerate(pairs)
    )


def test_loss_sanity_clean():
    pairs = [(s, 1.0) for s in range(101)]
    packet = review_loss_log(_log(pairs))
    assert packet["status"] == "clean"
    assert packet["spikes"] == []
    assert packet["flags"] == []


def test_loss_sanity_recovered_spike_ok():
    pairs = [(s, 1.0) for s in range(0, 16)] + [(16, 5.0), (17, 1.0), (18, 1.0)]
    packet = review_loss_log(_log(pairs))
    # the spike was recorded but flagged as recovered -> no review required
    assert packet["unrecovered_spikes"] == []
    assert packet["status"] == "clean"


def test_loss_sanity_unrecovered_spike_flag():
    pairs = [(s, 1.0) for s in range(0, 80)] + [(s, 1.0) for s in range(80, 95)] + [(95, 50.0)]
    packet = review_loss_log(_log(pairs))
    assert packet["status"] == "review_required"
    assert any("unrecovered" in f for f in packet["flags"])
    assert packet["unrecovered_spikes"]


def test_loss_sanity_recovered_in_window():
    pairs = [(s, 1.0) for s in range(0, 80)] + [(80, 50.0), (81, 1.0), (82, 1.0)]
    packet = review_loss_log(_log(pairs))
    assert packet["status"] == "clean"


def test_loss_sanity_insufficient_steps():
    packet = review_loss_log(_log([(0, 1.0), (1, 1.0), (2, 1.0)]))
    assert packet["status"] == "review_required"
    assert any(f.startswith("insufficient") for f in packet["flags"])


def test_loss_sanity_malformed_log():
    with pytest.raises(ValueError, match="no train_mm loss lines"):
        review_loss_log("nothing to see here")


def test_loss_sanity_utf16_log(tmp_path):
    from evaluation.loss_sanity import read_log_text

    log = _log([(s, 1.0) for s in range(6)]).encode("utf-16")
    p = tmp_path / "log.txt"
    p.write_bytes(log)
    packet = review_loss_log(read_log_text(p))
    assert packet["total_steps_logged"] == 6