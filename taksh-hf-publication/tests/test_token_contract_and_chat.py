"""Token-artifact contract (Workstream A) and chat-template compatibility (Workstream B).

Offline throughout: no network, no corpus generation, no training run.
"""

import json
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from data.encode_dataset import main as encode_main
from training.chat_template import (
    BOS,
    EOS,
    REQUIRED_MARKERS,
    SYSTEM_PROMPT,
    IncompatibleTokenizer,
    prompt_token_length,
    render_completion,
    render_prompt,
    validate_chat_template,
)
from training.data import TokenArtifactIncompatible, TokenDataset

CODE_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"
GENERAL_TOKENIZER = "tokenizer/qm_bpe_v0.json"


@pytest.fixture(scope="module")
def code_tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(CODE_TOKENIZER)


@pytest.fixture(scope="module")
def general_tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(GENERAL_TOKENIZER)


def _encode(corpus, out, tokenizer=GENERAL_TOKENIZER, extra=()):
    import sys

    argv = sys.argv
    sys.argv = [
        "encode_dataset",
        "--corpus", str(corpus),
        "--tokenizer", tokenizer,
        "--out", str(out),
        *extra,
    ]
    try:
        encode_main()
    finally:
        sys.argv = argv
    return out.with_suffix(".meta.json")


# --------------------------------------------------------------------------- #
# A -- token artifact contract
# --------------------------------------------------------------------------- #
def test_pilot_config_resolves_to_an_artifact_that_exists():
    from training.pretrain import load_yaml, resolve_tokens_path, validate_train_config

    cfg = load_yaml("configs/train/pretrain_pilot.yaml")
    assert validate_train_config(cfg) == []
    tokens = resolve_tokens_path(cfg, None)
    if not tokens.exists():
        pytest.skip("token artifacts are not part of the source release")
    assert tokens.exists(), f"pilot config points at a missing artifact: {tokens}"
    assert Path("data/corpus_v0.txt").exists(), "documented regeneration input must exist"


def test_pilot_artifact_matches_model_vocabulary():
    from training.pretrain import (
        flatten_model_cfg,
        load_yaml,
        resolve_tokens_path,
    )

    cfg = load_yaml("configs/train/pretrain_pilot.yaml")
    model_cfg = flatten_model_cfg(cfg)
    tokens = resolve_tokens_path(cfg, None)
    if not tokens.exists():
        pytest.skip("token artifacts are not part of the source release")
    dataset = TokenDataset(str(tokens))
    problems = dataset.validate_for_model(model_cfg["vocab_size"])
    assert not any("outside" in p for p in problems)
    assert dataset.vocab_size == model_cfg["vocab_size"]


def test_generated_artifact_records_full_provenance(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        '{"text": "def a():\\n    return 1\\n"}\n'
        '{"text": "def b():\\n    return 2\\n"}\n',
        encoding="utf-8",
    )
    meta = json.loads(
        _encode(corpus, tmp_path / "s.bin", tokenizer=CODE_TOKENIZER).read_text(
            encoding="utf-8"
        )
    )
    assert meta["schema"] == "quantsmind.token_stream/1"
    assert meta["tokenizer"] == CODE_TOKENIZER
    assert len(meta["tokenizer_sha256"]) == 64
    assert len(meta["corpus_sha256"]) == 64
    assert meta["num_documents"] == 2
    assert meta["document_separator"] == EOS
    assert meta["document_separator_id"] == 2
    assert "never truncated" in meta["oversized_document_policy"]
    assert "not pre-packed sequences" in meta["representation"]


def test_generated_artifact_is_consumable_by_the_loader(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        "".join(f'{{"text": "alpha beta gamma delta {i}\\n"}}\n' for i in range(20)),
        encoding="utf-8",
    )
    _encode(corpus, tmp_path / "s.bin", tokenizer=CODE_TOKENIZER)

    dataset = TokenDataset(str(tmp_path / "s.bin"))
    assert len(dataset) > 0
    assert dataset.validate_for_model(5822, 5822) == []
    x, y = dataset.batch(2, 16)
    assert x.shape == (2, 16) and y.shape == (2, 16)


def test_window_larger_than_the_stream_is_reported_actionably(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "tiny\\n"}\n', encoding="utf-8")
    _encode(corpus, tmp_path / "s.bin", tokenizer=CODE_TOKENIZER)
    dataset = TokenDataset(str(tmp_path / "s.bin"))
    with pytest.raises(ValueError) as exc:
        dataset.batch(2, 4096)
    assert "at least" in str(exc.value)
    assert "lower" in str(exc.value)


def test_missing_artifact_is_reported_actionably(tmp_path):
    missing = tmp_path / "nope.bin"
    with pytest.raises(Exception) as exc:
        TokenDataset(str(missing))
    assert "nope.bin" in str(exc.value)


def test_artifact_from_a_wider_tokenizer_is_rejected(tmp_path):
    """A different tokenizer's artifact must not be silently reinterpreted."""
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "hello world\\n"}\n', encoding="utf-8")
    _encode(corpus, tmp_path / "s.bin", tokenizer=GENERAL_TOKENIZER)

    dataset = TokenDataset(str(tmp_path / "s.bin"))
    problems = dataset.validate_for_model(5822, 5822)
    assert any("exceeds model vocab_size" in p for p in problems)
    assert any("different tokenizer" in p for p in problems)


def test_unknown_dtype_is_refused(tmp_path):
    bad = tmp_path / "s.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(bad)
    bad.with_suffix(".meta.json").write_text(
        json.dumps({"dtype": "float64", "vocab_size": 10}), encoding="utf-8"
    )
    with pytest.raises(TokenArtifactIncompatible) as exc:
        TokenDataset(str(bad))
    assert "unknown dtype" in str(exc.value)


def test_shipped_pilot_artifact_is_now_fully_verifiable():
    """Since the 2026-10-18 regeneration, the artifact carries real provenance.

    The legacy-artifact detector must stay covered, but against a synthetic
    fixture: asserting that the *shipped* artifact is legacy would simply
    re-assert a fact that regeneration deliberately ended.
    """
    if not Path("data/corpus_v0.tokens.bin").exists():
        pytest.skip("token artifacts are not part of the source release")
    dataset = TokenDataset("data/corpus_v0.tokens.bin")
    problems = dataset.validate_for_model(49152, 49152)
    assert problems == [], problems
    assert dataset.meta["schema"] == "quantsmind.token_stream/1"
    assert dataset.meta["document_separator"] == "<eos>"
    assert dataset.meta["num_documents"] > 0


def test_legacy_artifact_is_still_detected_and_reported(tmp_path):
    """The detection branch remains live, exercised on a synthetic artifact."""
    legacy = tmp_path / "legacy.bin"
    np.asarray(np.arange(64, dtype=np.uint16)).tofile(legacy)
    legacy.with_suffix(".meta.json").write_text(
        json.dumps({"vocab_size": 49152, "num_tokens": 64, "dtype": "uint16"}),
        encoding="utf-8",
    )
    problems = TokenDataset(str(legacy)).validate_for_model(49152)
    assert any("LEGACY ARTIFACT" in p for p in problems)
    assert any("run together" in p for p in problems)
    # Still non-fatal when the vocabulary matches.
    assert not any("outside" in p for p in problems)


def test_artifact_without_metadata_is_reported():
    bare = Path(tempfile.mkdtemp()) / "b.bin"
    np.asarray([1, 2, 3, 4], dtype=np.uint16).tofile(bare)
    problems = TokenDataset(str(bare)).validate_for_model(49152)
    assert any("no metadata" in p for p in problems)


def test_no_separator_flag_reproduces_legacy_encoding(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "aa\\n"}\n{"text": "bb\\n"}\n', encoding="utf-8")
    meta = json.loads(
        _encode(corpus, tmp_path / "s.bin", tokenizer=CODE_TOKENIZER,
                extra=("--no-separator",)).read_text(encoding="utf-8")
    )
    assert meta["document_separator"] is None
    assert "pre-B6" in meta["separator_policy"]


def test_generation_is_reproducible(tmp_path):
    """Same input, same tokenizer, same bytes."""
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "deterministic output\\n"}\n', encoding="utf-8")
    _encode(corpus, tmp_path / "a.bin", tokenizer=CODE_TOKENIZER)
    _encode(corpus, tmp_path / "b.bin", tokenizer=CODE_TOKENIZER)
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    meta_a = json.loads((tmp_path / "a.meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((tmp_path / "b.meta.json").read_text(encoding="utf-8"))
    assert meta_a["corpus_sha256"] == meta_b["corpus_sha256"]
    assert meta_a["tokenizer_sha256"] == meta_b["tokenizer_sha256"]


# --------------------------------------------------------------------------- #
# B -- C3 chat template compatibility
# --------------------------------------------------------------------------- #
def test_legacy_template_markers_are_absent_from_both_tokenizers(code_tok, general_tok):
    """Establishes the defect: the old markers were never real tokens."""
    for tok in (code_tok, general_tok):
        for marker in ("<|system|>", "<|user|>", "<|assistant|>", "<|eos|>"):
            assert tok.token_to_id(marker) is None, f"{marker} unexpectedly exists"


def test_legacy_template_shreds_into_byte_fragments(code_tok):
    legacy = "<|system|>hi<|assistant|>yo<|eos|>"
    tokens = code_tok.encode(legacy).tokens
    assert "<|system|>" not in tokens
    assert "<|" in tokens, "the marker was shredded rather than tokenized"


@pytest.mark.parametrize("tok_name", [CODE_TOKENIZER, GENERAL_TOKENIZER])
def test_template_markers_are_registered(tok_name):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_name)
    resolved = validate_chat_template(tok)
    assert set(resolved) == set(REQUIRED_MARKERS)
    for marker, token_id in resolved.items():
        assert tok.id_to_token(token_id) == marker


def test_tokenizer_without_markers_is_refused_with_a_clear_error():
    class Fake:
        def token_to_id(self, name):
            return None

    with pytest.raises(IncompatibleTokenizer) as exc:
        validate_chat_template(Fake())
    assert "<bos>" in str(exc.value)
    assert "<eos>" in str(exc.value)
    assert "byte-level" in str(exc.value)


def test_rendered_completion_uses_exactly_two_special_tokens(code_tok):
    text = render_completion(SYSTEM_PROMPT, "add two numbers", "def add(a,b): return a+b")
    enc = code_tok.encode(text)
    specials = [t for t in enc.tokens if t in ("<bos>", "<eos>")]
    assert specials == ["<bos>", "<eos>"], f"unexpected specials: {specials}"
    assert enc.ids[0] == code_tok.token_to_id(BOS)
    assert enc.ids[-1] == code_tok.token_to_id(EOS)


def test_prompt_is_a_literal_prefix_of_the_completion(code_tok):
    """Label masking is only correct if the prompt is the completion prefix."""
    prompt = render_prompt(SYSTEM_PROMPT, "hello")
    completion = render_completion(SYSTEM_PROMPT, "hello", "world")
    assert completion.startswith(prompt)
    assert completion.endswith(EOS)


def test_prompt_token_length_masks_exactly_the_prompt(code_tok):
    prompt = render_prompt(SYSTEM_PROMPT, "hello")
    length = prompt_token_length(code_tok, SYSTEM_PROMPT, "hello")
    assert length == len(code_tok.encode(prompt).ids) - 1

    completion_ids = code_tok.encode(
        render_completion(SYSTEM_PROMPT, "hello", "world")
    ).ids
    masked_text = code_tok.decode(completion_ids[1 : 1 + length])
    assert "hello" in masked_text
    remainder = code_tok.decode(completion_ids[1 + length :])
    assert "world" in remainder
    assert "hello" not in remainder


def test_no_role_markers_survive_in_any_render(code_tok):
    for text in (
        render_prompt(SYSTEM_PROMPT, "q"),
        render_completion(SYSTEM_PROMPT, "q", "a"),
    ):
        assert "<|" not in text
        for marker in ("<|system|>", "<|user|>", "<|assistant|>", "<|eos|>"):
            assert marker not in text


def test_template_survives_encode_decode_roundtrip(code_tok):
    text = render_completion(SYSTEM_PROMPT, "q", "a")
    decoded = code_tok.decode(
        code_tok.encode(text).ids, skip_special_tokens=False
    )
    assert decoded.startswith(BOS)
    assert decoded.endswith(EOS)
    assert "a" in decoded


# --------------------------------------------------------------------------- #
# B -- SFT and inference share one template
# --------------------------------------------------------------------------- #
def test_sft_dataset_masks_the_prompt_and_supervises_the_response(code_tok, tmp_path):
    from training.sft import SFTPairDataset

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {"instruction": "UNIQUEQUESTION", "response": "UNIQUERESPONSE"}
        )
        + "\n",
        encoding="utf-8",
    )
    ds = SFTPairDataset(str(path), code_tok, max_length=512)
    input_ids, labels = ds[0]

    assert input_ids.shape == labels.shape
    masked = code_tok.decode(
        input_ids[labels == -100].tolist()
    )
    supervised = code_tok.decode(
        input_ids[labels != -100].tolist()
    )
    assert "UNIQUEQUESTION" in masked
    assert "UNIQUERESPONSE" in supervised
    # labels = ids[1:], so the EOS is the final *target* the model is trained to
    # predict -- that is what inference stops on.
    assert labels[labels != -100][-1].item() == code_tok.token_to_id(EOS)


def test_sft_dataset_refuses_incompatible_tokenizer(tmp_path):
    from training.sft import SFTPairDataset

    path = tmp_path / "pairs.jsonl"
    path.write_text('{"instruction": "q", "response": "a"}\n', encoding="utf-8")

    class NoSpecials:
        def token_to_id(self, name):
            return None

    with pytest.raises(IncompatibleTokenizer):
        SFTPairDataset(str(path), NoSpecials())


def test_inference_prompt_matches_the_training_prompt(code_tok):
    """Inference must feed the model the exact prefix SFT trained on."""
    from serving.code_chat import CODE_SYSTEM_PROMPT, format_prompt
    from training.chat_template import render_prompt

    # No context: the user turn is the message verbatim.
    served = format_prompt("write a function", "python", "")
    assert served == render_prompt(CODE_SYSTEM_PROMPT, "write a function")
    assert served.startswith(BOS)
    assert "<|" not in served

    # With context, the same rule applies to the wrapped user turn.
    with_ctx = format_prompt("explain this", "javascript", "const a = 1;")
    expected_user = "Context (javascript):\n```\nconst a = 1;\n```\n\nQuestion: explain this"
    assert with_ctx == render_prompt(CODE_SYSTEM_PROMPT, expected_user)


def test_inference_prompt_is_a_prefix_of_the_training_completion():
    from serving.code_chat import CODE_SYSTEM_PROMPT, format_prompt

    prompt = format_prompt("add two numbers", "python", "")
    completion = render_completion(CODE_SYSTEM_PROMPT, "add two numbers", "def add():")
    assert completion.startswith(prompt)


def test_template_does_not_exceed_model_context(code_tok):
    """A template that cannot fit the context would silently truncate the eos."""
    from training.model import QMModel

    cfg = {"model": {
        "name": "qm-code-tiny", "vocab_size": 5822, "d_model": 32, "n_layers": 1,
        "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 64, "max_seq_len": 512,
        "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True,
    }}
    model = QMModel(cfg["model"])
    emb = model.tok_emb.weight if hasattr(model, "tok_emb") else None
    assert emb is None or emb.shape[0] == 5822, "embedding width must match tokenizer vocab"
    ids = code_tok.encode(
        render_completion(SYSTEM_PROMPT, "q" * 50, "a" * 50)
    ).ids
    assert len(ids) < 512