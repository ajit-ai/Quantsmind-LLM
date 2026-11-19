"""Tests for M4: dataset contract, formatting, loss masking and code checks."""

from __future__ import annotations

import pytest
from tokenizers import models, pre_tokenizers, trainers

from data.instruction_schema import (
    contains_control_characters,
    stable_id,
    validate_record,
)
from training.instruction_format import (
    IGNORE_INDEX,
    FormattingError,
    collate,
    encode_example,
    render_prompt,
)


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """A real tokenizer whose boundary tokens the tests can rely on."""
    from tokenizers import Tokenizer as T

    corpus = tmp_path_factory.mktemp("tok") / "corpus.txt"
    corpus.write_text("\n".join(
        f"def f{i}(a, b):\n    return a + b * {i}\n" for i in range(64)
    ), encoding="utf-8")
    tok = T(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=64, special_tokens=["<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(corpus)], trainer)
    return tok


def good_record(**overrides):
    record = {
        "id": stable_id("code_generation", "do a thing", "print hi"),
        "instruction": "Print hi.",
        "input": "",
        "response": "print('hi')",
        "category": "coding",
        "task": "code_generation",
        "language": "python",
        "source": "org-a",
        "license": "MIT",
        "split": "train",
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Dataset contract
# --------------------------------------------------------------------------- #
def test_valid_record_passes():
    assert validate_record(good_record()) == []


def test_missing_required_field_is_reported():
    record = good_record()
    del record["response"]
    assert any("response" in p for p in validate_record(record))


def test_empty_response_is_rejected():
    assert any("empty" in p for p in validate_record(good_record(response="   ")))


def test_malformed_id_is_rejected():
    assert any("stable form" in p
               for p in validate_record(good_record(id="not-a-stable-id")))


def test_unknown_category_and_task_are_rejected():
    problems = validate_record(good_record(category="nonsense", task="nope"))
    assert any("category" in p for p in problems)
    assert any("task" in p for p in problems)


def test_unsupported_language_is_rejected():
    assert validate_record(good_record(language="brainfuck"))


def test_missing_provenance_is_rejected_by_default():
    problems = validate_record(good_record(source=None, license=None))
    assert any("provenance" in p for p in problems)


def test_stable_id_is_deterministic_and_content_addressed():
    first = stable_id("code_generation", "a", "b")
    assert first == stable_id("code_generation", "a", "b")
    assert first != stable_id("code_generation", "a", "c")
    assert first != stable_id("sql", "a", "b")


def test_control_characters_are_detected():
    assert contains_control_characters("bad\x00text")
    assert not contains_control_characters("fine\ntext\ttab\r")


# --------------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------------- #
def test_prompt_contains_the_three_sections():
    text = render_prompt("Do it.", "some context")
    assert "### Instruction" in text
    assert "### Input" in text
    assert text.rstrip().endswith("### Response")


def test_input_section_is_omitted_when_empty():
    assert "### Input" not in render_prompt("Do it.", "")
    assert "### Input" not in render_prompt("Do it.", "   ")


def test_empty_instruction_is_rejected():
    with pytest.raises(FormattingError):
        render_prompt("")


def test_empty_response_is_rejected_by_encode(tiny_tokenizer):
    with pytest.raises(FormattingError, match="empty"):
        encode_example(tiny_tokenizer, "do it", "   ")


# --------------------------------------------------------------------------- #
# Loss masking
# --------------------------------------------------------------------------- #
def test_prompt_is_masked_and_response_is_supervised(tiny_tokenizer):
    example = encode_example(
        tiny_tokenizer, "Write a function.", "def f():\n    return 1")
    labels, prompt_len = example["labels"], example["prompt_tokens"]
    assert all(label == IGNORE_INDEX for label in labels[:prompt_len])
    assert all(label != IGNORE_INDEX for label in labels[prompt_len:])
    assert example["supervised_tokens"] == len(labels) - prompt_len


def test_mask_boundary_decodes_to_the_response(tiny_tokenizer):
    """Supervision covers the response *except its first token*.

    That is correct next-token behaviour, not an off-by-one: the first
    response token is the model's *input* for predicting the second, so it is
    never a target. Byte-level BPE emits one token per unseen character here,
    which is why the marker loses its leading "U".
    """
    example = encode_example(tiny_tokenizer, "Explain.", "UNIQUE_MARKER_TEXT")
    supervised = tiny_tokenizer.decode(
        example["labels"][example["prompt_tokens"]:])
    collapsed = "".join(supervised.split())
    assert collapsed == "NIQUE_MARKER_TEXT"
    assert "MARKER" in collapsed


def test_labels_are_shifted_like_a_causal_lm(tiny_tokenizer):
    """input_ids[i+1] must equal labels[i] wherever the label is supervised.

    The final position has no successor, so the comparison stops one short.
    """
    example = encode_example(tiny_tokenizer, "Explain.", "RESPONSE_BODY")
    input_ids, labels = example["input_ids"], example["labels"]
    assert len(input_ids) == len(labels)
    compared = 0
    for index in range(len(labels) - 1):
        if labels[index] != IGNORE_INDEX:
            assert int(labels[index]) == int(input_ids[index + 1]), index
            compared += 1
    assert compared > 0, "no supervised positions to compare"


def test_mask_alignment_holds_with_an_input_field(tiny_tokenizer):
    with_input = encode_example(tiny_tokenizer, "Explain.", "RESPONSE_A",
                                "CONTEXT_X")
    without = encode_example(tiny_tokenizer, "Explain.", "RESPONSE_A")
    # The context must be masked, so supervision starts later.
    assert with_input["prompt_tokens"] > without["prompt_tokens"]
    assert all(l == IGNORE_INDEX
               for l in with_input["labels"][:with_input["prompt_tokens"]])


def test_supervise_prompt_option_supervises_everything(tiny_tokenizer):
    example = encode_example(tiny_tokenizer, "Explain.", "RESPONSE",
                             supervise_prompt=True)
    assert all(label != IGNORE_INDEX for label in example["labels"])


def test_padding_is_masked_by_collate(tiny_tokenizer):
    import torch

    short = encode_example(tiny_tokenizer, "Hi.", "ok")
    long = encode_example(tiny_tokenizer, "Hi.", "a much longer response " * 6)
    batch = collate([short, long], pad_id=0)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    pads = batch["attention_mask"] == 0
    assert pads.any(), "fixture should have produced padding"
    assert bool((batch["labels"][pads] == IGNORE_INDEX).all())




def test_multiline_and_long_responses_are_supervised(tiny_tokenizer):
    for response in ("line one\nline two\nline three", "x = 1\n" * 60):
        example = encode_example(tiny_tokenizer, "Write code.", response,
                                 max_length=512)
        assert example["supervised_tokens"] > 0


def test_truncation_preserves_the_response(tiny_tokenizer):
    example = encode_example(tiny_tokenizer, "context " * 2000, "KEEP_ME",
                             max_length=128)
    assert example["truncated"] is True
    assert example["truncated_prompt_tokens"] > 0
    supervised = "".join(
        tiny_tokenizer.decode(
            example["labels"][example["prompt_tokens"]:]).split())
    # First response token is the input, not a target (see the masking test),
    # so the retained target text starts at "EEP_ME".
    assert "EEP_ME" in supervised, supervised[:80]


def test_oversized_response_is_rejected_not_truncated(tiny_tokenizer):
    with pytest.raises(FormattingError, match="refusing to truncate"):
        encode_example(tiny_tokenizer, "x", "y " * 5000, max_length=128)


def test_tokenizer_without_boundary_tokens_is_refused():
    class Fake:
        def token_to_id(self, name):
            return None

    with pytest.raises(FormattingError, match="refusing to fabricate"):
        encode_example(Fake(), "a", "b")


def test_labels_are_one_token_shifted(tiny_tokenizer):
    example = encode_example(tiny_tokenizer, "Explain.", "RESPONSE")
    assert len(example["input_ids"]) == len(example["labels"])
    assert example["total_tokens"] == len(example["input_ids"])