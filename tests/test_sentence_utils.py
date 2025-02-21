import pytest
import torch

from finephrase.sentence_utils import (
    _add_special_tokens,
    _pad,
    _split_long_sentences,
    get_sentence_offsets,
    get_sentence_offsets_blingfire,
    get_sentence_offsets_nltk,
    get_sentence_offsets_syntok,
)


def test_get_sentence_offsets_syntok_returns_tensor():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_syntok(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_syntok_non_empty():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_syntok(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_syntok_empty_text():
    pytest.importorskip("syntok")
    text = ""
    offsets = get_sentence_offsets_syntok(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_blingfire_returns_tensor():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_blingfire(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_blingfire_non_empty():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_blingfire(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_blingfire_empty_text():
    text = ""
    offsets = get_sentence_offsets_blingfire(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_nltk_returns_tensor():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_nltk(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_nltk_non_empty():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_nltk(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_nltk_empty_text():
    pytest.importorskip("nltk")
    text = ""
    offsets = get_sentence_offsets_nltk(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_blingfire_method():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="blingfire")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_nltk_method():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="nltk")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_syntok_method():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="syntok")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_invalid_method():
    text = "Hello world. This is a test."
    with pytest.raises(ValueError, match="Invalid method: 'invalid'"):
        get_sentence_offsets(text, method="invalid")


def test_get_sentence_offsets_parallel():
    texts = ["Hello world. This is a test.", "Another sentence. And another one."]
    offsets = get_sentence_offsets(texts, method="blingfire", n_jobs=2)
    # Check that the result is a list of tensors
    assert isinstance(offsets, list)
    assert all(isinstance(o, torch.Tensor) for o in offsets)
    assert all(o.ndim == 2 and o.size(1) == 2 for o in offsets)


def test_add_special_tokens_no_special_tokens():
    input_ids = torch.tensor([1, 2, 3, 4])
    result = _add_special_tokens(input_ids)
    expected = torch.tensor([1, 2, 3, 4])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_cls_token():
    input_ids = torch.tensor([1, 2, 3, 4])
    cls_token_id = 101
    result = _add_special_tokens(input_ids, cls_token_id=cls_token_id)
    expected = torch.tensor([101, 1, 2, 3, 4])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_sep_token():
    input_ids = torch.tensor([1, 2, 3, 4])
    sep_token_id = 102
    result = _add_special_tokens(input_ids, sep_token_id=sep_token_id)
    expected = torch.tensor([1, 2, 3, 4, 102])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_cls_and_sep_tokens():
    input_ids = torch.tensor([1, 2, 3, 4])
    cls_token_id = 101
    sep_token_id = 102
    result = _add_special_tokens(
        input_ids, cls_token_id=cls_token_id, sep_token_id=sep_token_id
    )
    expected = torch.tensor([101, 1, 2, 3, 4, 102])
    assert torch.equal(result, expected)


def test_pad_longest():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    padded = _pad(input_ids, pad_token_id, strategy="longest")
    expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    assert torch.equal(padded, expected)


def test_pad_max_length():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    max_length = 4
    padded = _pad(input_ids, pad_token_id, strategy="max_length", max_length=max_length)
    expected = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 0, 0, 0]])
    assert torch.equal(padded, expected)


def test_pad_no_padding():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    padded = _pad(input_ids, pad_token_id, strategy=None)
    assert padded == input_ids


def test_pad_empty_input():
    input_ids = []
    pad_token_id = 0
    with pytest.raises(ValueError, match="Input list must not be empty."):
        _pad(input_ids, pad_token_id)


def test_pad_max_length_exceeds():
    input_ids = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6])]
    pad_token_id = 0
    max_length = 3
    with pytest.raises(
        ValueError, match=r"Input sequence length \d+ exceeds `max_length`."
    ):
        _pad(input_ids, pad_token_id, strategy="max_length", max_length=max_length)


def test_pad_invalid_strategy():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    pad_token_id = 0
    with pytest.raises(ValueError, match="Invalid value 'invalid' for `strategy`."):
        _pad(input_ids, pad_token_id, strategy="invalid")


def test_split_long_sentences_no_split_needed():
    sentence_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    max_length = 3
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_split_needed():
    sentence_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_multiple_splits():
    sentence_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_with_padding():
    sentence_ids = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 2, 2, 3, -1, -1], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_empty_input():
    sentence_ids = torch.tensor([], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([], dtype=torch.int32)
    assert torch.equal(result, expected)
