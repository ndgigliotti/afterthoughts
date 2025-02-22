import polars as pl
import pytest
import torch
from transformers import AutoTokenizer

from finephrase.tokenize import (
    TokenizedDataset,
    _tokenize_batch,
    dynamic_pad_collate,
    get_max_length,
    pad,
    tokenize_docs,
)

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def test_pad_longest():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    padded = pad(input_ids, pad_token_id, strategy="longest")
    expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    assert torch.equal(padded, expected)


def test_pad_max_length():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    max_length = 4
    padded = pad(input_ids, pad_token_id, strategy="max_length", max_length=max_length)
    expected = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 0, 0, 0]])
    assert torch.equal(padded, expected)


def test_pad_no_padding():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    pad_token_id = 0
    padded = pad(input_ids, pad_token_id, strategy=None)
    assert padded == input_ids


def test_pad_empty_input():
    input_ids = []
    pad_token_id = 0
    with pytest.raises(ValueError, match="Input list must not be empty."):
        pad(input_ids, pad_token_id)


def test_pad_max_length_exceeds():
    input_ids = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6])]
    pad_token_id = 0
    max_length = 3
    with pytest.raises(
        ValueError, match=r"Input sequence length \d+ exceeds `max_length`."
    ):
        pad(input_ids, pad_token_id, strategy="max_length", max_length=max_length)


def test_pad_invalid_strategy():
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    pad_token_id = 0
    with pytest.raises(ValueError, match="Invalid value 'invalid' for `strategy`."):
        pad(input_ids, pad_token_id, strategy="invalid")


def test_dynamic_pad_collate():
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
        {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
        {"input_ids": torch.tensor([6]), "attention_mask": torch.tensor([1])},
    ]
    pad_token_id = 0
    collated = dynamic_pad_collate(batch, pad_token_id)
    expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    assert torch.equal(collated["input_ids"], expected_input_ids)
    assert torch.equal(collated["attention_mask"], expected_attention_mask)


def test_dynamic_pad_collate_empty_batch():
    batch = []
    pad_token_id = 0
    collated = dynamic_pad_collate(batch, pad_token_id)
    assert collated == {}


def test_get_max_length_with_max_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 128
    result = get_max_length(max_length, tokenizer)
    assert result == max_length


def test_get_max_length_without_max_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = get_max_length(None, tokenizer)
    assert result == tokenizer.model_max_length


def test_get_max_length_required_with_max_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 128
    result = get_max_length(max_length, tokenizer, required=True)
    assert result == max_length


def test_get_max_length_required_without_max_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = get_max_length(None, tokenizer, required=True)
    assert result == tokenizer.model_max_length


def test_get_max_length_required_without_model_max_length():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = None
    with pytest.raises(
        ValueError,
        match="The `max_length` parameter must be specified if the tokenizer does not have a `model_max_length`.",
    ):
        get_max_length(None, tokenizer, required=True)


def test_tokenized_dataset_init():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data)
    assert len(dataset) == 3
    assert dataset.keys() == ["input_ids", "attention_mask"]


def test_tokenized_dataset_validate_data():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data)
    dataset.validate_data()  # Should not raise any exceptions


def test_tokenized_dataset_invalid_data():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
        ],  # Different length
    }
    with pytest.raises(
        ValueError, match="All lists in the data must have the same length."
    ):
        TokenizedDataset(data)


def test_tokenized_dataset_get_sort_idx():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data)
    sort_idx = dataset.get_sort_idx()
    assert sort_idx.tolist() == [0, 1, 2]


def test_tokenized_dataset_getitem():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data)
    item = dataset[0]
    assert torch.equal(item["input_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(item["attention_mask"], torch.tensor([1, 1, 1]))


def test_tokenized_dataset_sorting():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data, sort_by_token_count=True)
    sorted_data = [dataset[i] for i in range(len(dataset))]
    assert torch.equal(sorted_data[0]["input_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(sorted_data[1]["input_ids"], torch.tensor([4, 5]))
    assert torch.equal(sorted_data[2]["input_ids"], torch.tensor([6]))


def test_tokenized_dataset_unsorted():
    data = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])],
        "attention_mask": [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1]),
        ],
    }
    dataset = TokenizedDataset(data, sort_by_token_count=False)
    unsorted_data = [dataset[i] for i in range(len(dataset))]
    assert torch.equal(unsorted_data[0]["input_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(unsorted_data[1]["input_ids"], torch.tensor([4, 5]))
    assert torch.equal(unsorted_data[2]["input_ids"], torch.tensor([6]))


def test_tokenize_batch_basic():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(docs, sample_idx, tokenizer)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 2
    assert len(result["attention_mask"]) == 2


def test_tokenize_batch_max_length():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 5
    result = _tokenize_batch(docs, sample_idx, tokenizer, max_length=max_length)
    assert all(len(ids) <= max_length for ids in result["input_ids"])


def test_tokenize_batch_padding():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(
        docs, sample_idx, tokenizer, padding="max_length", max_length=10
    )
    assert all(len(ids) == 10 for ids in result["input_ids"])


def test_tokenize_batch_truncation():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(docs, sample_idx, tokenizer, truncation=True, max_length=3)
    assert all(len(ids) == 3 for ids in result["input_ids"])


def test_tokenize_batch_overflowing_tokens():
    docs = [
        "This is a longer example with more text to test the tokenizer's "
        "ability to handle overflow.",
        "This is another test with even more words to see how the tokenizer manages "
        "longer sequences and overflow tokens.",
    ]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(
        docs,
        sample_idx,
        tokenizer,
        max_length=10,
        return_overflowing_tokens=True,
        stride=2,
    )
    assert "overflow_to_sample_mapping" in result


def test_tokenize_batch_return_tensors():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(
        docs, sample_idx, tokenizer, return_tensors="pt", padding=True
    )
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)


def test_tokenize_batch_add_special_tokens():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(docs, sample_idx, tokenizer, add_special_tokens=False)
    set_input_ids = {y for x in result["input_ids"] for y in x}
    for special_id in tokenizer.all_special_ids:
        assert (
            special_id not in set_input_ids
        ), f"Special token {special_id} found in input_ids"


def test_tokenize_batch_return_attention_mask():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(docs, sample_idx, tokenizer, return_attention_mask=True)
    assert "attention_mask" in result


def test_tokenize_batch_return_offsets_mapping():
    docs = ["Hello world", "This is a test"]
    sample_idx = [0, 1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = _tokenize_batch(docs, sample_idx, tokenizer, return_offsets_mapping=True)
    assert "offset_mapping" in result
    assert isinstance(result["offset_mapping"][0], list)
    assert isinstance(result["offset_mapping"][0][0], tuple)


def test_tokenize_docs_basic():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 2
    assert len(result["attention_mask"]) == 2


def test_tokenize_docs_max_length():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 5
    result = tokenize_docs(docs, tokenizer, max_length=max_length)
    assert all(len(ids) <= max_length for ids in result["input_ids"])


def test_tokenize_docs_truncation():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer, max_length=3, truncation=True)
    assert all(len(ids) == 3 for ids in result["input_ids"])


def test_tokenize_docs_overflowing_tokens():
    docs = [
        "This is a longer example with more text to test the tokenizer's ability to handle overflow.",
        "This is another test with even more words to see how the tokenizer manages longer sequences and overflow tokens.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer, max_length=10, chunk_docs=True, overlap=2)
    assert "sequence_idx" in result


def test_tokenize_docs_return_attention_mask():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer, return_attention_mask=True)
    assert "attention_mask" in result


def test_tokenize_docs_return_offsets_mapping():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer, return_offsets_mapping=True)
    assert "offset_mapping" in result
    assert isinstance(result["offset_mapping"][0], list)
    assert isinstance(result["offset_mapping"][0][0], tuple)


def test_tokenize_docs_return_tokenized_dataset():
    docs = ["Hello world", "This is a test"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    result = tokenize_docs(docs, tokenizer, return_tokenized_dataset=True)
    assert isinstance(result, TokenizedDataset)
    assert len(result) == 2
    assert result[0]["input_ids"].shape == (6,)
    assert result[0]["attention_mask"].shape == (6,)
