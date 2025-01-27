import numpy as np
import pytest
import torch

from finephrase import FinePhrase
from finephrase.finephrase import (
    TokenizedDataset,
    _move_or_convert_results,
    get_phrase_idx,
)

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def test_finephrase_init():
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    reduce_precision = True
    truncate_dims = 128
    normalize_embeds = True
    pca = 50
    pca_fit_batch_count = 1.0
    device = "cuda"

    finephrase = FinePhrase(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        reduce_precision=reduce_precision,
        truncate_dims=truncate_dims,
        normalize_embeds=normalize_embeds,
        pca=pca,
        pca_fit_batch_count=pca_fit_batch_count,
        device=device,
    )

    assert finephrase.tokenizer is not None
    assert finephrase.model is not None
    assert finephrase.amp == amp
    assert finephrase.amp_dtype == amp_dtype
    assert finephrase.reduce_precision == reduce_precision
    assert finephrase.truncate_dims == truncate_dims
    assert finephrase.normalize_embeds == normalize_embeds
    assert finephrase.pca == pca
    assert finephrase.pca_fit_batch_count == pca_fit_batch_count
    assert finephrase.device.type == device

    # Test invalid truncate_dims and pca combination
    with pytest.raises(ValueError):
        FinePhrase(
            model_name=model_name,
            truncate_dims=10,
            pca=50,
        )


def test_tokenized_dataset_init():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    dataset = TokenizedDataset(inputs)
    assert len(dataset) == 2
    assert torch.equal(dataset.inputs["input_ids"], inputs["input_ids"])
    assert torch.equal(dataset.inputs["attention_mask"], inputs["attention_mask"])


def test_tokenized_dataset_shuffle():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    dataset = TokenizedDataset(inputs, shuffle=True)


def test_tokenized_dataset_indexing():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    dataset = TokenizedDataset(inputs)
    idx, data = dataset[0]
    assert idx == 0
    assert torch.equal(data["input_ids"], inputs["input_ids"][0])
    assert torch.equal(data["attention_mask"], inputs["attention_mask"][0])

    idx, data = dataset[1]
    assert idx == 1
    assert torch.equal(data["input_ids"], inputs["input_ids"][1])
    assert torch.equal(data["attention_mask"], inputs["attention_mask"][1])


def test_tokenized_dataset_shuffle_indexing():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    dataset = TokenizedDataset(inputs, shuffle=True, random_state=42)
    idx, data = dataset[0]
    assert idx in [0, 1]
    assert torch.equal(data["input_ids"], inputs["input_ids"][idx])
    assert torch.equal(data["attention_mask"], inputs["attention_mask"][idx])

    idx, data = dataset[1]
    assert idx in [0, 1]
    assert torch.equal(data["input_ids"], inputs["input_ids"][idx])
    assert torch.equal(data["attention_mask"], inputs["attention_mask"][idx])


def test_get_phrase_idx_single_size():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    phrase_sizes = 3
    overlap = 0.5
    phrase_min_token_ratio = 0.5

    result = get_phrase_idx(
        input_ids, attention_mask, phrase_sizes, overlap, phrase_min_token_ratio
    )

    assert "phrase_idx" in result
    assert "phrase_ids" in result
    assert "valid_phrase_mask" in result
    assert "sequence_idx" in result
    assert len(result["phrase_idx"]) == 1
    assert len(result["phrase_ids"]) == 1
    assert len(result["valid_phrase_mask"]) == 1
    assert len(result["sequence_idx"]) == 1


def test_get_phrase_idx_multiple_sizes():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    phrase_sizes = [3, 5]
    overlap = 0.5
    phrase_min_token_ratio = 0.5

    result = get_phrase_idx(
        input_ids, attention_mask, phrase_sizes, overlap, phrase_min_token_ratio
    )

    assert "phrase_idx" in result
    assert "phrase_ids" in result
    assert "valid_phrase_mask" in result
    assert "sequence_idx" in result
    assert len(result["phrase_idx"]) == 2
    assert len(result["phrase_ids"]) == 2
    assert len(result["valid_phrase_mask"]) == 2
    assert len(result["sequence_idx"]) == 2


def test_get_phrase_idx_with_sequence_idx():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    phrase_sizes = 3
    overlap = 0.5
    phrase_min_token_ratio = 0.5
    sequence_idx = torch.tensor([0])

    result = get_phrase_idx(
        input_ids,
        attention_mask,
        phrase_sizes,
        overlap,
        phrase_min_token_ratio,
        sequence_idx,
    )

    assert "phrase_idx" in result
    assert "phrase_ids" in result
    assert "valid_phrase_mask" in result
    assert "sequence_idx" in result
    assert len(result["phrase_idx"]) == 1
    assert len(result["phrase_ids"]) == 1
    assert len(result["valid_phrase_mask"]) == 1
    assert len(result["sequence_idx"]) == 1


def test_get_phrase_idx_invalid_overlap():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    phrase_sizes = 3
    overlap = 1.0
    phrase_min_token_ratio = 0.5

    with pytest.raises(ValueError):
        get_phrase_idx(
            input_ids, attention_mask, phrase_sizes, overlap, phrase_min_token_ratio
        )


def test_get_phrase_idx_invalid_phrase_min_token_ratio():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    phrase_sizes = 3
    overlap = 0.5
    phrase_min_token_ratio = 0.0

    with pytest.raises(ValueError):
        get_phrase_idx(
            input_ids, attention_mask, phrase_sizes, overlap, phrase_min_token_ratio
        )


def test_move_or_convert_results_to_cpu():
    results = {
        "phrase_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "phrase_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
    }
    expected_results = {
        "phrase_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cpu"),
        "sequence_idx": torch.tensor([0, 1], device="cpu"),
        "phrase_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cpu"),
    }
    converted_results = _move_or_convert_results(results, move_results_to_cpu=True)
    for key in results:
        assert torch.equal(converted_results[key], expected_results[key])


def test_move_or_convert_results_to_numpy():
    results = {
        "phrase_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "phrase_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
    }
    expected_results = {
        "phrase_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": np.array([0, 1]),
        "phrase_embeds": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = _move_or_convert_results(
        results, return_tensors="np", move_results_to_cpu=True
    )
    for key in results:
        np.testing.assert_allclose(converted_results[key], expected_results[key])


def test_move_or_convert_results_invalid_return_tensors():
    results = {
        "phrase_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "phrase_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
    }
    with pytest.raises(ValueError):
        _move_or_convert_results(results, return_tensors="invalid")


def test_finephrase_to_cpu():
    model_name = MODEL_NAME
    device = "cuda"
    finephrase = FinePhrase(model_name=model_name, device=device, amp=False)
    assert finephrase.device.type == "cuda"

    finephrase.to("cpu")
    assert finephrase.device.type == "cpu"


def test_finephrase_to_cuda():
    model_name = MODEL_NAME
    device = "cpu"
    finephrase = FinePhrase(model_name=model_name, device=device, amp=False)
    assert finephrase.device.type == "cpu"

    finephrase.to("cuda")
    assert finephrase.device.type == "cuda"


def test_finephrase_to_device():
    model_name = MODEL_NAME
    device = "cpu"
    finephrase = FinePhrase(model_name=model_name, device=device, amp=False)
    assert finephrase.device.type == "cpu"

    finephrase.to(torch.device("cuda"))
    assert finephrase.device.type == "cuda"

    finephrase.to(torch.device("cpu"))
    assert finephrase.device.type == "cpu"


def test_finephrase_encode():
    docs = ["This is a test document.", "Another test document."]
    finephrase = FinePhrase(model_name=MODEL_NAME, device="cpu", amp=False)
    results = finephrase.encode(docs, phrase_sizes=3, max_length=10, batch_size=1)

    assert "sample_idx" in results
    assert "phrases" in results
    assert "phrase_embeds" in results
    assert len(results["sample_idx"]) > 0
    assert len(results["phrases"]) > 0
    assert len(results["phrase_embeds"]) > 0


def test_finephrase_encode_queries():
    queries = ["What is the capital of France?", "How to bake a cake?"]
    finephrase = FinePhrase(model_name=MODEL_NAME, device="cpu", amp=False)
    query_embeds = finephrase.encode_queries(queries, max_length=10, batch_size=1)

    assert query_embeds is not None
    assert len(query_embeds) == len(queries)


def test_finephrase_reduce_precision_if_needed():
    finephrase = FinePhrase(model_name=MODEL_NAME, device="cpu", reduce_precision=True)
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = finephrase.reduce_precision_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    finephrase.reduce_precision = False
    non_reduced_embeds = finephrase.reduce_precision_if_needed(embeds)
    assert non_reduced_embeds.dtype == torch.float32


def test_finephrase_truncate_dims_if_needed():
    finephrase = FinePhrase(model_name=MODEL_NAME, device="cpu", truncate_dims=5)
    embeds = torch.randn(10, 10)
    truncated_embeds = finephrase.truncate_dims_if_needed(embeds)
    assert truncated_embeds.shape[1] == 5

    finephrase.truncate_dims = None
    non_truncated_embeds = finephrase.truncate_dims_if_needed(embeds)
    assert non_truncated_embeds.shape[1] == 10


def test_finephrase_normalize_if_needed():
    finephrase = FinePhrase(model_name=MODEL_NAME, device="cpu", normalize_embeds=True)
    embeds = torch.randn(10, 10)
    normalized_embeds = finephrase.normalize_if_needed(embeds)
    norms = torch.norm(normalized_embeds, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    finephrase.normalize_embeds = False
    non_normalized_embeds = finephrase.normalize_if_needed(embeds)
    norms = torch.norm(non_normalized_embeds, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
