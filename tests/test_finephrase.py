import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch

from finephrase import FinePhrase
from finephrase.phrase_utils import get_phrase_idx
from finephrase.utils import _build_results_dataframe, move_or_convert_tensors

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
    num_token_jobs = 8
    num_loader_jobs = 4

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
        num_token_jobs=num_token_jobs,
        num_loader_jobs=num_loader_jobs,
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
    assert finephrase.num_token_jobs == num_token_jobs
    assert finephrase.num_loader_jobs == num_loader_jobs

    # Test invalid truncate_dims and pca combination
    with pytest.raises(ValueError):
        FinePhrase(
            model_name=model_name,
            truncate_dims=10,
            pca=50,
        )


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
    converted_results = move_or_convert_tensors(results, move_to_cpu=True)
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
    converted_results = move_or_convert_tensors(
        results, return_tensors="np", move_to_cpu=True
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
        move_or_convert_tensors(results, return_tensors="invalid")


@pytest.fixture
def model():
    return FinePhrase(
        model_name=MODEL_NAME,
        device="cpu",
        amp=False,
        num_token_jobs=1,
        num_loader_jobs=1,
    )


def test_finephrase_to_cpu():
    model_name = MODEL_NAME
    device = "cuda"
    finephrase = FinePhrase(
        model_name=model_name,
        device=device,
    )
    assert finephrase.device.type == "cuda"

    finephrase.to("cpu")
    assert finephrase.device.type == "cpu"


def test_finephrase_to_cuda(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    assert model.device.type == "cpu"

    model.to("cuda")
    assert model.device.type == "cuda"


def test_finephrase_to_device(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    assert model.device.type == "cpu"

    model.to(torch.device("cuda"))
    assert model.device.type == "cuda"

    model.to(torch.device("cpu"))
    assert model.device.type == "cpu"


def test_finephrase_encode(model):
    docs = ["This is a test document.", "Another test document."]
    df, X = model.encode(docs, phrase_sizes=3, max_length=10, batch_size=1)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)


def test_finephrase_encode_multiple_phrase_sizes():
    docs = ["This is a test document.", "Another test document."]
    finephrase = FinePhrase(
        model_name=MODEL_NAME,
        device="cpu",
        amp=False,
        num_token_jobs=1,
        num_loader_jobs=1,
    )
    phrase_sizes = [3, 5]
    df, X = finephrase.encode(
        docs, phrase_sizes=phrase_sizes, max_length=10, batch_size=1
    )
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert all(size in df["phrase_size"].unique() for size in phrase_sizes)


def test_finephrase_encode_queries(model):
    queries = ["What is the capital of France?", "How to bake a cake?"]
    query_embeds = model.encode_queries(queries, max_length=10, batch_size=1)

    assert query_embeds is not None
    assert len(query_embeds) == len(queries)


def test_finephrase_reduce_precision_if_needed():
    model = FinePhrase(model_name=MODEL_NAME, device="cpu", reduce_precision=True)
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = model.reduce_precision_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    model.reduce_precision = False
    non_reduced_embeds = model.reduce_precision_if_needed(embeds)
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


@pytest.mark.parametrize(
    "return_frame, convert_to_numpy",
    [
        ("polars", True),
        ("arrow", True),
        ("polars", False),
        ("teddies", True),
    ],
)
def test_build_results_dataframe(return_frame, convert_to_numpy):
    # Determine the expected dataframe type based on the return_frame parameter
    if return_frame == "polars":
        expected_df_type = pl.DataFrame
    elif return_frame == "arrow":
        expected_df_type = pa.Table

    # Determine the expected embeddings type based on the convert_to_numpy parameter
    expected_embeds_type = torch.Tensor
    if convert_to_numpy:
        expected_embeds_type = np.ndarray

    # Define the results dictionary with sample data
    results = {
        "embed_idx": torch.tensor([0, 1]),
        "sample_idx": torch.tensor([0, 1]),
        "sequence_idx": torch.tensor([0, 1]),
        "batch_idx": torch.tensor([0, 1]),
        "phrase_size": torch.tensor([3, 3]),
        "phrase": ["phrase1", "phrase2"],
        "phrase_embeds": torch.randn(2, 10),
    }

    # Test for invalid return_frame value
    if return_frame == "teddies":
        with pytest.raises(ValueError, match="Invalid value for"):
            _build_results_dataframe(results, return_frame, convert_to_numpy)
    else:
        # Build the results dataframe and check the types
        expected_length = len(results["sample_idx"])
        df, embeds = _build_results_dataframe(results, return_frame, convert_to_numpy)
        assert isinstance(df, expected_df_type)
        assert isinstance(embeds, expected_embeds_type)
        assert len(df) == len(embeds) == expected_length


def test_build_results_dataframe_pandas():
    pd = pytest.importorskip("pandas")
    return_frame = "pandas"
    convert_to_numpy = True

    expected_df_type = pd.DataFrame
    expected_embeds_type = np.ndarray

    results = {
        "embed_idx": torch.tensor([0, 1]),
        "sample_idx": torch.tensor([0, 1]),
        "sequence_idx": torch.tensor([0, 1]),
        "batch_idx": torch.tensor([0, 1]),
        "phrase_size": torch.tensor([3, 3]),
        "phrases": ["phrase1", "phrase2"],
        "phrase_embeds": torch.randn(2, 10),
    }

    expected_length = len(results["sample_idx"])
    df, embeds = _build_results_dataframe(results, return_frame, convert_to_numpy)
    assert isinstance(df, expected_df_type)
    assert isinstance(embeds, expected_embeds_type)
    assert len(df) == len(embeds) == expected_length
