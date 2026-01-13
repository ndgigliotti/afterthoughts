import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from finephrase import FinePhrase, FinePhraseLite
from finephrase.sentence_utils import get_segment_idx
from finephrase.utils import _build_results_dataframe, move_or_convert_tensors

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def test_finephrase_init():
    """Test basic FinePhrase initialization."""
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    normalize_embeds = True
    device = "cuda"
    num_token_jobs = 8

    finephrase = FinePhrase(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        normalize_embeds=normalize_embeds,
        device=device,
        num_token_jobs=num_token_jobs,
    )

    assert finephrase.tokenizer is not None
    assert finephrase.model is not None
    assert finephrase.amp == amp
    assert finephrase.amp_dtype == amp_dtype
    assert finephrase.normalize_embeds == normalize_embeds
    assert finephrase.device.type == device
    assert finephrase.num_token_jobs == num_token_jobs


def test_finephrase_lite_init():
    """Test FinePhraseLite initialization with lossy params."""
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

    finephrase = FinePhraseLite(
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

    # Test invalid truncate_dims and pca combination
    with pytest.raises(ValueError):
        FinePhraseLite(
            model_name=model_name,
            truncate_dims=10,
            pca=50,
        )


def test_get_segment_idx_single_size():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    segment_sizes = 2
    overlap = 0.5

    result = get_segment_idx(input_ids, sentence_ids, segment_sizes, overlap)

    assert "segment_token_idx" in result
    assert "segment_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "segment_size" in result


def test_get_segment_idx_multiple_sizes():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    segment_sizes = [1, 2]
    overlap = 0.5

    result = get_segment_idx(input_ids, sentence_ids, segment_sizes, overlap)

    assert "segment_token_idx" in result
    assert "segment_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "segment_size" in result


def test_move_or_convert_results_to_cpu():
    results = {
        "segment_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "segment_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
    }
    expected_results = {
        "segment_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cpu"),
        "sequence_idx": torch.tensor([0, 1], device="cpu"),
        "segment_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cpu"),
    }
    converted_results = move_or_convert_tensors(results, move_to_cpu=True)
    for key in results:
        assert torch.equal(converted_results[key], expected_results[key])


def test_move_or_convert_results_to_numpy():
    results = {
        "segment_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "segment_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
    }
    expected_results = {
        "segment_token_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": np.array([0, 1]),
        "segment_embeds": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = move_or_convert_tensors(results, return_tensors="np", move_to_cpu=True)
    for key in results:
        np.testing.assert_allclose(converted_results[key], expected_results[key])


def test_move_or_convert_results_invalid_return_tensors():
    results = {
        "segment_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"),
        "sequence_idx": torch.tensor([0, 1], device="cuda"),
        "segment_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda"),
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
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    df, X = model.encode(docs, segment_sizes=1, max_length=64, batch_max_tokens=256)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert "segment" in df.columns
    assert "segment_size" in df.columns


def test_finephrase_encode_multiple_segment_sizes():
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    finephrase = FinePhrase(
        model_name=MODEL_NAME,
        device="cpu",
        amp=False,
        num_token_jobs=1,
    )
    segment_sizes = [1, 2]
    df, X = finephrase.encode(
        docs, segment_sizes=segment_sizes, max_length=64, batch_max_tokens=256
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert all(size in df["segment_size"].unique() for size in segment_sizes)


def test_finephrase_encode_queries(model):
    queries = ["What is the capital of France?", "How to bake a cake?"]
    query_embeds = model.encode_queries(queries, max_length=10, batch_size=1)

    assert query_embeds is not None
    assert len(query_embeds) == len(queries)


def test_finephrase_lite_reduce_precision_if_needed():
    model = FinePhraseLite(model_name=MODEL_NAME, device="cpu", reduce_precision=True)
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = model.reduce_precision_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    model.reduce_precision = False
    non_reduced_embeds = model.reduce_precision_if_needed(embeds)
    assert non_reduced_embeds.dtype == torch.float32


def test_finephrase_lite_truncate_dims_if_needed():
    finephrase = FinePhraseLite(model_name=MODEL_NAME, device="cpu", truncate_dims=5)
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
        ("pandas", True),
        ("polars", True),
        ("arrow", True),
        ("polars", False),
        ("teddies", True),
    ],
)
def test_build_results_dataframe(return_frame, convert_to_numpy):
    # Determine the expected dataframe type based on the return_frame parameter
    if return_frame == "pandas":
        expected_df_type = pd.DataFrame
    elif return_frame == "polars":
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
        "segment_size": torch.tensor([2, 2]),
        "segment": ["segment1", "segment2"],
        "segment_embeds": torch.randn(2, 10),
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
        "segment_size": torch.tensor([2, 2]),
        "segment": ["segment1", "segment2"],
        "segment_embeds": torch.randn(2, 10),
    }

    expected_length = len(results["sample_idx"])
    df, embeds = _build_results_dataframe(results, return_frame, convert_to_numpy)
    assert isinstance(df, expected_df_type)
    assert isinstance(embeds, expected_embeds_type)
    assert len(df) == len(embeds) == expected_length
