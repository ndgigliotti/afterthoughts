import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from afterthoughts import Encoder, LiteEncoder
from afterthoughts.chunk import get_chunk_idx
from afterthoughts.utils import _build_results_dataframe, move_or_convert_tensors

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
def test_encoder_init():
    """Test basic Encoder initialization."""
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    normalize = True
    device = "cuda"
    _num_token_jobs = 8

    encoder = Encoder(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        normalize=normalize,
        device=device,
        _num_token_jobs=_num_token_jobs,
    )

    assert encoder.tokenizer is not None
    assert encoder.model is not None
    assert encoder.amp == amp
    assert encoder.amp_dtype == amp_dtype
    assert encoder.normalize == normalize
    assert encoder.device.type == device
    assert encoder._num_token_jobs == _num_token_jobs


@requires_cuda
def test_encoder_lite_init():
    """Test LiteEncoder initialization with lossy params."""
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    half_embeds = True
    truncate_dims = 128
    normalize = True
    pca = 50
    pca_early_stop = 1.0
    device = "cuda"
    _num_token_jobs = 8

    encoder = LiteEncoder(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        half_embeds=half_embeds,
        truncate_dims=truncate_dims,
        normalize=normalize,
        pca=pca,
        pca_early_stop=pca_early_stop,
        device=device,
        _num_token_jobs=_num_token_jobs,
    )

    assert encoder.tokenizer is not None
    assert encoder.model is not None
    assert encoder.amp == amp
    assert encoder.amp_dtype == amp_dtype
    assert encoder.half_embeds == half_embeds
    assert encoder.truncate_dims == truncate_dims
    assert encoder.normalize == normalize
    assert encoder.pca == pca
    assert encoder.pca_early_stop == pca_early_stop
    assert encoder.device.type == device
    assert encoder._num_token_jobs == _num_token_jobs

    # Test invalid truncate_dims and pca combination
    with pytest.raises(ValueError):
        LiteEncoder(
            model_name=model_name,
            truncate_dims=10,
            pca=50,
        )


def test_get_chunk_idx_single_size():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    num_sents = 2
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, num_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "chunk_size" in result


def test_get_chunk_idx_multiple_sizes():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    num_sents = [1, 2]
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, num_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "chunk_size" in result


def test_move_or_convert_results_to_cpu():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected_results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = move_or_convert_tensors(results, move_to_cpu=True)
    for key in results:
        assert torch.equal(converted_results[key], expected_results[key])


def test_move_or_convert_results_to_numpy():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected_results = {
        "chunk_token_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": np.array([0, 1]),
        "chunk_embeds": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = move_or_convert_tensors(results, return_tensors="np", move_to_cpu=True)
    for key in results:
        np.testing.assert_allclose(converted_results[key], expected_results[key])


def test_move_or_convert_results_invalid_return_tensors():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    with pytest.raises(ValueError):
        move_or_convert_tensors(results, return_tensors="invalid")


@pytest.fixture
def model():
    return Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        amp=False,
        _num_token_jobs=1,
    )


@requires_cuda
def test_encoder_to_cpu():
    model_name = MODEL_NAME
    device = "cuda"
    encoder = Encoder(
        model_name=model_name,
        device=device,
    )
    assert encoder.device.type == "cuda"

    encoder.to("cpu")
    assert encoder.device.type == "cpu"


@requires_cuda
def test_encoder_to_cuda(model):
    assert model.device.type == "cpu"

    model.to("cuda")
    assert model.device.type == "cuda"


@requires_cuda
def test_encoder_to_device(model):
    assert model.device.type == "cpu"

    model.to(torch.device("cuda"))
    assert model.device.type == "cuda"

    model.to(torch.device("cpu"))
    assert model.device.type == "cpu"


def test_encoder_encode(model):
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    df, X = model.encode(docs, num_sents=1, max_length=64, batch_tokens=256)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert "chunk" in df.columns
    assert "chunk_size" in df.columns


def test_encoder_encode_multiple_num_sents():
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    encoder = Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        amp=False,
        _num_token_jobs=1,
    )
    num_sents = [1, 2]
    df, X = encoder.encode(docs, num_sents=num_sents, max_length=64, batch_tokens=256)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert all(size in df["chunk_size"].unique() for size in num_sents)


def test_encoder_encode_queries(model):
    queries = ["What is the capital of France?", "How to bake a cake?"]
    query_embeds = model.encode_queries(queries, max_length=10, batch_size=1)

    assert query_embeds is not None
    assert len(query_embeds) == len(queries)


def test_encoder_lite_half_embeds_if_needed():
    model = LiteEncoder(model_name=MODEL_NAME, device="cpu", half_embeds=True)
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = model.half_embeds_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    model.half_embeds = False
    non_reduced_embeds = model.half_embeds_if_needed(embeds)
    assert non_reduced_embeds.dtype == torch.float32


def test_encoder_lite_truncate_dims_if_needed():
    encoder = LiteEncoder(model_name=MODEL_NAME, device="cpu", truncate_dims=5)
    embeds = torch.randn(10, 10)
    truncated_embeds = encoder.truncate_dims_if_needed(embeds)
    assert truncated_embeds.shape[1] == 5

    encoder.truncate_dims = None
    non_truncated_embeds = encoder.truncate_dims_if_needed(embeds)
    assert non_truncated_embeds.shape[1] == 10


def test_encoder_normalize_if_needed():
    encoder = Encoder(model_name=MODEL_NAME, device="cpu", normalize=True)
    embeds = torch.randn(10, 10)
    normalized_embeds = encoder.normalize_if_needed(embeds)
    norms = torch.norm(normalized_embeds, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    encoder.normalize = False
    non_normalized_embeds = encoder.normalize_if_needed(embeds)
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
        "chunk_size": torch.tensor([2, 2]),
        "chunk": ["segment1", "segment2"],
        "chunk_embeds": torch.randn(2, 10),
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
        "chunk_size": torch.tensor([2, 2]),
        "chunk": ["segment1", "segment2"],
        "chunk_embeds": torch.randn(2, 10),
    }

    expected_length = len(results["sample_idx"])
    df, embeds = _build_results_dataframe(results, return_frame, convert_to_numpy)
    assert isinstance(df, expected_df_type)
    assert isinstance(embeds, expected_embeds_type)
    assert len(df) == len(embeds) == expected_length
